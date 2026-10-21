//! The risk-guided harvest (docs/m3-hybrid/o3-rounds.md): the leaf-risk model G2
//! (`nn::leafrisk`) chooses which frontier positions a harvest labels, so the new training rows
//! go where the network O_k is wrong, in proportion to how often each kind of position occurs
//! and how wrong it is there.
//!
//! **The method** (Yoo & Kweon 2019, "Learning Loss for Active Learning": a second model predicts
//! the main model's loss and the examples it predicts worst are labelled next; Schaul et al.
//! 2016, prioritized replay: sample with probability ∝ priority^α; Katharopoulos & Fleuret 2018,
//! importance sampling by the loss; Hesterberg 1995, defensive importance sampling: a share drawn
//! the plain way so a wrong predictor cannot starve any kind of position):
//! 1. **Pilot** ([`pilot_scores`]): the harvest's own search on `decisions` decisions of a seed of
//!    their own, every distinct reached position scored by the model (no labels).
//! 2. **Risk profile** ([`build_profile`]): the scores cut into `bands` bands of equal count (by
//!    the scores' quantiles, lowest risk first); each band's natural share `n_b / N` and its mean
//!    predicted squared error `r_b`.
//! 3. **Target mix**: band `b`'s share of the new rows is
//!    `(1 − f) · n_b · r_b^α / Σ_c n_c · r_c^α + f · n_b / N`
//!    (`f` the plain floor, `α` the risk power in quarters: 0, 0.25, 0.5, 0.75 or 1; α = 1 gives
//!    each band its share of the predicted total error, α = 0.5 is Neyman allocation).
//! 4. **Acceptance** before labelling: a reached position of band `b` is kept with probability
//!    `a_b ∝ target_b / n_b`, scaled so the most favoured band keeps every position; the coin is
//!    the decision's own random stream, so the rows are a pure function of the seed, the profile
//!    and the model, on any machine and thread count. The accepted positions are then sampled up
//!    to the harvest's cap per decision (variety), as every harvest is.
//!
//! Everything is whole numbers: the scores are the model's integers, the powers integer square
//! roots, the shares and acceptances `u128` arithmetic, the acceptance in units of 2^-32.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use engine::game::HandStartState;
use engine::rng::Rng;
use nn::leafrisk::RiskModel;
use serde_json::{Value, json};
use solver::SolveMem;

use crate::oracle_labels::{Driver, GOLDEN_SEED, HarvestJob, harvest_plan};
use crate::pimc::SharedLeaf;
use crate::worlds::BeliefSource;

/// The profile file's format.
pub const PROFILE_FORMAT: &str = "41bots-risk-profile-v1";

/// Seed label of a pilot's decisions (never the training decisions').
const RISK_PILOT: u64 = 0x2154_5050;

/// Acceptance 1 (every position of the band kept), in units of 2^-32.
pub const ACCEPT_ONE: u64 = 1 << 32;

/// The default number of bands.
pub const RISK_BANDS: usize = 20;

/// The default plain floor, parts per million.
pub const RISK_FLOOR_PPM: u64 = 200_000;

/// The default pilot decisions.
pub const RISK_PILOT_DECISIONS: u64 = 2000;

/// A risk profile: the bands, what the pilot saw in each and each band's acceptance.
#[derive(Clone, Debug, PartialEq)]
pub struct RiskProfile {
    /// The model's SHA-256.
    pub model_sha256: String,
    /// The network the model was made for (and whose values the harvest's search uses).
    pub net_sha256: String,
    /// α in quarters (0..=4).
    pub power_q: u8,
    /// The plain floor `f`, parts per million.
    pub floor_ppm: u64,
    /// Upper score of bands `0..B−1` (ascending); band `B − 1` is everything above.
    pub edges: Vec<i64>,
    /// Pilot positions in each band.
    pub counts: Vec<u64>,
    /// Mean predicted squared error in each band (units of 2^-40, rounded down).
    pub mean: Vec<i64>,
    /// Each band's acceptance (units of 2^-32; [`ACCEPT_ONE`] keeps every position).
    pub accept: Vec<u64>,
    /// The pilot's seed and decisions (reports).
    pub pilot_seed: u64,
    pub pilot_decisions: u64,
}

/// Parse α: `0`, `0.25`, `0.5`, `0.75` or `1` (quarters, so its powers are integer roots).
pub fn parse_power(s: &str) -> Result<u8, String> {
    match s.trim() {
        "0" | "0.0" => Ok(0),
        "0.25" => Ok(1),
        "0.5" | "0.50" => Ok(2),
        "0.75" => Ok(3),
        "1" | "1.0" => Ok(4),
        x => Err(format!(
            "risk power {x:?}: one of 0, 0.25, 0.5, 0.75, 1 (integer roots keep the profile \
             exact)"
        )),
    }
}

/// α as text.
pub fn power_text(q: u8) -> &'static str {
    ["0", "0.25", "0.5", "0.75", "1"][usize::from(q.min(4))]
}

/// Parse a share `0` to `1` with at most 6 decimals into parts per million.
pub fn parse_ppm(s: &str) -> Result<u64, String> {
    let t = s.trim();
    let (int, frac) = t.split_once('.').unwrap_or((t, ""));
    let digits = |x: &str| x.bytes().all(|b| b.is_ascii_digit());
    if int.is_empty() || !digits(int) || !digits(frac) || frac.len() > 6 || int.len() > 1 {
        return Err(format!(
            "{s:?}: a share from 0 to 1 with at most 6 decimals"
        ));
    }
    let v = int.parse::<u64>().expect("digits") * 1_000_000
        + format!("{frac:0<6}").parse::<u64>().expect("digits");
    if v > 1_000_000 {
        return Err(format!("{s:?}: a share from 0 to 1"));
    }
    Ok(v)
}

/// `r^(q/4)` up to one common factor (the same for every `r` at a given `q`): integer square
/// roots of `r · 2^32` (so small risks keep their precision). Never 0 for `r > 0`.
fn power(r: i64, q: u8) -> u128 {
    let x = (r.max(0) as u128) << 32;
    let half = x.isqrt();
    let quarter = half.isqrt();
    match q {
        0 => 1,
        1 => quarter,
        2 => half,
        3 => half * quarter,
        _ => x,
    }
}

/// The band of a score: the first band whose upper edge is at least the score, else the last.
pub fn band_of(edges: &[i64], s: i64) -> usize {
    edges.partition_point(|&e| e < s)
}

/// The profile of the pilot's scores (in any order): `bands` bands of equal count, their counts,
/// mean scores and acceptances for the risk power `power_q` and the floor `floor_ppm`.
///
/// # Panics
/// With no scores, fewer than 1 band, a power above 4 or a floor above one million.
pub fn build_profile(
    mut scores: Vec<i64>,
    bands: usize,
    power_q: u8,
    floor_ppm: u64,
) -> (Vec<i64>, Vec<u64>, Vec<i64>, Vec<u64>) {
    assert!(!scores.is_empty() && bands >= 1 && power_q <= 4 && floor_ppm <= 1_000_000);
    scores.sort_unstable();
    let n = scores.len();
    let edges: Vec<i64> = (1..bands).map(|k| scores[k * n / bands - 1]).collect();
    let mut counts = vec![0u64; bands];
    let mut sums = vec![0i128; bands];
    for &s in &scores {
        let b = band_of(&edges, s);
        counts[b] += 1;
        sums[b] += i128::from(s);
    }
    let mean: Vec<i64> = counts
        .iter()
        .zip(&sums)
        .map(|(&c, &s)| {
            if c == 0 {
                0
            } else {
                (s / i128::from(c)) as i64
            }
        })
        .collect();
    // a_b ∝ target_b / n_b = (1 − f) · r_b^α / W + f / N, times N · W:
    // (1e6 − f) · p_b · N + f · W, with W = Σ n_c · p_c.
    let p: Vec<u128> = mean.iter().map(|&r| power(r, power_q)).collect();
    let total = n as u128;
    let w: u128 = counts
        .iter()
        .zip(&p)
        .map(|(&c, &x)| u128::from(c) * x)
        .sum();
    let score: Vec<u128> = p
        .iter()
        .zip(&counts)
        .map(|(&x, &c)| {
            if c == 0 {
                0
            } else {
                u128::from(1_000_000 - floor_ppm) * x * total + u128::from(floor_ppm) * w
            }
        })
        .collect();
    let max = score.iter().copied().max().unwrap_or(0);
    let accept = if max == 0 {
        // Nothing to prefer (every risk 0 and no floor): keep everything.
        vec![ACCEPT_ONE; bands]
    } else {
        let sh = (128 - max.leading_zeros()).saturating_sub(32);
        let den = max >> sh;
        score
            .iter()
            .map(|&s| (((s >> sh) << 32) / den) as u64)
            .collect()
    };
    (edges, counts, mean, accept)
}

impl RiskProfile {
    /// The band of a score.
    pub fn band(&self, s: i64) -> usize {
        band_of(&self.edges, s)
    }

    /// Bands.
    pub fn bands(&self) -> usize {
        self.accept.len()
    }

    /// Each band's target share of the new rows (parts per million): its natural share times
    /// its acceptance, normalised.
    pub fn target_ppm(&self) -> Vec<u64> {
        let kept: Vec<u128> = self
            .counts
            .iter()
            .zip(&self.accept)
            .map(|(&c, &a)| u128::from(c) * u128::from(a))
            .collect();
        let t: u128 = kept.iter().sum();
        kept.iter()
            .map(|&k| (k * 1_000_000).checked_div(t).unwrap_or(0) as u64)
            .collect()
    }

    /// The share of the pilot's positions a harvest keeps before its cap (parts per million).
    pub fn yield_ppm(&self) -> u64 {
        let n: u128 = self.counts.iter().map(|&c| u128::from(c)).sum();
        let kept: u128 = self
            .counts
            .iter()
            .zip(&self.accept)
            .map(|(&c, &a)| u128::from(c) * u128::from(a))
            .sum();
        if n == 0 {
            0
        } else {
            (kept * 1_000_000 / (n * u128::from(ACCEPT_ONE))) as u64
        }
    }

    /// The profile as JSON (the file `search risk-profile` writes and `--risk-profile` reads),
    /// with a plain table of the bands for the reports.
    pub fn to_json(&self) -> Value {
        let n: u64 = self.counts.iter().sum();
        let err: Vec<i128> = self
            .counts
            .iter()
            .zip(&self.mean)
            .map(|(&c, &m)| i128::from(c) * i128::from(m))
            .collect();
        let err_total: i128 = err.iter().sum();
        let target = self.target_ppm();
        let table: Vec<Value> = (0..self.bands())
            .map(|b| {
                json!({
                    "band": b,
                    "upper_score": self.edges.get(b),
                    "natural_share": if n == 0 { 0.0 } else { self.counts[b] as f64 / n as f64 },
                    "mean_predicted_sq_error": self.mean[b],
                    "predicted_error_share": if err_total == 0 { 0.0 } else { err[b] as f64 / err_total as f64 },
                    "accept": self.accept[b] as f64 / ACCEPT_ONE as f64,
                    "target_share": target[b] as f64 / 1e6,
                })
            })
            .collect();
        json!({
            "format": PROFILE_FORMAT,
            "model_sha256": self.model_sha256,
            "net_sha256": self.net_sha256,
            "power": power_text(self.power_q),
            "floor_ppm": self.floor_ppm,
            "edges": self.edges,
            "counts": self.counts,
            "mean": self.mean,
            "accept": self.accept,
            "pilot_seed": self.pilot_seed,
            "pilot_decisions": self.pilot_decisions,
            "pilot_positions": n,
            "yield_ppm": self.yield_ppm(),
            "what": "bands of the leaf-risk model's score of the positions the harvest's search \
                     reaches (equal counts in the pilot, lowest risk first); each band's share of \
                     the new rows is (1 - f) x frequency x risk^power / sum + f x frequency, kept \
                     before labelling with the acceptance shown (docs/m3-hybrid/o3-rounds.md)",
            "table": table,
        })
    }

    /// Read a profile file's JSON.
    pub fn from_json(v: &Value) -> Result<RiskProfile, String> {
        if v["format"].as_str() != Some(PROFILE_FORMAT) {
            return Err(format!("not a risk profile (format {PROFILE_FORMAT})"));
        }
        let ints = |k: &str| -> Result<Vec<i64>, String> {
            v[k].as_array()
                .ok_or_else(|| format!("risk profile: `{k}` must be a list"))?
                .iter()
                .map(|x| {
                    x.as_i64()
                        .ok_or_else(|| format!("risk profile: `{k}`: whole numbers"))
                })
                .collect()
        };
        let uints = |k: &str| -> Result<Vec<u64>, String> {
            v[k].as_array()
                .ok_or_else(|| format!("risk profile: `{k}` must be a list"))?
                .iter()
                .map(|x| {
                    x.as_u64()
                        .ok_or_else(|| format!("risk profile: `{k}`: whole numbers"))
                })
                .collect()
        };
        let p = RiskProfile {
            model_sha256: v["model_sha256"].as_str().unwrap_or("").to_string(),
            net_sha256: v["net_sha256"].as_str().unwrap_or("").to_string(),
            power_q: parse_power(v["power"].as_str().unwrap_or(""))?,
            floor_ppm: v["floor_ppm"].as_u64().ok_or("risk profile: floor_ppm")?,
            edges: ints("edges")?,
            counts: uints("counts")?,
            mean: ints("mean")?,
            accept: uints("accept")?,
            pilot_seed: v["pilot_seed"].as_u64().unwrap_or(0),
            pilot_decisions: v["pilot_decisions"].as_u64().unwrap_or(0),
        };
        let b = p.accept.len();
        if b == 0
            || p.edges.len() + 1 != b
            || p.counts.len() != b
            || p.mean.len() != b
            || p.edges.windows(2).any(|w| w[0] > w[1])
            || p.accept.iter().any(|&a| a > ACCEPT_ONE)
        {
            return Err(
                "risk profile: bands, edges, counts, means and acceptances disagree".into(),
            );
        }
        Ok(p)
    }
}

/// A risk-guided harvest's selection ([`HarvestJob::risk`]).
#[derive(Clone, Debug)]
pub struct RiskKeep {
    /// The leaf-risk model (made for the network that drives the harvest).
    pub model: Arc<RiskModel>,
    /// The bands and their acceptances.
    pub profile: Arc<RiskProfile>,
}

impl RiskKeep {
    /// A selection, refused unless the profile was made with this model for this network.
    pub fn new(model: RiskModel, profile: RiskProfile, net_sha256: &str) -> Result<Self, String> {
        if model.net_sha256 != net_sha256 {
            return Err(format!(
                "the leaf-risk model was made for the network {}, not the harvest's {net_sha256}",
                short(&model.net_sha256)
            ));
        }
        if profile.model_sha256 != model.sha256 || profile.net_sha256 != net_sha256 {
            return Err(format!(
                "the risk profile was made with the model {} for the network {}, not the model \
                 {} for {}",
                short(&profile.model_sha256),
                short(&profile.net_sha256),
                short(&model.sha256),
                short(net_sha256)
            ));
        }
        Ok(RiskKeep {
            model: Arc::new(model),
            profile: Arc::new(profile),
        })
    }
}

fn short(s: &str) -> &str {
    &s[..s.len().min(12)]
}

/// The pilot's seed for a run seeded `seed` (never the golden seed).
pub fn pilot_seed(seed: u64) -> u64 {
    let s = Rng::derive(seed, &[RISK_PILOT]).next_u64();
    if s == GOLDEN_SEED { s + 1 } else { s }
}

/// The pilot ([`build_profile`]'s scores): decisions `0..decisions` of `job` (its seed should be
/// [`pilot_seed`] of the run's; its selection and cap are ignored) searched as the harvest
/// searches them with `driver` (the network the model was made for), every distinct reached
/// position scored by `model` with the network's own value there. The scores in decision order,
/// the same for any thread count.
///
/// # Panics
/// If `driver` is not a network: the model scores the network's values.
#[allow(clippy::too_many_arguments)]
pub fn pilot_scores(
    job: &HarvestJob,
    decisions: u64,
    model: &RiskModel,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    driver: Driver<'_>,
    tt_mb: usize,
    threads: usize,
) -> Vec<i64> {
    assert!(
        matches!(driver, Driver::Net(_)),
        "a risk pilot scores the network's values: give the network"
    );
    let plain = HarvestJob {
        cap: 1,
        race: None,
        keep: None,
        risk: None,
        siblings: false,
        ..job.clone()
    };
    let source = BeliefSource::new();
    let next = AtomicU64::new(0);
    let per: Vec<Vec<(u64, Vec<i64>)>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads.max(1))
            .map(|_| {
                let (plain, source, next) = (&plain, &source, &next);
                s.spawn(move || {
                    let (mut sm, mut lm) = (SolveMem::new(), SolveMem::new());
                    let mut got = Vec::new();
                    loop {
                        let j = next.fetch_add(1, Ordering::Relaxed);
                        if j >= decisions {
                            break;
                        }
                        let (_, reached) = harvest_plan(
                            plain, j, starts, leaf, source, driver, tt_mb, &mut sm, &mut lm,
                        );
                        let sc = reached
                            .iter()
                            .map(|r| model.score(&r.canon, &r.ctx, leaf, &r.value))
                            .collect();
                        got.push((j, sc));
                    }
                    got
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("a pilot thread panicked"))
            .collect()
    });
    let mut all: Vec<(u64, Vec<i64>)> = per.into_iter().flatten().collect();
    all.sort_unstable_by_key(|(j, _)| *j);
    all.into_iter().flat_map(|(_, s)| s).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powers_and_shares_parse() {
        assert_eq!(parse_power("1"), Ok(4));
        assert_eq!(parse_power("0.5"), Ok(2));
        assert_eq!(parse_power("0"), Ok(0));
        assert!(parse_power("0.6").is_err());
        assert_eq!(parse_ppm("0.2"), Ok(200_000));
        assert_eq!(parse_ppm("1"), Ok(1_000_000));
        assert_eq!(parse_ppm("0"), Ok(0));
        assert!(parse_ppm("1.5").is_err());
        assert!(parse_ppm(".5").is_err());
        assert!(parse_ppm("0.1234567").is_err());
    }

    #[test]
    fn bands_hold_equal_counts_and_the_riskiest_band_keeps_everything() {
        // 1,000 scores 1..=1000: 4 bands of 250.
        let s: Vec<i64> = (1..=1000).rev().collect();
        let (edges, counts, mean, accept) = build_profile(s, 4, 4, 0);
        assert_eq!(edges, vec![250, 500, 750]);
        assert_eq!(counts, vec![250; 4]);
        assert_eq!(mean, vec![125, 375, 625, 875]);
        assert_eq!(accept[3], ACCEPT_ONE);
        // α = 1, no floor: acceptance ∝ the band's mean risk.
        for b in 0..3 {
            let want = ACCEPT_ONE as f64 * mean[b] as f64 / 875.0;
            assert!((accept[b] as f64 - want).abs() <= 2.0, "band {b}");
        }
        assert_eq!(band_of(&edges, 250), 0);
        assert_eq!(band_of(&edges, 251), 1);
        assert_eq!(band_of(&edges, 10_000), 3);
    }

    #[test]
    fn the_mix_follows_the_formula() {
        // Two bands: 900 positions of risk 1 and 100 of risk 81.
        let mut s = vec![1i64; 900];
        s.extend([81i64; 100]);
        for (q, floor) in [(4u8, 0u64), (4, 200_000), (2, 200_000), (0, 0)] {
            let (edges, counts, mean, accept) = build_profile(s.clone(), 10, q, floor);
            let p = RiskProfile {
                model_sha256: String::new(),
                net_sha256: String::new(),
                power_q: q,
                floor_ppm: floor,
                edges,
                counts,
                mean,
                accept,
                pilot_seed: 0,
                pilot_decisions: 0,
            };
            let t = p.target_ppm();
            let top: u64 = t[9];
            let f = floor as f64 / 1e6;
            let pw = |r: f64| match q {
                4 => r,
                2 => r.sqrt(),
                _ => 1.0,
            };
            let w = 900.0 * pw(1.0) + 100.0 * pw(81.0);
            let want = (1.0 - f) * 100.0 * pw(81.0) / w + f * 0.1;
            assert!(
                (top as f64 / 1e6 - want).abs() < 2e-4,
                "q {q} floor {floor}: {top} against {want}"
            );
        }
    }

    #[test]
    fn a_profile_reads_back() {
        let (edges, counts, mean, accept) = build_profile((0..100).collect(), 5, 2, 200_000);
        let p = RiskProfile {
            model_sha256: "m".into(),
            net_sha256: "n".into(),
            power_q: 2,
            floor_ppm: 200_000,
            edges,
            counts,
            mean,
            accept,
            pilot_seed: 7,
            pilot_decisions: 3,
        };
        assert_eq!(RiskProfile::from_json(&p.to_json()), Ok(p.clone()));
        let mut bad = p.to_json();
        bad["accept"] = json!([1, 2]);
        assert!(RiskProfile::from_json(&bad).is_err());
    }
}
