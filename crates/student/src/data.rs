//! Teacher shards read together (docs/STUDENT.md §3, §5): stamps checked, the records of one
//! decision merged across shards by deal index, and each merged decision encoded as a feature row.
//!
//! **Merging.** Records of the same decision (the same position, decision seed and candidates)
//! may come from several shards (more deals added later for close calls): their deal ranges are
//! put in deal order and must not overlap; the values are joined in that order, so the mean is
//! exactly the mean over all the deals. Decisions come out in the order of their first record
//! (shards in the order given, records in shard order): the same rows for the same shards.
//!
//! **Weights** ([`DealWeights`], docs/STUDENT.md §3 item 2): a card-reading model's importance
//! weights per stored deal turn the plain means into weighted ones without a new solve.

use std::collections::BTreeMap;
use std::path::Path;

use engine::game::Action;
use engine::rules::legal_plays;
use equity::MoneyChance;
use solver::Leaf;

use crate::encode::{encode_cand, encode_view, rotate};
use crate::features::FeatureRow;
use crate::position::cand_code;
use crate::record::{RecordHead, ShardReader, check_stamps, means_of, weighted_means_of};

/// Weights for stored deals: `None` for plain means.
pub trait DealWeights {
    /// The weights of deals `deals` of the decision with position `pos_sha256` and seed `dseed`,
    /// or `None` when there are none (plain means). An error stops the encoding.
    fn weights(
        &self,
        pos_sha256: &[u8; 32],
        dseed: u64,
        deals: &[u32],
    ) -> Result<Option<Vec<u64>>, String>;
}

/// Plain means.
pub struct Uniform;

impl DealWeights for Uniform {
    fn weights(&self, _: &[u8; 32], _: u64, _: &[u32]) -> Result<Option<Vec<u64>>, String> {
        Ok(None)
    }
}

/// Weights from a file: a `.npy` of records `[('pos_sha256', '|u1', (32,)), ('dseed', '<u8'),
/// ('deal', '<u4'), ('weight', '<u8')]`; every deal of every decision must have one.
pub struct WeightsFile {
    map: BTreeMap<([u8; 32], u64, u32), u64>,
    /// The file's SHA-256.
    pub sha256: String,
}

/// The weights file's record type.
pub const WEIGHTS_DESCR: &str =
    "[('pos_sha256', '|u1', (32,)), ('dseed', '<u8'), ('deal', '<u4'), ('weight', '<u8')]";

impl WeightsFile {
    /// Read a weights file.
    pub fn read(path: &Path) -> Result<WeightsFile, String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
            return Err(format!("{}: not a .npy file", path.display()));
        }
        let hl = usize::from(u16::from_le_bytes([b[8], b[9]]));
        let h = std::str::from_utf8(&b[10..10 + hl]).map_err(|_| "header")?;
        if !h.contains(WEIGHTS_DESCR) {
            return Err(format!("{}: not a deal-weights file", path.display()));
        }
        let rb = 32 + 8 + 4 + 8;
        let body = &b[10 + hl..];
        if !body.len().is_multiple_of(rb) {
            return Err(format!("{}: size", path.display()));
        }
        let mut map = BTreeMap::new();
        for r in body.chunks(rb) {
            let pos: [u8; 32] = r[..32].try_into().expect("32");
            let dseed = u64::from_le_bytes(r[32..40].try_into().expect("8"));
            let deal = u32::from_le_bytes(r[40..44].try_into().expect("4"));
            let w = u64::from_le_bytes(r[44..52].try_into().expect("8"));
            if map.insert((pos, dseed, deal), w).is_some() {
                return Err(format!("{}: a deal weighted twice", path.display()));
            }
        }
        Ok(WeightsFile {
            map,
            sha256: equity::sha256::sha256_hex(&b),
        })
    }
}

impl DealWeights for WeightsFile {
    fn weights(
        &self,
        pos_sha256: &[u8; 32],
        dseed: u64,
        deals: &[u32],
    ) -> Result<Option<Vec<u64>>, String> {
        let w = deals
            .iter()
            .map(|&d| {
                self.map
                    .get(&(*pos_sha256, dseed, d))
                    .copied()
                    .ok_or_else(|| format!("deal {d} of a decision has no weight"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if w.iter().all(|&x| x == 0) {
            return Err("a decision whose deals all weigh 0".into());
        }
        Ok(Some(w))
    }
}

/// One decision, merged over its records.
#[derive(Clone, Debug)]
pub struct Merged {
    /// Its first record's head.
    pub head: RecordHead,
    /// The deals' indices, ascending.
    pub deals: Vec<u32>,
    /// `values[j][c]`: deal `deals[j]`, candidate `c`, real pair order.
    pub values: Vec<Vec<MoneyChance>>,
}

impl Merged {
    /// Per candidate, the plain mean over every deal.
    pub fn means(&self) -> Vec<MoneyChance> {
        means_of(&self.values, self.head.cands.len())
    }
}

/// Teacher shards opened together.
pub struct Shards {
    /// The shards, in the order given.
    pub readers: Vec<ShardReader>,
    /// Decisions: the (shard, record) of each of its records, in deal order.
    pub groups: Vec<Vec<(usize, usize)>>,
}

impl Shards {
    /// Open shards; their stamps must agree (but those in `allow`), and so must their roles.
    pub fn open(paths: &[&Path], allow: &[String]) -> Result<Shards, String> {
        let readers = paths
            .iter()
            .map(|p| ShardReader::open(p))
            .collect::<Result<Vec<_>, _>>()?;
        let names: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
        let sides: Vec<(&str, &serde_json::Value)> = names
            .iter()
            .zip(&readers)
            .map(|(n, r)| (n.as_str(), &r.sidecar))
            .collect();
        check_stamps(&sides, allow)?;
        if readers
            .iter()
            .any(|r| r.sidecar["role"] != readers[0].sidecar["role"])
        {
            return Err("test and training shards cannot be read together".into());
        }
        type Key = ([u8; 32], u64, Vec<u16>);
        let mut index: BTreeMap<Key, usize> = BTreeMap::new();
        let mut groups: Vec<Vec<(usize, usize)>> = Vec::new();
        for (s, r) in readers.iter().enumerate() {
            for (i, h) in r.records.iter().enumerate() {
                let key = (
                    h.pos_sha256,
                    h.dseed,
                    h.cands.iter().map(|&c| cand_code(c)).collect(),
                );
                match index.get(&key) {
                    Some(&g) => groups[g].push((s, i)),
                    None => {
                        index.insert(key, groups.len());
                        groups.push(vec![(s, i)]);
                    }
                }
            }
        }
        for g in &mut groups {
            g.sort_by_key(|&(s, i)| readers[s].records[i].deal_from);
            for w in g.windows(2) {
                let a = &readers[w[0].0].records[w[0].1];
                let b = &readers[w[1].0].records[w[1].1];
                if a.deal_from + a.n_deals > b.deal_from {
                    return Err(format!(
                        "deals {}..{} and {}..{} of one decision overlap",
                        a.deal_from,
                        a.deal_from + a.n_deals,
                        b.deal_from,
                        b.deal_from + b.n_deals
                    ));
                }
            }
        }
        Ok(Shards { readers, groups })
    }

    /// The role of the shards (`train` or `test`).
    pub fn role(&self) -> String {
        self.readers
            .first()
            .and_then(|r| r.sidecar["role"].as_str())
            .unwrap_or("train")
            .to_string()
    }

    /// Decision `g`, merged (its values read and checked).
    pub fn merged(&mut self, g: usize) -> Result<Merged, String> {
        let mut deals = Vec::new();
        let mut values = Vec::new();
        for &(s, i) in &self.groups[g] {
            let h = &self.readers[s].records[i];
            deals.extend(h.deal_from..h.deal_from + h.n_deals);
            values.extend(self.readers[s].values(i)?);
        }
        let (s, i) = self.groups[g][0];
        Ok(Merged {
            head: self.readers[s].records[i].clone(),
            deals,
            values,
        })
    }
}

/// Check that a record's candidates are legal in its position (they were stored by a generator;
/// a reader never takes them on trust).
fn check_cands(
    h: &RecordHead,
    view: &engine::view::SeatView,
    legal: &[Action],
) -> Result<(), String> {
    let trump = view.public.trump.ok_or("trumps unknown")?;
    let led = view.public.trick_cards.first().map(|&(_, c)| c);
    for &(a, p) in &h.cands {
        if !legal.contains(&a) {
            return Err(format!("candidate {a:?} is not legal"));
        }
        if let (Action::Burn(b), Some(c)) = (a, p) {
            let rest = view.hand & !engine::cards::bit(b);
            if !engine::cards::has(legal_plays(rest, led, trump), c) {
                return Err(format!("joint candidate ({b}, {c}) is not legal"));
            }
        }
    }
    Ok(())
}

/// A merged decision as a feature row: the view and candidates encoded, the targets the means
/// (weighted when `weights` gives weights) rotated to the deciding pair first.
pub fn feature_row(
    m: &Merged,
    leaf: &dyn Leaf,
    weights: &dyn DealWeights,
) -> Result<FeatureRow, String> {
    let h = &m.head;
    let r = h.pos.replay()?;
    check_cands(h, &r.view, &r.legal)?;
    let view = encode_view(&r.view, leaf)?;
    let cands = h.cands.iter().map(|&c| encode_cand(&r.view, c)).collect();
    let means = match weights.weights(&h.pos_sha256, h.dseed, &m.deals)? {
        None => m.means(),
        Some(w) => weighted_means_of(&m.values, &w, h.cands.len()),
    };
    let team = engine::team_of(h.pos.seat);
    let t = usize::from(team);
    let n = m.values.len() as u128;
    let sd = (0..h.cands.len())
        .map(|c| {
            let (s, s2) = m.values.iter().fold((0u128, 0u128), |(a, b), row| {
                let x = u128::from(row[c].0[t]);
                (a + x, b + x * x)
            });
            if n >= 2 {
                crate::record::isqrt((n * s2 - s * s) / (n * (n - 1))) as u64
            } else {
                0
            }
        })
        .collect();
    Ok(FeatureRow {
        view,
        cands,
        targets: means.iter().map(|&x| rotate(x, team)).collect(),
        sd,
        n_deals: m.values.len() as u32,
        trick: h.trick,
        place: h.place,
        stratum: h.stratum,
        source: h.source,
        kind: h.kind,
        pos_key: u64::from_le_bytes(h.pos_sha256[..8].try_into().expect("8")),
    })
}
