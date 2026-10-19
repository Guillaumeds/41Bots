//! Hybrid training labels (`label_source`; docs/m3-hybrid/audit-loop.md §12, owner 2026-10-19):
//! the training shards of a rung at cut `k` labelled by the hybrid H_{k+1} (exact search through
//! trick `k`, the stored integer network O_{k+1} at the start of trick `k + 1`: `search labels
//! --label-net`) instead of exact search to the end of the hand. Bootstrapping from the next
//! trick's network, as TreeStrap (Veness, Silver, Uther & Blair, NeurIPS 2009) and TD-Leaf do:
//! at trick 2 an exact label costs about 3 thread-seconds and a hybrid one a few milliseconds.
//!
//! ```json
//! "label_source": "hybrid",
//! "label_net": "inputs/sha256/<hex of the integer O_{cut+1}>",
//! "hybrid_exact_share": 0.1
//! ```
//! - `label_source` (`"exact"`, the default: every label exact, the chunk exactly as before; or
//!   `"hybrid"`);
//! - `label_net` (required with `"hybrid"`, refused otherwise): the stored integer network of the
//!   trick after the cut (`rung.json`'s `net.int_key` of that trick's run), fetched at step 0 and
//!   checked to be a trick-`cut + 1` network before anything is installed;
//! - `hybrid_exact_share` (0; refused without `"hybrid"`): the exact anchor, a share (0 to 1, six
//!   decimals at most) of each hybrid shard's rows labelled exactly all the same, drawn by each
//!   row's key (`search::oracle_labels::anchored`: deterministic, the same for any thread count).
//!
//! **Only training shards.** Hybrid labels go to the generated training shards (`random`,
//! `rules`, `harvest`, `race`) and the audit loop's targeted harvests. The validation shard, the
//! audit shards, the golden set and the mini golden sets, the rough test and every standard stay
//! exact, always: no setting asks otherwise, a `label_inputs` validation entry marked hybrid is
//! refused, and a validation or audit shard whose sidecar says hybrid stops the chunk.
//!
//! **Provenance.** A hybrid shard's sidecar says `label_source` `"hybrid"`, the network's SHA-256
//! (`label_net_sha256`) and a values version of its own (`"2+O3:<sha256>"`), so nothing that
//! reads exact labels takes it for one; `rung.json`'s `labels` records and `label_sources`, and the
//! chunk's result, list every shard by label source. `label_inputs` entries say `"label_source":
//! "hybrid"` for a stored hybrid shard (exact by default) and the sidecar must agree
//! (`reuse::check_sidecar`): sources are never mixed in silence.

use serde_json::{Map, Value, json};

use super::Rung;
use crate::remote::{input_hex, input_key};

/// The settings of this module.
pub(crate) const KEYS: [&str; 3] = ["label_source", "label_net", "hybrid_exact_share"];

/// `label_source` `"hybrid"`'s settings.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HybridLabels {
    /// SHA-256 of the integer O_{cut+1} (`label_net`).
    pub net: String,
    /// `hybrid_exact_share` in parts per million.
    pub exact_ppm: u32,
}

impl HybridLabels {
    /// The arguments `search labels` takes for a training shard.
    pub fn args(&self, net_path: &std::path::Path) -> Vec<String> {
        let mut v = vec!["--label-net".to_string(), net_path.display().to_string()];
        if self.exact_ppm > 0 {
            v.push("--exact-share".into());
            v.push(share_text(self.exact_ppm));
        }
        v
    }
}

/// A share in parts per million as `search labels --exact-share` reads it ("0.100000").
pub(crate) fn share_text(ppm: u32) -> String {
    format!("{}.{:06}", ppm / 1_000_000, ppm % 1_000_000)
}

/// The settings as `rung.json` records them.
pub(crate) fn to_json(h: Option<&HybridLabels>) -> Value {
    match h {
        None => json!({ "label_source": "exact" }),
        Some(h) => json!({ "label_source": "hybrid", "label_net": input_key(&h.net),
                           "hybrid_exact_share": share_text(h.exact_ppm),
                           "applies_to": "the training shards (random, rules, harvest, race) and \
                                          the audit loop's targeted harvests; validation, audit \
                                          and golden sets stay exact" }),
    }
}

/// Reads the settings (`None`: exact labels); `cut`: the rung's.
pub(crate) fn parse(o: &Map<String, Value>, cut: u64) -> Result<Option<HybridLabels>, String> {
    let source = match o.get("label_source") {
        None | Some(Value::Null) => "exact",
        Some(Value::String(s)) if s == "exact" || s == "hybrid" => s.as_str(),
        Some(v) => {
            return Err(format!(
                "hybrid_rung: label_source {v}: \"exact\" (the default) or \"hybrid\""
            ));
        }
    };
    if source == "exact" {
        if let Some(k) = ["label_net", "hybrid_exact_share"]
            .iter()
            .find(|k| o.get(**k).is_some_and(|v| !v.is_null()))
        {
            return Err(format!(
                "hybrid_rung: {k} is for hybrid labels: it needs label_source \"hybrid\" (exact \
                 labels use no network)"
            ));
        }
        return Ok(None);
    }
    if cut >= 5 {
        return Err(format!(
            "hybrid_rung: label_source \"hybrid\" at cut {cut}: hybrid labels need the network \
             of the trick after the cut, and there is none after trick 5"
        ));
    }
    let next = cut + 1;
    let net = match o.get("label_net") {
        Some(Value::String(v)) => input_hex(v).ok_or_else(|| {
            format!(
                "hybrid_rung: label_net {v:?} must be a stored input's key, \
                 \"inputs/sha256/<64 hex digits>\" (the integer O_{next})"
            )
        })?,
        _ => {
            return Err(format!(
                "hybrid_rung: label_source \"hybrid\" needs label_net, the stored integer \
                 network of trick {next} (O_{next}), \"inputs/sha256/<64 hex digits>\""
            ));
        }
    };
    let exact_ppm = match o.get("hybrid_exact_share") {
        None | Some(Value::Null) => 0,
        Some(v) => {
            let x = v
                .as_f64()
                .filter(|x| x.is_finite() && (0.0..=1.0).contains(x));
            let ppm = x.map(|x| x * 1e6).filter(|p| (p - p.round()).abs() < 1e-6);
            match ppm {
                Some(p) => p.round() as u32,
                None => {
                    return Err(format!(
                        "hybrid_rung: hybrid_exact_share {v}: a share from 0 to 1 with at most \
                         six decimals (e.g. 0.1)"
                    ));
                }
            }
        }
    };
    Ok(Some(HybridLabels { net, exact_ppm }))
}

/// A shard's label source as its sidecar records it (`"exact"` for sidecars written before
/// 2026-10-19, which have no `label_source`: every label was exact then).
pub(crate) fn sidecar_source(side: &Value) -> &str {
    side["label_source"].as_str().unwrap_or("exact")
}

/// The values version a sidecar of label source `source` must record at cut `cut`: the exact
/// solver's (a number), or for hybrid labels `"<exact>+O<cut+1>:<label_net_sha256>"` (the
/// sidecar's own network, which must be a SHA-256).
pub(crate) fn want_values_version(side: &Value, source: &str, cut: u64) -> Result<Value, String> {
    let v = search::measure::VALUES_VERSION;
    if source == "exact" {
        return Ok(json!(v));
    }
    let sha = side["label_net_sha256"]
        .as_str()
        .filter(|s| input_hex(&format!("sha256:{s}")).is_some())
        .ok_or("its sidecar says hybrid labels but names no label_net_sha256")?;
    Ok(json!(format!("{v}+O{}:{sha}", cut + 1)))
}

/// What `rung.json` records of a shard's labels (`label_source`, and the hybrid facts).
pub(crate) fn record(side: &Value) -> Value {
    let source = sidecar_source(side);
    let mut v = json!({ "label_source": source });
    if source == "hybrid" {
        for k in [
            "label_net_sha256",
            "exact_share_ppm",
            "rows_hybrid",
            "rows_exact",
        ] {
            v[k] = side[k].clone();
        }
    }
    v
}

impl Rung<'_> {
    /// `search labels` arguments for a shard of set `set`: the hybrid's (`--label-net`, and
    /// `--exact-share`) for a training shard of a hybrid run, nothing otherwise (the validation
    /// and audit shards are always exact).
    pub(super) fn label_args(&self, set: &str) -> Vec<String> {
        match (&self.p.hybrid_labels, self.inputs.get("label_net")) {
            (Some(h), Some(net)) if !matches!(set, "val" | "audit") => h.args(net),
            _ => Vec::new(),
        }
    }

    /// Notes a label shard of the run (made or reused) with its set and its label source, read
    /// from its sidecar, for `rung.json`'s `label_sources`; returns what its own record adds. A
    /// validation or audit shard with hybrid labels stops the chunk (module notes).
    pub(super) fn note_labels(
        &mut self,
        file: &str,
        set: &str,
        side: &Value,
    ) -> Result<Value, String> {
        let rec = record(side);
        if sidecar_source(side) != "exact" && matches!(set, "val" | "audit") {
            return Err(format!(
                "hybrid_rung: {file}: the {} shard has {} labels; validation and audit shards \
                 are always exact",
                if set == "val" { "validation" } else { "audit" },
                sidecar_source(side)
            ));
        }
        let mut e = rec.clone();
        e["file"] = json!(file);
        e["set"] = json!(set);
        self.label_kinds.push(e);
        Ok(rec)
    }

    /// The summary's line on the labels: the run's label source, and with hybrid labels the
    /// network and which shards have them.
    pub(super) fn label_line(&self) -> String {
        match &self.p.hybrid_labels {
            None => "labels: exact (every shard)\n".to_string(),
            Some(h) => {
                let (mut hy, mut ex) = (Vec::new(), Vec::new());
                for e in &self.label_kinds {
                    let f = e["file"].as_str().unwrap_or("?").to_string();
                    if e["label_source"] == "hybrid" {
                        hy.push(f);
                    } else {
                        ex.push(f);
                    }
                }
                format!(
                    "labels: hybrid training labels (H_{} with the network {}, exact share \
                     {}): hybrid {}; exact {}\n",
                    self.p.cut + 1,
                    input_key(&h.net),
                    share_text(h.exact_ppm),
                    if hy.is_empty() {
                        "none".into()
                    } else {
                        hy.join(", ")
                    },
                    if ex.is_empty() {
                        "none".into()
                    } else {
                        ex.join(", ")
                    }
                )
            }
        }
    }

    /// `rung.json`'s `label_sources`: the run's setting, the shards by label source, and each
    /// shard's record.
    pub(super) fn label_sources(&self) -> Value {
        let files = |k: &str| -> Vec<Value> {
            self.label_kinds
                .iter()
                .filter(|e| e["label_source"] == k)
                .map(|e| e["file"].clone())
                .collect()
        };
        json!({ "setting": to_json(self.p.hybrid_labels.as_ref()),
                "hybrid": files("hybrid"), "exact": files("exact"),
                "shards": self.label_kinds })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn o(v: Value) -> Map<String, Value> {
        v.as_object().cloned().expect("object")
    }

    fn key(c: char) -> String {
        format!("inputs/sha256/{}", c.to_string().repeat(64))
    }

    #[test]
    fn settings_are_read_and_checked() {
        assert_eq!(parse(&o(json!({})), 2), Ok(None));
        assert_eq!(parse(&o(json!({ "label_source": "exact" })), 2), Ok(None));
        let h = parse(
            &o(json!({ "label_source": "hybrid", "label_net": key('a'),
                       "hybrid_exact_share": 0.1 })),
            2,
        )
        .expect("hybrid")
        .expect("some");
        assert_eq!(h.net, "a".repeat(64));
        assert_eq!(h.exact_ppm, 100_000);
        assert_eq!(
            h.args(std::path::Path::new("/n")),
            ["--label-net", "/n", "--exact-share", "0.100000"]
        );
        let plain = parse(
            &o(json!({ "label_source": "hybrid", "label_net": key('b') })),
            3,
        )
        .expect("hybrid")
        .expect("some");
        assert_eq!(plain.exact_ppm, 0);
        assert_eq!(
            plain.args(std::path::Path::new("/n")),
            ["--label-net", "/n"]
        );
        assert_eq!(to_json(None)["label_source"], "exact");
        assert_eq!(to_json(Some(&h))["label_net"], key('a'));
        for (v, cut, want) in [
            (json!({ "label_source": "both" }), 2, "\"exact\""),
            (json!({ "label_source": "hybrid" }), 2, "needs label_net"),
            (
                json!({ "label_source": "hybrid", "label_net": "x" }),
                2,
                "stored input's key",
            ),
            (
                json!({ "label_source": "hybrid", "label_net": key('a') }),
                5,
                "after trick 5",
            ),
            (
                json!({ "label_net": key('a') }),
                2,
                "label_source \"hybrid\"",
            ),
            (
                json!({ "label_source": "exact", "hybrid_exact_share": 0.1 }),
                2,
                "label_source \"hybrid\"",
            ),
            (
                json!({ "label_source": "hybrid", "label_net": key('a'), "hybrid_exact_share": 1.5 }),
                2,
                "share from 0 to 1",
            ),
            (
                json!({ "label_source": "hybrid", "label_net": key('a'),
                        "hybrid_exact_share": 0.12345678 }),
                2,
                "six decimals",
            ),
        ] {
            let e = parse(&o(v.clone()), cut).expect_err("refused");
            assert!(e.contains(want), "{v}: {e}");
        }
        assert_eq!(share_text(1_000_000), "1.000000");
        assert_eq!(share_text(5), "0.000005");
        for ppm in [0, 5, 100_000, 999_999, 1_000_000] {
            assert_eq!(
                search::oracle_labels::parse_exact_share(&share_text(ppm)),
                Ok(ppm)
            );
        }
    }

    #[test]
    fn sidecars_say_their_source() {
        let exact = json!({ "values_version": 2 });
        assert_eq!(sidecar_source(&exact), "exact");
        assert_eq!(
            want_values_version(&exact, "exact", 2),
            Ok(json!(search::measure::VALUES_VERSION))
        );
        let sha = "c".repeat(64);
        let hybrid = json!({ "label_source": "hybrid", "label_net_sha256": sha,
                             "rows_hybrid": 9, "rows_exact": 1, "exact_share_ppm": 100000 });
        assert_eq!(sidecar_source(&hybrid), "hybrid");
        assert_eq!(
            want_values_version(&hybrid, "hybrid", 2),
            Ok(json!(format!(
                "{}+O3:{sha}",
                search::measure::VALUES_VERSION
            )))
        );
        assert!(want_values_version(&json!({ "label_source": "hybrid" }), "hybrid", 2).is_err());
        let r = record(&hybrid);
        assert_eq!(r["label_source"], "hybrid");
        assert_eq!(r["rows_hybrid"], 9);
        assert_eq!(record(&exact), json!({ "label_source": "exact" }));
    }
}
