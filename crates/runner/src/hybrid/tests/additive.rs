//! The audit loop's additive mode and stored label shards reused, with the fake programs of the
//! step tests (docs/m3-hybrid/audit-loop.md §10).

use super::*;
use crate::remote::{DirRemote, put_input};

#[cfg(unix)]
/// A safetensors network of hidden widths `widths` at trick 4 (outcome head): the integer form
/// (`int`) with the format and the training record, or the float form with the same record (and
/// another seed when `other`).
fn net_file(path: &Path, widths: &[usize], int: bool, other: bool) {
    net_file_at(path, widths, int, other, 4);
}

/// [`net_file`] for frontier trick `trick`.
#[cfg(unix)]
fn net_file_at(path: &Path, widths: &[usize], int: bool, other: bool, trick: u8) {
    use nn::safetensors::{Dtype, File, Tensor, write};
    let mut f = File::default();
    let mut n_in = nn::encode::N_FEATURES;
    let outs: Vec<usize> = widths
        .iter()
        .copied()
        .chain([nn::outcome::vocab(trick)])
        .collect();
    for (k, &n) in outs.iter().enumerate() {
        let (wd, bd, wb, bb) = if int {
            (Dtype::I32, Dtype::I64, 4, 8)
        } else {
            (Dtype::F32, Dtype::F32, 4, 4)
        };
        f.tensors.insert(
            format!("w{k}"),
            Tensor {
                dtype: wd,
                shape: vec![n_in, n],
                data: vec![0; n_in * n * wb],
            },
        );
        f.tensors.insert(
            format!("b{k}"),
            Tensor {
                dtype: bd,
                shape: vec![n],
                data: vec![0; n * bb],
            },
        );
        n_in = n;
    }
    let mut meta = BTreeMap::new();
    meta.insert("trick".to_string(), trick.to_string());
    meta.insert("head".to_string(), "outcome-v1".to_string());
    meta.insert(
        "seed".to_string(),
        if other { "2" } else { "1" }.to_string(),
    );
    meta.insert("val_loss".to_string(), "1.25".to_string());
    if int {
        meta.insert("format".to_string(), nn::oracle::FORMAT.to_string());
        meta.insert("shifts".to_string(), "[]".to_string());
    }
    f.meta = meta;
    fs::write(path, write(&f)).expect("write net");
}

#[cfg(unix)]
/// Stores a file in the fake remote; its SHA-256.
fn store(f: &Fakes, path: &Path) -> String {
    let remote = DirRemote {
        root: f.base.join("remote"),
    };
    put_input(&remote, path, None).expect("put").0
}

#[cfg(unix)]
/// The integer and float networks of `init_net`, stored; their SHA-256.
fn init_nets(f: &Fakes, widths: &[usize], other_float: bool) -> (String, String) {
    let (i, fl) = (f.base.join("init-int"), f.base.join("init-float"));
    net_file(&i, widths, true, false);
    net_file(&fl, widths, false, other_float);
    (store(f, &i), store(f, &fl))
}

#[cfg(unix)]
/// A fake `search` around the audit loop's: its golden check also gives the overall row's bias
/// and the band of seats 1-2 (its bound for round k's network in `band-k`).
fn search_with_bands(f: &Fakes, inner: &Path) -> PathBuf {
    let arg = r#"arg() { n=$1; shift; while [ $# -gt 0 ]; do [ "$1" = "$n" ] && { echo "$2"; return; }; shift; done; }"#;
    fake_script(
        &f.bin,
        "search-additive",
        &format!(
            r#"{arg}
'{inner}' "$@" || exit 1
[ "$1" = hybrid-check ] || exit 0
case "$(arg --slices "$@")" in */mini/*) exit 0;; esac
net=$(arg --net "$@"); k=1
case "$net" in */audit/round-*) k=$(echo "$net" | sed 's|.*/audit/round-\([0-9]*\)/net/.*|\1|');; esac
[ -f '{base}/band-'$k ] || exit 0
b=$(cat '{base}/band-'$k)
sed -i "s|\"bias_pass\": true}}}}]|\"bias_pass\": true, \"bias\": [0.00001, 0.0, 0.0, 0.0]}}}}, {{\"group\": \"pos 1-2\", \"summary\": {{\"decisions\": 1, \"loss_upper95\": $b, \"bias\": [0.0, 0.0, 0.0, 0.0]}}}}]|" "$(arg --json "$@")"
"#,
            inner = inner.display(),
            base = f.base.display()
        ),
    )
}

#[cfg(unix)]
fn additive_params(f: &Fakes, extra: Value) -> Params {
    let (int, float) = init_nets(f, &[16, 8], false);
    let mut v = json!({ "labels_random": 7, "labels_rules": 0, "labels_harvest": 9,
                        "val_count": 3, "golden_count": 2, "golden_worlds": 4,
                        "widths": "16,8", "audit_rows": 50, "audit_mini_decisions": 2,
                        "audit_target_rows": 1000, "audit_rounds": 3,
                        "train_mode": "additive", "init_net": format!("inputs/sha256/{int}"),
                        "init_net_float": format!("inputs/sha256/{float}"),
                        "shrink_perturb": 0.9, "harvest_seat_weights": "2,1,1" });
    if let (Some(o), Value::Object(e)) = (v.as_object_mut(), extra) {
        o.extend(e);
    }
    parse(&v).expect("params")
}

/// A fake additive loop of three rounds on a (fake) GPU: round 1 is `init_net` as it is
/// (`--epochs 0`), round 2 continues it but is worse on the seats 1-2 band, so it is NOT adopted
/// (although its overall bound is the lowest: the scratch ranking would have kept it), and round 3
/// continues round 1 again, on the targeted shards of rounds 1 and 2, and is adopted; the chunk's
/// network is round 3's. The rounds run one after another (no training beside the CPU steps);
/// the group the best still fails stays failing although the rejected round 2 passed it; the
/// harvests carry the seat weights; the reports carry the adoption and the leader focus.
#[cfg(unix)]
#[test]
fn additive_loop_keeps_only_rounds_no_worse() {
    let f = fakes("additive");
    fs::write(f.base.join("audit-1.json"), GROUP_ONE).expect("write");
    fs::write(f.base.join("overall-bounds"), "0.00005 0.00003 0.00004").expect("write");
    fs::write(f.base.join("mini-bounds"), "0.0002 0.00004 0.0002").expect("write");
    for (k, b) in [(1, "0.0001"), (2, "0.0002"), (3, "0.00009")] {
        fs::write(f.base.join(format!("band-{k}")), b).expect("write");
    }
    let tools = audit_tools(&f, true, None);
    let tools = Tools {
        search: search_with_bands(&f, &tools.search),
        ..tools
    };
    let params = additive_params(&f, json!({ "upload_shards": true }));
    let root = f.base.join("run");
    let mut ctx = Ctx::local("hybrid-test", &root, 16);
    let r = chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect("the loop runs");
    assert_eq!(r["verdict"], "FAIL", "{r}");
    assert!(
        r["audit"]["stop"]
            .as_str()
            .is_some_and(|s| s.starts_with("round limit (3 rounds")),
        "{r}"
    );
    assert_eq!(r["audit"]["best_round"], 3, "{r}");
    let out = std::path::absolute(root.join("out/hybrid")).expect("abs");
    let inputs = std::path::absolute(root.join("hybrid-work/inputs")).expect("abs");
    let rung = read_json(&out.join("rung.json")).expect("rung.json");
    assert_eq!(rung["net"]["dir"], "audit/round-3/net", "the adopted best");
    assert_eq!(r["net_int_key"], rung["net_r3"]["int_key"]);
    assert!(
        rung["net"]["float_key"]
            .as_str()
            .is_some_and(|k| k.starts_with("inputs/sha256/")),
        "the float weights are shared too: {}",
        rung["net"]
    );
    assert_eq!(rung["params"]["train_mode"]["train_mode"], "additive");
    assert_eq!(rung["additive"]["init_widths"], "16,8");
    // The trainings: round 1 the starting network as it is; rounds 2 and 3 both from round 1's
    // network (round 2 was not adopted), perturbed with their own seeds; never beside the CPU
    // steps, even on the GPU.
    let log = fs::read_to_string(&f.calls).expect("calls");
    let trains: Vec<&str> = log
        .lines()
        .filter(|l| l.starts_with("python -m fortyone_nn.train"))
        .collect();
    assert_eq!(trains.len(), 3, "{log}");
    let (int, float) = (
        params.additive.as_ref().expect("additive").init_net.clone(),
        params
            .additive
            .as_ref()
            .expect("additive")
            .init_float
            .clone(),
    );
    assert!(trains[0].contains("--epochs 0 "), "{}", trains[0]);
    assert!(
        trains[0].contains(&format!(
            "--init {} --init-int {}",
            inputs.join(&float).display(),
            inputs.join(&int).display()
        )) && !trains[0].contains("--shrink-perturb"),
        "{}",
        trains[0]
    );
    let from_round_1 = format!(
        "--init {} --init-int {} --shrink-perturb 0.9 --perturb-seed",
        out.join("net/float.safetensors").display(),
        out.join("net/int.safetensors").display()
    );
    for (t, seed) in [(trains[1], 1002), (trains[2], 1003)] {
        assert!(
            t.contains("--epochs 8 ") && t.contains("--lr 0.0003 "),
            "{t}"
        );
        assert!(t.contains(&format!("{from_round_1} {seed}")), "{t}");
        assert!(
            t.contains("--widths 16,8 ") && t.contains("--device cuda"),
            "{t}"
        );
    }
    assert!(!log.contains("--watch-dir"), "sequential rounds: {log}");
    assert!(trains[2].contains("t4-target-r1-s") && trains[2].contains("t4-target-r2-s"));
    // Every harvest carries the seat weights; the audit shards do not.
    for l in log.lines().filter(|l| l.starts_with("search labels")) {
        let audit_shard = l.contains("-audit-r");
        assert_eq!(
            l.contains("--seat-weights 2,1,1"),
            !audit_shard && l.contains("--source harvest"),
            "{l}"
        );
    }
    let steps = steps_of(&rung);
    let at = |s: &str| {
        steps
            .iter()
            .position(|x| x == s)
            .unwrap_or_else(|| panic!("no step {s:?} in {steps:?}"))
    };
    assert!(at("targeted harvest (round 1)") < at("train (round 2)"));
    assert!(at("targeted harvest (round 2)") < at("train (round 3)"));
    // The adoption, in the summary and the round reports.
    let summary = fs::read_to_string(out.join("audit/audit-summary.txt")).expect("summary");
    for want in [
        "round 1: the starting network (init_net) taken as it is: the best so far",
        "round 2: NOT adopted (pos 1-2 bound 0.0200% > the best's 0.0100%); the best stays \
         round 1, which round 3 continues",
        "round 3: adopted (no worse than round 1",
        "best: round 3",
    ] {
        assert!(summary.contains(want), "{want} in\n{summary}");
    }
    let r2 = read_json(&out.join("audit/round-2.json")).expect("round 2");
    assert_eq!(r2["additive"]["adopted"], false);
    assert_eq!(r2["additive"]["best_round"], 1);
    assert_eq!(r2["leader_focus"][0]["group"], "pos 1-2");
    // The rejected round passed g-one's mini set, but the best fails it: still failing, still a
    // target.
    assert_eq!(
        r2["groups_failing"][0]["group"], "g-one",
        "{}",
        r2["groups_failing"]
    );
    assert_eq!(r2["targeted"]["why"], "re-targeting 1 groups still failing");
    let t2 = fs::read_to_string(out.join("audit/round-2.txt")).expect("txt");
    assert!(
        t2.contains("Additive: NOT adopted, the best stays round 1: pos 1-2 bound"),
        "{t2}"
    );
    assert!(
        t2.contains("Leader focus (the golden set by the decider's seat; seat 1 leads the trick): seat 1-2: 1 decisions, 95% upper 0.0200%"),
        "{t2}"
    );
    // upload_shards: the targeted shards are kept for a later run (set `extra`, weight 2).
    let kept = read_json(&out.join("audit/label_inputs.json")).expect("label_inputs.json");
    let kept = kept.as_array().expect("entries");
    assert_eq!(kept.len(), 2, "{kept:?}");
    assert_eq!(
        (kept[0]["set"].clone(), kept[0]["weight"].clone()),
        (json!("extra"), json!(2.0))
    );
    assert_eq!(
        kept[1]["seed"],
        json!(params.audit.as_ref().expect("audit").target_seed(2))
    );
    let r3 = read_json(&out.join("audit/round-3.json")).expect("round 3");
    assert_eq!(r3["additive"]["adopted"], true);
    assert_eq!(r3["shape"], "16,8", "the ladder is off");
    let _ = fs::remove_dir_all(&f.base);
}

/// At trick 3 the additive loop needs no `harvest_net`: round 1's audit shard is driven by
/// `init_net`, round 2's by round 1's network (`init_net` again, taken as it is).
#[cfg(unix)]
#[test]
fn additive_loop_at_trick_3_drives_round_1_with_init_net() {
    let f = fakes("additivet3");
    let (i, fl) = (f.base.join("init-int"), f.base.join("init-float"));
    net_file_at(&i, &[16, 8], true, false, 3);
    net_file_at(&fl, &[16, 8], false, false, 3);
    let (int, float) = (store(&f, &i), store(&f, &fl));
    fs::write(f.base.join("audit-1.json"), GROUP_ONE).expect("write");
    fs::write(f.base.join("mini-fail-all"), "").expect("write");
    let tools = audit_tools(&f, false, None);
    let params = parse(
        &json!({ "cut": 3, "labels_random": 7, "labels_rules": 0, "val_count": 3,
                                "golden_count": 2, "golden_worlds": 4, "audit_rows": 50,
                                "audit_mini_decisions": 2, "audit_target_rows": 1000,
                                "audit_rounds": 2, "train_mode": "additive",
                                "init_net": format!("inputs/sha256/{int}"),
                                "init_net_float": format!("inputs/sha256/{float}") }),
    )
    .expect("params");
    assert_eq!(params.widths, vec![1024, 512, 128], "widths not given");
    let root = f.base.join("run");
    let mut ctx = Ctx::local("hybrid-test", &root, 4);
    chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect("the loop runs");
    let log = fs::read_to_string(&f.calls).expect("calls");
    let shards: Vec<&str> = log
        .lines()
        .filter(|l| l.starts_with("search labels") && l.contains("-audit-r"))
        .collect();
    assert_eq!(shards.len(), 2, "{log}");
    let inputs = std::path::absolute(root.join("hybrid-work/inputs")).expect("abs");
    assert!(
        shards[0].contains(&format!("--harvest-net {}", inputs.join(&int).display())),
        "{}",
        shards[0]
    );
    let trains: Vec<&str> = log
        .lines()
        .filter(|l| l.starts_with("python -m fortyone_nn.train"))
        .collect();
    assert!(
        trains.iter().all(|t| t.contains("--widths 16,8 ")),
        "init_net's shape when widths is not given: {trains:?}"
    );
    let _ = fs::remove_dir_all(&f.base);
}

/// The starting network is checked before anything is installed: `widths` that disagree with
/// it, and float weights of another training, stop the chunk at step 0.
#[cfg(unix)]
#[test]
fn additive_init_is_checked_first() {
    let f = fakes("additiveinit");
    let tools = audit_tools(&f, false, None);
    let params = additive_params(&f, json!({ "widths": "32,8" }));
    let mut ctx = Ctx::local("hybrid-test", &f.base.join("a"), 4);
    let e = chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect_err("widths");
    assert!(
        e.contains("widths 32,8 disagree with init_net's 16,8"),
        "{e}"
    );
    let (int, float) = init_nets(&f, &[16, 8], true);
    let mut params = additive_params(&f, json!({}));
    if let Some(a) = params.additive.as_mut() {
        a.init_net = int;
        a.init_float = float;
    }
    let mut ctx = Ctx::local("hybrid-test", &f.base.join("b"), 4);
    let e = chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect_err("record");
    assert!(e.contains("is not init_net's float weights"), "{e}");
    let log = fs::read_to_string(&f.calls).unwrap_or_default();
    assert!(!log.contains("uv sync") && !log.contains("search"), "{log}");
    let _ = fs::remove_dir_all(&f.base);
}

/// The new settings through `parse`: the additive mode with the audit loop (not needing
/// `harvest_net` at trick 3), the reused shards (their sets not generated, their settings and
/// colliding seeds refused) and the seat weights.
#[test]
fn additive_and_reuse_settings() {
    let key = |c: char| format!("inputs/sha256/{}", c.to_string().repeat(64));
    let p = parse(
        &json!({ "cut": 3, "audit_rounds": 2, "train_mode": "additive",
                           "init_net": key('a'), "init_net_float": key('b') }),
    )
    .expect("additive at trick 3 without harvest_net");
    assert_eq!(p.additive.as_ref().map(|a| a.epochs), Some(8));
    let e = parse(&json!({ "cut": 3, "audit_rounds": 2 })).expect_err("scratch needs it");
    assert!(e.contains("needs harvest_net"), "{e}");
    assert!(parse(&json!({})).expect("defaults").additive.is_none());
    let e = parse(&json!({ "train_mode": "additive", "init_net": key('a'),
                           "init_net_float": key('b') }))
    .expect_err("no loop");
    assert!(e.contains("needs the audit loop"), "{e}");
    // Reused shards: the random set and the validation shard are not generated.
    let reuse = json!([{ "set": "random", "seed": 321, "shard": key('c'), "sidecar": key('d') },
                       { "set": "val", "seed": 401, "shard": key('e'), "sidecar": key('f') }]);
    let p = parse(&json!({ "label_inputs": reuse, "labels_rules": 0 })).expect("reuse");
    assert!(p.shards().is_empty(), "{:?}", p.shards());
    assert_eq!(p.label_inputs.len(), 2);
    assert_eq!(p.to_json()["label_inputs"][1]["seed"], 401);
    let p = parse(&json!({ "label_inputs": reuse })).expect("rules still generated");
    assert_eq!(
        p.shards().iter().map(|j| j.file(4)).collect::<Vec<_>>(),
        ["t4-rules-s102.npy"]
    );
    let harvest =
        json!([{ "set": "harvest", "seed": 313, "shard": key('c'), "sidecar": key('d') }]);
    let p = parse(&json!({ "label_inputs": harvest, "harvest_cap_per_decision": 32 }))
        .expect("harvest reused");
    assert_eq!(
        p.reused_harvest_args.as_deref().map(|a| a[1].as_str()),
        Some("32"),
        "the audit loop's harvests keep the harvest's settings"
    );
    for (extra, want) in [
        (
            json!({ "labels_random": 5 }),
            "labels_random is set, but label_inputs gives",
        ),
        (
            json!({ "val_seed": 9 }),
            "val_seed is set, but label_inputs gives",
        ),
        (json!({ "labels_rules_seed": 401 }), "share seed 401"),
        (
            json!({ "audit_rounds": 1, "audit_seed": 4 }),
            "would have seed 401",
        ),
        (json!({ "label_parts": 2 }), "label_parts"),
    ] {
        let mut v = json!({ "label_inputs": reuse });
        for (k, x) in extra.as_object().expect("object") {
            v[k] = x.clone();
        }
        let e = parse(&v).expect_err("refused");
        assert!(e.contains(want), "{extra}: {e}");
    }
    let only_val = json!([{ "set": "val", "seed": 401, "shard": key('e'), "sidecar": key('f') }]);
    let e = parse(&json!({ "label_inputs": only_val, "labels_random": 0, "labels_rules": 0 }))
        .expect_err("no training labels");
    assert!(e.contains("no training labels"), "{e}");
    assert_eq!(
        parse(&json!({ "harvest_seat_weights": "2,1,1" }))
            .expect("weights")
            .seat_weights
            .as_deref(),
        Some("2,1,1")
    );
    assert!(parse(&json!({ "harvest_seat_weights": "2,1" })).is_err());
    let e = parse(&json!({ "train_mode": "fast" })).expect_err("mode");
    assert!(e.contains("train_mode"), "{e}");
}

#[cfg(unix)]
/// A label shard of `rows` zero records at trick 4 and its sidecar, stored; (shard, sidecar)
/// SHA-256.
fn stored_shard(f: &Fakes, name: &str, rows: usize, seed: u64, source: &str) -> (String, String) {
    let v = nn::outcome::vocab(4);
    let path = f.base.join(name);
    let mut bytes = nn::shard::header(rows, v);
    bytes.extend(vec![0u8; rows * nn::shard::row_bytes_for(v)]);
    fs::write(&path, &bytes).expect("shard");
    let sha = store(f, &path);
    let side = f.base.join(format!("{name}.json"));
    let doc = json!({ "format": "41bots-label-shard-v1", "encoder": nn::encode::ENCODER,
                      "n_features": nn::encode::N_FEATURES, "trick": 4,
                      "outcomes": v, "values_version": search::measure::VALUES_VERSION,
                      "seed": seed, "source": source, "count": rows, "shard_sha256": sha });
    fs::write(&side, doc.to_string()).expect("sidecar");
    (sha, store(f, &side))
}

/// Stored shards reused: fetched, checked and trained on (reused first), nothing generated for
/// their sets; a run with `upload_shards` lists every shard as a `label_inputs` entry (the
/// sidecars shared too); a sidecar that does not fit stops the chunk before any training.
#[cfg(unix)]
#[test]
fn stored_label_shards_are_reused() {
    let f = fakes("reuse");
    let (rs, rside) = stored_shard(&f, "r.npy", 7, 321, "random");
    let (vs, vside) = stored_shard(&f, "v.npy", 3, 401, "random");
    let k = |h: &str| format!("inputs/sha256/{h}");
    let reuse = json!([{ "set": "random", "seed": 321, "rows": 7, "shard": k(&rs), "sidecar": k(&rside) },
                       { "set": "val", "seed": 401, "shard": k(&vs), "sidecar": k(&vside) }]);
    let params = parse(
        &json!({ "check": false, "label_inputs": reuse, "labels_rules": 5,
                                "labels_rules_seed": 512, "widths": "16,8", "epochs": 2,
                                "upload_shards": true }),
    )
    .expect("params");
    let remote: Arc<dyn Remote> = Arc::new(DirRemote {
        root: f.base.join("remote"),
    });
    let tools = Tools {
        search: f.search.clone(),
        uv: f.uv.clone(),
        python_dir: f.py.clone(),
        nvidia_smi: f.base.join("no-such-nvidia-smi"),
        remote: Some(Arc::clone(&remote)),
    };
    let root = f.base.join("run");
    let mut ctx = Ctx::local("hybrid-test", &root, 4);
    let r = chunk_with(&mut ctx, &params, &tools, &f.table, json!({})).expect("the chunk runs");
    assert_eq!(r["verdict"], TRAIN_ONLY);
    let log = fs::read_to_string(&f.calls).expect("calls");
    let labels: Vec<&str> = log
        .lines()
        .filter(|l| l.starts_with("search labels"))
        .collect();
    assert_eq!(labels.len(), 1, "only the rules set is made: {log}");
    assert!(labels[0].contains("--source rules"), "{}", labels[0]);
    let work = std::path::absolute(root.join("hybrid-work/labels")).expect("abs");
    let train = log
        .lines()
        .find(|l| l.starts_with("python -m fortyone_nn.train"))
        .expect("trained");
    assert!(
        train.contains(&format!(
            "--train {} {} --val {}",
            work.join("t4-random-s321.npy").display(),
            work.join("t4-rules-s512.npy").display(),
            work.join("t4-val-s401.npy").display()
        )),
        "{train}"
    );
    let out = root.join("out/hybrid");
    let rung = read_json(&out.join("rung.json")).expect("rung.json");
    assert_eq!(rung["label_inputs"]["entries"][0]["seed"], 321);
    assert_eq!(rung["labels"][0]["reused"]["shard"], k(&rs).as_str());
    assert!(out.join("labels/t4-val-s401.npy.json").is_file());
    let steps = steps_of(&rung);
    assert!(steps.iter().any(|s| s == "label inputs"), "{steps:?}");
    // The run's shards, ready for a later run's label_inputs: the reused ones by the same keys.
    let shards = read_json(&out.join("shards.json")).expect("shards.json");
    let entries = shards["label_inputs"].as_array().expect("entries");
    assert_eq!(entries.len(), 3);
    assert_eq!(
        (
            entries[0]["set"].clone(),
            entries[0]["seed"].clone(),
            entries[0]["shard"].clone()
        ),
        (json!("random"), json!(321), json!(k(&rs)))
    );
    assert_eq!(entries[0]["sidecar"], k(&rside).as_str());
    assert_eq!(
        (entries[1]["set"].clone(), entries[1]["seed"].clone()),
        (json!("rules"), json!(512))
    );
    assert!(
        remote
            .exists(entries[1]["sidecar"].as_str().expect("key"))
            .expect("listed"),
        "the generated shard's sidecar is shared"
    );
    assert_eq!(entries[2]["set"], "val");
    let again = parse(
        &json!({ "check": false, "label_inputs": entries, "widths": "16,8",
                               "epochs": 2 }),
    )
    .expect("the entries are a valid label_inputs");
    assert_eq!(again.label_inputs.len(), 3);
    assert!(again.shards().is_empty());
    // A sidecar that does not fit (another seed): refused before any training.
    let bad = json!([{ "set": "random", "seed": 322, "shard": k(&rs), "sidecar": k(&rside) },
                     { "set": "val", "seed": 401, "shard": k(&vs), "sidecar": k(&vside) }]);
    let params = parse(
        &json!({ "check": false, "label_inputs": bad, "labels_rules": 0,
                                "widths": "16,8", "epochs": 2 }),
    )
    .expect("params");
    let _ = fs::remove_file(&f.calls);
    let mut ctx = Ctx::local("hybrid-test", &f.base.join("bad"), 4);
    let e = chunk_with(&mut ctx, &params, &tools, &f.table, json!({})).expect_err("refused");
    assert!(
        e.contains("does not fit this run: its sidecar's seed is 321, not 322"),
        "{e}"
    );
    let log = fs::read_to_string(&f.calls).unwrap_or_default();
    assert!(!log.contains("fortyone_nn.train"), "{log}");
    let _ = fs::remove_dir_all(&f.base);
}

/// Tools for the additive loop at trick 2: the audit loop's fakes, with a `python` whose
/// `fortyone_nn.audit --overall-only` (the paired check) gives round k's network the overall SD in
/// `sd-k` (round 1: `out/hybrid/net`) on any shard.
#[cfg(unix)]
fn tools_with_paired_check(f: &Fakes) -> Tools {
    let inner = audit_tools(f, false, None);
    let arg = r#"arg() { n=$1; shift; while [ $# -gt 0 ]; do [ "$1" = "$n" ] && { echo "$2"; return; }; shift; done; }"#;
    let py = fake_script(
        &f.bin,
        "fake-python-paired",
        &format!(
            r#"{arg}
case " $* " in
  *" --overall-only "*)
    echo "python $*" >> '{calls}'
    net=$(arg --net "$@"); k=1
    case "$net" in */audit/round-*) k=$(echo "$net" | sed 's|.*/audit/round-\([0-9]*\)/net/.*|\1|');; esac
    out=$(arg --out "$@"); mkdir -p "$out"
    echo "{{\"overall_only\": true, \"overall\": {{\"rows\": 50, \"sd\": $(cat '{base}/sd-'$k), \"bias\": [0.0, 0.0, 0.0, 0.0]}}}}" > "$out/report.json"
    exit 0;;
esac
exec '{inner}' "$@"
"#,
            calls = f.calls.display(),
            base = f.base.display(),
            inner = f.bin.join("fake-python-audit").display()
        ),
    );
    let uv = fake_script(
        &f.bin,
        "uv-paired",
        &format!(
            r#"echo "uv $*" >> '{calls}'
case $1 in
  --version) echo "uv 0.13.0";;
  sync) mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"; cp '{py}' "$UV_PROJECT_ENVIRONMENT/bin/python";;
esac
"#,
            calls = f.calls.display(),
            py = py.display()
        ),
    );
    Tools { uv, ..inner }
}

/// The additive loop at trick 2, where no golden set exists: each round's network and the best
/// so far are compared on the same positions, the round's fresh audit shard (the paired check,
/// `--overall-only`, before the shard goes). Round 2 is worse there (overall SD 1.2% against the
/// best's 1.0%): NOT adopted, and round 3 continues round 1 again. Then either round 3 is
/// better and confirms nothing (flat after round 3, its network kept), or it is worse and
/// confirms nothing (the loop ends: nothing new to train on; round 1's network kept).
#[cfg(unix)]
#[test]
fn additive_loop_at_trick_2_compares_on_the_same_audit_shard() {
    for (case, sd3) in [("better", "0.009"), ("worse", "0.011")] {
        let f = fakes(&format!("additivet2{case}"));
        for k in 1..=2 {
            fs::write(f.base.join(format!("audit-{k}.json")), GROUP_ONE).expect("write");
        }
        for (k, sd) in [(1, "0.01"), (2, "0.012"), (3, sd3)] {
            fs::write(f.base.join(format!("sd-{k}")), sd).expect("write");
        }
        let tools = tools_with_paired_check(&f);
        let (i, fl) = (f.base.join("init-int"), f.base.join("init-float"));
        net_file_at(&i, &[16, 8], true, false, 2);
        net_file_at(&fl, &[16, 8], false, false, 2);
        let (int, float) = (store(&f, &i), store(&f, &fl));
        let decisions: String = (0..32)
            .map(|i| format!("{{\"id\":{i},\"trick\":1}}\n"))
            .collect();
        let dec = f.base.join("decisions-t1.jsonl");
        fs::write(&dec, decisions).expect("write");
        let tables = f.base.join("tables.ckpt");
        fs::write(&tables, b"{\"id\":0,\"w\":3}\n").expect("write");
        let (dec, tables) = (store(&f, &dec), store(&f, &tables));
        let params = parse(&json!({
            "cut": 2, "labels_random": 7, "labels_rules": 0, "val_count": 3, "audit_rounds": 4,
            "audit_rows": 50, "audit_target_rows": 1000, "train_mode": "additive",
            "init_net": format!("inputs/sha256/{int}"),
            "init_net_float": format!("inputs/sha256/{float}"),
            "t1_decisions_input": format!("inputs/sha256/{dec}"),
            "t1_saved_inputs": [ { "run": "m3-t1-tables-1", "input": format!("sha256:{tables}") } ] }))
        .expect("params");
        let root = f.base.join("run");
        let mut ctx = Ctx::local("hybrid-test", &root, 4);
        let r = chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect("the loop runs");
        let out = std::path::absolute(root.join("out/hybrid")).expect("abs");
        let stop = r["audit"]["stop"].as_str().expect("stop");
        let rung = read_json(&out.join("rung.json")).expect("rung.json");
        if case == "better" {
            assert!(stop.starts_with("flat after round 3"), "{stop}");
            assert_eq!(rung["net"]["dir"], "audit/round-3/net");
        } else {
            assert_eq!(
                stop,
                "round 3 confirmed no group on its audit shard but was not adopted (worse than \
                 the best on the same positions): nothing new to train on"
            );
            assert_eq!(rung["net"]["dir"], "net", "round 1's network is kept");
            assert_eq!(r["verdict"], "FAIL");
        }
        let log = fs::read_to_string(&f.calls).expect("calls");
        // The paired checks: round 1's network (the best) on rounds 2 and 3's audit shards, and
        // each round's own (the stand-in auditor gives no overall numbers).
        let paired: Vec<&str> = log
            .lines()
            .filter(|l| l.contains("--overall-only"))
            .collect();
        assert_eq!(paired.len(), 4, "{log}");
        assert!(
            paired[0].contains(&format!(
                "--net {}",
                out.join("net/int.safetensors").display()
            )) && paired[0].contains("t2-audit-r2-"),
            "{}",
            paired[0]
        );
        let trains: Vec<&str> = log
            .lines()
            .filter(|l| l.starts_with("python -m fortyone_nn.train"))
            .collect();
        assert_eq!(trains.len(), 3, "{log}");
        assert!(trains[0].contains("--epochs 0 "), "{}", trains[0]);
        for t in &trains[1..] {
            assert!(
                t.contains(&format!(
                    "--init {}",
                    out.join("net/float.safetensors").display()
                )),
                "round 2 was not adopted, so round 3 continues round 1 too: {t}"
            );
        }
        let summary = fs::read_to_string(out.join("audit/audit-summary.txt")).expect("summary");
        assert!(
            summary.contains(
                "round 2: NOT adopted (overall SD 1.2000% > the best's 1.0000%); the best stays \
                 round 1"
            ),
            "{summary}"
        );
        let r2 = read_json(&out.join("audit/round-2.json")).expect("round 2");
        assert_eq!(r2["additive"]["adopted"], false);
        assert_eq!(
            r2["additive"]["compared_with"]["on_this_shard"]["loss_upper95"],
            0.01
        );
        let t2 = fs::read_to_string(out.join("audit/round-2.txt")).expect("txt");
        assert!(
            t2.contains("Additive: NOT adopted, the best stays round 1"),
            "{t2}"
        );
        let _ = fs::remove_dir_all(&f.base);
    }
}
