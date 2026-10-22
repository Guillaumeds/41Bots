//! The O3G2 round (`hybrid/round.rs`, docs/m3-hybrid/o3-rounds.md) end to end with fake programs:
//! the risk profile, the nested parts, the candidates trained and checked, the winner, the next
//! G2 and the record a next round starts from.

#[cfg(unix)]
use super::additive::{net_file_at, store};
use super::*;

/// A stored trick-3 label shard of `rows` zero rows and its sidecar; their SHA-256.
#[cfg(unix)]
fn stored_t3(f: &Fakes, name: &str, rows: usize, seed: u64, source: &str) -> (String, String) {
    let v = nn::outcome::vocab(3);
    let path = f.base.join(name);
    let mut bytes = nn::shard::header(rows, v);
    bytes.extend(vec![0u8; rows * nn::shard::row_bytes_for(v)]);
    fs::write(&path, &bytes).expect("shard");
    let sha = store(f, &path);
    let side = f.base.join(format!("{name}.json"));
    let doc = json!({ "format": "41bots-label-shard-v1", "encoder": nn::encode::ENCODER,
                      "n_features": nn::encode::N_FEATURES, "trick": 3,
                      "outcomes": v, "values_version": search::measure::VALUES_VERSION,
                      "seed": seed, "source": source, "count": rows, "shard_sha256": sha });
    fs::write(&side, doc.to_string()).expect("sidecar");
    (sha, store(f, &side))
}

/// Fake programs for a round: a `search` whose labels (and their sidecars) come from the step
/// tests' fake, with the risk table when `--harvest-risk` is given; whose `risk-profile` writes a
/// profile; whose `hybrid-check` gives an overall 95% bound by network (the incumbent 0.00007,
/// candidate 1 0.00005, candidate 2 0.0002, candidate 3 0.00006: candidate 2 loses the overall
/// pass) with the seat bands; and whose `hybrid-paired` writes a comparison. A `python` (on a GPU)
/// whose trainer writes a network, proves an init, and whose leaf-risk steps and label comparison
/// write their files.
#[cfg(unix)]
fn round_tools(f: &Fakes) -> Tools {
    let arg = r#"arg() { n=$1; shift; while [ $# -gt 0 ]; do [ "$1" = "$n" ] && { echo "$2"; return; }; shift; done; }"#;
    let calls = f.calls.display().to_string();
    let search = fake_script(
        &f.bin,
        "search-round",
        &format!(
            r#"{arg}
case $1 in
  labels)
    '{search}' "$@" || exit 1
    out=$(arg --out "$@")
    if [ -n "$(arg --harvest-risk "$@")" ]; then
      n=$(arg --count "$@")
      echo '{{"shard_sha256": "{sha}", "count": '$n', "rows": '$n', "risk": {{"rows_by_band": ['$n']}}}}' > "$out.json"
    fi;;
  risk-profile)
    echo "search $*" >> '{calls}'
    echo '{{"format": "41bots-risk-profile-v1", "power": "1", "table": [], "yield_ppm": 1000}}' > "$(arg --out "$@")";;
  hybrid-paired)
    echo "search $*" >> '{calls}'
    echo '[{{"measure": "loss all", "diff": -0.00001}}]' > "$(arg --json "$@")"
    echo "paired: report";;
  hybrid-check)
    echo "search $*" >> '{calls}'
    net=$(arg --net "$@")
    case "$net" in
      */round/c1/*) up=0.00005;;
      */round/c2/*) up=0.0002;;
      */round/c3/*) up=0.00006;;
      *) up=0.00007;;
    esac
    ok=$(awk -v u="$up" 'BEGIN {{ print (u <= 0.0001) ? "true" : "false" }}')
    echo "{{\"pass\": $ok, \"groups\": [{{\"group\": \"all\", \"summary\": {{\"decisions\": 2, \"loss_mean\": 0.00002, \"loss_upper95\": $up, \"loss_pass\": $ok, \"bias\": [0.00001, 0.0, 0.0, 0.0], \"bias_pass\": true}}}}, {{\"group\": \"pos 1-2\", \"summary\": {{\"loss_upper95\": 0.00008}}}}, {{\"group\": \"pos 1\", \"summary\": {{\"loss_mean\": 0.00004, \"loss_upper95\": 0.00009}}}}], \"net\": {{\"sha256\": \"abc\"}}, \"values\": {{}}, \"time\": {{\"hybrid_ms_per_world\": 1.0}}, \"decisions\": []}}" > "$(arg --json "$@")"
    echo "standard: $ok";;
  *) exec '{search}' "$@";;
esac
"#,
            search = f.search.display(),
            sha = f.sha,
        ),
    );
    let py = fake_script(
        &f.bin,
        "fake-python-round",
        &format!(
            r#"{arg}
echo "python $*" >> '{calls}'
case "$2" in
  fortyone_nn.device) echo '{{"device": "cuda", "gpu_name": "RTX A2000", "reason": "every GPU check passed"}}' > "$(arg --json "$@")";;
  fortyone_nn.train)
    out=$(arg --out "$@"); mkdir -p "$out"
    case "$*" in *--check-init-only*) echo '{{"passed": true}}' > "$out/init-check.json"; exit 0;; esac
    echo "int $out" > "$out/int.safetensors"; echo "float $out" > "$out/float.safetensors"
    echo '{{"history": [{{"val_loss": 1.5}}], "best_epoch": 0, "float": {{}}, "int": {{"error": {{"sd": 0.01}}}}, "minutes": 0.1, "device": {{}}}}' > "$out/report.json"
    echo '{{"step": "done", "epoch": 1}}' > "$(arg --progress "$@")";;
  fortyone_nn.leafrisk)
    case "$3" in
      prepare) echo npz > "$(arg --out "$@")";;
      train) echo '{{"format": "41bots-leafrisk-v1", "trees": [], "base": 0}}' > "$(arg --out "$@")";;
      calibrate) echo '{{"thresholds": []}}' > "$(arg --out "$@")";;
    esac;;
  fortyone_nn.labelcmp)
    echo '{{"all": {{"rows": 3, "rms_pct": 0.5, "largest_abs_bias_pct": 0.01}}}}' > "$(arg --out "$@")";;
esac
"#,
        ),
    );
    let uv = fake_script(
        &f.bin,
        "uv-round",
        &format!(
            r#"echo "uv $*" >> '{calls}'
case $1 in
  --version) echo "uv 0.13.0";;
  sync) mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"; cp '{py}' "$UV_PROJECT_ENVIRONMENT/bin/python";;
esac
"#,
            py = py.display()
        ),
    );
    Tools {
        search,
        uv,
        python_dir: f.py.clone(),
        nvidia_smi: f.gpu.clone(),
        remote: Some(Arc::new(crate::remote::DirRemote {
            root: f.base.join("remote"),
        })),
    }
}

/// A round of three candidates on a (fake) GPU, from a stored base and a stored G2: the profile
/// first, then the three nested parts (2, 2 and 4 rows), each stored; candidate 1 trains as soon
/// as part 1 is in; candidates 1 and 2 continue the incumbent, candidate 3 trains a wider network
/// from scratch; every network and the incumbent get the golden check and the label check; the
/// candidates are compared with the incumbent; candidate 2 loses the overall pass and is not
/// adopted, candidates 1 and 3 are, and candidate 1 (the lower bound) wins; its G2 is built; the
/// record names the next round's start (the winner, its G2, the base and every part).
#[cfg(unix)]
#[test]
fn a_round_trains_checks_and_picks_the_winner() {
    let f = fakes("round");
    let (i, fl) = (f.base.join("init-int"), f.base.join("init-float"));
    net_file_at(&i, &[16, 8], true, false, 3);
    net_file_at(&fl, &[16, 8], false, false, 3);
    let (int, float) = (store(&f, &i), store(&f, &fl));
    let model = f.base.join("g2.json");
    fs::write(
        &model,
        json!({ "format": nn::leafrisk::FORMAT, "encoder": nn::encode::ENCODER,
                "n_inputs": nn::leafrisk::N_INPUTS, "extras": nn::leafrisk::EXTRA_NAMES,
                "net_sha256": int, "trick": 3, "base": 1,
                "trees": [{"f": [-1], "t": [0], "l": [0], "r": [0], "v": [0]}] })
        .to_string(),
    )
    .expect("model");
    let model = store(&f, &model);
    let (rs, rside) = stored_t3(&f, "r.npy", 7, 321, "random");
    let (vs, vside) = stored_t3(&f, "v.npy", 3, 401, "random");
    let check = f.base.join("check.npy");
    fs::write(&check, "exact\n").expect("shard");
    let check_side = f.base.join("check.npy.json");
    fs::write(
        &check_side,
        json!({ "trick": 2, "rows": 3, "seed": 601, "source": "random", "start_hands": 20000 })
            .to_string(),
    )
    .expect("sidecar");
    let (cs, cside) = (store(&f, &check), store(&f, &check_side));
    let dec = f.base.join("decisions-t2.jsonl");
    fs::write(&dec, b"{\"id\":0,\"trick\":2}\n{\"id\":1,\"trick\":2}\n").expect("write");
    let wor = f.base.join("exact-t2-n4.jsonl");
    fs::write(
        &wor,
        b"{\"id\":0,\"method\":\"exact\"}\n{\"id\":1,\"method\":\"exact\"}\n",
    )
    .expect("write");
    let (dec, wor) = (store(&f, &dec), store(&f, &wor));
    let k = |h: &str| format!("inputs/sha256/{h}");
    let params = parse(&json!({
        "cut": 3, "label_inputs": [
            { "set": "random", "seed": 321, "rows": 7, "shard": k(&rs), "sidecar": k(&rside) },
            { "set": "val", "seed": 401, "rows": 3, "shard": k(&vs), "sidecar": k(&vside) }],
        "labels_rules": 0, "golden_count": 2, "golden_worlds": 4,
        "golden_decisions_input": k(&dec), "golden_worlds_input": k(&wor), "verify_pct": 1,
        "train_mode": "additive", "init_net": k(&int), "init_net_float": k(&float),
        "upload_shards": true,
        "round": { "id": "o3r-t", "seed": 93000, "risk_model": k(&model),
                   "candidates": [ { "rows": 2 }, { "rows": 4 },
                                   { "rows": 8, "mode": "scratch", "widths": "32,16" } ],
                   "label_check": { "shard": k(&cs), "sidecar": k(&cside) },
                   "g2": { "splits": [ { "name": "train", "count": 6, "seed": 93101 },
                                       { "name": "cal", "count": 4, "seed": 93102 },
                                       { "name": "test", "count": 4, "seed": 93103 } ] } }
    }))
    .expect("params");
    let tools = round_tools(&f);
    let root = f.base.join("run");
    let mut ctx = Ctx::local("hybrid-test", &root, 16);
    let r = chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect("the round runs");
    assert_eq!(r["winner"], "c1", "{r}");
    assert_eq!(r["verdict"], "round o3r-t: winner c1");
    let out = std::path::absolute(root.join("out/hybrid/round")).expect("abs");
    let rec = read_json(&out.join("round.json")).expect("round.json");
    let adopted: Vec<Value> = rec["candidates"]
        .as_array()
        .expect("candidates")
        .iter()
        .map(|c| c["adopted"].clone())
        .collect();
    assert_eq!(adopted, vec![json!(true), json!(false), json!(true)]);
    assert_eq!(rec["winner"]["name"], "c1");
    assert_eq!(rec["parts"].as_array().map(Vec::len), Some(3));
    assert_eq!(
        rec["parts"]
            .as_array()
            .expect("parts")
            .iter()
            .map(|p| p["seed"].as_u64().expect("seed"))
            .collect::<Vec<_>>(),
        vec![93001, 93002, 93003]
    );
    assert!(rec["candidates"][0]["label_check"]["all"]["rms_pct"].is_number());
    assert!(rec["incumbent"]["label_check"]["all"]["rms_pct"].is_number());
    assert!(rec["candidates"][0]["paired_with_incumbent"].is_array());
    // The next round's start: the winner's networks, its new G2, the base and the three parts.
    let next = &rec["next"];
    assert_eq!(next["init_net"], rec["candidates"][0]["int_key"]);
    assert!(
        next["init_net"]
            .as_str()
            .is_some_and(|x| x.starts_with("inputs/sha256/"))
    );
    assert!(
        next["init_net_float"]
            .as_str()
            .is_some_and(|x| x.starts_with("inputs/sha256/"))
    );
    assert!(
        next["risk_model"]
            .as_str()
            .is_some_and(|x| x.starts_with("inputs/sha256/"))
    );
    assert_ne!(next["risk_model"], json!(k(&model)), "the winner's own G2");
    let ds = next["label_inputs"].as_array().expect("dataset");
    assert_eq!(ds.len(), 5, "{ds:?}");
    assert_eq!(ds[0]["shard"], json!(k(&rs)));
    assert_eq!(ds[1]["set"], "val");
    assert!(ds[2..].iter().all(|e| e["set"] == "extra"));
    assert!(next["dataset_key"].as_str().is_some());
    assert_eq!(next["golden_decisions_input"], json!(k(&dec)));
    // Stored where the next round reads it.
    let stored = f.base.join("remote/rounds/o3r-t/round.json");
    assert_eq!(
        read_json(&stored).expect("stored round.json")["winner"]["name"],
        "c1"
    );
    assert!(f.base.join("remote/rounds/o3r-t/report.txt").is_file());
    let text = fs::read_to_string(out.join("report.txt")).expect("report");
    assert!(text.contains("Winner: c1"), "{text}");
    assert!(text.contains("NOT adopted"), "{text}");
    // The commands.
    let log = fs::read_to_string(&f.calls).expect("calls");
    let at = |p: &str| {
        log.lines()
            .position(|l| l.contains(p))
            .unwrap_or_else(|| panic!("no {p:?} in {log}"))
    };
    assert!(at("search risk-profile") < at("--risk-profile"));
    let profile = log
        .lines()
        .find(|l| l.starts_with("search risk-profile"))
        .expect("profile");
    for want in [
        "--seed 93000",
        "--power 1 --floor 0.2",
        "--harvest-tricks 2",
        "--harvest-positions 0.4,0.3,0.3",
    ] {
        assert!(profile.contains(want), "{profile}");
    }
    let parts: Vec<&str> = log
        .lines()
        .filter(|l| l.starts_with("search labels") && l.contains("--harvest-risk"))
        .collect();
    assert_eq!(parts.len(), 3, "{log}");
    for (p, (rows, seed)) in parts.iter().zip([(2, 93001), (2, 93002), (4, 93003)]) {
        assert!(
            p.contains(&format!("--count {rows} --seed {seed}"))
                && p.contains("--risk-profile")
                && p.contains("--harvest-cap 1000"),
            "{p}"
        );
    }
    let trains: Vec<&str> = log
        .lines()
        .filter(|l| {
            l.starts_with("python -m fortyone_nn.train") && !l.contains("--check-init-only")
        })
        .collect();
    assert_eq!(trains.len(), 3, "{log}");
    let inputs = std::path::absolute(root.join("hybrid-work/inputs")).expect("abs");
    for (t, n) in trains[..2].iter().zip(1..) {
        assert!(
            t.contains("--epochs 8 ")
                && t.contains("--widths 16,8 ")
                && t.contains(&format!("--init {}", inputs.join(&float).display()))
                && t.contains("--device cuda"),
            "{t}"
        );
        // Candidate n trains on the base and parts 1..=n, then the validation shard.
        assert!(
            t.contains(&format!("-p{n}-s")) && !t.contains(&format!("-p{}-s", n + 1)),
            "{t}"
        );
    }
    assert!(
        trains[2].contains("--epochs 40 ")
            && trains[2].contains("--widths 32,16 ")
            && trains[2].contains("--lr 0.001 ")
            && !trains[2].contains("--init "),
        "{}",
        trains[2]
    );
    assert!(trains[2].contains("-p3-s93003"), "{}", trains[2]);
    // Candidate 1 trains beside the later parts: started before part 2's labels.
    assert!(at("c1/net") < at("--seed 93002"), "{log}");
    // The golden checks: the incumbent (rechecking 1% of the worlds), then the three candidates.
    let checks: Vec<&str> = log
        .lines()
        .filter(|l| l.starts_with("search hybrid-check"))
        .collect();
    assert_eq!(checks.len(), 4, "{log}");
    assert!(
        checks[0].contains("--verify-pct 1 ") && checks[0].contains(&int),
        "{}",
        checks[0]
    );
    assert!(checks[1..].iter().all(|c| c.contains("--verify-pct 0 ")));
    assert_eq!(
        log.lines()
            .filter(|l| l.starts_with("search hybrid-paired"))
            .count(),
        3
    );
    // The label checks: the stored shard's seed and source, labelled by each network.
    let lc: Vec<&str> = log
        .lines()
        .filter(|l| l.starts_with("search labels") && l.contains("--label-net"))
        .collect();
    assert_eq!(lc.len(), 4, "{log}");
    assert!(
        lc.iter()
            .all(|l| l.contains("--trick 2 --count 3 --seed 601 --source random"))
    );
    // The winner's G2: its splits harvested by the winner.
    assert!(
        log.lines().any(|l| l.starts_with("search labels")
            && l.contains("--seed 93101")
            && l.contains("round/c1/net/int.safetensors")),
        "{log}"
    );
    assert!(log.contains("fortyone_nn.leafrisk calibrate"));
    let _ = fs::remove_dir_all(&f.base);
}

/// A round needs additive mode, a stored golden set and stored shards, and refuses the audit loop.
#[test]
fn a_round_needs_its_rung_settings() {
    let k = |c: char| format!("inputs/sha256/{}", c.to_string().repeat(64));
    let base = || {
        json!({ "cut": 3, "train_mode": "additive", "init_net": k('a'), "init_net_float": k('b'),
                "golden_count": 2, "golden_decisions_input": k('c'), "golden_worlds_input": k('d'),
                "upload_shards": true, "labels_rules": 0,
                "label_inputs": [
                    { "set": "random", "seed": 321, "rows": 7, "shard": k('1'), "sidecar": k('2') },
                    { "set": "val", "seed": 401, "rows": 3, "shard": k('3'), "sidecar": k('4') }],
                "round": { "id": "o3r-1", "seed": 93000, "risk_model": k('e'),
                           "candidates": [ { "rows": 10 } ] } })
    };
    assert!(parse(&base()).expect("valid").round.is_some());
    let bad = |f: &dyn Fn(&mut Value), want: &str| {
        let mut v = base();
        f(&mut v);
        let e = parse(&v).expect_err("refused");
        assert!(e.contains(want), "{e}");
    };
    bad(
        &|v| {
            v.as_object_mut().expect("o").remove("train_mode");
            v.as_object_mut().expect("o").remove("init_net");
            v.as_object_mut().expect("o").remove("init_net_float");
        },
        "train_mode \"additive\"",
    );
    bad(
        &|v| {
            v.as_object_mut()
                .expect("o")
                .remove("golden_decisions_input");
            v.as_object_mut().expect("o").remove("golden_worlds_input");
        },
        "stored golden set",
    );
    bad(&|v| v["upload_shards"] = json!(false), "upload_shards");
    bad(&|v| v["round"]["seed"] = json!(320), "used twice");
    bad(&|v| v["labels_rules"] = json!(5), "its whole base stored");
}
