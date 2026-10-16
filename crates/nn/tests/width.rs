//! The width limit of the integer inference (`nn::MAX_WIDTH`), everywhere a width is accepted:
//! a weights file with a layer over the limit is refused with a message that names the layer and
//! the limit (this is what `search labels --harvest-net` and `search hybrid-check` print when
//! they load it), and `nn-tool` prints the limit for the Python parity test and refuses to build
//! a benchmark network over it.

use std::process::Command;
use std::sync::Arc;

use nn::mlp::{check_hidden_widths, width_error};
use nn::oracle::{self, Oracle};
use nn::safetensors::{self, Tensor};
use nn::synthetic::SyntheticLeaf;
use nn::{MAX_WIDTH, encode::N_FEATURES};

/// A weights file of the `oracle-v2` / `41bots-int-mlp-v1` format whose first hidden layer has
/// `width` units (all weights zero, so it is only the width that can be refused), then 4 logits.
fn wide_net_bytes(width: usize) -> Vec<u8> {
    let mut f = safetensors::File::default();
    f.tensors.insert(
        "w0".into(),
        Tensor::i32(vec![N_FEATURES, width], &vec![0; N_FEATURES * width]),
    );
    f.tensors
        .insert("b0".into(), Tensor::i64(vec![width], &vec![0; width]));
    f.tensors.insert(
        "w1".into(),
        Tensor::i32(vec![width, 4], &vec![0; width * 4]),
    );
    f.tensors.insert("b1".into(), Tensor::i64(vec![4], &[0; 4]));
    f.meta.insert("format".into(), oracle::FORMAT.into());
    f.meta.insert("encoder".into(), nn::encode::ENCODER.into());
    f.meta.insert("n_features".into(), N_FEATURES.to_string());
    f.meta.insert("trick".into(), "4".into());
    f.meta.insert("shifts".into(), "[24, 0]".into());
    f.meta.insert("head".into(), "money4".into());
    safetensors::write(&f)
}

#[test]
fn a_weights_file_over_the_limit_is_refused_with_the_limit_named() {
    let leaf = Arc::new(SyntheticLeaf);
    let e = Oracle::from_bytes(&wide_net_bytes(MAX_WIDTH + 1), leaf.clone())
        .err()
        .expect("a layer of 1025 units is refused");
    assert_eq!(e, width_error(0, MAX_WIDTH + 1));
    assert!(
        e.contains("wider than 1024") && e.contains("layer 0"),
        "{e}"
    );
    // A 2048-unit first layer, as the trick-3 run that failed after 27 minutes of training had.
    let e = Oracle::from_bytes(&wide_net_bytes(2048), leaf.clone())
        .err()
        .expect("refused");
    assert!(e.contains("2048 units") && e.contains("MAX_WIDTH"), "{e}");
    // The limit itself loads.
    assert!(Oracle::from_bytes(&wide_net_bytes(MAX_WIDTH), leaf).is_ok());
}

#[test]
fn nn_tool_prints_the_limit_and_refuses_a_wider_network() {
    let tool = env!("CARGO_BIN_EXE_nn-tool");
    let out = Command::new(tool).arg("limits").output().expect("runs");
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        format!("max_width={MAX_WIDTH}")
    );
    let dir = std::env::temp_dir().join(format!("nn-width-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let target = dir.join("net.safetensors");
    let run = |widths: &str| {
        Command::new(tool)
            .args(["random", "--trick", "4", "--widths", widths, "--out"])
            .arg(&target)
            .output()
            .expect("runs")
    };
    let bad = run("2048,1024,256");
    assert!(!bad.status.success());
    let err = String::from_utf8_lossy(&bad.stderr);
    assert!(
        err.contains("--widths 2048,1024,256") && err.contains(&width_error(0, 2048)),
        "{err}"
    );
    assert!(!target.exists(), "nothing is written for a refused width");
    assert!(run("1024,512,128").status.success());
    assert_eq!(check_hidden_widths(&[1024, 512, 128]), Ok(()));
    std::fs::remove_dir_all(&dir).ok();
}
