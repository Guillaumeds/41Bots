//! A root shortlist (`FastOptions::root_keep`, docs/m3-hybrid/path-forward.md step 0c): the cards
//! searched get exactly the values a full solve gives them, the choice is the best of them under
//! the same tie order, the sequential and the parallel solve agree, and fewer root cards cost
//! fewer nodes.

mod common;

use common::{leaf, position_mid};
use engine::cards::bit;
use solver::{FastOptions, TieOrder, from_match, solve_fast, solve_fast_par};

#[test]
fn shortlisted_root_cards_keep_their_full_solve_values() {
    let (mut full_nodes, mut kept_nodes, mut cases) = (0u64, 0u64, 0u32);
    for seed in 0..6 {
        for (k, extra) in [(4, 0), (3, 2), (5, 1), (3, 0)] {
            let m = position_mid(seed, k, extra);
            let (pos, ctx, pending) = from_match(&m).expect("position");
            assert!(pending.is_none());
            for order in [TieOrder::Ascending, TieOrder::Descending] {
                let base = FastOptions {
                    order,
                    tt_mb: 16,
                    ..FastOptions::default()
                };
                let full = solve_fast(&pos, ctx, &leaf(), base);
                let cards: Vec<u8> = full.children.iter().map(|&(c, _)| c).collect();
                if cards.len() < 3 {
                    continue;
                }
                // One card, the first two, the last two, and all but the first.
                let picks: Vec<Vec<u8>> = vec![
                    vec![cards[0]],
                    cards[..2].to_vec(),
                    cards[cards.len() - 2..].to_vec(),
                    cards[1..].to_vec(),
                    cards.clone(),
                ];
                for keep in picks {
                    let mask = keep.iter().fold(0, |m, &c| m | bit(c));
                    let opts = FastOptions {
                        root_keep: mask,
                        ..base
                    };
                    let seq = solve_fast(&pos, ctx, &leaf(), opts);
                    let par = solve_fast_par(&pos, ctx, &leaf(), opts, 3, None);
                    let what = format!("seed {seed} trick {k}+{extra} {order:?} keep {keep:?}");
                    assert_eq!(seq.children, par.children, "{what}: seq against par");
                    assert_eq!(seq.best, par.best, "{what}: best, seq against par");
                    for &c in &keep {
                        let a = full.children.iter().find(|&&(x, _)| x == c).unwrap();
                        let b = seq.children.iter().find(|&&(x, _)| x == c);
                        assert_eq!(Some(a), b, "{what}: the value of card {c}");
                    }
                    for child in &seq.children {
                        assert!(full.children.contains(child), "{what}: a value not in full");
                    }
                    // The choice is the first best in the tie order among the cards searched.
                    let team = engine::team_of(pos.to_act());
                    let top = seq.children.iter().map(|(_, v)| v.get(team)).max().unwrap();
                    assert_eq!(seq.value.unwrap().get(team), top, "{what}: value");
                    assert_eq!(
                        seq.children
                            .iter()
                            .find(|(_, v)| v.get(team) == top)
                            .map(|&(c, _)| c),
                        seq.best,
                        "{what}: choice"
                    );
                    if keep.len() == cards.len() {
                        assert_eq!(seq.children, full.children, "{what}: all cards");
                        assert_eq!(seq.best, full.best, "{what}: all cards, best");
                    } else if keep.len() == 1 {
                        full_nodes += full.stats.nodes;
                        kept_nodes += seq.stats.nodes;
                        cases += 1;
                    }
                }
            }
        }
    }
    assert!(cases > 10, "too few cases: {cases}");
    assert!(
        kept_nodes < full_nodes,
        "one root card should cost fewer nodes ({kept_nodes} against {full_nodes})"
    );
    eprintln!("one root card: {kept_nodes} nodes against {full_nodes} for all ({cases} cases)");
}
