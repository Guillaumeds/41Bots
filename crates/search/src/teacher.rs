//! Teacher hands and their output files (docs/M3-SPEC.md §5.10).
//!
//! One teacher hand starts with `Match::from_hand_start` from a stratified start state; all eight
//! seats are played by the PIMC teacher and every decision's values are recorded.
//! - `hands-<sha16>.jsonl`: one JSON line per hand (format version, rules version, equity sha256,
//!   start state, deck, actions, final money chance or successor state, bot name, git commit).
//! - `decisions-<sha16>.npy`: a little-endian structured array, one row per decision, with the
//!   fields of [`DECISION_DTYPE`]. Network inputs are not stored: they are rebuilt by replaying
//!   the hands.

use std::sync::Arc;

use bot::Bot;
use engine::cards::DECK_SIZE;
use engine::game::{Action, HandStartState, Match, Phase, RULES_VERSION, Rules};
use engine::rng::Rng;
use serde_json::json;

use crate::bot::{EQUITY_SHA256, PimcFactory};
use crate::labels;
use crate::pimc::{BeliefMode, DecisionKind, DecisionValues, Method};

/// Format version of the teacher files.
pub const FORMAT_VERSION: u32 = 1;

/// Maximum candidates per decision row.
pub const MAX_CANDS: usize = 32;

/// The numpy dtype of a decision row (packed, little-endian).
pub const DECISION_DTYPE: &str = "[('hand_id', '<u8'), ('action_index', '<u2'), ('seat', 'u1'), ('trick', 'u1'), ('kind', 'u1'), ('method', 'u1'), ('belief', 'u1'), ('n_worlds', '<u4'), ('log2_world_count_q8', '<u2'), ('cand_action', '<u2', (32,)), ('values', '<u8', (32, 4)), ('own_sd', '<u8', (32,)), ('chosen', 'u1')]";

/// Bytes per decision row.
pub const ROW_BYTES: usize = 8 + 2 + 1 + 1 + 1 + 1 + 1 + 4 + 2 + 2 * 32 + 8 * 32 * 4 + 8 * 32 + 1;

/// One decision of a teacher hand.
#[derive(Clone, Debug)]
pub struct TeacherDecision {
    /// Hand id.
    pub hand_id: u64,
    /// Index of the action in the hand.
    pub action_index: u16,
    /// Deciding seat.
    pub seat: u8,
    /// Trick.
    pub trick: u8,
    /// Values.
    pub dv: DecisionValues,
}

/// Encode one decision row.
pub fn encode_row(d: &TeacherDecision, out: &mut Vec<u8>) {
    let start = out.len();
    let dv = &d.dv;
    out.extend_from_slice(&d.hand_id.to_le_bytes());
    out.extend_from_slice(&d.action_index.to_le_bytes());
    out.push(d.seat);
    out.push(d.trick);
    out.push(match dv.kind {
        DecisionKind::Play => 0,
        DecisionKind::Burn => 1,
        DecisionKind::JointBurnPlay => 2,
        DecisionKind::DealerBurn => 3,
    });
    out.push(match dv.method {
        Method::RolloutPimc => 0,
        Method::ExactPimc => 1,
        Method::Forced => 2,
    });
    out.push(match dv.belief {
        BeliefMode::Uniform => 0,
    });
    out.extend_from_slice(&dv.n_worlds.to_le_bytes());
    // ⌊256·log2 N⌋ in integers (0 when not counted).
    let lq = dv.world_count.map_or(0u16, belief::log2_q8);
    out.extend_from_slice(&lq.to_le_bytes());
    assert!(dv.candidates.len() <= MAX_CANDS, "at most 32 candidates");
    for i in 0..MAX_CANDS {
        let code: u16 = dv.candidates.get(i).map_or(0, |&(a, p, _)| match (a, p) {
            (Action::Burn(b), Some(c)) => u16::from(b) * 64 + u16::from(c) + 1,
            (a, _) => u16::from(a.card()) + 1,
        });
        out.extend_from_slice(&code.to_le_bytes());
    }
    for i in 0..MAX_CANDS {
        let v = dv.candidates.get(i).map_or([0; 4], |c| c.2.0);
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    for i in 0..MAX_CANDS {
        out.extend_from_slice(&dv.own_sd.get(i).copied().unwrap_or(0).to_le_bytes());
    }
    out.push(dv.chosen as u8);
    debug_assert_eq!(out.len() - start, ROW_BYTES);
}

/// The `.npy` file of decision rows.
pub fn decisions_npy(rows: &[TeacherDecision]) -> Vec<u8> {
    let mut header = format!(
        "{{'descr': {DECISION_DTYPE}, 'fortran_order': False, 'shape': ({},), }}",
        rows.len()
    );
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + rows.len() * ROW_BYTES);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for r in rows {
        encode_row(r, &mut out);
    }
    out
}

/// Play one teacher hand from `start` with `deck`; returns the hand's JSON line and its decisions.
pub fn teacher_hand(
    hand_id: u64,
    start: HandStartState,
    deck: [u8; DECK_SIZE],
    factory: &PimcFactory,
    bot_seed: u64,
    commit: &str,
) -> (serde_json::Value, Vec<TeacherDecision>) {
    let mut m = Match::from_hand_start(start, deck, 0, Rules::HOUSE).expect("valid start");
    let log: crate::bot::DecisionLog = Arc::default();
    let mut bots: Vec<_> = (0..8u8)
        .map(|s| {
            let mut b = factory.build_pimc();
            b.log = Some(log.clone());
            b.start_match(
                s,
                Rules::HOUSE,
                Rng::derive(bot_seed, &[u64::from(s)]).next_u64(),
            );
            b
        })
        .collect();
    let mut decisions = Vec::new();
    let h = m.hand_index();
    let mut idx = 0u16;
    while m.phase() != Phase::MatchOver && m.hand_index() == h {
        let seat = m.to_act().expect("to act");
        let view = m.view(seat);
        let legal = m.legal_actions();
        let before = log.lock().map(|l| l.len()).unwrap_or(0);
        let a = bots[usize::from(seat)].act(&view, &legal);
        let l = log.lock().unwrap_or_else(|e| e.into_inner());
        if l.len() > before {
            decisions.push(TeacherDecision {
                hand_id,
                action_index: idx,
                seat,
                trick: view.public.trick_no,
                dv: l[before].1.clone(),
            });
        }
        drop(l);
        m.apply(a).expect("legal");
        idx += 1;
    }
    let end = match m.money() {
        Some(_) => json!({ "match_over": true, "money": m.money() }),
        None => {
            let p = m.public();
            json!({ "next": { "scores": p.scores, "games": p.games_won, "dealer": p.dealer } })
        }
    };
    let actions: Vec<serde_json::Value> = m
        .actions()
        .iter()
        .map(|a| match a {
            Action::Burn(c) => json!({ "burn": c }),
            Action::Play(c) => json!({ "play": c }),
        })
        .collect();
    let line = json!({
        "format": FORMAT_VERSION, "rules_version": RULES_VERSION, "equity_sha256": EQUITY_SHA256,
        "hand_id": hand_id, "start": start, "deck": deck.to_vec(), "actions": actions,
        "end": end, "bot": factory_name(factory), "git_commit": commit,
    });
    (line, decisions)
}

fn factory_name(f: &PimcFactory) -> String {
    bot::BotFactory::name(f).to_string()
}

/// The deck of teacher hand `i` of a chunk.
pub fn teacher_deck(chunk_seed: u64, i: u64) -> [u8; DECK_SIZE] {
    let mut d: [u8; DECK_SIZE] = core::array::from_fn(|k| k as u8);
    Rng::derive(chunk_seed, &[labels::TEACHER_HAND, i]).shuffle(&mut d);
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use equity::MoneyChance;

    #[test]
    fn row_size_matches_dtype() {
        let dv = DecisionValues {
            kind: DecisionKind::Play,
            method: Method::Forced,
            n_worlds: 0,
            world_count: Some(1 << 20),
            belief: BeliefMode::Uniform,
            candidates: vec![(Action::Play(3), None, MoneyChance::even())],
            chosen: 0,
            own_sd: vec![0],
        };
        let rows = vec![TeacherDecision {
            hand_id: 1,
            action_index: 2,
            seat: 3,
            trick: 4,
            dv,
        }];
        let b = decisions_npy(&rows);
        let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
        assert_eq!((10 + hl) % 64, 0);
        assert_eq!(b.len(), 10 + hl + ROW_BYTES);
        // log2 of 2^20 → 20·256.
        let off = 10 + hl + 8 + 2 + 5 + 4;
        assert_eq!(u16::from_le_bytes([b[off], b[off + 1]]), 20 * 256);
    }
}
