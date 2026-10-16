//! Adversarial probes of the exact solver (review 2026-10-15): hand-made positions with known
//! answers for the naive oracle, late-match positions (game 2 or 3 of a match, scores near 41,
//! games won mid-hand, certain money at the leaves), a degenerate leaf table full of certain
//! wins and exact half-splits (to hit every equality branch of shallow pruning and the
//! certain-win stop), the equivalent-card claim checked directly with the naive solver, and the
//! parallel solver on 1-4 threads with several split sizes.

mod common;

use common::{L, brute, leaf, position_mid, tie_leaf};
use engine::cards::{Bits, Card, JOKER, Suit, bit, has, parse};
use engine::game::{Action, Match, Phase, Rules};
use engine::rng::Rng;
use engine::rules::{ace_of_trumps, five_of};
use equity::hand::InHand;
use equity::{MatchState, MoneyChance, ONE};
use solver::equiv::Classes;
use solver::{
    FastOptions, FnLeaf, MatchCtx, Options, Position, Rob, Solution, SolveMem, Step, TieOrder,
    from_match, solve, solve_fast, solve_fast_par, solve_fast_par_split, solve_fast_with,
};

const ALL: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

fn same(a: &Solution, b: &Solution, what: &str) {
    assert!(!a.timed_out && !b.timed_out, "{what}: timed out");
    assert_eq!(a.value, b.value, "{what}: value");
    assert_eq!(a.best, b.best, "{what}: chosen card");
    assert_eq!(a.children, b.children, "{what}: root card values");
    assert_eq!(a.root_tied, b.root_tied, "{what}: root ties");
}

/// A degenerate leaf: certain wins, exact two-way and three-way splits, near-certain values and
/// zeros, from a hash of the state (not symmetric in teams). Every value sums to `ONE`.
fn extreme_value(s: &MatchState) -> MoneyChance {
    let mut h: u64 = 0x9E37_79B9_7F4A_7C15;
    for b in s
        .scores
        .iter()
        .chain(&s.games)
        .chain(core::iter::once(&s.dealer_team))
    {
        h = (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01B3);
        h ^= h >> 29;
    }
    let a = (h >> 8) as usize % 4;
    let b = (a + 1 + (h >> 12) as usize % 3) % 4;
    let mut w = [0u128; 4];
    match h % 6 {
        0 => w[a] = 1,
        1 => {
            w[a] = 1;
            w[b] = 1;
        }
        2 => {
            w = [1, 1, 1, 1];
            w[a] = 0;
        }
        3 => {
            w[a] = u128::from(ONE) - 1;
            w[b] = 1;
        }
        4 => {
            w[a] = 3;
            w[b] = 1;
        }
        _ => w = core::array::from_fn(|t| u128::from((h >> (16 * t)) & 0x3)),
    }
    if w.iter().all(|&x| x == 0) {
        w[a] = 1;
    }
    MoneyChance::from_weights(w)
}

fn extreme_leaf() -> L {
    FnLeaf(extreme_value)
}

fn leaves() -> [L; 3] {
    [leaf(), tie_leaf(), extreme_leaf()]
}

/// A random-play position late in a match: at the start of trick `k` of a hand in which some
/// team has at least `min_score` and at least `min_games` games have been won, then `extra`
/// random plays inside the hand. `None` if none was met within the action budget.
fn late_position(seed: u64, k: u8, extra: u8, min_score: u8, min_games: u8) -> Option<Match> {
    let mut rng = Rng::derive(seed, &[0x1a7e, u64::from(k), u64::from(min_score)]);
    let mut m = Match::new(rng.next_u64(), Rules::HOUSE);
    for _ in 0..200_000 {
        if m.phase() == Phase::MatchOver {
            m = Match::new(rng.next_u64(), Rules::HOUSE);
            continue;
        }
        let p = m.public();
        if p.trick_no == k
            && p.trick_cards.is_empty()
            && matches!(m.phase(), Phase::AwaitPlay(_))
            && m.scores().iter().any(|&s| s >= min_score)
            && m.games_won().iter().sum::<u8>() >= min_games
        {
            let hand0 = m.hand_index();
            let mut r2 = Rng::derive(seed, &[0x2b, u64::from(extra)]);
            for _ in 0..extra {
                let acts = m.legal_actions();
                let mut n = m.clone();
                n.apply(acts[r2.bounded(acts.len() as u64) as usize])
                    .expect("legal");
                if n.phase() == Phase::MatchOver || n.hand_index() != hand0 {
                    break;
                }
                m = n;
            }
            return Some(m);
        }
        let acts = m.legal_actions();
        m.apply(acts[rng.bounded(acts.len() as u64) as usize])
            .expect("legal");
    }
    None
}

/// Naive (table on and off agree) against every listed switch combination, both tie orders,
/// plus the parallel solver for the all-on combination.
fn identity(pos: &Position, ctx: MatchCtx, l: &L, combos: &[u8], par: &[usize], what: &str) {
    for order in [TieOrder::Ascending, TieOrder::Descending] {
        let naive = solve(
            pos,
            ctx,
            l,
            Options {
                tt: true,
                order,
                ..Options::default()
            },
        );
        for &bits in combos {
            for tt_mb in [0usize, 64] {
                let opts = FastOptions {
                    order,
                    tt_mb,
                    ..FastOptions::default()
                }
                .combo(bits);
                if tt_mb == 0 && bits & 1 == 0 {
                    continue;
                }
                let f = solve_fast(pos, ctx, l, opts);
                same(
                    &naive,
                    &f,
                    &format!("{what} {order:?} combo {bits} tt {tt_mb}"),
                );
                if bits == 15 || bits == 13 {
                    for &threads in par {
                        let p = solve_fast_par(pos, ctx, l, opts, threads, None);
                        same(
                            &naive,
                            &p,
                            &format!("{what} {order:?} c{bits} par {threads}"),
                        );
                    }
                }
            }
        }
    }
}

fn set(s: &str) -> Bits {
    s.split_whitespace()
        .map(|n| bit(parse(n).unwrap_or_else(|| panic!("card {n}"))))
        .fold(0, |a, b| a | b)
}

fn c(s: &str) -> Card {
    parse(s).unwrap_or_else(|| panic!("card {s}"))
}

/// A hand-made position at a trick boundary: one hand string per seat.
fn made(
    trump: Suit,
    leader: u8,
    hands: [&str; 8],
    scores: [u8; 4],
    tricks: [u8; 4],
    best: Option<(u8, u8)>,
    five_played: bool,
) -> Position {
    let mut hand = InHand::start(scores);
    hand.tricks = tricks;
    hand.tricks_done = tricks.iter().sum();
    hand.five_played = five_played;
    Position {
        hands: hands.map(set),
        trump,
        leader,
        trick: [0; 8],
        n_played: 0,
        hand,
        best,
        rob: None,
    }
}

fn state(scores: [u8; 4], games: [u8; 4], dealer_team: u8) -> MatchState {
    MatchState {
        scores,
        games,
        dealer_team,
    }
}

// ---------- 1. the naive oracle on hand-made positions with known answers ----------

/// Last trick, every hand one card: the trick winner, the end-of-hand bonus, 41 by a trick,
/// 41 by the end bonus, a jink, the 5's immediate bonus, certain money at two games.
#[test]
fn naive_known_answers_last_trick() {
    let l = leaf();
    let fv = common::fake_value;
    let ctx = MatchCtx {
        games: [0, 0, 0, 0],
        dealer_team: 2,
    };
    let t = Suit::Diamonds;
    // Seat 0 leads the KC; seat 5 trumps with the 2D (lowest red trump), seat 6 overtrumps with
    // the 3D. No trump played earlier: the 3D is the best trump of the hand, team 2 gets trick
    // and bonus (+5 +6).
    let hands = ["KC", "QC", "JC", "10C", "9C", "2D", "3D", "8C"];
    let p = made(t, 0, hands, [10, 5, 0, 0], [1, 1, 1, 1], None, false);
    let s = solve(&p, ctx, &l, Options::default());
    assert_eq!(s.value, Some(fv(&state([10, 5, 11, 0], [0; 4], 3))));
    // Same, with the QD played earlier by team 1 as best so far: the 3D is lower (red trumps
    // run 10 down to 2 below the Q), so team 1 keeps the bonus.
    let p = made(
        t,
        0,
        hands,
        [10, 5, 0, 0],
        [1, 1, 1, 1],
        Some((9, 1)),
        false,
    );
    let s = solve(&p, ctx, &l, Options::default());
    assert_eq!(s.value, Some(fv(&state([10, 11, 5, 0], [0; 4], 3))));
    // The 5 was played earlier (its bonus counted then): no end bonus at all.
    let p = made(
        t,
        0,
        hands,
        [10, 16, 0, 0],
        [1, 1, 1, 1],
        Some((15, 1)),
        true,
    );
    let s = solve(&p, ctx, &l, Options::default());
    assert_eq!(s.value, Some(fv(&state([10, 16, 5, 0], [0; 4], 3))));
    // Team 2 on 40 wins the trick: 45, game won at once; team 1's pending bonus is lost. With
    // one game already, that is certain money.
    let p = made(
        t,
        0,
        hands,
        [10, 5, 40, 0],
        [1, 1, 1, 1],
        Some((9, 1)),
        false,
    );
    let s = solve(&p, ctx, &l, Options::default());
    assert_eq!(s.value, Some(fv(&state([0; 4], [0, 0, 1, 0], 3))));
    let ctx1 = MatchCtx {
        games: [1, 0, 1, 0],
        dealer_team: 2,
    };
    let s = solve(&p, ctx1, &l, Options::default());
    assert_eq!(s.value, Some(MoneyChance::certain(2)));
    // Team 1 on 35 holds the best trump; team 2 wins the last trick (to 25): the end bonus takes
    // team 1 to 41 (35 + 6) and wins the game.
    let p = made(
        t,
        0,
        hands,
        [10, 35, 20, 0],
        [1, 1, 1, 1],
        Some((9, 1)),
        false,
    );
    let s = solve(&p, ctx, &l, Options::default());
    assert_eq!(s.value, Some(fv(&state([0; 4], [0, 1, 0, 0], 3))));
    // Team 1 on 36 (bonus already taken this game) gets 5, not 6: 41 still.
    let p = made(
        t,
        0,
        hands,
        [10, 36, 20, 0],
        [1, 1, 1, 1],
        Some((9, 1)),
        false,
    );
    let s = solve(&p, ctx, &l, Options::default());
    assert_eq!(s.value, Some(fv(&state([0; 4], [0, 1, 0, 0], 3))));
    // Team 1 on 30 (no bonus yet): 36, no game.
    let p = made(
        t,
        0,
        hands,
        [10, 30, 20, 0],
        [1, 1, 1, 1],
        Some((9, 1)),
        false,
    );
    let s = solve(&p, ctx, &l, Options::default());
    assert_eq!(s.value, Some(fv(&state([10, 36, 25, 0], [0; 4], 3))));
    // Jink: team 2 has the first four tricks and takes the fifth, from 0 points to 20 + bonus.
    let p = made(t, 0, hands, [0, 0, 20, 0], [0, 0, 4, 0], None, false);
    let s = solve(&p, ctx, &l, Options::default());
    assert_eq!(s.value, Some(fv(&state([0; 4], [0, 0, 1, 0], 3))));
    // The 5 of trumps in the last trick: its bonus counts at once (team 3 on 30: 30 + 5 + 6).
    let hands5 = ["KC", "QC", "JC", "5D", "9C", "2D", "3D", "8C"];
    let p = made(
        t,
        0,
        hands5,
        [10, 5, 0, 30],
        [1, 1, 1, 1],
        Some((9, 1)),
        false,
    );
    let s = solve(&p, ctx, &l, Options::default());
    assert_eq!(s.value, Some(fv(&state([0; 4], [0, 0, 0, 1], 3))));
    // ... and from 20: 31, and no end bonus for team 1 (the 5 was played).
    let p = made(
        t,
        0,
        hands5,
        [10, 5, 0, 20],
        [1, 1, 1, 1],
        Some((9, 1)),
        false,
    );
    let s = solve(&p, ctx, &l, Options::default());
    assert_eq!(s.value, Some(fv(&state([10, 5, 0, 31], [0; 4], 3))));
    // Every solver agrees on all of these (the last-trick shortcut included).
    for bits in ALL {
        let f = solve_fast(&p, ctx, &l, FastOptions::default().combo(bits));
        assert_eq!(f.value, s.value);
    }
}

/// Two cards each: the max^n choice and the team backup, worked out by hand. Clubs trumps
/// (black: 2 beats 10). Seat 0 holds the AH and the 4S and leads; seat 1 (team 1) holds the 10C
/// and the KS; everyone else holds plain hearts and diamonds that cannot win.
#[test]
fn naive_known_answers_two_cards() {
    let t = Suit::Clubs;
    // Team 1 needs only one trick to reach 41 (scores 36 + 5); team 0 needs both (31 + 10).
    let hands = [
        "AH 4S", "10C KS", "2H 3H", "4H 6H", "7H 8H", "9H 10H", "2D 3D", "4D 6D",
    ];
    let p = made(
        t,
        0,
        hands,
        [31, 36, 0, 0],
        [2, 1, 0, 0],
        Some((9, 0)),
        false,
    );
    let ctx = MatchCtx {
        games: [1, 1, 0, 0],
        dealer_team: 0,
    };
    let s = solve(&p, ctx, &leaf(), Options::default());
    // Leading the AH (a trump lead: seat 1 must follow with the 10C, the AH wins, 36), then the
    // 4S: seat 1 must follow with the KS and wins trick 5 (team 1 to 41, the money). Leading
    // the 4S first: seat 1 wins with the KS or the 10C at once (41). Either way team 1 wins.
    assert_eq!(s.value, Some(MoneyChance::certain(1)));
    let mut want = vec![
        (c("4S"), MoneyChance::certain(1)),
        (c("AH"), MoneyChance::certain(1)),
    ];
    want.sort_by_key(|x| x.0);
    assert_eq!(s.children, want);
    // Give seat 0 the KS and seat 1 the 4S. Leading the AH forces the 10C (an ordinary trump
    // cannot be held back) and wins (36); then the KS beats the 4S (41): the money for team 0.
    // Leading the KS first lets seat 1 trump it with the 10C (rule 14: suit or a trump) and
    // reach 41 at once. So the AH is the only winning lead.
    let hands = [
        "AH KS", "10C 4S", "2H 3H", "4H 6H", "7H 8H", "9H 10H", "2D 3D", "4D 6D",
    ];
    let p = made(
        t,
        0,
        hands,
        [31, 36, 0, 0],
        [2, 1, 0, 0],
        Some((9, 0)),
        false,
    );
    let s = solve(&p, ctx, &leaf(), Options::default());
    assert_eq!(s.value, Some(MoneyChance::certain(0)));
    assert_eq!(s.best, Some(c("AH")));
    let ks = s.children.iter().find(|x| x.0 == c("KS")).expect("legal").1;
    assert_eq!(ks, MoneyChance::certain(1));
    for bits in ALL {
        let f = solve_fast(&p, ctx, &leaf(), FastOptions::default().combo(bits));
        same(&s, &f, &format!("combo {bits}"));
    }
}

// ---------- 2. late-match positions: oracle vs engine, fast vs oracle ----------

#[test]
fn late_match_naive_matches_engine_brute_force() {
    let mut met = 0;
    let mut certain_leaves = 0;
    for seed in 0..40u64 {
        for (k, extra) in [(5u8, 0u8), (5, 3), (4, 0), (4, 5)] {
            let Some(m) = late_position(seed, k, extra, 30, 1) else {
                continue;
            };
            let (pos, ctx, pending) = from_match(&m).expect("position");
            assert!(pending.is_none());
            for (li, l) in leaves().iter().enumerate() {
                let want = brute(&m, m.hand_index(), l);
                let got = solve(&pos, ctx, l, Options::default());
                assert_eq!(got.value, Some(want), "seed {seed} t{k}+{extra} leaf {li}");
                for &(card, v) in &got.children {
                    let mut n = m.clone();
                    n.apply(Action::Play(card)).expect("legal");
                    assert_eq!(v, brute(&n, m.hand_index(), l), "seed {seed} card {card}");
                    certain_leaves += u32::from(v.0.contains(&ONE));
                }
            }
            met += 1;
        }
    }
    assert!(met >= 120, "{met}");
    assert!(certain_leaves >= 20, "{certain_leaves}");
}

#[test]
fn late_match_identity_tricks_4_and_5_every_combination() {
    let mut met = 0;
    for seed in 0..40u64 {
        for (k, extra) in [(5u8, 0u8), (5, 2), (4, 0), (4, 3), (4, 6)] {
            let Some(m) = late_position(seed, k, extra, 26, 1) else {
                continue;
            };
            let (pos, ctx, _) = from_match(&m).expect("position");
            for (li, l) in leaves().iter().enumerate() {
                identity(
                    &pos,
                    ctx,
                    l,
                    &ALL,
                    &[2, 3],
                    &format!("late seed {seed} t{k}+{extra} leaf {li}"),
                );
            }
            met += 1;
        }
    }
    assert!(met >= 150, "{met}");
}

#[test]
fn late_match_identity_trick_3() {
    let mut met = 0;
    for seed in 0..12u64 {
        let extra = (seed % 5) as u8;
        let Some(m) = late_position(seed, 3, extra, 25, 1) else {
            continue;
        };
        let (pos, ctx, _) = from_match(&m).expect("position");
        for (li, l) in leaves().iter().enumerate() {
            identity(
                &pos,
                ctx,
                l,
                &[15, 13, 12, 9, 8, 11, 3, 1],
                &[2, 4],
                &format!("late t3 seed {seed}+{extra} leaf {li}"),
            );
        }
        met += 1;
    }
    assert!(met >= 10, "{met}");
}

/// Positions where a game can be won in the middle of the hand (some team on 31 or more at the
/// start of trick 3 or 4, so a trick or the 5's bonus ends the game early), with games won
/// before in the match (so a won game can be the money).
#[test]
fn game_won_mid_hand_identity() {
    let mut met = 0;
    for seed in 100..130u64 {
        for (k, extra) in [(3u8, 1u8), (4, 2), (4, 7)] {
            let Some(m) = late_position(seed, k, extra, 31, 2) else {
                continue;
            };
            let (pos, ctx, _) = from_match(&m).expect("position");
            if k == 3 && !seed.is_multiple_of(3) {
                continue;
            }
            for (li, l) in leaves().iter().enumerate() {
                identity(
                    &pos,
                    ctx,
                    l,
                    &[15, 8, 9, 12, 13, 1, 0],
                    &[3],
                    &format!("gw seed {seed} t{k}+{extra} leaf {li}"),
                );
            }
            met += 1;
        }
    }
    assert!(met >= 40, "{met}");
}

// ---------- 3. degenerate leaves on ordinary positions ----------

#[test]
fn extreme_leaf_identity_tricks_3_to_5() {
    let l = extreme_leaf();
    for seed in 0..60u64 {
        for (k, extra) in [(5u8, 1u8), (4, 0), (4, 4)] {
            let m = position_mid(seed, k, extra);
            let (pos, ctx, _) = from_match(&m).expect("position");
            // Late in a match too: one game each for two teams, so game ends are the money.
            for games in [[0, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 1]] {
                let ctx = MatchCtx { games, ..ctx };
                identity(
                    &pos,
                    ctx,
                    &l,
                    &ALL,
                    &[2],
                    &format!("x seed {seed} t{k}+{extra} {games:?}"),
                );
            }
        }
    }
    for seed in 0..6u64 {
        let m = position_mid(seed, 3, (seed % 3) as u8);
        let (pos, ctx, _) = from_match(&m).expect("position");
        let ctx = MatchCtx {
            games: [1, 0, 1, 0],
            ..ctx
        };
        identity(
            &pos,
            ctx,
            &l,
            &[15, 8, 12, 13, 9],
            &[3],
            &format!("x t3 seed {seed}"),
        );
    }
}

// ---------- 4. the equivalent-card claim, checked with the naive solver ----------

/// Every pair of cards that the classes merge in the mover's legal set has the identical value
/// vector under the naive solver (no merging there), at the root of many positions, with and
/// without a pending rob, late in a match and with the degenerate leaf. This tests the claim
/// itself, not only the fast solver's use of it.
#[test]
fn merged_cards_have_identical_naive_values() {
    let mut pairs = 0u32;
    let mut check = |pos: &Position, ctx: MatchCtx, what: &str| {
        let classes = Classes::new(pos.trump);
        let hand = pos.hands[usize::from(pos.to_act())];
        let mut g = [0; 16];
        let n = classes.groups(hand, pos.live(), &mut g);
        let legal = pos.legal();
        for l in leaves().iter() {
            let s = solve(pos, ctx, l, Options::default());
            for &grp in &g[..n] {
                let grp = grp & legal;
                if grp.count_ones() < 2 {
                    continue;
                }
                let vals: Vec<MoneyChance> = engine::cards::cards(grp)
                    .map(|c| s.children.iter().find(|x| x.0 == c).expect("legal").1)
                    .collect();
                assert!(
                    vals.windows(2).all(|w| w[0] == w[1]),
                    "{what}: group {grp:#x} values {vals:?}"
                );
                pairs += 1;
            }
        }
    };
    for seed in 0..80u64 {
        for (k, extra) in [(4u8, 0u8), (4, 3), (3, 0), (3, 5)] {
            if k == 3 && !seed.is_multiple_of(4) {
                continue;
            }
            let m = position_mid(seed, k, extra);
            let (pos, ctx, _) = from_match(&m).expect("position");
            check(&pos, ctx, &format!("seed {seed} t{k}+{extra}"));
            // The same with a pending rob of a dead card by a later seat.
            if let Some((rp, rctx)) = with_rob(&pos, ctx, seed) {
                check(&rp, rctx, &format!("rob seed {seed} t{k}+{extra}"));
            }
            if let Some(m) = late_position(seed, k, extra, 28, 1) {
                let (pos, ctx, _) = from_match(&m).expect("position");
                check(&pos, ctx, &format!("late seed {seed} t{k}+{extra}"));
            }
        }
    }
    assert!(pairs >= 150, "{pairs}");
}

/// The position that exposed the tie-order flaw in equivalent-card merging: clubs trumps, the
/// mover (seat 3) holds the 4♣ 5♣ 7♣ with the 6♣ gone. The old rule merged the 4♣ and the 7♣,
/// but the 5♣ lies between them in card index, so the tie rule later breaks a tie differently
/// (for the mover's own team the 7♣ is worth 0 with the 4♣'s value copied as 0.2). Fast,
/// every combination, and parallel must equal the naive solver.
#[test]
fn identity_when_tie_order_separates_merged_cards() {
    let m = position_mid(52, 3, 5);
    let (pos, ctx, _) = from_match(&m).expect("position");
    assert_eq!(pos.trump, Suit::Clubs);
    assert_eq!(pos.hands[usize::from(pos.to_act())], set("4C 5C 7C"));
    identity(
        &pos,
        ctx,
        &extreme_leaf(),
        &ALL,
        &[2, 4],
        "seed 52 trick 3+5",
    );
}

/// Every pair of burns that the fast solver merges (one group of the robber's hand after the
/// take) leaves positions of identical naive value.
#[test]
fn merged_burns_have_identical_naive_values() {
    let mut pairs = 0u32;
    for seed in 0..120u64 {
        for (k, extra) in [(5u8, 0u8), (4, 0), (4, 2)] {
            let m = position_mid(seed, k, extra);
            let (pos, ctx, _) = from_match(&m).expect("position");
            let Some((mut rp, ctx)) = with_rob(&pos, ctx, seed) else {
                continue;
            };
            // Bring the robber to act: play the cards before it as the naive solver's tie
            // order would (the lowest legal card).
            let robber = rp.rob.expect("rob").seat;
            while rp.to_act() != robber {
                let c = rp.legal().trailing_zeros() as Card;
                match rp.play(c) {
                    Step::Next(q) => rp = q,
                    _ => unreachable!("mid-trick"),
                }
            }
            let (h, burns) = rp.burn_choice().expect("robber to act");
            let mut g = [0; 16];
            let n = Classes::new(rp.trump).groups(h, rp.live(), &mut g);
            for l in leaves().iter() {
                for &grp in &g[..n] {
                    let grp = grp & burns;
                    if grp.count_ones() < 2 {
                        continue;
                    }
                    let vals: Vec<MoneyChance> = engine::cards::cards(grp)
                        .map(|b| {
                            let q = rp.burnt(b);
                            // The robber's own choice after the burn is a root solve.
                            solve(&q, ctx, l, Options::default()).value.expect("solved")
                        })
                        .collect();
                    assert!(
                        vals.windows(2).all(|w| w[0] == w[1]),
                        "seed {seed} t{k}+{extra}: burns {grp:#x} values {vals:?}"
                    );
                    pairs += 1;
                }
            }
        }
    }
    assert!(pairs >= 50, "{pairs}");
}

/// A pending rob of a dead card by a seat after the root mover in this trick (as in burn.rs),
/// the robber given the ace of trumps when it is out of play. `None` if no seat is left to act.
fn with_rob(pos: &Position, ctx: MatchCtx, seed: u64) -> Option<(Position, MatchCtx)> {
    let mut pos = *pos;
    let room = 8 - u64::from(pos.n_played);
    if room < 2 {
        return None;
    }
    let mut rng = Rng::derive(seed, &[0x70b, u64::from(pos.n_played)]);
    let seat = (pos.to_act() + 1 + rng.bounded(room - 1) as u8) % 8;
    let live = pos.live();
    let dead: Vec<Card> = (0..=JOKER).filter(|&c| !has(live, c)).collect();
    let table = dead[rng.bounded(dead.len() as u64) as usize];
    if pos.hand.five_played && table == five_of(pos.trump) {
        return None;
    }
    let a = ace_of_trumps(pos.trump);
    if a != table && !has(live, a) && seed.is_multiple_of(2) {
        let h = pos.hands[usize::from(seat)];
        let x = h.trailing_zeros() as Card;
        pos.hands[usize::from(seat)] = (h & !bit(x)) | bit(a);
    }
    pos.rob = Some(Rob { seat, card: table });
    // Synthetic later-trick robs only where the burn merge is proved sound (see the helper).
    common::burn_merge_sound(&pos).then_some((pos, ctx))
}

/// Pending robs (any dead card, the 5 of trumps included when it was not played, and the Joker)
/// with the degenerate leaf and late-match contexts: fast, every combination, and parallel
/// against naive.
#[test]
fn pending_rob_identity_with_degenerate_leaves() {
    let mut met = 0;
    for seed in 0..60u64 {
        for (k, extra) in [(5u8, 0u8), (4, 0), (4, 3)] {
            let m = position_mid(seed, k, extra);
            let (pos, ctx, _) = from_match(&m).expect("position");
            let Some((rp, ctx)) = with_rob(&pos, ctx, seed) else {
                continue;
            };
            for games in [[0, 0, 0, 0], [1, 0, 1, 0]] {
                let ctx = MatchCtx { games, ..ctx };
                for (li, l) in leaves().iter().enumerate() {
                    identity(
                        &rp,
                        ctx,
                        l,
                        &ALL,
                        &[2, 3],
                        &format!("rob seed {seed} t{k}+{extra} {games:?} leaf {li}"),
                    );
                }
            }
            met += 1;
        }
    }
    assert!(met >= 100, "{met}");
}

// ---------- 5. parallel solver: thread counts and split sizes ----------

#[test]
fn par_identity_threads_and_splits_late_match() {
    let mut mem = SolveMem::new();
    for seed in 0..8u64 {
        let (k, extra) = if seed.is_multiple_of(2) {
            (3u8, 0u8)
        } else {
            (4, 1)
        };
        let Some(m) = late_position(seed, k, extra, 25, 1) else {
            continue;
        };
        let (pos, ctx, _) = from_match(&m).expect("position");
        let l = extreme_leaf();
        for order in [TieOrder::Ascending, TieOrder::Descending] {
            let opts = FastOptions {
                order,
                tt_mb: 4,
                ..FastOptions::default()
            };
            let seq = solve_fast_with(&pos, ctx, &l, opts, &mut mem);
            let naive = solve(
                &pos,
                ctx,
                &l,
                Options {
                    order,
                    tt: true,
                    ..Options::default()
                },
            );
            same(&naive, &seq, &format!("seed {seed} {order:?} seq"));
            for threads in 1..=4 {
                for split in [1usize, 3, 64] {
                    for bits in [15u8, 11, 14] {
                        let p = solve_fast_par_split(
                            &pos,
                            ctx,
                            &l,
                            opts.combo(bits),
                            threads,
                            split,
                            None,
                        );
                        same(
                            &naive,
                            &p,
                            &format!("seed {seed} {order:?} t{threads} split {split} c{bits}"),
                        );
                    }
                }
            }
        }
    }
}

// ---------- 6. fixed point: every reported value sums to exactly ONE ----------

#[test]
fn every_reported_value_sums_to_one() {
    for seed in 0..30u64 {
        let m = position_mid(seed, 4, (seed % 8) as u8);
        let (pos, ctx, _) = from_match(&m).expect("position");
        for l in leaves().iter() {
            for bits in ALL {
                let f = solve_fast(&pos, ctx, l, FastOptions::default().combo(bits));
                for (_, v) in &f.children {
                    assert_eq!(v.0.iter().sum::<u64>(), ONE);
                }
            }
        }
    }
}

// ---------- 7. the real match-equity table (optional, needs a built table) ----------

/// With `PROBE_EQUITY=<equity-vN.npy>` set: the naive and the default fast solver on random
/// positions of tricks 3-5 (some late in a match) with the real table as the leaf, counting
/// disagreements (none allowed). Skipped without the table.
#[test]
#[ignore = "needs PROBE_EQUITY=<path to a built equity table>"]
fn real_table_identity() {
    let Ok(path) = std::env::var("PROBE_EQUITY") else {
        return;
    };
    let bytes = std::fs::read(&path).expect("table file");
    let table = equity::EquityTable::from_npy(&bytes).expect("table");
    let n: u64 = std::env::var("PROBE_N")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(200);
    let (mut roots, mut bad, mut bad_mover) = (0u32, 0u32, 0u32);
    for seed in 0..n {
        let k = 3 + (seed % 3) as u8;
        let extra = (seed % 7) as u8;
        let m = if seed.is_multiple_of(2) {
            position_mid(seed, k, extra)
        } else {
            match late_position(seed, k, extra, 30, 2) {
                Some(m) => m,
                None => continue,
            }
        };
        let (pos, ctx, _) = from_match(&m).expect("position");
        let naive = solve(
            &pos,
            ctx,
            &table,
            Options {
                tt: true,
                ..Options::default()
            },
        );
        let fast = solve_fast(&pos, ctx, &table, FastOptions::default());
        roots += 1;
        if naive.children != fast.children {
            bad += 1;
            let t = usize::from(engine::team_of(pos.to_act()));
            if naive
                .children
                .iter()
                .zip(&fast.children)
                .any(|(a, b)| a.1.0[t] != b.1.0[t])
            {
                bad_mover += 1;
            }
            eprintln!(
                "seed {seed} t{k}+{extra}: naive {:?} fast {:?}",
                naive.best, fast.best
            );
        }
    }
    eprintln!("roots {roots}, differing {bad}, differing for the mover's team {bad_mover}");
    assert_eq!(bad, 0);
}

/// The cost of the equivalent-card rule: total nodes of default solves at tricks 2-3 (printed;
/// for comparing rule changes).
#[test]
#[ignore = "measurement"]
fn equiv_cost_measurement() {
    let l = leaf();
    for k in [3u8, 2] {
        let (mut nodes, mut skipped) = (0u64, 0u64);
        let n = if k == 3 { 40 } else { 4 };
        for seed in 0..n {
            let m = position_mid(seed, k, (seed % 3) as u8);
            let (pos, ctx, _) = from_match(&m).expect("position");
            let s = solve_fast(&pos, ctx, &l, FastOptions::default());
            nodes += s.stats.nodes;
            skipped += s.stats.equiv_skipped;
        }
        eprintln!("trick {k}: {n} roots, nodes {nodes}, equiv_skipped {skipped}");
    }
}

/// The degenerate leaf on many more positions (release: about a minute).
#[test]
#[ignore = "long: cargo test --release -p solver --test probe -- --ignored extreme_leaf_sweep_long"]
fn extreme_leaf_sweep_long() {
    let l = extreme_leaf();
    for seed in 1000..1600u64 {
        let k = 3 + (seed % 3) as u8;
        if k == 3 && !seed.is_multiple_of(4) {
            continue;
        }
        let extra = (seed % 8) as u8;
        let m = if seed.is_multiple_of(2) {
            position_mid(seed, k, extra)
        } else {
            match late_position(seed, k, extra, 28, 1) {
                Some(m) => m,
                None => continue,
            }
        };
        let (pos, ctx, _) = from_match(&m).expect("position");
        let combos: &[u8] = if k == 3 { &[15, 8, 12, 1] } else { &ALL };
        identity(
            &pos,
            ctx,
            &l,
            combos,
            &[3],
            &format!("sweep seed {seed} t{k}+{extra}"),
        );
        if let Some((rp, rctx)) = with_rob(&pos, ctx, seed) {
            identity(
                &rp,
                rctx,
                &l,
                combos,
                &[2],
                &format!("sweep rob seed {seed} t{k}"),
            );
        }
    }
}
