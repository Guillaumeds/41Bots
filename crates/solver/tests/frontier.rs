//! The hybrid frontier (docs/M3-HYBRID-SPEC.md §2, §6; `solver::frontier`):
//! - the optimised and parallel solvers with a frontier return exactly the naive solver's
//!   hybrid values (value, chosen card, every root card's value) under every switch
//!   combination, both tie orders, 1–4 threads, and with solver memory reused across hybrid and
//!   plain solves;
//! - the canonical form is complete: positions with equal canonical forms have equal exact
//!   values (frontier positions reached by every line from many roots, and positions with dead
//!   cards moved or the best trump changed), and a frontier that returns exact values makes the
//!   hybrid equal the full exact solve;
//! - the team rotation used by the encoder maps exact values as claimed.

mod common;

use std::collections::BTreeMap;

use common::{L, fake_value, leaf, position_mid};
use engine::cards::{Bits, Card, bit, cards, has};
use engine::game::{Match, Phase, Rules};
use engine::rng::Rng;
use engine::team_of;
use equity::{MatchState, MoneyChance};
use solver::frontier::{CanonPos, canonical_order};
use solver::{
    FastOptions, FnFrontier, FnLeaf, Frontier, FrontierCut, MatchCtx, Options, Position, Rob,
    Solution, SolveMem, Step, TieOrder, canonical, from_match, solve, solve_cut, solve_fast,
    solve_fast_cut, solve_fast_par_cut, solve_fast_with,
};

fn same(a: &Solution, b: &Solution, what: &str) {
    assert!(!a.timed_out && !b.timed_out, "{what}: timed out");
    assert_eq!(a.value, b.value, "{what}: value");
    assert_eq!(a.best, b.best, "{what}: chosen card");
    assert_eq!(a.children, b.children, "{what}: root card values");
    assert_eq!(a.root_tied, b.root_tied, "{what}: root ties");
}

/// A deterministic, arbitrary frontier: weights from a hash of every canonical field and the
/// match context (so any mix-up of positions shows).
fn hash_frontier(c: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |b: u8| h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3);
    eat(c.trump as u8);
    eat(c.leader);
    eat(c.tricks_done);
    for t in 0..4 {
        eat(c.scores[t]);
        eat(u8::from(c.flags[t]));
        eat(c.tricks[t]);
        eat(ctx.games[t]);
    }
    eat(u8::from(c.five_played));
    eat(c.end_bonus.unwrap_or(9));
    eat(ctx.dealer_team);
    for &s in &c.slots {
        eat(s);
    }
    let w: [u128; 4] = core::array::from_fn(|t| u128::from((h >> (16 * t)) & 0xFFFF) + 1);
    MoneyChance::from_weights(w)
}

/// A coarse frontier with very many exact ties (only the tricks won and who holds the best live
/// trump matter), to stress the tie rule at the nodes above the frontier.
fn tie_frontier(c: &CanonPos, _ctx: &MatchCtx) -> MoneyChance {
    let top = (0..15).find_map(|i| c.slot(i)).map(|(s, _)| team_of(s));
    MoneyChance::from_weights(core::array::from_fn(|t| {
        1 + u128::from(c.tricks[t]) * 2 + u128::from(top == Some(t as u8))
    }))
}

type F = FnFrontier<fn(&CanonPos, &MatchCtx) -> MoneyChance>;

fn frontiers() -> [F; 2] {
    [FnFrontier(hash_frontier), FnFrontier(tie_frontier)]
}

const ALL: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Naive hybrid (table on and off) vs the optimised hybrid in the given switch combinations and
/// the parallel hybrid on 1–4 threads, both tie orders.
fn identity(
    pos: &Position,
    ctx: MatchCtx,
    l: &L,
    f: &dyn Frontier,
    k: u8,
    combos: &[u8],
    what: &str,
) -> u64 {
    let cut = Some(FrontierCut::new(k, f));
    let mut evals = 0;
    for order in [TieOrder::Ascending, TieOrder::Descending] {
        let naive = solve_cut(
            pos,
            ctx,
            l,
            Options {
                tt: false,
                order,
                ..Options::default()
            },
            cut,
        );
        let naive_tt = solve_cut(
            pos,
            ctx,
            l,
            Options {
                tt: true,
                order,
                ..Options::default()
            },
            cut,
        );
        same(&naive, &naive_tt, &format!("{what} {order:?} naive table"));
        evals += naive.stats.frontier;
        for &bits in combos {
            let opts = FastOptions {
                order,
                ..FastOptions::default()
            }
            .combo(bits);
            let f = solve_fast_cut(pos, ctx, l, opts, cut, &mut SolveMem::new(), None);
            same(&naive, &f, &format!("{what} {order:?} combo {bits}"));
        }
        for threads in [1, 2, 3, 4] {
            for bits in [15, 11, 0] {
                let opts = FastOptions {
                    order,
                    tt_mb: if threads == 3 { 0 } else { 64 },
                    ..FastOptions::default()
                }
                .combo(bits);
                let p =
                    solve_fast_par_cut(pos, ctx, l, opts, threads, None, &mut SolveMem::new(), cut);
                same(
                    &naive,
                    &p,
                    &format!("{what} {order:?} combo {bits} threads {threads}"),
                );
            }
        }
    }
    evals
}

#[test]
fn hybrid_identity_tricks_2_to_4_every_combination() {
    let mut evals = 0;
    for seed in 0..24 {
        for (k, extra, cuts) in [
            (4, 0, &[5u8][..]),
            (4, 3, &[5]),
            (3, 2, &[4, 5]),
            (3, 5, &[4, 5]),
            (2, 4, &[3]),
        ] {
            let m = position_mid(seed, k, extra);
            let (pos, ctx, pending) = from_match(&m).expect("position");
            assert!(pending.is_none());
            for &cut in cuts {
                for (i, f) in frontiers().iter().enumerate() {
                    let combos: &[u8] = if k == 4 || seed < 6 {
                        &ALL
                    } else {
                        &[15, 1, 2, 4, 8, 14, 13, 11, 7]
                    };
                    evals += identity(
                        &pos,
                        ctx,
                        &leaf(),
                        f,
                        cut,
                        combos,
                        &format!("seed {seed} trick {k}+{extra} cut {cut} frontier {i}"),
                    );
                }
            }
        }
    }
    assert!(
        evals > 10_000,
        "the frontier is reached: {evals} evaluations"
    );
}

/// Trick-1 roots with a rob still to be made by a later seat (a burn node above the frontier),
/// frontier at the start of trick 2.
#[test]
fn hybrid_identity_trick_1_with_pending_rob() {
    let mut checked = 0;
    for seed in 0..400u64 {
        let Some((pos, ctx)) = trick1_with_rob(seed) else {
            continue;
        };
        for (i, f) in frontiers().iter().enumerate() {
            identity(
                &pos,
                ctx,
                &leaf(),
                f,
                2,
                &[15, 2, 0, 13],
                &format!("seed {seed} trick 1 rob frontier {i}"),
            );
        }
        checked += 1;
        if checked == 12 {
            break;
        }
    }
    assert_eq!(checked, 12, "enough trick-1 rob roots");
}

/// A trick-1 position after at least 4 cards, with the holder of the ace of trumps still to act
/// (its rob pending), as the yardstick builds it.
fn trick1_with_rob(seed: u64) -> Option<(Position, MatchCtx)> {
    let mut rng = Rng::derive(seed, &[0x7121]);
    let mut m = Match::new(rng.next_u64(), Rules::HOUSE);
    loop {
        if m.phase() == Phase::MatchOver || m.public().trick_no > 1 {
            return None;
        }
        let p = m.public();
        if p.trick_no == 1 && p.trick_cards.len() >= 4 && matches!(m.phase(), Phase::AwaitPlay(_)) {
            let (mut pos, ctx, pending) = from_match(&m).ok()?;
            let s = pending?;
            pos.rob = Some(Rob {
                seat: s,
                card: p.table_card?,
            });
            return Some((pos, ctx));
        }
        let acts = m.legal_actions();
        m.apply(acts[rng.bounded(acts.len() as u64) as usize])
            .expect("legal");
    }
}

/// Every frontier position (start of trick `k`) reached from `p` by any line, burns included.
fn frontier_positions(p: &Position, k: u8, out: &mut Vec<Position>) {
    if p.n_played == 0 && p.hand.tricks_done + 1 == k && p.rob.is_none() {
        out.push(*p);
        return;
    }
    if let Some((_, burns)) = p.burn_choice() {
        for b in cards(burns) {
            frontier_positions(&p.burnt(b), k, out);
        }
        return;
    }
    for c in cards(p.legal()) {
        if let Step::Next(q) = p.play(c) {
            frontier_positions(&q, k, out);
        }
    }
}

/// Exact values of frontier positions by canonical form.
struct Exact(BTreeMap<CanonPos, MoneyChance>);

impl Frontier for Exact {
    fn value(&self, pos: &CanonPos, _ctx: &MatchCtx) -> MoneyChance {
        *self.0.get(pos).expect("every frontier position was solved")
    }
}

/// Completeness on reached positions, and the frontier plumbing: every frontier position reached
/// from a root is solved exactly; positions with the same canonical form must have the same
/// exact value; then the hybrid with those exact values at the frontier equals the full exact
/// solve (value, chosen card, every root card's value), naive and optimised.
#[test]
fn exact_frontier_reproduces_the_exact_solve_and_canonical_form_is_complete() {
    let (mut roots, mut positions, mut shared) = (0, 0, 0);
    for seed in 0..40 {
        // Trick-2 roots solve every trick-3 frontier position exactly: a few only (debug builds).
        let t2: &[(u8, u8)] = if seed < 3 { &[(2, 6)] } else { &[] };
        for &(k, extra) in [(4, 0), (4, 4), (3, 3), (3, 6)].iter().chain(t2) {
            let m = position_mid(seed, k, extra);
            let (pos, ctx, _) = from_match(&m).expect("position");
            let cut = pos.hand.tricks_done + 2;
            let mut fr = Vec::new();
            frontier_positions(&pos, cut, &mut fr);
            for order in [TieOrder::Ascending, TieOrder::Descending] {
                let opts = FastOptions {
                    order,
                    ..FastOptions::default()
                };
                let mut map: BTreeMap<CanonPos, MoneyChance> = BTreeMap::new();
                for q in &fr {
                    let v = solve_fast(q, ctx, &leaf(), opts).value.expect("solved");
                    match map.insert(canonical(q), v) {
                        Some(old) => {
                            assert_eq!(old, v, "seed {seed}: equal canonical forms, equal values");
                            shared += 1;
                        }
                        None => positions += 1,
                    }
                }
                let ex = Exact(map);
                let full = solve(
                    &pos,
                    ctx,
                    &leaf(),
                    Options {
                        order,
                        ..Options::default()
                    },
                );
                let hybrid = solve_cut(
                    &pos,
                    ctx,
                    &leaf(),
                    Options {
                        order,
                        ..Options::default()
                    },
                    Some(FrontierCut::new(cut, &ex)),
                );
                same(&full, &hybrid, &format!("seed {seed} trick {k}+{extra}"));
                let fast = solve_fast_cut(
                    &pos,
                    ctx,
                    &leaf(),
                    opts,
                    Some(FrontierCut::new(cut, &ex)),
                    &mut SolveMem::new(),
                    None,
                );
                same(&full, &fast, &format!("seed {seed} trick {k}+{extra} fast"));
            }
            roots += 1;
        }
    }
    assert_eq!(roots, 163);
    assert!(
        positions > 5_000 && shared > 1_000,
        "{positions} positions, {shared} sharing a canonical form"
    );
}

/// The class of every card for a trump suit.
fn classes(trump: engine::cards::Suit) -> [u8; 64] {
    let mut out = [u8::MAX; 64];
    for &(c, cls) in canonical_order(trump) {
        out[usize::from(c)] = cls;
    }
    out
}

/// Move one live card of a hand to a dead card of its class with no live card of the class
/// between them in rank and no card of the hand between them in card index (the moves an
/// equivalent-card merge makes); `None` if the drawn card has no such move.
fn moved(p: &Position, rng: &mut Rng) -> Option<Position> {
    let live: Bits = p.hands.iter().fold(0, |a, &h| a | h);
    let all: Vec<Card> = cards(live).collect();
    let x = all[rng.bounded(all.len() as u64) as usize];
    let s = (0..8).find(|&s| has(p.hands[s], x)).expect("held");
    let cls = classes(p.trump);
    let ord: Vec<Card> = canonical_order(p.trump)
        .iter()
        .filter(|&&(_, k)| k == cls[usize::from(x)])
        .map(|&(c, _)| c)
        .collect();
    let ix = ord.iter().position(|&c| c == x).expect("in its class");
    let mut options = Vec::new();
    for dir in [-1i32, 1] {
        let mut j = ix as i32 + dir;
        while j >= 0 && (j as usize) < ord.len() && !has(live, ord[j as usize]) {
            options.push(ord[j as usize]);
            j += dir;
        }
    }
    let rest = p.hands[s] & !bit(x);
    let options: Vec<Card> = options
        .into_iter()
        .filter(|&d| {
            let (lo, hi) = (x.min(d), x.max(d));
            (lo + 1..hi).all(|c| !has(rest, c))
        })
        .collect();
    if options.is_empty() {
        return None;
    }
    let d = options[rng.bounded(options.len() as u64) as usize];
    let mut q = *p;
    q.hands[s] = rest | bit(d);
    Some(q)
}

/// The best trump so far changed to another dead trump's rank, and sometimes another team.
fn rebested(p: &Position, rng: &mut Rng) -> Option<Position> {
    let live: Bits = p.hands.iter().fold(0, |a, &h| a | h);
    let dead: Vec<u8> = (0..=engine::cards::JOKER)
        .filter(|&c| !has(live, c))
        .map(|c| engine::rules::trump_rank(c, p.trump))
        .filter(|&r| r > 0)
        .collect();
    if dead.is_empty() {
        return None;
    }
    let mut q = *p;
    let r = dead[rng.bounded(dead.len() as u64) as usize];
    let t = match p.best {
        Some((_, t)) if rng.bounded(2) == 0 => t,
        _ => rng.bounded(4) as u8,
    };
    q.best = Some((r, t));
    Some(q)
}

/// Completeness against moved cards: positions at the start of tricks 3–5 with dead cards moved
/// between live ones (as equivalent-card merges do) or the best trump changed; whenever the
/// canonical form is unchanged the exact value (root value and the multiset of root card values)
/// is unchanged, under both tie orders, with an equity-like and a tie-heavy leaf.
#[test]
fn canonical_form_is_complete_against_moved_cards_and_best_trumps() {
    let mut rng = Rng::new(0xF407);
    let (mut equal, mut differ) = (0, 0);
    let leaves = [leaf(), common::tie_leaf()];
    for seed in 0..150 {
        let k = 3 + (seed % 3) as u8;
        let m = common::position_at(seed, k);
        let (pos, ctx, _) = from_match(&m).expect("position");
        let c0 = canonical(&pos);
        for _ in 0..6 {
            let mut q = pos;
            let steps = 1 + rng.bounded(4);
            for _ in 0..steps {
                let next = if rng.bounded(3) == 0 {
                    rebested(&q, &mut rng)
                } else {
                    moved(&q, &mut rng)
                };
                if let Some(n) = next {
                    q = n;
                }
            }
            if q == pos {
                continue;
            }
            if canonical(&q) != c0 {
                differ += 1;
                continue;
            }
            equal += 1;
            for l in &leaves {
                for order in [TieOrder::Ascending, TieOrder::Descending] {
                    let opts = FastOptions {
                        order,
                        ..FastOptions::default()
                    };
                    let a = solve_fast(&pos, ctx, l, opts);
                    let b = solve_fast(&q, ctx, l, opts);
                    assert_eq!(a.value, b.value, "seed {seed} trick {k} {order:?}");
                    let mut va: Vec<[u64; 4]> = a.children.iter().map(|(_, v)| v.0).collect();
                    let mut vb: Vec<[u64; 4]> = b.children.iter().map(|(_, v)| v.0).collect();
                    va.sort_unstable();
                    vb.sort_unstable();
                    assert_eq!(va, vb, "seed {seed} trick {k} {order:?}: root card values");
                }
            }
        }
    }
    assert!(
        equal > 400 && differ > 20,
        "{equal} equal canonical forms, {differ} changed"
    );
}

/// A leaf that commutes with team rotation (as the equity table does: it is stored for dealer
/// team 0 and rotated).
fn rotation_value(s: &MatchState) -> MoneyChance {
    let (c, d) = s.canonical();
    fake_value(&c).rotate(d)
}

/// Seats moved `r` places to the left (teams `r mod 4`).
fn rotated(p: &Position, ctx: &MatchCtx, r: u8) -> (Position, MatchCtx) {
    let mut q = *p;
    let t = |x: u8| (x + r) % 4;
    for s in 0..8u8 {
        q.hands[usize::from((s + r) % 8)] = p.hands[usize::from(s)];
    }
    q.leader = (p.leader + r) % 8;
    for i in 0..4u8 {
        q.hand.scores[usize::from(t(i))] = p.hand.scores[usize::from(i)];
        q.hand.flags[usize::from(t(i))] = p.hand.flags[usize::from(i)];
        q.hand.tricks[usize::from(t(i))] = p.hand.tricks[usize::from(i)];
    }
    q.best = p.best.map(|(b, tm)| (b, t(tm)));
    q.rob = p.rob.map(|rb| Rob {
        seat: (rb.seat + r) % 8,
        card: rb.card,
    });
    let mut games = [0u8; 4];
    for i in 0..4u8 {
        games[usize::from(t(i))] = ctx.games[usize::from(i)];
    }
    (
        q,
        MatchCtx {
            games,
            dealer_team: t(ctx.dealer_team),
        },
    )
}

/// Team rotation, the symmetry the encoder uses (it rotates the leader to seat 0): rotating every
/// seat by `r` rotates the exact value of every root card by `r mod 4`, with a leaf that commutes
/// with rotation.
#[test]
fn rotation_maps_exact_values() {
    let l = FnLeaf(rotation_value as fn(&MatchState) -> MoneyChance);
    for seed in 0..60 {
        let k = 3 + (seed % 3) as u8;
        let m = position_mid(seed, k, (seed % 5) as u8);
        let (pos, ctx, _) = from_match(&m).expect("position");
        let base = solve_fast(&pos, ctx, &l, FastOptions::default());
        for r in 1..8u8 {
            let (q, qctx) = rotated(&pos, &ctx, r);
            let s = solve_fast(&q, qctx, &l, FastOptions::default());
            assert_eq!(s.best, base.best, "seed {seed} r {r}");
            for ((c1, v1), (c2, v2)) in base.children.iter().zip(&s.children) {
                assert_eq!(c1, c2);
                assert_eq!(v1.rotate(r % 4), *v2, "seed {seed} r {r} card {c1}");
            }
        }
    }
}

/// One solver memory kept across hybrid and plain solves (and frontiers at different tricks)
/// gives exactly what fresh memory gives: a plain solve after a hybrid one is the exact answer.
#[test]
fn memory_reused_across_hybrid_and_plain_solves() {
    let mut mem = SolveMem::new();
    let f = FnFrontier(hash_frontier as fn(&CanonPos, &MatchCtx) -> MoneyChance);
    for seed in 0..10 {
        let m = position_mid(seed, 3, (seed % 3) as u8);
        let (pos, ctx, _) = from_match(&m).expect("position");
        for (cut, tt_mb) in [
            (Some(4u8), 64),
            (None, 64),
            (Some(5), 1),
            (None, 0),
            (Some(4), 0),
        ] {
            let opts = FastOptions {
                tt_mb,
                ..FastOptions::default()
            };
            let c = cut.map(|k| FrontierCut::new(k, &f));
            let fresh = solve_fast_cut(&pos, ctx, &leaf(), opts, c, &mut SolveMem::new(), None);
            let reused = solve_fast_cut(&pos, ctx, &leaf(), opts, c, &mut mem, None);
            assert_eq!(fresh, reused, "seed {seed} cut {cut:?}");
            if cut.is_none() {
                let plain = solve_fast_with(&pos, ctx, &leaf(), opts, &mut mem);
                assert_eq!(plain, fresh, "seed {seed}: plain solve");
                assert_eq!(fresh.stats.frontier, 0);
            } else {
                assert!(fresh.stats.frontier > 0, "seed {seed}: frontier reached");
            }
        }
    }
}

/// A root at or after the frontier's trick is solved exactly (the root is never cut, and no
/// later position is on the frontier).
#[test]
fn root_at_or_after_the_frontier_is_exact() {
    let f = FnFrontier(hash_frontier as fn(&CanonPos, &MatchCtx) -> MoneyChance);
    for seed in 0..10 {
        let m = position_mid(seed, 4, (seed % 3) as u8);
        let (pos, ctx, _) = from_match(&m).expect("position");
        for cut in [2, 3, 4] {
            let h = solve_cut(
                &pos,
                ctx,
                &leaf(),
                Options::default(),
                Some(FrontierCut::new(cut, &f)),
            );
            same(&h, &solve(&pos, ctx, &leaf(), Options::default()), "exact");
            assert_eq!(h.stats.frontier, 0);
        }
    }
}

/// A frontier that values like [`hash_frontier`], logs every request the solver makes (the
/// canonical form, in order) and every batch it is handed ([`Frontier::prefetch`]), and asks
/// for batches from `depth` plays before the frontier.
struct Batching {
    depth: u8,
    trick: u8,
    asked: std::sync::Mutex<Vec<CanonPos>>,
    handed: std::sync::Mutex<Vec<Vec<CanonPos>>>,
}

impl Batching {
    fn new(depth: u8, trick: u8) -> Self {
        Batching {
            depth,
            trick,
            asked: std::sync::Mutex::new(Vec::new()),
            handed: std::sync::Mutex::new(Vec::new()),
        }
    }
}

impl Frontier for Batching {
    fn value(&self, pos: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        hash_frontier(pos, ctx)
    }

    fn value_at(&self, _pos: &Position, canon: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        self.asked.lock().expect("no panics").push(*canon);
        hash_frontier(canon, ctx)
    }

    fn prefetch_depth(&self) -> u8 {
        self.depth
    }

    fn prefetch(&self, pos: &[CanonPos], _ctx: &MatchCtx) {
        assert!(
            pos.iter().all(|c| c.tricks_done + 1 == self.trick),
            "a batch holds frontier positions only"
        );
        self.handed.lock().expect("no panics").push(pos.to_vec());
    }
}

/// The frontier batch (docs/ENGINEERING-2026-10-17.md, item B) changes nothing: for every depth
/// 1..=7, switch combination and tie order, the optimised solver's answer and every counter, and
/// the sequence of positions it asks the evaluator for, are exactly those of the solve without
/// batches; batches hold only frontier positions; and from depth 1 to 3 every position asked
/// for was handed over in a batch first (the gathering follows the search's own cards). The
/// parallel solver's answers are unchanged too.
#[test]
fn frontier_batches_change_nothing_and_cover_the_requests() {
    let mut asked_total = 0;
    let mut batches = 0;
    for seed in 0..10 {
        for (k, extra, cut) in [(4, 0, 5u8), (3, 2, 4), (3, 5, 4), (3, 7, 4), (2, 3, 3)] {
            let m = position_mid(seed, k, extra);
            let (pos, ctx, pending) = from_match(&m).expect("position");
            assert!(pending.is_none());
            for order in [TieOrder::Ascending, TieOrder::Descending] {
                for bits in [15u8, 0, 4, 11, 14] {
                    let opts = FastOptions {
                        order,
                        ..FastOptions::default()
                    }
                    .combo(bits);
                    let plain = Batching::new(0, cut);
                    let want = solve_fast_cut(
                        &pos,
                        ctx,
                        &leaf(),
                        opts,
                        Some(FrontierCut::new(cut, &plain)),
                        &mut SolveMem::new(),
                        None,
                    );
                    let want_asked = plain.asked.into_inner().expect("no panics");
                    assert!(plain.handed.into_inner().expect("no panics").is_empty());
                    asked_total += want_asked.len();
                    for depth in 1..=7u8 {
                        let what = format!(
                            "seed {seed} trick {k}+{extra} cut {cut} {order:?} combo {bits} depth {depth}"
                        );
                        let f = Batching::new(depth, cut);
                        let got = solve_fast_cut(
                            &pos,
                            ctx,
                            &leaf(),
                            opts,
                            Some(FrontierCut::new(cut, &f)),
                            &mut SolveMem::new(),
                            None,
                        );
                        same(&want, &got, &what);
                        assert_eq!(want.stats, got.stats, "{what}: counters");
                        let asked = f.asked.into_inner().expect("no panics");
                        assert_eq!(asked, want_asked, "{what}: requests");
                        let handed = f.handed.into_inner().expect("no panics");
                        batches += handed.len();
                        if depth <= 3 {
                            let all: std::collections::BTreeSet<CanonPos> =
                                handed.into_iter().flatten().collect();
                            assert!(
                                asked.iter().all(|c| all.contains(c)),
                                "{what}: a request not handed over first"
                            );
                        }
                        if depth == 2 && bits == 15 {
                            let f = Batching::new(depth, cut);
                            let p = solve_fast_par_cut(
                                &pos,
                                ctx,
                                &leaf(),
                                opts,
                                3,
                                None,
                                &mut SolveMem::new(),
                                Some(FrontierCut::new(cut, &f)),
                            );
                            same(&want, &p, &format!("{what} parallel"));
                        }
                    }
                }
            }
        }
    }
    // Trick-1 roots with a rob still to be made (burn nodes above the frontier).
    let mut checked = 0;
    for seed in 0..400u64 {
        let Some((pos, ctx)) = trick1_with_rob(seed) else {
            continue;
        };
        let opts = FastOptions::default();
        let plain = Batching::new(0, 2);
        let want = solve_fast_cut(
            &pos,
            ctx,
            &leaf(),
            opts,
            Some(FrontierCut::new(2, &plain)),
            &mut SolveMem::new(),
            None,
        );
        let want_asked = plain.asked.into_inner().expect("no panics");
        for depth in 1..=7u8 {
            let f = Batching::new(depth, 2);
            let got = solve_fast_cut(
                &pos,
                ctx,
                &leaf(),
                opts,
                Some(FrontierCut::new(2, &f)),
                &mut SolveMem::new(),
                None,
            );
            let what = format!("seed {seed} trick 1 rob depth {depth}");
            same(&want, &got, &what);
            assert_eq!(want.stats, got.stats, "{what}: counters");
            assert_eq!(
                f.asked.into_inner().expect("no panics"),
                want_asked,
                "{what}: requests"
            );
        }
        checked += 1;
        if checked == 6 {
            break;
        }
    }
    assert_eq!(checked, 6, "enough trick-1 rob roots");
    assert!(
        asked_total > 5_000,
        "the frontier is reached: {asked_total}"
    );
    assert!(batches > 1_000, "batches are made: {batches}");
}
