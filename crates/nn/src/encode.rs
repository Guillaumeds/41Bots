//! The oracle encoder (docs/M3-HYBRID-SPEC.md §6, encoder `oracle-v2`): a position at the start
//! of a trick, in the solver's canonical form ([`CanonPos`]), with the match context, as feature
//! indices for the network.
//!
//! **Rotation.** Seats are first rotated so the seat to lead is seat 0 (teams rotate with them);
//! the network's four outputs are in that rotated team order and [`encode`] returns the rotation
//! so the caller can turn them back. Rotating every seat maps exact values by the same team
//! rotation (tested in `solver/tests/frontier.rs`, `rotation_maps_exact_values`, with a leaf that
//! commutes with rotation, as the equity table does), so the network learns each situation once
//! instead of eight times.
//!
//! **Features** (all from the rotated position; binary unless stated):
//! - each canonical card slot × the seat holding it (54 × 8), and × the card's place in its
//!   holder's hand in card-index order (54 × 5): together exactly the live cards by class and
//!   rank, their holders, and every hand's tie order;
//! - the trump suit (4); tricks done (5);
//! - each team's score as one of the 15 reachable values (4 × 15), its bonus flag (4), its
//!   tricks this hand (4 × 5);
//! - the 5 of trumps played (1); the end-of-hand bonus team (none or a team: 5);
//! - each team with one game won (4); the dealer's team (4);
//! - dense, from the equity leaf (PLAN §6.5: "each team's equity if the hand ended now … and
//!   the reachable successor equities"), each money chance `p` given as `log2(p) / 32` (`p` at
//!   least 2^-32, so values lie in `[−1, 0]`), in units of 2^-24 ([`log2_feature`], integer
//!   only):
//!   - every team's money chance if the hand ended now with the current scores (4), and if team
//!     `t` won the game now, for each `t` (4 × 4);
//!   - the successor equities of the next trick: for each team `a`, the money chances after `a`
//!     wins the next trick (4 × 4), and of the next two tricks: after `a` then `b` win them
//!     (16 × 4). Each is scored exactly as the hand is (`equity::hand::InHand`: trick points, the
//!     5 of trumps' bonus at once in the first of these tricks its holder's team wins, 41, then
//!     after trick 5 the jink, the end-of-hand bonus to the end-bonus team and 41); when the hand
//!     goes on after those tricks the leaf is taken as if it ended there. With one trick left the
//!     two-trick block repeats the one-trick one.
//!
//!   These are functions of fields the binary features already hold; they let the network
//!   *select* an outcome instead of learning the match-equity table, and in log form a selected
//!   outcome's chances are exactly its logits.
//!
//! The binary features are a complete description of the canonical form and the match context
//! (see [`decode`], tested), so equal encodings mean equal canonical forms, and so equal exact
//! values (`solver::frontier`).

use engine::cards::Suit;
use equity::hand::{AfterHand, AfterTrick, InHand, Scoring};
use equity::state::{N_SCORES, SCORE_VALUES, score_index};
use equity::{MatchState, MoneyChance, ONE};
use solver::frontier::{CanonPos, SLOTS};
use solver::{Leaf, MatchCtx};

/// The encoder's name, recorded in every weights file and shard sidecar.
pub const ENCODER: &str = "oracle-v2";

/// Places a card can have in its holder's hand (a hand holds at most 4 cards at the start of
/// trick 2; 5 keeps room).
pub const PLACES: usize = 5;

/// Slot × holding seat.
pub const OFF_SLOT_SEAT: usize = 0;
/// Slot × place in the holder's hand.
pub const OFF_SLOT_PLACE: usize = OFF_SLOT_SEAT + SLOTS * 8;
/// Trump suit.
pub const OFF_TRUMP: usize = OFF_SLOT_PLACE + SLOTS * PLACES;
/// Tricks done (0..5).
pub const OFF_TRICKS_DONE: usize = OFF_TRUMP + 4;
/// Team × score index.
pub const OFF_SCORE: usize = OFF_TRICKS_DONE + 5;
/// Team's bonus flag.
pub const OFF_FLAG: usize = OFF_SCORE + 4 * N_SCORES;
/// Team × tricks this hand (0..5).
pub const OFF_TRICKS: usize = OFF_FLAG + 4;
/// The 5 of trumps played.
pub const OFF_FIVE_PLAYED: usize = OFF_TRICKS + 4 * 5;
/// End-of-hand bonus team: none, then teams 0..4.
pub const OFF_END_BONUS: usize = OFF_FIVE_PLAYED + 1;
/// Team has one game.
pub const OFF_GAMES: usize = OFF_END_BONUS + 5;
/// Dealer's team.
pub const OFF_DEALER: usize = OFF_GAMES + 4;
/// Binary features.
pub const N_BINARY: usize = OFF_DEALER + 4;
/// Dense: money chance if the hand ended now (team).
pub const DENSE_NOW: usize = 0;
/// Dense: money chance if team `t` won the game now (`t × 4 + team`).
pub const DENSE_GAME: usize = 4;
/// Dense: after team `a` wins the next trick (`a × 4 + team`).
pub const DENSE_NEXT1: usize = 20;
/// Dense: after `a` then `b` win the next two tricks (`a × 16 + b × 4 + team`).
pub const DENSE_NEXT2: usize = 36;
/// Dense features (they follow the binary ones).
pub const N_DENSE: usize = 100;
/// All features.
pub const N_FEATURES: usize = N_BINARY + N_DENSE;
/// Room for active binary features (at most 84 occur: 2 per live card, at most 32 live cards).
pub const MAX_ACTIVE: usize = 96;
/// Padding of the active list.
pub const PAD: u16 = u16::MAX;
/// The input scale: a binary feature that is on, and a dense value of 1, are 2^24.
pub const INPUT_SHIFT: u32 = 24;

/// An encoded position: the active binary features (ascending, then [`PAD`]) and the dense
/// values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Features {
    /// Active binary feature indices, ascending; `PAD` after the first `n_active`.
    pub active: [u16; MAX_ACTIVE],
    /// How many are active.
    pub n_active: u8,
    /// Dense values, `−2^24..=0` (`log2(p) / 32` in units of 2^-24).
    pub dense: [i32; N_DENSE],
}

impl Features {
    /// The active binary feature indices.
    pub fn active(&self) -> &[u16] {
        &self.active[..usize::from(self.n_active)]
    }

    fn push(&mut self, i: usize) {
        debug_assert!(i < N_BINARY);
        debug_assert!(
            self.n_active == 0 || usize::from(self.active[usize::from(self.n_active) - 1]) < i
        );
        self.active[usize::from(self.n_active)] = i as u16;
        self.n_active += 1;
    }
}

/// The rotation [`encode`] applies: seats move this many places so the leader is seat 0.
pub fn rotation(c: &CanonPos) -> u8 {
    (8 - c.leader % 8) % 8
}

/// A chance `v` (units of 2^-32, at least 1 counted) as a dense value: `log2(v / 2^32) / 32` in
/// units of 2^-24, so in `[−2^24, 0]`, rounded down. Integer only: the integer part from the
/// highest set bit, then 19 fraction bits by the classic squaring method (square the mantissa
/// in `[1, 2)`, held with 31 fraction bits and truncated after each square; a square of 2 or
/// more gives a fraction bit 1 and is halved).
pub fn log2_feature(v: u64) -> i32 {
    let v = v.clamp(1, ONE);
    let n = 63 - v.leading_zeros();
    let mut m: u64 = if n >= 31 {
        v >> (n - 31)
    } else {
        v << (31 - n)
    };
    let mut frac: i64 = 0;
    for i in (0..19).rev() {
        m = (m * m) >> 31;
        if m >= 1 << 32 {
            m >>= 1;
            frac |= 1 << i;
        }
    }
    (((i64::from(n) - 32) << 19) + frac) as i32
}

/// The money chance after the given teams win the next tricks in order, scored as the hand is
/// (see the module notes; `five`: the team holding the live 5 of trumps, whose bonus comes in
/// the first of these tricks that team wins; `end`: the end-of-hand bonus team, paid only if
/// that team has won a trick this hand: the best trump wins its own trick, so a line where its
/// team wins none cannot happen, and paying it would give a score no team can stand on). When
/// the hand goes on after them, the leaf of the scores reached (as if the hand ended there).
fn successor(
    h: InHand,
    winners: &[u8],
    five: Option<u8>,
    end: Option<u8>,
    games: [u8; 4],
    next_dealer: u8,
    leaf: &dyn Leaf,
) -> MoneyChance {
    let won = |t: u8| {
        let mut g = games;
        g[usize::from(t)] += 1;
        if g[usize::from(t)] >= engine::game::GAMES_TO_WIN {
            MoneyChance::certain(t)
        } else {
            leaf.value(&MatchState {
                scores: [0; 4],
                games: g,
                dealer_team: next_dealer,
            })
        }
    };
    let mut h = h;
    let mut five_due = five;
    for &w in winners {
        let five_here = five_due == Some(w);
        if five_here {
            five_due = None;
        }
        match h.play_trick(w, five_here, Scoring::HOUSE) {
            AfterTrick::GameWon(t, _) => return won(t),
            AfterTrick::Next(n) => h = n,
            AfterTrick::TricksDone(n) => {
                let end = end.filter(|&t| n.tricks[usize::from(t)] > 0);
                return match n.finish(end, Scoring::HOUSE) {
                    AfterHand::GameWon(t, _) => won(t),
                    AfterHand::HandOver(sc) => leaf.value(&MatchState {
                        scores: sc,
                        games,
                        dealer_team: next_dealer,
                    }),
                };
            }
        }
    }
    leaf.value(&MatchState {
        scores: h.scores,
        games,
        dealer_team: next_dealer,
    })
}

/// Encode a canonical position with its match context; `leaf` gives the dense equity features.
/// Returns the features and the rotation `r` applied: the network's output for team `t` is the
/// chance of the real team `(t + 4 − r mod 4) mod 4`, so `money.rotate((4 − r % 4) % 4)` turns
/// it back.
///
/// # Panics
/// If the position is not a trick-boundary position of a hand in progress (a score that cannot
/// stand between tricks, five tricks done, a card's place beyond [`PLACES`]).
pub fn encode(c: &CanonPos, ctx: &MatchCtx, leaf: &dyn Leaf) -> (Features, u8) {
    let r = rotation(c);
    let rt = |t: u8| usize::from((t + r) % 4);
    let mut f = Features {
        active: [PAD; MAX_ACTIVE],
        n_active: 0,
        dense: [0; N_DENSE],
    };
    for i in 0..SLOTS {
        if let Some((seat, _)) = c.slot(i) {
            f.push(OFF_SLOT_SEAT + i * 8 + usize::from((seat + r) % 8));
        }
    }
    for i in 0..SLOTS {
        if let Some((_, place)) = c.slot(i) {
            assert!(usize::from(place) < PLACES, "a hand of more than 5 cards");
            f.push(OFF_SLOT_PLACE + i * PLACES + usize::from(place));
        }
    }
    f.push(OFF_TRUMP + c.trump as usize);
    assert!(c.tricks_done < 5, "a hand in progress");
    f.push(OFF_TRICKS_DONE + usize::from(c.tricks_done));
    let mut scores = [0u8; 4];
    let mut flags = [false; 4];
    let mut tricks = [0u8; 4];
    let mut games = [0u8; 4];
    for t in 0..4u8 {
        scores[rt(t)] = c.scores[usize::from(t)];
        flags[rt(t)] = c.flags[usize::from(t)];
        tricks[rt(t)] = c.tricks[usize::from(t)];
        games[rt(t)] = ctx.games[usize::from(t)];
    }
    for (t, &s) in scores.iter().enumerate() {
        let i = score_index(s).expect("a score that can stand between tricks");
        f.push(OFF_SCORE + t * N_SCORES + i);
    }
    for (t, &fl) in flags.iter().enumerate() {
        if fl {
            f.push(OFF_FLAG + t);
        }
    }
    for (t, &k) in tricks.iter().enumerate() {
        assert!(k < 5, "at most four tricks before the last");
        f.push(OFF_TRICKS + t * 5 + usize::from(k));
    }
    if c.five_played {
        f.push(OFF_FIVE_PLAYED);
    }
    f.push(OFF_END_BONUS + c.end_bonus.map_or(0, |t| 1 + rt(t)));
    for (t, &g) in games.iter().enumerate() {
        assert!(g <= 1, "a match in progress");
        if g == 1 {
            f.push(OFF_GAMES + t);
        }
    }
    let dealer = rt(ctx.dealer_team) as u8;
    f.push(OFF_DEALER + usize::from(dealer));
    let next = (dealer + 1) % 4;
    let put = |f: &mut Features, at: usize, v: MoneyChance| {
        for (u, &x) in v.0.iter().enumerate() {
            f.dense[at + u] = log2_feature(x);
        }
    };
    let now = leaf.value(&MatchState {
        scores,
        games,
        dealer_team: next,
    });
    put(&mut f, DENSE_NOW, now);
    for t in 0..4u8 {
        let mut g = games;
        g[usize::from(t)] += 1;
        let v = if g[usize::from(t)] >= engine::game::GAMES_TO_WIN {
            MoneyChance::certain(t)
        } else {
            leaf.value(&MatchState {
                scores: [0; 4],
                games: g,
                dealer_team: next,
            })
        };
        put(&mut f, DENSE_GAME + 4 * usize::from(t), v);
    }
    let h = InHand {
        scores,
        flags,
        tricks,
        five_played: c.five_played,
        tricks_done: c.tricks_done,
    };
    // The live 5's holder is slot 0's; its team in the rotated order.
    let five = c.slot(0).map(|(s, _)| (s + r) % 4);
    let end = c.end_bonus.map(|t| rt(t) as u8);
    let two = c.tricks_done + 2 <= engine::game::TRICKS_PER_HAND;
    for a in 0..4u8 {
        let v1 = successor(h, &[a], five, end, games, next, leaf);
        put(&mut f, DENSE_NEXT1 + 4 * usize::from(a), v1);
        for b in 0..4u8 {
            let v2 = if two {
                successor(h, &[a, b], five, end, games, next, leaf)
            } else {
                v1
            };
            put(
                &mut f,
                DENSE_NEXT2 + 16 * usize::from(a) + 4 * usize::from(b),
                v2,
            );
        }
    }
    debug_assert!(f.dense.iter().all(|&d| (-(1 << 24)..=0).contains(&d)));
    (f, r)
}

/// Turn the network's rotated money chance back to real team order (`r` from [`encode`]).
pub fn unrotate(m: MoneyChance, r: u8) -> MoneyChance {
    m.rotate((4 - r % 4) % 4)
}

/// The canonical form (with the leader at seat 0) and the match context that the binary
/// features describe: the inverse of [`encode`] up to its rotation. `None` if the features are
/// not an encoding. Used to test that the encoding is complete.
pub fn decode(f: &Features) -> Option<(CanonPos, MatchCtx)> {
    let mut c = CanonPos {
        trump: Suit::Clubs,
        leader: 0,
        tricks_done: 0,
        scores: [0; 4],
        flags: [false; 4],
        tricks: [0; 4],
        five_played: false,
        end_bonus: None,
        slots: [0; SLOTS],
    };
    let mut ctx = MatchCtx {
        games: [0; 4],
        dealer_team: 0,
    };
    let mut seat = [None::<u8>; SLOTS];
    let mut place = [None::<u8>; SLOTS];
    let (mut trump, mut done, mut bonus, mut dealer) = (None, None, None, None);
    let mut score = [None::<u8>; 4];
    let mut tricks = [None::<u8>; 4];
    for &a in f.active() {
        let i = usize::from(a);
        if i < OFF_SLOT_PLACE {
            seat[i / 8] = Some((i % 8) as u8);
        } else if i < OFF_TRUMP {
            let j = i - OFF_SLOT_PLACE;
            place[j / PLACES] = Some((j % PLACES) as u8);
        } else if i < OFF_TRICKS_DONE {
            trump = Some(Suit::from_index((i - OFF_TRUMP) as u8));
        } else if i < OFF_SCORE {
            done = Some((i - OFF_TRICKS_DONE) as u8);
        } else if i < OFF_FLAG {
            let j = i - OFF_SCORE;
            score[j / N_SCORES] = Some(SCORE_VALUES[j % N_SCORES]);
        } else if i < OFF_TRICKS {
            c.flags[i - OFF_FLAG] = true;
        } else if i < OFF_FIVE_PLAYED {
            let j = i - OFF_TRICKS;
            tricks[j / 5] = Some((j % 5) as u8);
        } else if i < OFF_END_BONUS {
            c.five_played = true;
        } else if i < OFF_GAMES {
            bonus = Some(i - OFF_END_BONUS);
        } else if i < OFF_DEALER {
            ctx.games[i - OFF_GAMES] = 1;
        } else if i < N_BINARY {
            dealer = Some((i - OFF_DEALER) as u8);
        } else {
            return None;
        }
    }
    for i in 0..SLOTS {
        match (seat[i], place[i]) {
            (Some(s), Some(p)) => c.slots[i] = 1 + s + 8 * p,
            (None, None) => {}
            _ => return None,
        }
    }
    c.trump = trump?;
    c.tricks_done = done?;
    for t in 0..4 {
        c.scores[t] = score[t]?;
        c.tricks[t] = tricks[t]?;
    }
    c.end_bonus = match bonus? {
        0 => None,
        b => Some((b - 1) as u8),
    };
    ctx.dealer_team = dealer?;
    Some((c, ctx))
}

/// Rotate a canonical form and its match context by `r` seats (teams by `r mod 4`).
pub fn rotate_canon(c: &CanonPos, ctx: &MatchCtx, r: u8) -> (CanonPos, MatchCtx) {
    let rt = |t: u8| usize::from((t + r) % 4);
    let mut d = *c;
    d.leader = (c.leader + r) % 8;
    for i in 0..SLOTS {
        if let Some((s, p)) = c.slot(i) {
            d.slots[i] = 1 + (s + r) % 8 + 8 * p;
        }
    }
    let mut x = *ctx;
    for t in 0..4u8 {
        d.scores[rt(t)] = c.scores[usize::from(t)];
        d.flags[rt(t)] = c.flags[usize::from(t)];
        d.tricks[rt(t)] = c.tricks[usize::from(t)];
        x.games[rt(t)] = ctx.games[usize::from(t)];
    }
    d.end_bonus = c.end_bonus.map(|t| rt(t) as u8);
    x.dealer_team = rt(ctx.dealer_team) as u8;
    (d, x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log2_feature_matches_the_logarithm() {
        assert_eq!(log2_feature(ONE), 0);
        assert_eq!(log2_feature(1), -(1 << 24));
        assert_eq!(log2_feature(0), -(1 << 24));
        assert_eq!(log2_feature(ONE / 2), -(1 << 19));
        assert_eq!(log2_feature(ONE / 4), -(2 << 19));
        for v in [3u64, 1000, 123_456_789, ONE / 3, ONE - 1, 3_000_000_000] {
            let exact = ((v as f64) / (ONE as f64)).log2() * f64::from(1 << 19);
            let got = f64::from(log2_feature(v));
            assert!(
                got <= exact + 1e-6 && got > exact - 4.0,
                "{v}: {got} vs {exact}"
            );
        }
        // Monotone.
        let mut last = i32::MIN;
        for k in 0..2000u64 {
            let x = log2_feature(1 + k * (ONE / 2000));
            assert!(x >= last);
            last = x;
        }
    }
}
