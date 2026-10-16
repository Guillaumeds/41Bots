//! The outcome head (docs/M3-HYBRID-SPEC.md §6a): the network classifies how the hand ends, and
//! the value is the mixture of the exact money vectors of those endings.
//!
//! **Why it is complete.** In a perfect-information position the exact max^n value is the leaf
//! value of one terminal state of the hand (a backup only copies a child's vector). From the
//! start of trick `k` (with `r = 6 − k` tricks left) a terminal state is one of:
//! - **a game won by team `t`** (by 41 at any trick, or a jink): its value is the solver's
//!   game-won leaf, which depends on `t` alone (certain money for a second game, otherwise the
//!   equity of the new game's start);
//! - **the hand over** with final scores `scores + 5·n + bonus`, where `n` gives the tricks each
//!   team wins of the `r` left (a composition of `r` into four parts) and the bonus is fixed by
//!   the position: when the 5 of trumps is still live its holder's team scores it (the 5 wins its
//!   own trick, and on a line that ends the hand it has been played); when it was never played
//!   and is out of play, the end-of-hand bonus team of the canonical form scores it (the team of
//!   the highest of the best trump so far and every live trump: every live card is played by the
//!   end); when it was played, nobody does. The size is 6 for a team's first bonus and 5 later,
//!   by its current flag (no other bonus can come first).
//!
//! So the **vocabulary** for trick `k` is: 4 game-won outcomes (one per team), then the
//! compositions of `r` in a fixed order (`n0` from 0 up, then `n1`, then `n2`; `n3` the rest):
//! 8, 14, 24 and 39 outcomes for tricks 5, 4, 3 and 2. Teams are in the encoder's rotated order.
//!
//! **Masks** (exact: they only remove outcomes no line can reach): a game won by `t` needs
//! `scores[t] + 5r + (t's bonus) ≥ 41` or a possible jink (`t` has won every trick so far); a
//! hand-over composition is removed if a team would reach 41 or win all five tricks on it (the
//! game would have ended), or if the team holding the highest live trump wins no trick (that
//! trump wins the trick it is played in).
//!
//! **Labels**: the first outcome (in vocabulary order, mask on) whose money vector equals the
//! exact value; `search labels` refuses a position with none, and tests check every row.

use equity::{MatchState, MoneyChance};
use solver::frontier::CanonPos;
use solver::{Leaf, MatchCtx};

use crate::encode::rotation;

/// The most outcomes of any trick (39 at trick 2), with room.
pub const MAX_OUT: usize = 64;

/// Game-won outcomes come first.
pub const GAME_WON: usize = 4;

/// The compositions of `r` tricks into four teams, in vocabulary order.
pub fn compositions(r: u8) -> Vec<[u8; 4]> {
    let mut out = Vec::new();
    for a in 0..=r {
        for b in 0..=r - a {
            for c in 0..=r - a - b {
                out.push([a, b, c, r - a - b - c]);
            }
        }
    }
    out
}

/// The vocabulary size for positions at the start of trick `trick` (2..=5).
pub fn vocab(trick: u8) -> usize {
    assert!((2..=5).contains(&trick), "frontier tricks are 2..=5");
    GAME_WON + compositions(6 - trick).len()
}

/// The outcomes of a position: how many, which are possible (bit `o`), and the exact money vector
/// of each possible one (rotated team order; impossible ones hold an even split, never used).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcomes {
    /// Vocabulary size.
    pub n: usize,
    /// Possible outcomes.
    pub mask: u64,
    /// Money vectors.
    pub vecs: Vec<MoneyChance>,
}

impl Outcomes {
    /// The label of an exact value: the first possible outcome with exactly this vector.
    pub fn label(&self, v: &MoneyChance) -> Option<u8> {
        (0..self.n)
            .find(|&o| self.mask >> o & 1 == 1 && self.vecs[o] == *v)
            .map(|o| o as u8)
    }
}

/// The outcomes of a canonical position with its match context, in the encoder's rotated team
/// order (`crate::encode::rotation`).
pub fn outcomes(c: &CanonPos, ctx: &MatchCtx, leaf: &dyn Leaf) -> Outcomes {
    let r = rotation(c);
    let rt = |t: u8| usize::from((t + r) % 4);
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
    let next = ((rt(ctx.dealer_team) + 1) % 4) as u8;
    let left = engine::game::TRICKS_PER_HAND - c.tricks_done;
    // The bonus still to come and its team: the live 5's holder, else the end-bonus team.
    let five_holder = c.slot(0).map(|(s, _)| ((s + r) % 4) as usize);
    let bonus_team = if c.five_played {
        None
    } else {
        five_holder.or(c.end_bonus.map(rt))
    };
    let bonus_of = |t: usize| -> u8 {
        if bonus_team == Some(t) {
            if flags[t] {
                engine::game::BONUS_LATER
            } else {
                engine::game::BONUS_FIRST
            }
        } else {
            0
        }
    };
    // The team holding the highest live trump (classes 0–2 are slots 0..15, best first).
    let top = (0..15)
        .find_map(|i| c.slot(i))
        .map(|(s, _)| ((s + r) % 4) as usize);
    let comps = compositions(left);
    let n = GAME_WON + comps.len();
    let mut mask = 0u64;
    let mut vecs = Vec::with_capacity(n);
    for t in 0..4usize {
        let mut g = games;
        g[t] += 1;
        vecs.push(if g[t] >= engine::game::GAMES_TO_WIN {
            MoneyChance::certain(t as u8)
        } else {
            leaf.value(&MatchState {
                scores: [0; 4],
                games: g,
                dealer_team: next,
            })
        });
        let reach = u32::from(scores[t]) + 5 * u32::from(left) + u32::from(bonus_of(t));
        let jink = tricks[t] == c.tricks_done;
        if reach >= u32::from(engine::game::WINNING_SCORE) || jink {
            mask |= 1 << t;
        }
    }
    for (i, n4) in comps.iter().enumerate() {
        let mut fin = [0u8; 4];
        let mut ok = top.is_none_or(|t| n4[t] > 0);
        for t in 0..4 {
            fin[t] = scores[t] + 5 * n4[t] + bonus_of(t);
            if fin[t] >= engine::game::WINNING_SCORE
                || tricks[t] + n4[t] == engine::game::TRICKS_PER_HAND
            {
                ok = false;
            }
        }
        if ok {
            mask |= 1 << (GAME_WON + i);
            vecs.push(leaf.value(&MatchState {
                scores: fin,
                games,
                dealer_team: next,
            }));
        } else {
            vecs.push(MoneyChance::even());
        }
    }
    Outcomes { n, mask, vecs }
}

/// The value of the outcome head: a softmax over the possible outcomes' logits (the integer rule
/// of [`crate::money`]: `2^62 · e^(z_o − max z)` with the maximum over possible outcomes), then
/// the weighted mean of their money vectors in `u128` with largest-remainder rounding
/// ([`MoneyChance::mix_wide`]: sums to exactly 2^32, ties to the lower team).
pub fn money_from_outcomes(z: &[i64], o: &Outcomes) -> MoneyChance {
    let m = (0..o.n)
        .filter(|&i| o.mask >> i & 1 == 1)
        .map(|i| z[i])
        .max()
        .expect("a position has at least one possible outcome");
    let terms: Vec<(u128, MoneyChance)> = (0..o.n)
        .filter(|&i| o.mask >> i & 1 == 1)
        .map(|i| {
            (
                u128::from(crate::money::exp_weight((m - z[i]) as u64)),
                o.vecs[i],
            )
        })
        .collect();
    MoneyChance::mix_wide(&terms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_sizes() {
        assert_eq!([vocab(5), vocab(4), vocab(3), vocab(2)], [8, 14, 24, 39]);
        assert!(vocab(2) <= MAX_OUT);
        let c = compositions(2);
        assert_eq!(c[0], [0, 0, 0, 2]);
        assert_eq!(c[c.len() - 1], [2, 0, 0, 0]);
        assert!(c.iter().all(|x| x.iter().sum::<u8>() == 2));
    }
}
