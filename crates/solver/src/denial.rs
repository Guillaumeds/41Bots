//! The forced-denial check (PLAN amendment A2; for the §14 scenario tests only, **never a
//! decision rule**): in one fully known world, can the other three teams, playing together,
//! stop team `team` from winning any of the remaining tricks of this hand, whatever `team`
//! does?
//!
//! It is a plain two-sided brute force over the cards (no scores, no money chance): the hand is
//! played to its fifth trick even if a game would end earlier, because the question is about
//! tricks. It is exponential, so use it from trick 3 on, or on small hand-made deals.

use engine::cards::cards;
use engine::team_of;

use crate::{Position, Step};

/// Whether the other three teams can deny `team` every remaining trick of this hand.
pub fn forced_denial(p: &Position, team: u8) -> bool {
    if p.hands.iter().all(|&h| h == 0) {
        return true;
    }
    let seat = p.to_act();
    let mover_is_team = team_of(seat) == team;
    let mut any = false;
    for c in cards(p.legal()) {
        let ok = match after(p, c, team) {
            Some(q) => forced_denial(&q, team),
            None => false,
        };
        if mover_is_team && !ok {
            return false;
        }
        if !mover_is_team && ok {
            return true;
        }
        any |= ok;
    }
    if mover_is_team { true } else { any }
}

/// The position after `c`, ignoring game ends; `None` if `team` wins the trick `c` completes.
fn after(p: &Position, c: engine::cards::Card, team: u8) -> Option<Position> {
    let completes = p.n_played == 7;
    let mut q = *p;
    // Play the card without scoring: a copy of the trick bookkeeping, with a neutral hand
    // state so no game can end.
    q.hand.scores = [0; 4];
    q.hand.flags = [true; 4];
    match q.play(c) {
        Step::Next(n) => {
            if completes {
                let winner_team = team_of(n.leader);
                if winner_team == team {
                    return None;
                }
            }
            Some(n)
        }
        // The last trick of the hand: decide by the trick winner directly.
        Step::GameWon(_) | Step::HandOver(_) => {
            let mut t = q.trick;
            t[7] = c;
            let w = engine::rules::trick_winner(&t, q.trump);
            let winner = (q.leader + w as u8) % 8;
            if team_of(winner) == team {
                None
            } else {
                let mut e = q;
                e.hands[usize::from(q.to_act())] &= !engine::cards::bit(c);
                Some(e)
            }
        }
    }
}
