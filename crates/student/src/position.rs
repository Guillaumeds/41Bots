//! A decision position in raw form (docs/STUDENT.md §3): the hand's start state, the deck in deal
//! order, every action of the hand before the decision, and the deciding seat. Replaying it
//! rebuilds the true deal (for diagnostics) and the deciding seat's exact view, so a record never
//! needs to store encoded features: any later encoder reads the same positions.

use engine::cards::{Card, DECK_SIZE};
use engine::game::{Action, HandStartState, Match, Phase, Rules};
use engine::view::SeatView;
use sha2::{Digest, Sha256};

/// Most actions a hand can have before a decision (four tricks of eight plays, seven plays of
/// the fifth, up to three burns): 42, with room.
pub const MAX_ACTIONS: usize = 48;

/// A candidate of a decision: an action and, for a robber's joint decision, the card played
/// straight after the burn (as `search::pimc::Candidate`).
pub type Cand = (Action, Option<Card>);

/// The code of a hand action: a play of `c` is `c + 1` (1..=53), a burn of `c` is `65 + c`
/// (65..=117); 0 is "no action" (padding).
pub fn action_code(a: Action) -> u16 {
    match a {
        Action::Play(c) => u16::from(c) + 1,
        Action::Burn(c) => 65 + u16::from(c),
    }
}

/// The action of a code of [`action_code`].
pub fn action_from_code(x: u16) -> Option<Action> {
    match x {
        1..=53 => Some(Action::Play((x - 1) as Card)),
        65..=117 => Some(Action::Burn((x - 65) as Card)),
        _ => None,
    }
}

/// Joint (burn, play) candidates carry this bit.
pub const JOINT: u16 = 0x8000;
/// A burn with nothing after it (the dealer's burn before trick 1; never a student decision).
pub const LONE_BURN: u16 = 0x4000;

/// The code of a candidate: a play of `c` is `c + 1` (1..=53); a robber's burn of `b` then play
/// of `c` is `0x8000 + 64·b + c`; a lone burn of `b` is `0x4000 + b`. One code per candidate, so
/// a record's candidates read back without its decision kind (`search::teacher`'s older code
/// needed the kind).
pub fn cand_code(c: Cand) -> u16 {
    match c {
        (Action::Play(p), None) => u16::from(p) + 1,
        (Action::Burn(b), Some(p)) => JOINT + 64 * u16::from(b) + u16::from(p),
        (Action::Burn(b), None) => LONE_BURN + u16::from(b),
        (Action::Play(_), Some(_)) => panic!("a play has no joint card"),
    }
}

/// The candidate of a code of [`cand_code`].
pub fn cand_from_code(x: u16) -> Option<Cand> {
    if x & JOINT != 0 {
        let r = x & !JOINT;
        let (b, p) = (r / 64, r % 64);
        (b < 53 && p < 53).then_some((Action::Burn(b as Card), Some(p as Card)))
    } else if x & LONE_BURN != 0 {
        let b = x & !LONE_BURN;
        (b < 53).then_some((Action::Burn(b as Card), None))
    } else {
        (1..=53)
            .contains(&x)
            .then(|| (Action::Play((x - 1) as Card), None))
    }
}

/// A decision position in raw form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawPosition {
    /// The hand's start state.
    pub start: HandStartState,
    /// The hand's deck in deal order.
    pub deck: [u8; DECK_SIZE],
    /// The hand's actions before the decision.
    pub actions: Vec<Action>,
    /// The deciding seat.
    pub seat: u8,
}

/// A replayed position: the match at the decision, the seat's view and its legal actions.
pub struct Replayed {
    /// The match (the true deal, all cards known).
    pub m: Match,
    /// The deciding seat's view.
    pub view: SeatView,
    /// Its legal actions.
    pub legal: Vec<Action>,
}

impl RawPosition {
    /// The match at the decision (the true deal): the start state and deck, then the actions.
    pub fn replay_match(&self) -> Result<Match, String> {
        self.start
            .validate()
            .map_err(|e| format!("start state: {e}"))?;
        let mut m = Match::from_hand_start(self.start, self.deck, 0, Rules::HOUSE)
            .map_err(|e| format!("start: {e:?}"))?;
        let h = m.hand_index();
        for (i, &a) in self.actions.iter().enumerate() {
            m.apply(a)
                .map_err(|e| format!("action {i} ({a:?}) is illegal: {e:?}"))?;
            if m.hand_index() != h || m.phase() == Phase::MatchOver {
                return Err(format!("action {i} ends the hand"));
            }
        }
        Ok(m)
    }

    /// Replay to the decision; refused unless the deciding seat is the one to act.
    pub fn replay(&self) -> Result<Replayed, String> {
        let m = self.replay_match()?;
        if m.to_act() != Some(self.seat) {
            return Err(format!(
                "seat {} is not to act (seat {:?} is)",
                self.seat,
                m.to_act()
            ));
        }
        let view = m.view(self.seat);
        let legal = m.legal_actions();
        Ok(Replayed { m, view, legal })
    }

    /// The position's canonical bytes: start state (dealer, scores, bonus flags, games won, game
    /// index, hand index little-endian), the deck, the number of actions and their codes
    /// (little-endian), the seat.
    pub fn bytes(&self) -> Vec<u8> {
        let s = &self.start;
        let mut out = Vec::with_capacity(80 + 2 * self.actions.len());
        out.push(s.dealer);
        out.extend_from_slice(&s.scores);
        out.extend(s.bonus_taken.iter().map(|&b| u8::from(b)));
        out.extend_from_slice(&s.games_won);
        out.push(s.game_index);
        out.extend_from_slice(&s.hand_index.to_le_bytes());
        out.extend_from_slice(&self.deck);
        out.push(self.actions.len() as u8);
        for &a in &self.actions {
            out.extend_from_slice(&action_code(a).to_le_bytes());
        }
        out.push(self.seat);
        out
    }

    /// The position's identity: the SHA-256 of [`RawPosition::bytes`].
    pub fn sha256(&self) -> [u8; 32] {
        Sha256::digest(self.bytes()).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_round_trip_and_never_collide() {
        let mut seen = std::collections::BTreeSet::new();
        for c in 0..53u8 {
            for a in [Action::Play(c), Action::Burn(c)] {
                let x = action_code(a);
                assert_ne!(x, 0);
                assert_eq!(action_from_code(x), Some(a));
                assert!(seen.insert(x));
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for b in 0..53u8 {
            for cand in [(Action::Play(b), None), (Action::Burn(b), None)] {
                assert_eq!(cand_from_code(cand_code(cand)), Some(cand));
                assert!(seen.insert(cand_code(cand)));
            }
            for p in 0..53u8 {
                let cand = (Action::Burn(b), Some(p));
                assert_eq!(cand_from_code(cand_code(cand)), Some(cand));
                assert!(seen.insert(cand_code(cand)));
            }
        }
        assert_eq!(cand_from_code(0), None);
        assert_eq!(action_from_code(0), None);
    }

    #[test]
    fn a_position_replays_to_its_view() {
        let start = HandStartState {
            dealer: 3,
            scores: [0; 4],
            bonus_taken: [false; 4],
            games_won: [0; 4],
            hand_index: 0,
            game_index: 0,
        };
        let deck: [u8; DECK_SIZE] = core::array::from_fn(|c| c as u8);
        let mut m = Match::from_hand_start(start, deck, 0, Rules::HOUSE).expect("valid");
        let mut actions = Vec::new();
        for _ in 0..5 {
            let a = m.legal_actions()[0];
            m.apply(a).expect("legal");
            actions.push(a);
        }
        let seat = m.to_act().expect("to act");
        let p = RawPosition {
            start,
            deck,
            actions,
            seat,
        };
        let r = p.replay().expect("replays");
        assert_eq!(r.view, m.view(seat));
        assert_eq!(r.legal, m.legal_actions());
        let mut q = p.clone();
        q.seat = (seat + 1) % 8;
        assert!(q.replay().is_err());
        assert_ne!(p.sha256(), q.sha256());
    }
}
