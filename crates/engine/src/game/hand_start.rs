//! Starting a match at the start of any hand (docs/M3-SPEC.md §2.1): the PIMC search rebuilds
//! sampled worlds with it, and teacher hands start from stratified match states.

use super::{Event, HandState, Match, Phase, Rules, Truth, identity_deck, is_full_deck};
use crate::cards::{Card, DECK_SIZE};
use crate::rng::Rng;
use crate::{SEATS, TEAMS};

/// The public state at the start of a hand.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HandStartState {
    /// The dealer of the hand.
    pub dealer: u8,
    /// Each team's score in the current game.
    pub scores: [u8; 4],
    /// Whether each team has taken a best-trump bonus this game.
    pub bonus_taken: [bool; 4],
    /// Games won so far by each team.
    pub games_won: [u8; 4],
    /// Hands dealt before this one in the match.
    pub hand_index: u32,
    /// Games started before this one in the match.
    pub game_index: u8,
}

/// Why a hand-start state is not a reachable state of an unfinished match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateError(pub &'static str);

impl core::fmt::Display for StateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for StateError {}

impl HandStartState {
    /// Check that the state can occur at the start of a hand of an unfinished match.
    pub fn validate(&self) -> Result<(), StateError> {
        if usize::from(self.dealer) >= SEATS {
            return Err(StateError("no such dealer seat"));
        }
        for t in 0..TEAMS {
            let s = self.scores[t];
            if s >= super::WINNING_SCORE {
                return Err(StateError("a team at 41 or more has already won the game"));
            }
            // A score is 5k, or 5k+6 with k >= 1 (a first bonus comes with its trick): never 1
            // or 6 (RULES, "Reachable scores").
            if s % 5 > 1 || s == 1 || s == 6 {
                return Err(StateError(
                    "scores are tricks of 5 plus bonuses of 6 then 5",
                ));
            }
            if (s % 5 == 1) != self.bonus_taken[t] {
                return Err(StateError(
                    "a team has taken a bonus exactly when its score is 1 more than a multiple of 5",
                ));
            }
            if self.games_won[t] > 1 {
                return Err(StateError("a team with 2 games has already won the match"));
            }
        }
        // Note: M3-SPEC §2.1 says "at most 2 teams at 1 game", but with four partnerships three
        // or even all four can each hold one game while the match goes on (seen in play), so
        // that check is not made.
        if self.games_won.iter().sum::<u8>() != self.game_index {
            return Err(StateError("each finished game was won by one team"));
        }
        Ok(())
    }
}

impl Match {
    /// A match that starts at the start of a hand in state `s`, dealing `deck` (deal order,
    /// docs/M0-SPEC.md §1) as hand `s.hand_index`. Later hands are shuffled from `Rng::new(seed)`.
    /// The event log starts with `MatchStart { first_dealer: s.dealer }`, then the hand's events.
    pub fn from_hand_start(
        s: HandStartState,
        deck: [Card; DECK_SIZE],
        seed: u64,
        rules: Rules,
    ) -> Result<Match, StateError> {
        s.validate()?;
        if !is_full_deck(&deck) {
            return Err(StateError(
                "the deck must hold each of the 53 cards exactly once",
            ));
        }
        let mut m = Match {
            rules,
            seed,
            rng: Rng::new(seed),
            decks: vec![deck],
            next_deck: 0,
            hand_decks: None,
            dealer: s.dealer,
            hand_index: s.hand_index,
            game_index: s.game_index,
            scores: s.scores,
            games_won: s.games_won,
            bonus_taken: s.bonus_taken,
            money: None,
            phase: Phase::MatchOver,
            events: Vec::new(),
            actions: Vec::new(),
            extras: Vec::new(),
            truth: Truth {
                deck: identity_deck(),
                hands: [0; SEATS],
                burns: [0; SEATS],
                undealt: 0,
            },
            hand: HandState::new(s.dealer),
        };
        m.emit_public(Event::MatchStart {
            first_dealer: s.dealer,
        });
        m.start_hand();
        Ok(m)
    }

    /// The public state at the start of the current hand (the scores before its first trick).
    pub fn current_hand_start(&self) -> HandStartState {
        current_hand_start_of(&self.view(0))
    }
}

/// The public state at the start of the current hand, from any seat's view: the scores, bonus
/// flags and games are read back from the hand's public events.
pub fn current_hand_start_of(view: &crate::view::SeatView) -> HandStartState {
    let p = &view.public;
    let mut scores = p.scores;
    let mut bonus = p.bonus_taken;
    // Undo the hand's tricks and bonuses, newest first.
    let start = view
        .events
        .iter()
        .rposition(|e| matches!(e, Event::HandStart { .. }))
        .unwrap_or(0);
    let mut games = p.games_won;
    let mut game_index = p.game_index;
    // Partnerships turned this hand (rule 41): their score is their hand-start score from the
    // turn on, so their earlier points this hand are not undone (they were taken away).
    let mut fixed = [false; TEAMS];
    for e in view.events[start..].iter().rev() {
        match *e {
            Event::TrickWon { team, points, .. } | Event::TrickPassed { team, points, .. } => {
                if !fixed[usize::from(team)] {
                    scores[usize::from(team)] -= points;
                }
            }
            Event::Bonus { team, points, .. } => {
                if !fixed[usize::from(team)] {
                    scores[usize::from(team)] -= points;
                    if points == super::BONUS_FIRST {
                        bonus[usize::from(team)] = false;
                    }
                }
            }
            Event::PairTurned { team, score, .. } => {
                let t = usize::from(team);
                scores[t] = score;
                bonus[t] = score % 5 == 1;
                fixed[t] = true;
            }
            Event::GameWon {
                team, scores: s, ..
            } => {
                // The hand ended the game: the view already shows the next game (or the end).
                // Rebuild from the scores the game ended with.
                games[usize::from(team)] -= 1;
                game_index = games.iter().sum();
                scores = s;
                bonus = core::array::from_fn(|t| s[t] % 5 == 1);
            }
            _ => {}
        }
    }
    let (hand_index, dealer) = match view.events[start] {
        Event::HandStart {
            hand_index, dealer, ..
        } => (hand_index, dealer),
        _ => (p.hand_index, p.dealer),
    };
    HandStartState {
        dealer,
        scores,
        bonus_taken: bonus,
        games_won: games,
        hand_index,
        game_index,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Action;

    /// The hand-start state of the current hand, from the engine's accessors at its start.
    fn state_now(m: &Match) -> HandStartState {
        HandStartState {
            dealer: m.dealer(),
            scores: m.scores(),
            bonus_taken: m.bonus_taken(),
            games_won: m.games_won(),
            hand_index: m.hand_index(),
            game_index: m.game_index(),
        }
    }

    fn tail_from_hand_start(events: &[Event]) -> Vec<Event> {
        let i = events
            .iter()
            .rposition(|e| matches!(e, Event::HandStart { .. }))
            .expect("a hand started");
        events[i..].to_vec()
    }

    fn pick(m: &Match, i: u64, seed: u64) -> Action {
        let acts = m.legal_actions();
        acts[((i * 7 + seed) % acts.len() as u64) as usize]
    }

    #[test]
    fn from_hand_start_equals_played_match() {
        for seed in 0..300u64 {
            let mut m = Match::new(seed, Rules::HOUSE);
            let target = (seed % 9) as u32;
            let mut i = 0u64;
            while m.hand_index() < target && m.phase() != Phase::MatchOver {
                m.apply(pick(&m, i, seed)).expect("legal");
                i += 1;
            }
            if m.phase() == Phase::MatchOver {
                continue;
            }
            let s = state_now(&m);
            assert_eq!(m.current_hand_start(), s);
            let deck = m.truth().deck;
            let mut r = Match::from_hand_start(s, deck, seed ^ 99, Rules::HOUSE).expect("valid");
            let h = m.hand_index();
            while m.hand_index() == h && m.phase() != Phase::MatchOver {
                assert_eq!(m.current_hand_start(), s, "seed {seed}");
                for seat in 0..8 {
                    let (a, b) = (m.view(seat), r.view(seat));
                    assert_eq!(
                        tail_from_hand_start(&a.events),
                        tail_from_hand_start(&b.events)
                    );
                    assert_eq!(a.public, b.public);
                    assert_eq!(a.hand, b.hand);
                    assert_eq!(a.own_burns, b.own_burns);
                }
                let a = pick(&m, i, seed);
                i += 1;
                m.apply(a).expect("legal");
                r.apply(a).expect("legal");
            }
            assert_eq!(m.scores(), r.scores());
            assert_eq!(m.games_won(), r.games_won());
            assert_eq!(m.money(), r.money());
        }
    }

    #[test]
    fn hand_start_recovered_after_game_end() {
        // Play matches to the end and check the recovered start of the final hand.
        for seed in 0..40u64 {
            let mut m = Match::new(seed, Rules::HOUSE);
            let mut last = state_now(&m);
            let mut i = 0;
            while m.phase() != Phase::MatchOver {
                let h = m.hand_index();
                m.apply(pick(&m, i, seed)).expect("legal");
                i += 1;
                if m.phase() != Phase::MatchOver && m.hand_index() != h {
                    last = state_now(&m);
                }
            }
            assert_eq!(m.current_hand_start(), last);
        }
    }

    #[test]
    fn invalid_states_refused() {
        let ok = HandStartState {
            dealer: 3,
            scores: [10, 16, 0, 0],
            bonus_taken: [false, true, false, false],
            games_won: [1, 0, 0, 0],
            hand_index: 9,
            game_index: 1,
        };
        assert!(ok.validate().is_ok());
        let deck = identity_deck();
        assert!(Match::from_hand_start(ok, deck, 1, Rules::HOUSE).is_ok());
        let mut bad = ok;
        bad.scores[0] = 41;
        assert!(bad.validate().is_err());
        let mut bad = ok;
        bad.bonus_taken[1] = false;
        assert!(bad.validate().is_err());
        let mut bad = ok;
        bad.dealer = 8;
        assert!(bad.validate().is_err());
        let mut bad = ok;
        bad.games_won = [1, 1, 1, 0];
        bad.game_index = 2;
        assert!(bad.validate().is_err());
        let mut bad = ok;
        bad.scores[2] = 7;
        assert!(bad.validate().is_err());
        let mut d2 = deck;
        d2[0] = 1;
        assert!(Match::from_hand_start(ok, d2, 1, Rules::HOUSE).is_err());
    }
}
