//! Scoring inside one hand, trick by trick, in the engine's exact order (RULES 27–33 and the
//! "End checks" notes; engine `finish_trick` and `end_of_hand`):
//!
//! - after each trick: the trick's 5 points → if the 5 of trumps is in it, its best-trump bonus at
//!   once (rule 29) → a check for 41;
//! - after trick 5, in addition: the jink check → if the 5 of trumps was never played, the
//!   end-of-hand bonus to the team holding the best trump (rule 30) → a check for 41.
//!
//! A team's first bonus in a game is `bonus_first` (6) and every later one `bonus_later` (5).
//! Only the scoring team's score changes at each step, so only that team can reach 41.
//!
//! The sweep, the Monte-Carlo simulator and the engine-parity test all use these functions, so
//! the table scores hands exactly as the engine does.

use engine::game::{GameEndReason, Rules, TRICK_POINTS, TRICKS_PER_HAND, WINNING_SCORE};

use crate::state::has_first_bonus;

/// The bonus sizes the table is built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scoring {
    /// A team's first best-trump bonus in a game.
    pub bonus_first: u8,
    /// Each later one in the same game.
    pub bonus_later: u8,
}

impl Scoring {
    /// The house rules (6 then 5): the only setting where the bonus flag is `score mod 5 == 1`.
    pub const HOUSE: Scoring = Scoring {
        bonus_first: 6,
        bonus_later: 5,
    };

    /// The scoring of these rules, if the table can be built for them. Only the house rule
    /// (`bonus_first = 6`, `bonus_later = 5`) is supported: under any other setting the flags are
    /// not implied by the scores and the table would need explicit flags (RULES, `bonus_later`;
    /// not built at M1).
    pub fn for_rules(rules: &Rules) -> Result<Scoring, String> {
        if rules.bonus_first == 6 && rules.bonus_later == 5 {
            Ok(Scoring::HOUSE)
        } else {
            Err(format!(
                "the match-equity table is built for the house rule (bonus 6 then 5); \
                 bonus {} then {} needs explicit flags (not built at M1)",
                rules.bonus_first, rules.bonus_later
            ))
        }
    }
}

/// The scoring state inside a hand, in one team order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InHand {
    /// Each team's score now.
    pub scores: [u8; 4],
    /// Whether each team has taken a best-trump bonus in this game (kept explicitly and checked
    /// against `score mod 5 == 1`).
    pub flags: [bool; 4],
    /// Tricks each team has won this hand.
    pub tricks: [u8; 4],
    /// Whether the 5 of trumps has been played this hand.
    pub five_played: bool,
    /// Tricks completed this hand (0..=5).
    pub tricks_done: u8,
}

/// What a trick leads to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AfterTrick {
    /// The game is over: this team won it, for this reason.
    GameWon(u8, GameEndReason),
    /// The hand goes on to the next trick.
    Next(InHand),
    /// All five tricks are played and nobody has won the game yet: the end-of-hand bonus is next.
    TricksDone(InHand),
}

/// What the end of a hand leads to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AfterHand {
    /// The end-of-hand bonus won the game for this team.
    GameWon(u8, GameEndReason),
    /// The hand is over with these scores; the game goes on.
    HandOver([u8; 4]),
}

impl InHand {
    /// The start of a hand from the scores between hands (flags from `score mod 5 == 1`).
    pub fn start(scores: [u8; 4]) -> Self {
        InHand {
            scores,
            flags: scores.map(has_first_bonus),
            tricks: [0; 4],
            five_played: false,
            tricks_done: 0,
        }
    }

    /// The start of a hand with explicitly given flags (for checking the engine's own flags).
    pub fn start_with_flags(scores: [u8; 4], flags: [bool; 4]) -> Self {
        InHand {
            scores,
            flags,
            tricks: [0; 4],
            five_played: false,
            tricks_done: 0,
        }
    }

    /// Does every explicit flag agree with `score mod 5 == 1`? (the house-rule property)
    pub fn flags_match_scores(&self) -> bool {
        (0..4).all(|t| self.flags[t] == has_first_bonus(self.scores[t]))
    }

    /// The trick is won by `winner`; `five_here` if the 5 of trumps is in it.
    ///
    /// # Panics
    /// In debug builds, if five tricks are already done, or the 5 is played twice.
    pub fn play_trick(&self, winner: u8, five_here: bool, sc: Scoring) -> AfterTrick {
        debug_assert!(self.tricks_done < TRICKS_PER_HAND);
        debug_assert!(!(five_here && self.five_played), "only one 5 of trumps");
        let w = usize::from(winner);
        let mut s = *self;
        s.scores[w] += TRICK_POINTS;
        s.tricks[w] += 1;
        s.tricks_done += 1;
        let before_bonus = s.scores[w];
        if five_here {
            s.scores[w] += if s.flags[w] {
                sc.bonus_later
            } else {
                sc.bonus_first
            };
            s.flags[w] = true;
            s.five_played = true;
        }
        if s.scores[w] >= WINNING_SCORE {
            let reason = if five_here && before_bonus < WINNING_SCORE {
                GameEndReason::FortyOneByImmediateBonus
            } else {
                GameEndReason::FortyOneByTrick
            };
            return AfterTrick::GameWon(winner, reason);
        }
        if s.tricks_done < TRICKS_PER_HAND {
            return AfterTrick::Next(s);
        }
        if s.tricks[w] == TRICKS_PER_HAND {
            return AfterTrick::GameWon(winner, GameEndReason::Jink);
        }
        AfterTrick::TricksDone(s)
    }

    /// After trick 5 (no jink, nobody at 41): the end-of-hand bonus goes to `best_trump_team`
    /// if the 5 of trumps was never played (none if no trump was played, rule 31), then the check
    /// for 41.
    pub fn finish(&self, best_trump_team: Option<u8>, sc: Scoring) -> AfterHand {
        debug_assert_eq!(self.tricks_done, TRICKS_PER_HAND);
        let mut scores = self.scores;
        if !self.five_played
            && let Some(b) = best_trump_team
        {
            let bi = usize::from(b);
            scores[bi] += if self.flags[bi] {
                sc.bonus_later
            } else {
                sc.bonus_first
            };
            if scores[bi] >= WINNING_SCORE {
                return AfterHand::GameWon(b, GameEndReason::FortyOneByEndBonus);
            }
        }
        AfterHand::HandOver(scores)
    }

    /// The flags after the end-of-hand bonus to `team` (for property checks).
    pub fn flags_after_bonus(&self, team: Option<u8>) -> [bool; 4] {
        let mut f = self.flags;
        if !self.five_played
            && let Some(b) = team
        {
            f[usize::from(b)] = true;
        }
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SC: Scoring = Scoring::HOUSE;

    fn run(start: [u8; 4], winners: &[u8], five: Option<u8>, best: Option<u8>) -> AfterHand {
        let mut pos = InHand::start(start);
        for (i, &w) in winners.iter().enumerate() {
            match pos.play_trick(w, five == Some((i + 1) as u8), SC) {
                AfterTrick::GameWon(t, r) => return AfterHand::GameWon(t, r),
                AfterTrick::Next(p) | AfterTrick::TricksDone(p) => pos = p,
            }
        }
        pos.finish(best, SC)
    }

    #[test]
    fn plain_hand_with_end_bonus() {
        // A first bonus is 6, a later one 5 (house rule).
        assert_eq!(
            run([0; 4], &[0, 1, 2, 3, 0], None, Some(1)),
            AfterHand::HandOver([10, 11, 5, 5])
        );
        assert_eq!(
            run([0, 11, 0, 0], &[0, 1, 2, 3, 0], None, Some(1)),
            AfterHand::HandOver([10, 21, 5, 5])
        );
    }

    #[test]
    fn immediate_bonus_and_its_41() {
        // 30 + 5 + 6 = 41 at once with the 5 of trumps (rule 29).
        assert_eq!(
            run([30, 0, 0, 0], &[1, 0], Some(2), None),
            AfterHand::GameWon(0, GameEndReason::FortyOneByImmediateBonus)
        );
        // 36 + 5 = 41 with the trick itself, even though the 5 is in it.
        assert_eq!(
            run([36, 0, 0, 0], &[0], Some(1), None),
            AfterHand::GameWon(0, GameEndReason::FortyOneByTrick)
        );
        // With the 5 played there is no end-of-hand bonus.
        assert_eq!(
            run([0; 4], &[0, 1, 2, 3, 3], Some(4), Some(0)),
            AfterHand::HandOver([5, 5, 5, 16])
        );
    }

    #[test]
    fn end_checks_after_trick_five() {
        // Trick points first: a 41 by trick 5 wins before the pending end bonus for another team.
        assert_eq!(
            run([40, 35, 0, 0], &[2, 2, 3, 3, 0], None, Some(1)),
            AfterHand::GameWon(0, GameEndReason::FortyOneByTrick)
        );
        // The end bonus can win the game.
        assert_eq!(
            run([0, 30, 0, 0], &[0, 1, 3, 0, 2], None, Some(1)),
            AfterHand::GameWon(1, GameEndReason::FortyOneByEndBonus)
        );
        // A jink wins whatever the score, before the end bonus.
        assert_eq!(
            run([0; 4], &[3, 3, 3, 3, 3], None, Some(3)),
            AfterHand::GameWon(3, GameEndReason::Jink)
        );
        // Jink and 41 in one hand count once (the 41 is found first, after the fifth trick).
        assert_eq!(
            run([20, 0, 0, 0], &[0, 0, 0, 0, 0], None, Some(0)),
            AfterHand::GameWon(0, GameEndReason::FortyOneByTrick)
        );
    }

    #[test]
    fn scores_of_45_46_47() {
        // 40 + a plain trick = 45.
        let p = InHand::start([40, 0, 0, 0]);
        assert_eq!(
            p.play_trick(0, false, SC),
            AfterTrick::GameWon(0, GameEndReason::FortyOneByTrick)
        );
        // 36 (bonus already taken) + the 5's trick = 36 + 5 + 5 = 46 (47 under the flat rule).
        let p = InHand::start([36, 0, 0, 0]);
        match p.play_trick(0, true, SC) {
            AfterTrick::GameWon(0, _) => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn flags_follow_scores_under_house_rule() {
        let p = InHand::start([10, 11, 0, 35]);
        assert_eq!(p.flags, [false, true, false, false]);
        let AfterTrick::Next(q) = p.play_trick(0, true, SC) else {
            panic!("game should go on")
        };
        assert_eq!(q.scores[0], 21);
        assert!(q.flags_match_scores());
        assert!(Scoring::for_rules(&Rules::HOUSE).is_ok());
        assert!(Scoring::for_rules(&Rules::FLAT).is_err());
    }
}
