//! Rule-branch counters for a differential run.
//!
//! Each finished match is classified from the engine's event log, which the driver has already
//! checked against the reference's after every action. All arithmetic is in whole numbers.

use engine::cards::{ACE, Card, JOKER, rank_of, suit_of};
use engine::game::{BonusKind, Event, GameEndReason, Match as EMatch};
use engine::rules::{ACE_HEARTS, is_trump};

/// Counts for a differential run. Branch counts are per hand unless noted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// Matches played to the money.
    pub matches: u64,
    /// Games finished.
    pub games: u64,
    /// Hands dealt.
    pub hands: u64,
    /// Actions applied (burns and plays).
    pub actions: u64,
    /// The first turned card was the Joker (step 2).
    pub turned_joker: u64,
    /// The Joker, then an ace as the second turned card (step 2c).
    pub joker_then_ace: u64,
    /// The Joker, then a non-ace whose ace of trumps the dealer holds; the dealer robs at its
    /// own turn in trick 1 (H4 as amended on 2026-10-14).
    pub joker_dealer_holds_ace: u64,
    /// The card that fixed trumps (the turned card, or the second after a Joker) was the A♥.
    pub turned_ace_hearts: u64,
    /// The card that fixed trumps was a 5 (so it was the 5 of trumps).
    pub turned_five_of_trumps: u64,
    /// The card that fixed trumps was an ace other than the A♥.
    pub turned_plain_ace: u64,
    /// The dealer took a turned ace before trick 1 (steps 2c, 3).
    pub dealer_robbed_at_once: u64,
    /// The dealer held the ace of trumps under an ordinary turned card and robbed at their turn.
    pub dealer_robbed_in_turn: u64,
    /// A seat other than the dealer robbed.
    pub non_dealer_robbed: u64,
    /// Nobody robbed (the ace of trumps was undealt).
    pub nobody_robbed: u64,
    /// Games won by a jink (per game).
    pub jink: u64,
    /// Games that ended before the fifth trick of their last hand (per game).
    pub game_ended_mid_hand: u64,
    /// Hands with an immediate best-trump bonus (the 5 of trumps played).
    pub immediate_bonus: u64,
    /// Hands with an end-of-hand best-trump bonus.
    pub end_of_hand_bonus: u64,
    /// Hands that completed five tricks without a trump played (impossible with 8 players).
    pub no_trump_hand: u64,
    /// Matches that needed five games (per match).
    pub five_game_match: u64,
    /// Games won by reaching 41 on a trick's points (per game).
    pub won_by_trick: u64,
    /// Games won by reaching 41 with an immediate bonus (per game).
    pub won_by_immediate_bonus: u64,
    /// Games won by reaching 41 with an end-of-hand bonus (per game).
    pub won_by_end_bonus: u64,
    /// Burns of an ace (a plain ace, or a future ace of trumps burned before trumps were known).
    pub ace_burned: u64,
    /// Matches won by each team.
    pub match_wins: [u64; 4],
    /// Money won (positive) or lost by each team over all matches.
    pub money: [i64; 4],
}

/// What a branch count is a fraction of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Per {
    /// Hands dealt.
    Hand,
    /// Games finished.
    Game,
    /// Matches played.
    Match,
}

/// One printed line of the branch table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Line {
    /// Plain-English name of the branch.
    pub label: &'static str,
    /// How often it happened.
    pub count: u64,
    /// What the count is a fraction of.
    pub per: Per,
    /// The size of that base (hands, games or matches).
    pub out_of: u64,
    /// For gated branches: the minimum rate, in tenths of a percent, for runs of 100,000
    /// matches or more.
    pub minimum_per_mille: Option<u64>,
    /// The branch is impossible and must never be counted.
    pub must_be_zero: bool,
}

/// Runs of at least this many matches must meet the rare-branch minimums.
pub const COVERAGE_MIN_MATCHES: u64 = 100_000;

impl Report {
    /// Adds another report's counts to this one.
    pub fn add(&mut self, o: &Report) {
        self.matches += o.matches;
        self.games += o.games;
        self.hands += o.hands;
        self.actions += o.actions;
        self.turned_joker += o.turned_joker;
        self.joker_then_ace += o.joker_then_ace;
        self.joker_dealer_holds_ace += o.joker_dealer_holds_ace;
        self.turned_ace_hearts += o.turned_ace_hearts;
        self.turned_five_of_trumps += o.turned_five_of_trumps;
        self.turned_plain_ace += o.turned_plain_ace;
        self.dealer_robbed_at_once += o.dealer_robbed_at_once;
        self.dealer_robbed_in_turn += o.dealer_robbed_in_turn;
        self.non_dealer_robbed += o.non_dealer_robbed;
        self.nobody_robbed += o.nobody_robbed;
        self.jink += o.jink;
        self.game_ended_mid_hand += o.game_ended_mid_hand;
        self.immediate_bonus += o.immediate_bonus;
        self.end_of_hand_bonus += o.end_of_hand_bonus;
        self.no_trump_hand += o.no_trump_hand;
        self.ace_burned += o.ace_burned;
        self.five_game_match += o.five_game_match;
        self.won_by_trick += o.won_by_trick;
        self.won_by_immediate_bonus += o.won_by_immediate_bonus;
        self.won_by_end_bonus += o.won_by_end_bonus;
        for t in 0..4 {
            self.match_wins[t] += o.match_wins[t];
            self.money[t] += o.money[t];
        }
    }

    /// Counts one finished match from its event log.
    pub fn count_match(&mut self, m: &EMatch) {
        self.matches += 1;
        self.actions += m.actions().len() as u64;
        let mut hand = HandTally::default();
        let mut games_this_match = 0u32;
        for (_, e) in m.events() {
            match *e {
                Event::HandStart { dealer, .. } => {
                    self.hands += 1;
                    hand = HandTally {
                        dealer,
                        ..HandTally::default()
                    };
                }
                Event::Turned { card } => hand.turned.push(card),
                Event::Took { seat, card } => {
                    if card != JOKER {
                        hand.robber = Some((seat, hand.plays == 0));
                    }
                }
                Event::Played { card, .. } => {
                    hand.plays += 1;
                    if let Some(trump) = hand.trump_card().and_then(suit_of)
                        && is_trump(card, trump)
                    {
                        hand.trump_played = true;
                    }
                }
                Event::TrickWon { .. } => hand.tricks += 1,
                Event::Bonus { kind, .. } => match kind {
                    BonusKind::Immediate => self.immediate_bonus += 1,
                    BonusKind::EndOfHand => self.end_of_hand_bonus += 1,
                },
                Event::GameWon { reason, .. } => {
                    self.games += 1;
                    games_this_match += 1;
                    match reason {
                        GameEndReason::FortyOneByTrick => self.won_by_trick += 1,
                        GameEndReason::FortyOneByImmediateBonus => {
                            self.won_by_immediate_bonus += 1;
                        }
                        GameEndReason::FortyOneByEndBonus => self.won_by_end_bonus += 1,
                        GameEndReason::Jink => self.jink += 1,
                    }
                }
                Event::HandEnd { aborted, .. } => {
                    if aborted {
                        self.game_ended_mid_hand += 1;
                    }
                    hand.finish(self);
                }
                Event::MatchWon { team, money } => {
                    let t = usize::from(team);
                    self.match_wins[t] += 1;
                    for (total, &m) in self.money.iter_mut().zip(&money) {
                        *total += i64::from(m);
                    }
                    if games_this_match == 5 {
                        self.five_game_match += 1;
                    }
                }
                Event::Burned { card, .. } => {
                    if rank_of(card) == ACE {
                        self.ace_burned += 1;
                    }
                }
                Event::MatchStart { .. }
                | Event::Dealt { .. }
                | Event::BurnDone { .. }
                | Event::ProvedNoAce { .. }
                | Event::AceUndealt => {}
            }
        }
    }

    /// The branch table, in print order.
    pub fn lines(&self) -> Vec<Line> {
        let (h, g, m) = (self.hands, self.games, self.matches);
        let line = |label, count, per, out_of, minimum_per_mille| Line {
            label,
            count,
            per,
            out_of,
            minimum_per_mille,
            must_be_zero: false,
        };
        vec![
            line("turned Joker", self.turned_joker, Per::Hand, h, Some(15)),
            line(
                "Joker then turned ace",
                self.joker_then_ace,
                Per::Hand,
                h,
                Some(1),
            ),
            line(
                "Joker, dealer holds the new ace",
                self.joker_dealer_holds_ace,
                Per::Hand,
                h,
                Some(1),
            ),
            line("turned A♥", self.turned_ace_hearts, Per::Hand, h, Some(15)),
            line(
                "turned 5 of trumps",
                self.turned_five_of_trumps,
                Per::Hand,
                h,
                Some(60),
            ),
            line(
                "turned plain ace",
                self.turned_plain_ace,
                Per::Hand,
                h,
                None,
            ),
            line(
                "dealer robbed at once",
                self.dealer_robbed_at_once,
                Per::Hand,
                h,
                None,
            ),
            line(
                "dealer robbed at own turn",
                self.dealer_robbed_in_turn,
                Per::Hand,
                h,
                None,
            ),
            line(
                "non-dealer robbed",
                self.non_dealer_robbed,
                Per::Hand,
                h,
                None,
            ),
            line("nobody robbed", self.nobody_robbed, Per::Hand, h, None),
            line("immediate bonus", self.immediate_bonus, Per::Hand, h, None),
            line(
                "end-of-hand bonus",
                self.end_of_hand_bonus,
                Per::Hand,
                h,
                None,
            ),
            Line {
                must_be_zero: true,
                ..line("no-trump hand", self.no_trump_hand, Per::Hand, h, None)
            },
            line("jink", self.jink, Per::Game, g, None),
            line("41 on a trick", self.won_by_trick, Per::Game, g, None),
            line(
                "41 by immediate bonus",
                self.won_by_immediate_bonus,
                Per::Game,
                g,
                None,
            ),
            line(
                "41 by end-of-hand bonus",
                self.won_by_end_bonus,
                Per::Game,
                g,
                None,
            ),
            line(
                "game ended mid-hand",
                self.game_ended_mid_hand,
                Per::Game,
                g,
                None,
            ),
            line("five-game match", self.five_game_match, Per::Match, m, None),
            line("ace burned", self.ace_burned, Per::Hand, h, None),
        ]
    }

    /// Coverage problems, in plain English: a branch that must never happen did, or (when
    /// `check_minimums`) a rare branch fell below its minimum rate.
    pub fn coverage_problems(&self, check_minimums: bool) -> Vec<String> {
        let mut out = Vec::new();
        for l in self.lines() {
            if l.must_be_zero && l.count != 0 {
                out.push(format!(
                    "{}: counted {} times but is impossible",
                    l.label, l.count
                ));
            }
            if let (true, Some(min)) = (check_minimums, l.minimum_per_mille)
                && u128::from(l.count) * 1000 < u128::from(l.out_of) * u128::from(min)
            {
                out.push(format!(
                    "{}: {} of {} is below the minimum of {}.{}%",
                    l.label,
                    l.count,
                    l.out_of,
                    min / 10,
                    min % 10
                ));
            }
        }
        out
    }
}

/// What one hand did, read from its events.
#[derive(Default)]
struct HandTally {
    dealer: u8,
    turned: Vec<Card>,
    /// The seat that robbed, and whether it did so before the first card was played.
    robber: Option<(u8, bool)>,
    plays: u32,
    tricks: u32,
    trump_played: bool,
}

impl HandTally {
    /// The turned card that fixed trumps: the first, or the second after a Joker.
    fn trump_card(&self) -> Option<Card> {
        match self.turned.first() {
            Some(&JOKER) => self.turned.get(1).copied(),
            first => first.copied(),
        }
    }

    fn finish(&self, r: &mut Report) {
        let joker = self.turned.first() == Some(&JOKER);
        let dealer_at_once = matches!(self.robber, Some((s, true)) if s == self.dealer);
        if joker {
            r.turned_joker += 1;
            match self.turned.get(1) {
                Some(&c) if rank_of(c) == ACE => r.joker_then_ace += 1,
                Some(_) if matches!(self.robber, Some((s, _)) if s == self.dealer) => {
                    r.joker_dealer_holds_ace += 1
                }
                _ => {}
            }
        }
        if let Some(c) = self.trump_card() {
            if c == ACE_HEARTS {
                r.turned_ace_hearts += 1;
            } else if rank_of(c) == ACE {
                r.turned_plain_ace += 1;
            }
            if rank_of(c) == 5 {
                r.turned_five_of_trumps += 1;
            }
        }
        match self.robber {
            None => r.nobody_robbed += 1,
            Some((s, _)) if s != self.dealer => r.non_dealer_robbed += 1,
            Some(_) if dealer_at_once => r.dealer_robbed_at_once += 1,
            Some(_) => r.dealer_robbed_in_turn += 1,
        }
        if self.tricks == 5 && !self.trump_played {
            r.no_trump_hand += 1;
        }
    }
}
