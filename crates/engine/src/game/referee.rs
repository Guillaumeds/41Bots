//! Turned partnerships and the app's referee (rules 32–34 and 37–41; the owner's decisions of
//! 2026-10-17 in docs/APP-ENGINE-CONTRACT.md §7).
//!
//! The engine's own players never act illegally (house rule H1): [`Match::legal_actions`] offers
//! only legal actions and nothing here is reachable from them. The app's Real Life mode lets a
//! player renege, and the engine must referee the rest of such a hand exactly as the app does
//! (fivefingers `src/engine/game.ts`: `penalise`, `credit`, `catchHiddenAces`, `resolveTrick`,
//! `advance`, `finishHand`). This module gives the app three ways in, kept apart from the bots'
//! actions and recorded in the match's record ([`Match::extras`]):
//!
//! - [`Match::turn_team`]: the app turned a partnership before a play, or while a rob is due
//!   (the app's `declineRob`, `leaveJoker`, and Guide's play without robbing: the automatic take
//!   is undone, as the take never happened at the app's table);
//! - [`Match::play_then_turn`]: a card is placed, the partnership is turned, then the trick goes on
//!   (the app catches a pair right after the card that proves it, before the trick is resolved);
//! - [`Match::let_through`]: what the app's referee lets through ([`Lapse`]): an illegal card, a
//!   play without robbing, a rob without the ace, a burned ace. Each is kept as a hidden
//!   [`Renege`], and the engine turns the pair exactly when the app would: the moment a later card
//!   of that seat proves it (rule 37), when the ace of trumps shows from another hand (a rob
//!   without the ace), when the pair would claim the game (rule 38: it is turned instead, the
//!   points not credited), or at the end of the hand (a rob without the ace, a burned ace).
//!
//! Once a partnership is turned (`turn_pair`, the app's `penalise`): its score and bonus flag go
//! back to their hand-start values (rule 41); each trick it holds passes, in trick order, to the
//! partnership still in play that played the strongest card in it, judged against the card led,
//! and when none of their cards is a trump or of the suit led, to the one that played the first
//! of them (rule 39, owner 2026-10-17: "first card played takes it", as the app does); each pass
//! is credited at once, so the first partnership to reach 41 wins the game and later passes are
//! not credited (rule 32, owner 2026-10-17: "same as the app"); a trick in which no partnership
//! still in play played stays with nobody, for no points. A turned seat plays no more cards
//! (rule 40); its cards in the trick in progress stay there but do not count. With one
//! partnership left it plays the hand out alone and takes every trick, which is a jink at the
//! end of the hand (rule 34, as the app plays it); with none left the hand ends and nobody scores
//! it. A hand with a turned pair or a let-through ends as the app's `finishHand` does: hidden
//! false robs and burned aces are caught, then the best-trump bonus, then the jink.
//!
//! One case stays open and is refused: a partnership turned while the trick it led is still in
//! progress (owner 2026-10-17: "ask the pros first"). The app keeps the turned card as the card
//! led; the engine waits for the pub's ruling ([`TurnRefusal::LedTrickInProgress`]).

use super::{
    BONUS_FIRST, BONUS_LATER, BonusKind, Event, GameEndReason, IllegalAction, Match, Phase, Rules,
    TRICK_POINTS, TRICKS_PER_HAND, WINNING_SCORE, left_of,
};
use crate::cards::{Card, JOKER, Suit, bit, has};
use crate::game::{Action, AfterBurn};
use crate::rules::{
    EffSuit, ace_of_trumps, effective_suit, is_top_trump, legal_plays, plain_rank, trump_rank,
};
use crate::{TEAMS, team_of};

/// The holder of a trick nobody still in play played in (rule 40).
pub(super) const NOBODY: u8 = TEAMS as u8;

/// Whether the hand goes on after a scoring step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Flow {
    /// It goes on.
    Go,
    /// The game ended, or the hand did (the next hand is dealt): nothing more of this hand.
    HandOver,
}

/// Why and when a partnership was turned ([`Event::PairTurned`]).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnCause {
    /// The app turned it before a play ([`Match::turn_team`]).
    BeforePlay,
    /// The app turned it right after a card, before the trick went on ([`Match::play_then_turn`]).
    WithPlay,
    /// A card of its own proved a renege it had made (rule 37).
    Proved,
    /// The ace of trumps showed from another hand: it had robbed without the ace (rule 38).
    AceShown,
    /// It would have reached 41 with a hidden rob without the ace or a burned ace (rule 38).
    Claim,
    /// The hand ended with a rob without the ace or a burned ace of its never shown (rule 38).
    HandEnd,
    /// It did not rob at its turn: the app turned it before its card (Guide).
    NoRob,
    /// The dealer refused the turned ace (rule 26, the app's `declineRob`).
    DeclinedRob,
    /// The dealer left the turned Joker on the deck (rule 24, the app's `leaveJoker`).
    LeftJoker,
}

impl TurnCause {
    /// The word used in canonical event strings.
    pub const fn canonical(self) -> &'static str {
        match self {
            TurnCause::BeforePlay => "before_play",
            TurnCause::WithPlay => "with_play",
            TurnCause::Proved => "proved",
            TurnCause::AceShown => "ace_shown",
            TurnCause::Claim => "claim",
            TurnCause::HandEnd => "hand_end",
            TurnCause::NoRob => "no_rob",
            TurnCause::DeclinedRob => "declined_rob",
            TurnCause::LeftJoker => "left_joker",
        }
    }
}

/// What a hidden renege was (the app's `Renege.kind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenegeKind {
    /// A plain suit was led and the seat, holding it, played another plain card (rule 14). A
    /// later card of that suit proves it.
    Follow(Suit),
    /// This trump was led and the seat, holding a trump it could not hold back, played a plain
    /// card (rules 16–17). A later trump it could not have held back against that card proves
    /// it.
    TrumpLead(Card),
    /// The seat held the ace of trumps and played without robbing (rules 20, 25). Playing the ace
    /// proves it.
    NoRob,
    /// The seat took the turned card without the ace of trumps (rule 38). Caught when the ace
    /// shows from another hand, when its pair would claim the game, or at the end of the hand.
    FalseRob,
    /// The robber burned the ace of trumps (rules 21, 24). Caught when its pair would claim the
    /// game, or at the end of the hand.
    BurnedAce,
}

impl RenegeKind {
    /// A short word for records and tests.
    pub fn canonical(self) -> String {
        match self {
            RenegeKind::Follow(s) => format!("follow:{}", s.letter()),
            RenegeKind::TrumpLead(c) => format!("trump_lead:{}", crate::cards::name(c)),
            RenegeKind::NoRob => "no_rob".to_string(),
            RenegeKind::FalseRob => "false_rob".to_string(),
            RenegeKind::BurnedAce => "burned_ace".to_string(),
        }
    }
}

/// A renege the app let through, hidden from the table until it is proved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Renege {
    /// The seat that reneged.
    pub seat: u8,
    /// What it did.
    pub kind: RenegeKind,
    /// Whether it has been proved (its pair turned, or already turned).
    pub caught: bool,
}

/// What the app's referee lets through (Real Life, a real table: nobody can see a hand), for
/// replaying app records only. Never offered to a bot ([`Match::legal_actions`] knows nothing of
/// it) and never searched.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lapse {
    /// The seat to play plays this card it holds although it is not a legal card.
    Play(Card),
    /// The seat whose rob is due (the engine has taken the turned card for it and waits for its
    /// burn) plays this card without robbing: the take is undone and the card played.
    PlayWithoutRob(Card),
    /// The seat to play in trick 1, without the ace of trumps, takes the turned card still on the
    /// table and burns this card (any card it then holds but the ace of trumps, H3), then plays.
    FalseRob(Card),
    /// The robber burns the ace of trumps.
    BurnAce,
}

/// Why [`Match::let_through`] refused. The match is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LapseRefusal {
    /// That cannot happen at this moment of the hand.
    NotNow,
    /// The card is not in the seat's hand (or, for a burn, is the ace of trumps).
    NotInHand,
    /// The card is a legal play: play it with [`Match::apply`].
    Legal,
    /// Open question: the card proves a renege of the partnership that led the trick still in
    /// progress (see [`TurnRefusal::LedTrickInProgress`]).
    LedTrickInProgress,
}

impl LapseRefusal {
    /// A short stable word for the refusal (compared with the reference implementation's).
    pub const fn code(self) -> &'static str {
        match self {
            LapseRefusal::NotNow => "not_now",
            LapseRefusal::NotInHand => "not_in_hand",
            LapseRefusal::Legal => "legal",
            LapseRefusal::LedTrickInProgress => "led_trick_in_progress",
        }
    }
}

/// A lapse [`Match::let_through`] refused, and why. The match is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LapseRefused {
    /// The refused lapse.
    pub lapse: Lapse,
    /// Why.
    pub why: LapseRefusal,
}

impl core::fmt::Display for LapseRefused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?} refused ({})", self.lapse, self.why.code())
    }
}

impl std::error::Error for LapseRefused {}

/// Why [`Match::turn_team`] or [`Match::play_then_turn`] refused. The match is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnRefusal {
    /// The team number is not 0..=3.
    NoSuchTeam,
    /// A partnership is turned before a play, or (its own) while its rob is due; not during
    /// another seat's burn, nor once the match is over.
    NotBeforeAPlay,
    /// The partnership is already turned this hand.
    AlreadyTurned,
    /// The card for [`Match::play_then_turn`] is not a legal play of the seat to act.
    IllegalCard,
    /// Open question (owner 2026-10-17, "ask the pros first"): the partnership would be turned
    /// while the trick it led is still in progress (not yet resolved, even when its last card is
    /// down). Rule 40 says the trick goes on without the turned cards, but not which suit the
    /// players still to play must follow, nor against which card the trick is won. (The app
    /// keeps the turned card as the card led.) Not refused when every partnership is then turned
    /// (nobody scores the hand) or when a passed trick ends the game first.
    LedTrickInProgress,
}

impl TurnRefusal {
    /// A short stable word for the refusal (compared with the reference implementation's).
    pub const fn code(self) -> &'static str {
        match self {
            TurnRefusal::NoSuchTeam => "no_such_team",
            TurnRefusal::NotBeforeAPlay => "not_before_a_play",
            TurnRefusal::AlreadyTurned => "already_turned",
            TurnRefusal::IllegalCard => "illegal_card",
            TurnRefusal::LedTrickInProgress => "led_trick_in_progress",
        }
    }

    /// Why, in plain English.
    pub const fn reason(self) -> &'static str {
        match self {
            TurnRefusal::NoSuchTeam => "partnerships are numbered 0 to 3",
            TurnRefusal::NotBeforeAPlay => {
                "a partnership is turned before a play, or while its own rob is due"
            }
            TurnRefusal::AlreadyTurned => "that partnership is already turned this hand",
            TurnRefusal::IllegalCard => "that card is not a legal play of the seat to act",
            TurnRefusal::LedTrickInProgress => {
                "open rules question (owner: ask the pros first): the partnership that led the \
                 trick still in progress would be turned"
            }
        }
    }
}

/// A turn [`Match::turn_team`] or [`Match::play_then_turn`] refused, and why. The match is
/// unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TurnRefused {
    /// The partnership that was to be turned.
    pub team: u8,
    /// Why it was refused.
    pub why: TurnRefusal,
}

impl core::fmt::Display for TurnRefused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "partnership {} cannot be turned now ({}): {}",
            self.team,
            self.why.code(),
            self.why.reason()
        )
    }
}

impl std::error::Error for TurnRefused {}

/// A step of a match's record that is not one of the bots' actions ([`Match::extras`]).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extra {
    /// [`Match::turn_team`].
    Turn(u8),
    /// [`Match::play_then_turn`].
    PlayThenTurn(Card, u8),
    /// [`Match::let_through`].
    Lapse(Lapse),
}

/// Why a record ([`Match::apply_record`]) could not be replayed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecordError {
    /// An action was refused.
    Action(IllegalAction),
    /// A turn was refused.
    Turn(TurnRefused),
    /// A lapse was refused.
    Lapse(LapseRefused),
    /// The extras are not in order, or one points past the end of the actions.
    BadExtraIndex,
}

impl core::fmt::Display for RecordError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RecordError::Action(e) => write!(f, "{e}"),
            RecordError::Turn(e) => write!(f, "{e}"),
            RecordError::Lapse(e) => write!(f, "{e}"),
            RecordError::BadExtraIndex => f.write_str(
                "the extras must be in order, each after at most as many actions as were recorded",
            ),
        }
    }
}

impl std::error::Error for RecordError {}

/// The index (in play order) of the strongest card of a trick among the seats `in_play`
/// accepts, judged against the card led (`cards[0]`, which may be a turned one): the highest
/// trump, else the highest card of the suit led; when none of their cards is a trump or of the
/// suit led, the first of their cards (rule 39 as the owner ruled on 2026-10-17, the app's
/// `standingWinnerIndex`). `None` when none of the seats played in it.
pub(super) fn strongest_in_play(
    cards: &[(u8, Card)],
    trump: Suit,
    in_play: impl Fn(u8) -> bool,
) -> Option<usize> {
    let led = effective_suit(cards.first()?.1, trump);
    let mut best: Option<(u8, usize)> = None;
    for (i, &(seat, c)) in cards.iter().enumerate() {
        if !in_play(seat) {
            continue;
        }
        let r = trump_rank(c, trump);
        let strength = if r > 0 {
            32 + r
        } else if effective_suit(c, trump) == led {
            plain_rank(c)
        } else {
            0
        };
        if best.is_none_or(|(b, _)| strength > b) {
            best = Some((strength, i));
        }
    }
    best.map(|(_, i)| i)
}

/// Does `card`, played by the reneging seat, prove the renege (the app's `provesRenege`)?
fn proves(kind: RenegeKind, card: Card, trump: Suit) -> bool {
    match kind {
        RenegeKind::Follow(s) => effective_suit(card, trump) == EffSuit::Plain(s),
        RenegeKind::TrumpLead(led) => {
            let r = trump_rank(card, trump);
            r > 0 && !(is_top_trump(card, trump) && r > trump_rank(led, trump))
        }
        RenegeKind::NoRob => card == ace_of_trumps(trump),
        RenegeKind::FalseRob | RenegeKind::BurnedAce => false,
    }
}

impl Match {
    // ----- the app's ways in -----

    /// Turn partnership `team` now (the app caught it; rules 37–38 are the app's). Allowed before
    /// the play the match awaits (between tricks or mid-trick), or, for the partnership of the
    /// seat whose rob is due, instead of that rob: the robber at its turn in trick 1 does not
    /// play (Guide's play without robbing), the dealer refuses a turned ace (`declineRob`: the
    /// ace stays on the pack, out of play), or the dealer leaves a turned Joker (`leaveJoker`: the
    /// Joker stays face up on the deck and the next card is turned; with the dealer's pair turned
    /// an ace turned next stays on the pack unrobbed). The rest of the hand follows the module
    /// notes. Refused, with the match unchanged, when nothing may be turned now, when the
    /// partnership is already turned, and in the open case of a partnership turned while the
    /// trick it led is still in progress.
    pub fn turn_team(&mut self, team: u8) -> Result<(), TurnRefused> {
        let refuse = |why| Err(TurnRefused { team, why });
        if usize::from(team) >= TEAMS {
            return refuse(TurnRefusal::NoSuchTeam);
        }
        match self.phase {
            Phase::AwaitPlay(_) if self.hand.trump.is_some() => {}
            Phase::AwaitBurn(s) if team_of(s) == team => {}
            _ => return refuse(TurnRefusal::NotBeforeAPlay),
        }
        if !self.hand.in_play(team) {
            return refuse(TurnRefusal::AlreadyTurned);
        }
        let mut m = self.clone();
        let at = m.actions.len() as u32;
        if let Err(why) = m.turn_now(team) {
            return refuse(why);
        }
        m.extras.push((at, Extra::Turn(team)));
        *self = m;
        Ok(())
    }

    /// The seat to act plays `card` (a legal play), then partnership `team` is turned at once,
    /// before the trick goes on or is resolved (the app turns a pair right after the card that
    /// proves its renege: a trick that card completes is then won among the others). Refused,
    /// with the match unchanged, as [`Match::turn_team`] is, and when the card is not legal.
    pub fn play_then_turn(&mut self, card: Card, team: u8) -> Result<(), TurnRefused> {
        let refuse = |why| Err(TurnRefused { team, why });
        if usize::from(team) >= TEAMS {
            return refuse(TurnRefusal::NoSuchTeam);
        }
        let Phase::AwaitPlay(seat) = self.phase else {
            return refuse(TurnRefusal::NotBeforeAPlay);
        };
        if self.hand.trump.is_none() {
            return refuse(TurnRefusal::NotBeforeAPlay);
        }
        if !self.hand.in_play(team) {
            return refuse(TurnRefusal::AlreadyTurned);
        }
        if usize::from(card) >= crate::DECK_SIZE || !has(self.legal_mask(), card) {
            return refuse(TurnRefusal::IllegalCard);
        }
        let mut m = self.clone();
        let at = m.actions.len() as u32;
        if let Err(why) = m.place_then_turn(seat, card, &[(team, TurnCause::WithPlay)]) {
            return refuse(why);
        }
        m.extras.push((at, Extra::PlayThenTurn(card, team)));
        *self = m;
        Ok(())
    }

    /// Something the app's referee let through ([`Lapse`]); for replaying app records only.
    /// Refused, with the match unchanged, when it cannot happen now, and in the open case where
    /// the card proves a renege of the partnership that led the trick still in progress.
    pub fn let_through(&mut self, lapse: Lapse) -> Result<(), LapseRefused> {
        let refuse = |why| Err(LapseRefused { lapse, why });
        let Some(trump) = self.hand.trump else {
            return refuse(LapseRefusal::NotNow);
        };
        let mut m = self.clone();
        let at = m.actions.len() as u32;
        match (lapse, self.phase) {
            (Lapse::Play(c), Phase::AwaitPlay(s)) => {
                if usize::from(c) >= crate::DECK_SIZE || !has(self.hand_of(s), c) {
                    return refuse(LapseRefusal::NotInHand);
                }
                if has(self.legal_mask(), c) {
                    return refuse(LapseRefusal::Legal);
                }
                m.record_illegal(s, c, trump);
                if m.play(s, c).is_err() {
                    return refuse(LapseRefusal::LedTrickInProgress);
                }
            }
            (Lapse::PlayWithoutRob(c), Phase::AwaitBurn(s))
                if self.hand.after_burn == AfterBurn::PlayFirst =>
            {
                let taken = self.hand.table_card_taken_by(s);
                if usize::from(c) >= crate::DECK_SIZE
                    || !has(self.hand_of(s), c)
                    || Some(c) == taken
                {
                    return refuse(LapseRefusal::NotInHand);
                }
                m.untake(s);
                m.hand.reneges.push(Renege {
                    seat: s,
                    kind: RenegeKind::NoRob,
                    caught: false,
                });
                if !has(m.legal_mask(), c) {
                    m.record_illegal(s, c, trump);
                }
                if m.play(s, c).is_err() {
                    return refuse(LapseRefusal::LedTrickInProgress);
                }
            }
            (Lapse::FalseRob(b), Phase::AwaitPlay(s)) => {
                let Some(card) = self.hand.table_card else {
                    return refuse(LapseRefusal::NotNow);
                };
                if self.hand.trick_no != 1
                    || self.hand.robber.is_some()
                    || has(self.hand_of(s), ace_of_trumps(trump))
                {
                    return refuse(LapseRefusal::NotNow);
                }
                // H3: any card but the ace of trumps, the card just taken included (the turned
                // card may be the ace of trumps itself, left on the table by a dealer who refused
                // it: it is never burned).
                if usize::from(b) >= crate::DECK_SIZE
                    || !has(self.hand_of(s) | bit(card), b)
                    || b == ace_of_trumps(trump)
                {
                    return refuse(LapseRefusal::NotInHand);
                }
                m.hand.reneges.push(Renege {
                    seat: s,
                    kind: RenegeKind::FalseRob,
                    caught: false,
                });
                m.take(s, card);
                m.hand.after_burn = AfterBurn::PlayFirst;
                m.burn(s, b);
            }
            (Lapse::BurnAce, Phase::AwaitBurn(s))
                if self.hand.after_burn == AfterBurn::PlayFirst
                    && has(self.hand_of(s), ace_of_trumps(trump)) =>
            {
                m.hand.reneges.push(Renege {
                    seat: s,
                    kind: RenegeKind::BurnedAce,
                    caught: false,
                });
                m.burn(s, ace_of_trumps(trump));
            }
            _ => return refuse(LapseRefusal::NotNow),
        }
        m.extras.push((at, Extra::Lapse(lapse)));
        *self = m;
        Ok(())
    }

    /// For rebuilding a world from a seat's view (the search, `search::worlds::rebuild`), never
    /// for a record: the seat to act plays `card` as the table saw it, with no check that it was
    /// legal and nothing recorded as a renege (a seat whose partnership the view shows turned may
    /// have reneged), its automatic rob undone first if one is waiting for its burn; then the
    /// partnerships in `turns` are turned with their causes before the trick goes on (a proof
    /// first, before the engine's own catches of reneges already recorded; the others after).
    pub fn observe_play(
        &mut self,
        card: Card,
        turns: &[(u8, TurnCause)],
    ) -> Result<(), TurnRefusal> {
        let seat = match self.phase {
            Phase::AwaitPlay(s) => s,
            Phase::AwaitBurn(s) if self.hand.after_burn == AfterBurn::PlayFirst => s,
            _ => return Err(TurnRefusal::NotBeforeAPlay),
        };
        if self.hand.trump.is_none()
            || usize::from(card) >= crate::DECK_SIZE
            || !has(self.hand_of(seat), card)
            || turns
                .iter()
                .any(|&(t, _)| usize::from(t) >= TEAMS || !self.hand.in_play(t))
        {
            return Err(TurnRefusal::IllegalCard);
        }
        let mut m = self.clone();
        if matches!(m.phase, Phase::AwaitBurn(_)) {
            if m.hand.table_card_taken_by(seat) == Some(card) {
                return Err(TurnRefusal::IllegalCard);
            }
            m.untake(seat);
        }
        // Listed with the actions (so `actions()` holds every card played), though a match built
        // this way is never a record: its actions need not replay.
        m.actions.push(Action::Play(card));
        m.place_then_turn(seat, card, turns)?;
        *self = m;
        Ok(())
    }

    // ----- the record -----

    /// Every step of the match's record that is not one of the bots' actions (turns, plays with
    /// a turn, lapses), with the number of actions applied before it. With [`Match::actions`] it
    /// is the match's record ([`Match::apply_record`]). Empty unless the app turned a pair.
    pub fn extras(&self) -> &[(u32, Extra)] {
        &self.extras
    }

    /// The hidden reneges of the current hand (for tests and the differential test; never shown
    /// to a seat).
    pub fn reneges(&self) -> &[Renege] {
        &self.hand.reneges
    }

    /// Partnerships turned in the current hand.
    pub fn turned_teams(&self) -> [bool; TEAMS] {
        core::array::from_fn(|t| self.hand.turned_pairs & (1 << t) != 0)
    }

    /// Rebuild a match from its seed and its record ([`Match::actions`] and [`Match::extras`]).
    /// Fails on the first refused step.
    pub fn replay_record(
        seed: u64,
        rules: Rules,
        actions: &[Action],
        extras: &[(u32, Extra)],
    ) -> Result<Match, RecordError> {
        let mut m = Match::new(seed, rules);
        m.apply_record(actions, extras)?;
        Ok(m)
    }

    /// Apply a record to this match: the actions in order, each extra `(i, step)` applied once the
    /// match has `i` actions in all (counting from its start, as [`Match::extras`] records them).
    /// Extras must be in order and none may point past the last action. Stops at the first
    /// refusal.
    pub fn apply_record(
        &mut self,
        actions: &[Action],
        extras: &[(u32, Extra)],
    ) -> Result<(), RecordError> {
        let mut next = 0;
        for &a in actions {
            next = self.extras_due(extras, next)?;
            self.apply(a).map_err(RecordError::Action)?;
        }
        next = self.extras_due(extras, next)?;
        if next < extras.len() {
            return Err(RecordError::BadExtraIndex);
        }
        Ok(())
    }

    /// Apply the extras from `extras[next..]` that are due now; returns the index of the first one
    /// not yet due.
    fn extras_due(
        &mut self,
        extras: &[(u32, Extra)],
        mut next: usize,
    ) -> Result<usize, RecordError> {
        while let Some(&(at, step)) = extras.get(next) {
            if at as usize != self.actions.len() {
                break;
            }
            match step {
                Extra::Turn(t) => self.turn_team(t).map_err(RecordError::Turn)?,
                Extra::PlayThenTurn(c, t) => {
                    self.play_then_turn(c, t).map_err(RecordError::Turn)?
                }
                Extra::Lapse(l) => self.let_through(l).map_err(RecordError::Lapse)?,
            }
            next += 1;
        }
        Ok(next)
    }

    // ----- the referee -----

    /// A seat's current hand.
    fn hand_of(&self, seat: u8) -> crate::cards::Bits {
        self.truth.hands[usize::from(seat)]
    }

    /// Record the renege an illegal card makes (the app's classification): to a trump lead, a
    /// missing trump; to a plain lead, the suit it held.
    fn record_illegal(&mut self, seat: u8, card: Card, trump: Suit) {
        let led = self.hand.trick.first().map(|&(_, c)| c);
        debug_assert!(!has(legal_plays(self.hand_of(seat), led, trump), card));
        let Some(led) = led else { return };
        let kind = match effective_suit(led, trump) {
            EffSuit::Trump => RenegeKind::TrumpLead(led),
            EffSuit::Plain(s) => RenegeKind::Follow(s),
        };
        self.hand.reneges.push(Renege {
            seat,
            kind,
            caught: false,
        });
    }

    /// Undo the engine's automatic take of the turned card by `seat` (its last event): the card
    /// goes back face up where it was, as the take never happened at the app's table, and the
    /// seat is to play (the callers move the play on from there).
    fn untake(&mut self, seat: u8) {
        let Some((_, Event::Took { seat: s, card })) = self.events.pop() else {
            unreachable!("a burn is due right after the take");
        };
        debug_assert_eq!(s, seat);
        self.truth.hands[usize::from(seat)] &= !bit(card);
        self.hand.table_card = Some(card);
        if let Some(last) = self.hand.turned.last_mut() {
            last.1 = None;
        }
        self.hand.took[usize::from(seat)] = false;
        if card != JOKER {
            self.hand.robber = None;
        }
        self.phase = Phase::AwaitPlay(seat);
    }

    /// Turn `team` (validated by the caller) at the current moment.
    fn turn_now(&mut self, team: u8) -> Result<(), TurnRefusal> {
        match self.phase {
            Phase::AwaitBurn(s) => {
                let after = self.hand.after_burn;
                let cause = match after {
                    AfterBurn::PlayFirst => TurnCause::NoRob,
                    AfterBurn::StartTricks => TurnCause::DeclinedRob,
                    AfterBurn::TurnSecond => TurnCause::LeftJoker,
                };
                self.untake(s);
                if self.turn_pair(team, cause) == Flow::HandOver {
                    return Ok(());
                }
                match after {
                    // The seat does not play; the play goes on from the next seat in play.
                    AfterBurn::PlayFirst => self.carry_on(s),
                    AfterBurn::StartTricks => {
                        self.start_tricks();
                        Ok(())
                    }
                    AfterBurn::TurnSecond => {
                        self.turn_second();
                        Ok(())
                    }
                }
            }
            Phase::AwaitPlay(s) => {
                if self.turn_pair(team, TurnCause::BeforePlay) == Flow::HandOver {
                    return Ok(());
                }
                self.carry_on(s)
            }
            Phase::MatchOver => unreachable!("validated by the caller"),
        }
    }

    /// Place `card` for `seat` (no legality check here), then turn the partnerships in `turns`
    /// that are still in play, then go on. A turn named as a proof ([`TurnCause::Proved`], when a
    /// world is rebuilt from a view: the engine never saw that renege) comes before the engine's
    /// own catches, as the engine's own proofs do; the others come after them, in order.
    fn place_then_turn(
        &mut self,
        seat: u8,
        card: Card,
        turns: &[(u8, TurnCause)],
    ) -> Result<(), TurnRefusal> {
        self.place(seat, card);
        let proof = |&&(_, c): &&(u8, TurnCause)| c == TurnCause::Proved;
        for &(team, cause) in turns.iter().filter(proof) {
            if self.hand.in_play(team) && self.turn_pair(team, cause) == Flow::HandOver {
                return Ok(());
            }
        }
        if !self.hand.reneges.is_empty() && self.catches_after(seat, card) == Flow::HandOver {
            return Ok(());
        }
        for &(team, cause) in turns.iter().filter(|t| !proof(t)) {
            if self.hand.in_play(team) && self.turn_pair(team, cause) == Flow::HandOver {
                return Ok(());
            }
        }
        self.carry_on(left_of(seat))
    }

    /// The play goes on from seat `from` (the seat due to act, or the one after the seat that just
    /// played): the next seat still in play that has not played to the trick acts, or the trick
    /// is complete and resolved; a trick not yet led is led by the first seat in play from
    /// `from` (rule 40). With nobody in play the trick in progress, if any, goes to its best card
    /// for no points (rule 40, as the app's `resolveTrick`) and the hand ends. Refused when the
    /// partnership that led the trick in progress is turned before the trick is resolved (the
    /// open question: which suit the players still to play follow, and against which card the
    /// trick is won).
    pub(super) fn carry_on(&mut self, from: u8) -> Result<(), TurnRefusal> {
        let Some(&(lead_seat, _)) = self.hand.trick.first() else {
            if !self.hand.anyone_in_play() {
                self.end_of_hand();
                return Ok(());
            }
            let leader = self.hand.in_play_from(from);
            self.hand.leader = leader;
            self.turn_to(leader);
            return Ok(());
        };
        if !self.hand.anyone_in_play() {
            self.finish_trick();
            return Ok(());
        }
        if !self.hand.in_play(lead_seat) {
            return Err(TurnRefusal::LedTrickInProgress);
        }
        match self.next_in_trick(from) {
            Some(n) => self.turn_to(n),
            None => self.finish_trick(),
        }
        Ok(())
    }

    /// After `seat` played `card` with hidden reneges about: a renege of that seat the card proves
    /// turns its pair (all of that seat's reneges are then caught), then the ace of trumps
    /// showing catches every other seat that robbed without it (the app's `play`).
    pub(super) fn catches_after(&mut self, seat: u8, card: Card) -> Flow {
        let trump = self.hand.trump.expect("trumps are known during play");
        let proved = self
            .hand
            .reneges
            .iter()
            .any(|r| r.seat == seat && !r.caught && proves(r.kind, card, trump));
        if proved {
            for r in self.hand.reneges.iter_mut().filter(|r| r.seat == seat) {
                r.caught = true;
            }
            if self.hand.in_play(seat)
                && self.turn_pair(team_of(seat), TurnCause::Proved) == Flow::HandOver
            {
                return Flow::HandOver;
            }
        }
        if card == ace_of_trumps(trump) {
            for i in 0..self.hand.reneges.len() {
                let r = self.hand.reneges[i];
                if r.kind != RenegeKind::FalseRob || r.seat == seat || r.caught {
                    continue;
                }
                self.hand.reneges[i].caught = true;
                if self.hand.in_play(r.seat)
                    && self.turn_pair(team_of(r.seat), TurnCause::AceShown) == Flow::HandOver
                {
                    return Flow::HandOver;
                }
            }
        }
        Flow::Go
    }

    /// Would crediting `points` to `team` take it to 41 while it has a hidden rob without the
    /// ace or burned ace not yet caught? (rule 38: it is then turned instead, the app's `credit`)
    pub(super) fn caught_at_claim(&self, team: u8, points: u8) -> bool {
        !self.hand.reneges.is_empty()
            && self.scores[usize::from(team)].saturating_add(points) >= WINNING_SCORE
            && self.hand.reneges.iter().any(|r| {
                !r.caught
                    && team_of(r.seat) == team
                    && matches!(r.kind, RenegeKind::FalseRob | RenegeKind::BurnedAce)
            })
    }

    /// Catch the hidden robs without the ace and burned aces (of `only`'s partnership, or all),
    /// in the order they were made, turning each partnership still in play (the app's
    /// `catchHiddenAces`).
    pub(super) fn catch_hidden_aces(&mut self, only: Option<u8>, cause: TurnCause) -> Flow {
        for i in 0..self.hand.reneges.len() {
            let r = self.hand.reneges[i];
            if r.caught || !matches!(r.kind, RenegeKind::FalseRob | RenegeKind::BurnedAce) {
                continue;
            }
            let team = team_of(r.seat);
            if only.is_some_and(|o| o != team) {
                continue;
            }
            self.hand.reneges[i].caught = true;
            if self.hand.in_play(r.seat) && self.turn_pair(team, cause) == Flow::HandOver {
                return Flow::HandOver;
            }
        }
        Flow::Go
    }

    /// Rules 39–41 for `team`, turned now (the app's `penalise`): its score and flag back to
    /// their hand-start values, then each trick it holds passes in trick order, credited at once.
    pub(super) fn turn_pair(&mut self, team: u8, cause: TurnCause) -> Flow {
        let t = usize::from(team);
        self.hand.turned_pairs |= 1 << team;
        self.scores[t] = self.hand.start_scores[t];
        self.bonus_taken[t] = self.hand.start_bonus[t];
        self.hand.tricks_won[t] = 0;
        self.emit_public(Event::PairTurned {
            team,
            score: self.scores[t],
            cause,
        });
        let Some(trump) = self.hand.trump else {
            return Flow::Go;
        };
        for i in 0..self.hand.trick_winners.len() {
            if self.hand.done[i].holder != team {
                continue;
            }
            let d = self.hand.done[i];
            match strongest_in_play(d.cards(), trump, |s| self.hand.in_play(s)) {
                None => self.hand.done[i].holder = NOBODY,
                Some(j) => {
                    let u = team_of(d.cards[j].0);
                    self.hand.done[i].holder = u;
                    self.hand.tricks_won[usize::from(u)] += 1;
                    if self.credit_pass(i, u) == Flow::HandOver {
                        return Flow::HandOver;
                    }
                }
            }
        }
        self.recount_trumps(trump);
        Flow::Go
    }

    /// Credit passed trick `i` to `team`: its 5 points, unless the team is caught at the claim
    /// instead; a team that reaches 41 wins the game at once.
    fn credit_pass(&mut self, i: usize, team: u8) -> Flow {
        let trick = i as u8 + 1;
        if self.caught_at_claim(team, TRICK_POINTS) {
            self.emit_public(Event::TrickPassed {
                trick,
                team,
                points: 0,
            });
            return self.catch_hidden_aces(Some(team), TurnCause::Claim);
        }
        let t = usize::from(team);
        self.scores[t] = self.scores[t].saturating_add(TRICK_POINTS);
        self.emit_public(Event::TrickPassed {
            trick,
            team,
            points: TRICK_POINTS,
        });
        if self.scores[t] >= WINNING_SCORE {
            self.end_game(team, GameEndReason::FortyOneByPassedTrick);
            return Flow::HandOver;
        }
        Flow::Go
    }

    /// The best trump played this hand and whether the 5 of trumps is among the cards played,
    /// counting only the cards of partnerships still in play (rule 39), from the completed
    /// tricks and the trick in progress.
    pub(super) fn recount_trumps(&mut self, trump: Suit) {
        let h = &self.hand;
        let k = h.trick_winners.len();
        let mut best: Option<(Card, u8)> = None;
        let mut five = false;
        let played = h.done[..k]
            .iter()
            .flat_map(|d| d.cards().iter().copied())
            .chain(h.trick.iter().copied());
        for (seat, c) in played {
            if !h.in_play(seat) {
                continue;
            }
            let r = trump_rank(c, trump);
            if r > 0 && best.is_none_or(|(b, _)| r > trump_rank(b, trump)) {
                best = Some((c, seat));
            }
            if c == crate::rules::five_of(trump) {
                five = true;
            }
        }
        self.hand.best_trump = best;
        self.hand.five_played = five;
    }

    /// The end of a hand with a turned partnership or a let-through, in the app's order
    /// (`finishHand`): the hidden robs without the ace and burned aces are caught, then the
    /// best-trump bonus (rule 30) goes to the best trump of a partnership still in play, then a
    /// partnership holding all five tricks has a jink (rules 33–34); every credit can still be
    /// caught at the claim (rule 38).
    pub(super) fn end_of_refereed_hand(&mut self) {
        if self.catch_hidden_aces(None, TurnCause::HandEnd) == Flow::HandOver {
            return;
        }
        if !self.hand.five_played
            && let Some((card, seat)) = self.hand.best_trump
        {
            let team = team_of(seat);
            let extra = if self.bonus_taken[usize::from(team)] {
                BONUS_LATER
            } else {
                BONUS_FIRST
            };
            if self.caught_at_claim(team, extra) {
                // The bonus is lost with the pair (as the app: it is not found again).
                if self.catch_hidden_aces(Some(team), TurnCause::Claim) == Flow::HandOver {
                    return;
                }
            } else {
                self.award_bonus(team, BonusKind::EndOfHand, card);
                if self.scores[usize::from(team)] >= WINNING_SCORE {
                    self.end_game(team, GameEndReason::FortyOneByEndBonus);
                    return;
                }
            }
        }
        let k = self.hand.trick_winners.len();
        if k == usize::from(TRICKS_PER_HAND) {
            let j = self.hand.done[0].holder;
            if j != NOBODY
                && self.hand.in_play(j)
                && self.hand.done[..k].iter().all(|d| d.holder == j)
            {
                let need = WINNING_SCORE.saturating_sub(self.scores[usize::from(j)]);
                if self.caught_at_claim(j, need.max(10)) {
                    if self.catch_hidden_aces(Some(j), TurnCause::Claim) == Flow::HandOver {
                        return;
                    }
                } else {
                    self.end_game(j, GameEndReason::Jink);
                    return;
                }
            }
        }
        self.close_hand();
    }
}

impl super::HandState {
    /// The turned card `seat` has just taken (its rob waiting for the burn), if any.
    fn table_card_taken_by(&self, seat: u8) -> Option<Card> {
        match self.turned.last() {
            Some(&(c, Some(s))) if s == seat => Some(c),
            _ => None,
        }
    }
}
