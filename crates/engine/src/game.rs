//! Hand, game and match flow: the state machine of docs/M0-SPEC.md §2 and the API of §3.
//!
//! A [`Match`] runs from the first deal to the money. It stops only to ask the seat to act for a
//! decision (which card to burn, or which card to play). Everything forced (dealing, turning,
//! taking a turned card, scoring, ending games, moving the deal) happens automatically inside
//! [`Match::apply`], so an illegal action is impossible: `apply` rejects it and changes nothing.
//! Every step is written to an event log with a visibility tag; each seat's view is derived from
//! those tags (see [`crate::view`]).

use crate::cards::{
    ACE, ALL_CARDS, Bits, Card, DECK_SIZE, JACK, JOKER, KING, QUEEN, Suit, bit, cards, count, has,
    name, names, rank_of, suit_of,
};
use crate::rng::Rng;
use crate::rules::{
    EffSuit, ace_of_trumps, effective_suit, five_of, legal_burns, legal_plays, plain_rank,
    trick_winner, trump_rank,
};
use crate::view::{BurnKind, PublicState, SeatView};
use crate::{SEATS, TEAMS, team_of};
use std::sync::Arc;

/// Version of the rules this engine implements. Every game record carries it; it is bumped
/// whenever an owner answer changes the engine (RULES.md, "Records"). Version 2 (2026-10-14):
/// the owner withdrew the "no ace may be burned" amendment to H3 (any card but the ace of trumps
/// may be burned, as rules 21 and 24 say); after a
/// turned Joker and an ordinary card, a dealer holding the ace of trumps robs at their own turn
/// (H4); the `bonus_later` switch was removed (bonuses are always 6 then 5).
pub const RULES_VERSION: u32 = 2;

/// Points for each trick (rule 27).
pub const TRICK_POINTS: u8 = 5;

/// The score that wins a game: 41 or more (rule 32).
pub const WINNING_SCORE: u8 = 41;

/// Games needed to win the match (rule 35, H6).
pub const GAMES_TO_WIN: u8 = 2;

/// Tricks in a hand.
pub const TRICKS_PER_HAND: u8 = 5;

/// Cards dealt to each seat.
pub const HAND_SIZE: u8 = 5;

/// Deck position of the turned card.
const TURNED_POS: usize = 40;

/// Label of the per-hand deck streams of a duplicate block (docs/M1-SPEC.md §1): the deck for hand
/// index `k` is the identity deck shuffled by `Rng::derive(block_seed, &[BLOCK_DECK_LABEL, k])`.
pub const BLOCK_DECK_LABEL: u64 = 0xDEC;

/// Label of a duplicate block's first-dealer draw (docs/M1-SPEC.md §1).
pub const BLOCK_DEALER_LABEL: u64 = 0xDEA;

/// Bonus for a partnership's first best-trump trick in a game (rule 28: 6, so the trick is worth
/// 11).
pub const BONUS_FIRST: u8 = 6;

/// Bonus for each later best-trump trick of that partnership in the same game (house rule: 5, so
/// the trick is worth 10). The flat-11 variant was considered and rejected by the owner on
/// 2026-10-14; there is no switch.
pub const BONUS_LATER: u8 = 5;

/// Match settings (docs/M0-SPEC.md §1). Only the stake can vary; the scoring is fixed.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rules {
    /// The stake each player of a losing partnership pays (H6).
    pub stake: i32,
}

impl Rules {
    /// The owner's house rules: first bonus 6, later bonuses 5 (fixed), stake 5.
    pub const HOUSE: Rules = Rules { stake: 5 };
}

impl Default for Rules {
    fn default() -> Self {
        Rules::HOUSE
    }
}

/// A decision by the seat to act. Robbing and taking are never actions: the engine does them.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Action {
    /// Burn this card face down (after taking a turned card).
    Burn(Card),
    /// Play this card to the current trick.
    Play(Card),
}

impl Action {
    /// The card the action names.
    pub const fn card(self) -> Card {
        match self {
            Action::Burn(c) | Action::Play(c) => c,
        }
    }
}

/// What the match is waiting for.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// This seat must burn a card.
    AwaitBurn(u8),
    /// This seat must play a card.
    AwaitPlay(u8),
    /// The match is over; nothing more can happen.
    MatchOver,
}

/// Why a game ended.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameEndReason {
    /// A trick's 5 points took the team to 41 or more.
    FortyOneByTrick,
    /// The bonus for the trick holding the 5 of trumps took the team to 41 or more.
    FortyOneByImmediateBonus,
    /// The end-of-hand best-trump bonus took the team to 41 or more.
    FortyOneByEndBonus,
    /// The team won all five tricks of a hand (rule 33), passed tricks included, or every other
    /// partnership has been turned (rule 34).
    Jink,
    /// Tricks passed from a turned partnership (rule 39) took the team to 41 or more (rule 32:
    /// "the game ends immediately at 41, a passed trick included").
    FortyOneByPassedTrick,
}

impl GameEndReason {
    /// The word used in canonical event strings.
    pub const fn canonical(self) -> &'static str {
        match self {
            GameEndReason::FortyOneByTrick => "trick",
            GameEndReason::FortyOneByImmediateBonus => "immediate_bonus",
            GameEndReason::FortyOneByEndBonus => "end_bonus",
            GameEndReason::Jink => "jink",
            GameEndReason::FortyOneByPassedTrick => "passed_trick",
        }
    }
}

/// When a best-trump bonus was awarded.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BonusKind {
    /// At once, for the trick holding the 5 of trumps (rule 29).
    Immediate,
    /// At the end of the hand, for the trick holding the highest trump played (rule 30).
    EndOfHand,
}

impl BonusKind {
    /// The word used in canonical event strings.
    pub const fn canonical(self) -> &'static str {
        match self {
            BonusKind::Immediate => "immediate",
            BonusKind::EndOfHand => "end",
        }
    }
}

/// Who may see an event.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Visibility {
    /// Every seat.
    Public,
    /// Only this seat.
    Private(u8),
}

impl Visibility {
    /// May this seat see the event?
    pub const fn visible_to(self, seat: u8) -> bool {
        match self {
            Visibility::Public => true,
            Visibility::Private(s) => s == seat,
        }
    }
}

/// One step of the match, as recorded in the event log.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// The match begins with this dealer.
    MatchStart {
        /// The first dealer.
        first_dealer: u8,
    },
    /// A hand begins.
    HandStart {
        /// Hands dealt before this one in the match.
        hand_index: u32,
        /// Games started before this one in the match.
        game_index: u8,
        /// The dealer.
        dealer: u8,
    },
    /// A seat's five cards (private to that seat).
    Dealt {
        /// The seat.
        seat: u8,
        /// Its five cards.
        cards: Bits,
    },
    /// A card is turned face up.
    Turned {
        /// The card.
        card: Card,
    },
    /// A seat takes the turned card into its hand (a rob, or the dealer taking a turned Joker).
    Took {
        /// The seat.
        seat: u8,
        /// The card taken.
        card: Card,
    },
    /// A seat burns a card face down (private to that seat).
    Burned {
        /// The seat.
        seat: u8,
        /// The card burned.
        card: Card,
    },
    /// A seat burns a card face down (the public twin of `Burned`).
    BurnDone {
        /// The seat.
        seat: u8,
    },
    /// A card is played to the current trick.
    Played {
        /// The seat.
        seat: u8,
        /// The card.
        card: Card,
    },
    /// A seat has shown it does not hold the ace of trumps (it played, or passed its forced rob,
    /// without robbing).
    ProvedNoAce {
        /// The seat.
        seat: u8,
    },
    /// Everyone played to trick 1 without a rob: the ace of trumps is out of play, in the undealt
    /// pack or (after a turned Joker) in the dealer's first burn, made before trumps were known
    /// (nothing is forbidden then, H3). The name is kept for stability.
    AceUndealt,
    /// A trick is complete.
    TrickWon {
        /// The trick number, 1..=5.
        trick: u8,
        /// The winning seat.
        seat: u8,
        /// The winning team.
        team: u8,
        /// Points for the trick (always 5).
        points: u8,
    },
    /// A best-trump bonus is awarded.
    Bonus {
        /// The team receiving it.
        team: u8,
        /// The bonus points (`BONUS_FIRST` or `BONUS_LATER`).
        points: u8,
        /// Immediate (5 of trumps) or at the end of the hand.
        kind: BonusKind,
        /// The best trump that earned it.
        card: Card,
    },
    /// A team wins a game.
    GameWon {
        /// The winning team.
        team: u8,
        /// Why the game ended.
        reason: GameEndReason,
        /// All four scores at that moment.
        scores: [u8; 4],
    },
    /// A hand ends.
    HandEnd {
        /// The hand's index.
        hand_index: u32,
        /// True if the hand stopped before its fifth trick because the game ended.
        aborted: bool,
    },
    /// A partnership is turned (rules 39-41, owner decision of 2026-10-17 in
    /// docs/APP-ENGINE-CONTRACT.md §7): its cards no longer count, it plays no more cards this
    /// hand and it scores nothing for the hand, so its score goes back to `score`, its score when
    /// the hand started (rule 41). Never caused by the engine's own players (H1): the app hands
    /// the engine a position in which it has caught a pair ([`Match::turn_team`]).
    PairTurned {
        /// The turned partnership.
        team: u8,
        /// Its score now: its score at the start of the hand.
        score: u8,
    },
    /// A trick the turned partnership held passes, with its points, to the partnership still in
    /// play that played the strongest card in it, judged against the suit led (rule 39), or to
    /// the only partnership left in play (rule 34).
    TrickPassed {
        /// The trick number, 1..=5.
        trick: u8,
        /// The partnership that takes it now.
        team: u8,
        /// The trick's points (always 5).
        points: u8,
    },
    /// A team wins the match: the money.
    MatchWon {
        /// The winning team.
        team: u8,
        /// Money per team (winner +6 stakes, each loser −2 stakes).
        money: [i32; 4],
    },
}

/// Card names joined by commas, in ascending index order.
fn joined_names(set: Bits) -> String {
    cards(set).map(name).collect::<Vec<_>>().join(",")
}

impl Event {
    /// The canonical one-line form of the event (docs/M0-SPEC.md §5), shared with the reference
    /// implementation so the two can be compared event by event.
    pub fn canonical(&self) -> String {
        match *self {
            Event::MatchStart { first_dealer } => format!("MatchStart first_dealer={first_dealer}"),
            Event::HandStart {
                hand_index,
                game_index,
                dealer,
            } => format!("HandStart hand={hand_index} game={game_index} dealer={dealer}"),
            Event::Dealt { seat, cards } => {
                format!("Dealt seat={seat} cards={}", joined_names(cards))
            }
            Event::Turned { card } => format!("Turned card={}", name(card)),
            Event::Took { seat, card } => format!("Took seat={seat} card={}", name(card)),
            Event::Burned { seat, card } => format!("Burned seat={seat} card={}", name(card)),
            Event::BurnDone { seat } => format!("BurnDone seat={seat}"),
            Event::Played { seat, card } => format!("Played seat={seat} card={}", name(card)),
            Event::ProvedNoAce { seat } => format!("ProvedNoAce seat={seat}"),
            Event::AceUndealt => "AceUndealt".to_string(),
            Event::TrickWon {
                trick,
                seat,
                team,
                points,
            } => format!("TrickWon trick={trick} seat={seat} team={team} points={points}"),
            Event::Bonus {
                team,
                points,
                kind,
                card,
            } => format!(
                "Bonus team={team} points={points} kind={} card={}",
                kind.canonical(),
                name(card)
            ),
            Event::GameWon {
                team,
                reason,
                scores,
            } => format!(
                "GameWon team={team} reason={} scores={},{},{},{}",
                reason.canonical(),
                scores[0],
                scores[1],
                scores[2],
                scores[3]
            ),
            Event::HandEnd {
                hand_index,
                aborted,
            } => format!("HandEnd hand={hand_index} aborted={aborted}"),
            Event::MatchWon { team, money } => format!(
                "MatchWon team={team} money={},{},{},{}",
                money[0], money[1], money[2], money[3]
            ),
            Event::PairTurned { team, score } => format!("PairTurned team={team} score={score}"),
            Event::TrickPassed {
                trick,
                team,
                points,
            } => format!("TrickPassed trick={trick} team={team} points={points}"),
        }
    }
}

/// An action the engine refused, and why. The match is unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IllegalAction {
    /// The refused action.
    pub action: Action,
    /// Why it was refused, in plain English.
    pub reason: &'static str,
}

impl core::fmt::Display for IllegalAction {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (verb, card) = match self.action {
            Action::Burn(c) => ("burn", c),
            Action::Play(c) => ("play", c),
        };
        let card = if usize::from(card) < DECK_SIZE {
            name(card)
        } else {
            format!("card #{card}")
        };
        write!(f, "illegal {verb} of {card}: {}", self.reason)
    }
}

impl std::error::Error for IllegalAction {}

/// Why [`Match::turn_team`] refused to turn a partnership. The match is unchanged.
///
/// Three refusals are cases the rules' wording leaves open (docs/RULES.md rules 39 and 40); they
/// are refused rather than guessed, until the owner settles them: [`TurnRefusal::LedTrickInProgress`],
/// [`TurnRefusal::NoStandingWinner`] and [`TurnRefusal::SeveralReachFortyOne`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnRefusal {
    /// The team number is not 0..=3.
    NoSuchTeam,
    /// A partnership is turned only while a play is awaited (not during a burn, not once the
    /// match is over).
    NotBeforeAPlay,
    /// The partnership is already turned this hand.
    AlreadyTurned,
    /// Open question: the partnership led the trick in progress. Rule 40 says the trick goes on
    /// without the turned cards, but not which suit the players still to play must then follow,
    /// nor against which card the trick is won.
    LedTrickInProgress,
    /// Open question: a trick the partnership holds would pass (rule 39), but no card of a
    /// partnership still in play in it is a trump or of the suit led, and two or more
    /// partnerships are still in play, so "the strongest card in that trick" is not defined.
    NoStandingWinner {
        /// The trick, 1..=5.
        trick: u8,
    },
    /// Open question: the passed tricks would take two or more partnerships to 41 at the same
    /// moment, and the rules do not say which of them wins the game.
    SeveralReachFortyOne,
}

impl TurnRefusal {
    /// A short stable word for the refusal (compared with the reference implementation's).
    pub const fn code(self) -> &'static str {
        match self {
            TurnRefusal::NoSuchTeam => "no_such_team",
            TurnRefusal::NotBeforeAPlay => "not_before_a_play",
            TurnRefusal::AlreadyTurned => "already_turned",
            TurnRefusal::LedTrickInProgress => "led_trick_in_progress",
            TurnRefusal::NoStandingWinner { .. } => "no_standing_winner",
            TurnRefusal::SeveralReachFortyOne => "several_reach_41",
        }
    }

    /// Why, in plain English.
    pub const fn reason(self) -> &'static str {
        match self {
            TurnRefusal::NoSuchTeam => "partnerships are numbered 0 to 3",
            TurnRefusal::NotBeforeAPlay => {
                "a partnership is turned only before a play (not during a burn or after the match)"
            }
            TurnRefusal::AlreadyTurned => "that partnership is already turned this hand",
            TurnRefusal::LedTrickInProgress => {
                "open rules question (owner): the turned partnership led the trick in progress, \
                 and rule 40 does not say what the players still to play must follow"
            }
            TurnRefusal::NoStandingWinner { .. } => {
                "open rules question (owner): a trick would pass, but no card of a partnership \
                 still in play in it is a trump or of the suit led (rule 39)"
            }
            TurnRefusal::SeveralReachFortyOne => {
                "open rules question (owner): the passed tricks would take two partnerships to 41 \
                 at the same moment"
            }
        }
    }
}

/// A turn [`Match::turn_team`] refused, and why. The match is unchanged.
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
        )?;
        if let TurnRefusal::NoStandingWinner { trick } = self.why {
            write!(f, " (trick {trick})")?;
        }
        Ok(())
    }
}

impl std::error::Error for TurnRefused {}

/// Why a record ([`Match::apply_record`]) could not be replayed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecordError {
    /// An action was refused.
    Action(IllegalAction),
    /// A turn was refused.
    Turn(TurnRefused),
    /// The turns are not in order, or one points past the end of the actions.
    BadTurnIndex,
}

impl core::fmt::Display for RecordError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RecordError::Action(e) => write!(f, "{e}"),
            RecordError::Turn(e) => write!(f, "{e}"),
            RecordError::BadTurnIndex => f.write_str(
                "the turns must be in order, each after at most as many actions as were recorded",
            ),
        }
    }
}

impl std::error::Error for RecordError {}

/// The hidden truth of the current hand, for tests and the differential test only. Bots must
/// never read it; they get [`SeatView`]s.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Truth {
    /// The current hand's deck in deal order (docs/M0-SPEC.md §1).
    pub deck: [Card; DECK_SIZE],
    /// Each seat's current hand.
    pub hands: [Bits; SEATS],
    /// Each seat's burned cards this hand.
    pub burns: [Bits; SEATS],
    /// The undealt pack (cards still face down in the pack, not counting burns).
    pub undealt: Bits,
}

/// What happens once the pending burn is made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AfterBurn {
    /// The dealer burned after taking a turned Joker: turn the next card (step 2b).
    TurnSecond,
    /// The dealer robbed at once: start trick 1 (step 5).
    StartTricks,
    /// A robber burned at their first turn in trick 1: they now play (step 6).
    PlayFirst,
}

/// A completed trick of the hand: the cards that counted in it when it was won (play order, the
/// led card first) and the partnership that holds it now (its winner's, or the one it passed to
/// under rules 39 and 34).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DoneTrick {
    cards: [(u8, Card); SEATS],
    n: u8,
    holder: u8,
}

impl DoneTrick {
    const EMPTY: DoneTrick = DoneTrick {
        cards: [(0, 0); SEATS],
        n: 0,
        holder: 0,
    };

    fn cards(&self) -> &[(u8, Card)] {
        &self.cards[..usize::from(self.n)]
    }
}

/// The state of the hand in progress (or of the last hand, once the match is over).
#[derive(Clone, Debug, PartialEq, Eq)]
struct HandState {
    trump: Option<Suit>,
    table_card: Option<Card>,
    turned: Vec<(Card, Option<u8>)>,
    joker_turned: bool,
    played: Bits,
    trick_no: u8,
    leader: u8,
    trick: Vec<(u8, Card)>,
    trick_winners: Vec<u8>,
    tricks_won: [u8; TEAMS],
    best_trump: Option<(Card, u8)>,
    five_played: bool,
    robber: Option<u8>,
    pending_robber: Option<u8>,
    took: [bool; SEATS],
    burn_order: Vec<(u8, BurnKind)>,
    proved_no_ace: [bool; SEATS],
    ace_undealt: bool,
    after_burn: AfterBurn,
    /// Partnerships turned this hand: bit `t` for team `t` (rules 39-41, 34). 0 unless the app
    /// has turned one ([`Match::turn_team`]).
    turned_pairs: u8,
    /// The completed tricks, in order (the first `trick_winners.len()` entries).
    done: [DoneTrick; TRICKS_PER_HAND as usize],
    /// Each team's score and bonus flag when the hand started (rule 41 gives them back to a
    /// turned partnership).
    start_scores: [u8; TEAMS],
    start_bonus: [bool; TEAMS],
}

impl HandState {
    fn new(dealer: u8) -> Self {
        HandState {
            trump: None,
            table_card: None,
            turned: Vec::with_capacity(2),
            joker_turned: false,
            played: 0,
            trick_no: 0,
            leader: left_of(dealer),
            trick: Vec::with_capacity(SEATS),
            trick_winners: Vec::with_capacity(usize::from(TRICKS_PER_HAND)),
            tricks_won: [0; TEAMS],
            best_trump: None,
            five_played: false,
            robber: None,
            pending_robber: None,
            took: [false; SEATS],
            burn_order: Vec::with_capacity(2),
            proved_no_ace: [false; SEATS],
            ace_undealt: false,
            after_burn: AfterBurn::StartTricks,
            turned_pairs: 0,
            done: [DoneTrick::EMPTY; TRICKS_PER_HAND as usize],
            start_scores: [0; TEAMS],
            start_bonus: [false; TEAMS],
        }
    }

    /// Is the seat's partnership still in play (not turned)?
    fn in_play(&self, seat: u8) -> bool {
        self.turned_pairs & (1 << team_of(seat)) == 0
    }
}

/// The seat on the left (the next seat clockwise).
const fn left_of(seat: u8) -> u8 {
    (seat + 1) % SEATS as u8
}

/// The deck 0, 1, …, 52, which every hand's shuffle starts from.
const fn identity_deck() -> [Card; DECK_SIZE] {
    let mut d = [0u8; DECK_SIZE];
    let mut i = 0;
    while i < DECK_SIZE {
        d[i] = i as Card;
        i += 1;
    }
    d
}

/// The deck for hand index `k` (counted across the games of the match) of every match in the
/// duplicate block with this seed: the cards 0..53 shuffled by
/// `Rng::derive(block_seed, &[BLOCK_DECK_LABEL, k])` (docs/M1-SPEC.md §1). The standard deck
/// closure for [`Match::new_block`]: `move |k| block_deck(block_seed, k)`.
pub fn block_deck(block_seed: u64, k: u32) -> [Card; DECK_SIZE] {
    let mut d = identity_deck();
    Rng::derive(block_seed, &[BLOCK_DECK_LABEL, u64::from(k)]).shuffle(&mut d);
    d
}

/// The first dealer of every match in the duplicate block with this seed:
/// `Rng::derive(block_seed, &[BLOCK_DEALER_LABEL]).bounded(8)` (docs/M1-SPEC.md §1).
pub fn block_first_dealer(block_seed: u64) -> u8 {
    Rng::derive(block_seed, &[BLOCK_DEALER_LABEL]).bounded(SEATS as u64) as u8
}

/// The deck source of a [`Match::new_block`] match: the deck for each hand index, asked for when
/// the hand starts. Two of them are equal only if they are the same closure (one match is a clone
/// of the other), since closures cannot be compared.
#[derive(Clone)]
struct HandDecks(Arc<dyn Fn(u32) -> [Card; DECK_SIZE] + Send + Sync>);

impl core::fmt::Debug for HandDecks {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("HandDecks(..)")
    }
}

impl PartialEq for HandDecks {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for HandDecks {}

/// Is the deck an ordering of all 53 cards?
fn is_full_deck(deck: &[Card; DECK_SIZE]) -> bool {
    let mut seen: Bits = 0;
    for &c in deck {
        if usize::from(c) >= DECK_SIZE || has(seen, c) {
            return false;
        }
        seen |= bit(c);
    }
    seen == ALL_CARDS
}

/// A whole match of 41, from the first deal to the money.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Match {
    rules: Rules,
    seed: u64,
    rng: Rng,
    decks: Vec<[Card; DECK_SIZE]>,
    next_deck: usize,
    hand_decks: Option<HandDecks>,
    dealer: u8,
    hand_index: u32,
    game_index: u8,
    scores: [u8; TEAMS],
    games_won: [u8; TEAMS],
    bonus_taken: [bool; TEAMS],
    money: Option<[i32; TEAMS]>,
    phase: Phase,
    events: Vec<(Visibility, Event)>,
    actions: Vec<Action>,
    /// Every partnership turned so far ([`Match::turn_team`]): the number of actions applied
    /// before the turn, and the team. Empty in a match where nobody is turned.
    turns: Vec<(u32, u8)>,
    truth: Truth,
    hand: HandState,
}

impl Match {
    /// A new match from a seed: the first dealer is `rng.bounded(8)`, then each hand's deck is
    /// the cards 0..53 shuffled by `rng.shuffle`, all from `Rng::new(seed)`. The match runs to the
    /// first decision.
    pub fn new(seed: u64, rules: Rules) -> Match {
        let mut rng = Rng::new(seed);
        let first_dealer = rng.bounded(SEATS as u64) as u8;
        Match::start(first_dealer, Vec::new(), None, rng, seed, rules)
    }

    /// A match with chosen decks (for tests, hand-made deals and the differential test): the
    /// given decks are used for the first hands, in order, then decks are shuffled by
    /// `Rng::new(seed)` (no dealer is drawn from it, since the first dealer is given).
    ///
    /// # Panics
    /// If `first_dealer` is not a seat or a deck is not an ordering of all 53 cards.
    pub fn with_decks(
        first_dealer: u8,
        decks: Vec<[Card; DECK_SIZE]>,
        seed: u64,
        rules: Rules,
    ) -> Match {
        assert!(usize::from(first_dealer) < SEATS, "no such dealer seat");
        assert!(
            decks.iter().all(is_full_deck),
            "every deck must hold each of the 53 cards exactly once"
        );
        Match::start(first_dealer, decks, None, Rng::new(seed), seed, rules)
    }

    /// A match of a duplicate block (docs/M1-SPEC.md §1–2): like [`Match::with_decks`], but the
    /// deck for hand index `k` (counted across the games of the match, from 0) is asked from
    /// `hand_decks(k)` when that hand starts, for as many hands as the match lasts. The standard
    /// closure is `move |k| block_deck(block_seed, k)`, with `block_first_dealer(block_seed)` as
    /// the first dealer; then every match of the block deals the same cards in hand `k`.
    /// [`Match::seed`] returns `block_seed`. The closure must be a pure function of `k` (it is
    /// shared, not re-run, when the match is cloned) so the match replays exactly.
    ///
    /// # Panics
    /// If `first_dealer` is not a seat; and, when a hand starts (so possibly inside
    /// [`Match::apply`]), if the closure returns a deck that is not an ordering of all 53 cards.
    pub fn new_block(
        block_seed: u64,
        hand_decks: impl Fn(u32) -> [Card; DECK_SIZE] + Send + Sync + 'static,
        first_dealer: u8,
        rules: Rules,
    ) -> Match {
        assert!(usize::from(first_dealer) < SEATS, "no such dealer seat");
        let decks = Some(HandDecks(Arc::new(hand_decks)));
        Match::start(
            first_dealer,
            Vec::new(),
            decks,
            Rng::new(block_seed),
            block_seed,
            rules,
        )
    }

    /// Rebuild a match from its seed and recorded actions. Fails on the first illegal action.
    pub fn replay(seed: u64, rules: Rules, actions: &[Action]) -> Result<Match, IllegalAction> {
        let mut m = Match::new(seed, rules);
        for &a in actions {
            m.apply(a)?;
        }
        Ok(m)
    }

    fn start(
        first_dealer: u8,
        decks: Vec<[Card; DECK_SIZE]>,
        hand_decks: Option<HandDecks>,
        rng: Rng,
        seed: u64,
        rules: Rules,
    ) -> Match {
        let mut m = Match {
            rules,
            seed,
            rng,
            decks,
            next_deck: 0,
            hand_decks,
            dealer: first_dealer,
            hand_index: 0,
            game_index: 0,
            scores: [0; TEAMS],
            games_won: [0; TEAMS],
            bonus_taken: [false; TEAMS],
            money: None,
            phase: Phase::MatchOver,
            events: Vec::new(),
            actions: Vec::new(),
            turns: Vec::new(),
            truth: Truth {
                deck: identity_deck(),
                hands: [0; SEATS],
                burns: [0; SEATS],
                undealt: 0,
            },
            hand: HandState::new(first_dealer),
        };
        m.emit_public(Event::MatchStart { first_dealer });
        m.start_hand();
        m
    }

    // ----- accessors -----

    /// The rule switches.
    pub fn rules(&self) -> Rules {
        self.rules
    }

    /// The seed the match was created with (the block seed for [`Match::new_block`]).
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// What the match is waiting for.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// The seat to act, or `None` once the match is over.
    pub fn to_act(&self) -> Option<u8> {
        match self.phase {
            Phase::AwaitBurn(s) | Phase::AwaitPlay(s) => Some(s),
            Phase::MatchOver => None,
        }
    }

    /// The cards the seat to act may burn or play (empty once the match is over).
    pub fn legal_mask(&self) -> Bits {
        match self.phase {
            Phase::AwaitBurn(s) => legal_burns(self.truth.hands[usize::from(s)], self.hand.trump),
            Phase::AwaitPlay(s) => match self.hand.trump {
                Some(trump) => legal_plays(
                    self.truth.hands[usize::from(s)],
                    self.hand.trick.first().map(|&(_, c)| c),
                    trump,
                ),
                None => 0,
            },
            Phase::MatchOver => 0,
        }
    }

    /// The legal actions, sorted by card index. Never empty before the match is over.
    pub fn legal_actions(&self) -> Vec<Action> {
        let mask = self.legal_mask();
        match self.phase {
            Phase::AwaitBurn(_) => cards(mask).map(Action::Burn).collect(),
            Phase::AwaitPlay(_) => cards(mask).map(Action::Play).collect(),
            Phase::MatchOver => Vec::new(),
        }
    }

    /// The event log, with visibility tags, from `MatchStart` on.
    pub fn events(&self) -> &[(Visibility, Event)] {
        &self.events
    }

    /// Every action applied so far, in order.
    pub fn actions(&self) -> &[Action] {
        &self.actions
    }

    /// The dealer of the current hand.
    pub fn dealer(&self) -> u8 {
        self.dealer
    }

    /// The current hand's index in the match, from 0.
    pub fn hand_index(&self) -> u32 {
        self.hand_index
    }

    /// The current game's index in the match, from 0.
    pub fn game_index(&self) -> u8 {
        self.game_index
    }

    /// Each team's score in the current game.
    pub fn scores(&self) -> [u8; TEAMS] {
        self.scores
    }

    /// Games won by each team.
    pub fn games_won(&self) -> [u8; TEAMS] {
        self.games_won
    }

    /// Whether each team has taken a best-trump bonus in the current game.
    pub fn bonus_taken(&self) -> [bool; TEAMS] {
        self.bonus_taken
    }

    /// The money per team once the match is over.
    pub fn money(&self) -> Option<[i32; TEAMS]> {
        self.money
    }

    /// The hidden truth (deck, hands, burns, undealt pack). For tests and the differential test
    /// only: bots must use [`Match::view`].
    pub fn truth(&self) -> &Truth {
        &self.truth
    }

    /// Everything every seat knows.
    pub fn public(&self) -> PublicState {
        let h = &self.hand;
        let hand_sizes: [u8; SEATS] = core::array::from_fn(|s| count(self.truth.hands[s]));
        let burn_counts: [u8; SEATS] = core::array::from_fn(|s| count(self.truth.burns[s]));
        let hidden_count = count(self.truth.undealt) + burn_counts.iter().sum::<u8>();
        PublicState {
            phase: self.phase,
            to_act: self.to_act(),
            dealer: self.dealer,
            hand_index: self.hand_index,
            game_index: self.game_index,
            scores: self.scores,
            games_won: self.games_won,
            bonus_taken: self.bonus_taken,
            trump: h.trump,
            table_card: h.table_card,
            turned: h.turned.clone(),
            joker_turned: h.joker_turned,
            hand_sizes,
            burn_counts,
            burn_order: h.burn_order.clone(),
            hidden_count,
            played: h.played,
            trick_no: h.trick_no,
            leader: h.leader,
            trick_cards: h.trick.clone(),
            trick_winners: h.trick_winners.clone(),
            tricks_won: h.tricks_won,
            best_trump: h.best_trump,
            five_played: h.five_played,
            robber: h.robber,
            proved_no_ace: h.proved_no_ace,
            ace_undealt: h.ace_undealt,
            turned_teams: self.turned_teams(),
        }
    }

    /// One seat's view: its hand, its burns, the public state and the events it may see.
    ///
    /// # Panics
    /// If `seat` is not 0..=7.
    pub fn view(&self, seat: u8) -> SeatView {
        assert!(usize::from(seat) < SEATS, "no such seat");
        SeatView {
            seat,
            hand: self.truth.hands[usize::from(seat)],
            own_burns: self.truth.burns[usize::from(seat)],
            public: self.public(),
            events: self
                .events
                .iter()
                .filter(|(v, _)| v.visible_to(seat))
                .map(|(_, e)| e.clone())
                .collect(),
        }
    }

    // ----- acting -----

    /// Apply the seat-to-act's decision. An illegal action is refused with a reason and leaves
    /// the match unchanged; this never panics, whatever the action.
    pub fn apply(&mut self, action: Action) -> Result<(), IllegalAction> {
        let reject = |reason: &'static str| Err(IllegalAction { action, reason });
        if usize::from(action.card()) >= DECK_SIZE {
            return reject("no such card: card indices run from 0 to 52");
        }
        match (self.phase, action) {
            (Phase::MatchOver, _) => reject("the match is over"),
            (Phase::AwaitBurn(_), Action::Play(_)) => {
                reject("a burn is due: the player who took the turned card must burn first")
            }
            (Phase::AwaitPlay(_), Action::Burn(_)) => {
                reject("a card must be played: nobody has a burn to make")
            }
            (Phase::AwaitBurn(seat), Action::Burn(c)) => {
                let hand = self.truth.hands[usize::from(seat)];
                if !has(hand, c) {
                    return reject("that card is not in the burning player's hand");
                }
                if !has(legal_burns(hand, self.hand.trump), c) {
                    return reject("the ace of trumps may never be burned (rules 21 and 24)");
                }
                self.actions.push(action);
                self.burn(seat, c);
                Ok(())
            }
            (Phase::AwaitPlay(seat), Action::Play(c)) => {
                if !has(self.truth.hands[usize::from(seat)], c) {
                    return reject("that card is not in the hand of the player to act");
                }
                if !has(self.legal_mask(), c) {
                    return reject(self.illegal_play_reason());
                }
                self.actions.push(action);
                self.play(seat, c);
                Ok(())
            }
        }
    }

    // ----- turned partnerships (rules 39-41 and 34; docs/APP-ENGINE-CONTRACT.md §7) -----

    /// Turn partnership `team` now, before the play the match is waiting for (in the middle of a
    /// trick or before its lead). The engine's own players never act illegally (H1), so the engine
    /// never turns anyone itself: the app tells it which pair it has caught, and when (rules 37
    /// and 38 stay the app's), and the hand then goes on exactly as rules 39, 40, 41 and 34 say:
    ///
    /// - rule 41: the partnership scores nothing for the hand: its score and bonus flag go back to
    ///   what they were when the hand started (an 11 it took this hand is not spent);
    /// - rule 39: none of its cards counts any more. Every trick it holds passes, with its 5
    ///   points, to the partnership still in play that played the strongest card in it, judged
    ///   against the suit led (the strongest trump, else the highest card of the suit led). The
    ///   best-trump bonus goes to the highest trump played in the hand by a partnership still in
    ///   play: at once if that is the 5 of trumps (rule 29, already paid), else at the end of the
    ///   hand (rule 30), worth that partnership's 11 or 10 (rule 28);
    /// - rule 32: a partnership the passed tricks take to 41 wins the game at once;
    /// - rule 34: when every other partnership is turned, the one still in play takes every trick
    ///   of the hand, which is a jink: it wins the game at once (after the 41 check above);
    /// - rule 40: its cards in the trick in progress leave it and the trick goes on among the
    ///   other players; a turned seat never plays again; if the seat to lead is turned, the lead
    ///   passes to the next seat on the left still in play.
    ///
    /// Passed tricks count as the receiving partnership's tricks (rules 33 and 34). Events:
    /// [`Event::PairTurned`], then one [`Event::TrickPassed`] per passed trick in trick order,
    /// then whatever follows (a game won, the trick in progress completed, and so on). The turn
    /// is recorded in [`Match::turns`], so the match replays exactly ([`Match::apply_record`]).
    ///
    /// Refused (the match unchanged) when no play is awaited, when the partnership is already
    /// turned, and in the three cases the rules leave open (see [`TurnRefusal`]).
    pub fn turn_team(&mut self, team: u8) -> Result<(), TurnRefused> {
        let refuse = |why| Err(TurnRefused { team, why });
        if usize::from(team) >= TEAMS {
            return refuse(TurnRefusal::NoSuchTeam);
        }
        let (Phase::AwaitPlay(to_act), Some(trump)) = (self.phase, self.hand.trump) else {
            return refuse(TurnRefusal::NotBeforeAPlay);
        };
        if self.hand.turned_pairs & (1 << team) != 0 {
            return refuse(TurnRefusal::AlreadyTurned);
        }
        if self
            .hand
            .trick
            .first()
            .is_some_and(|&(s, _)| team_of(s) == team)
        {
            return refuse(TurnRefusal::LedTrickInProgress);
        }
        let turned = self.hand.turned_pairs | (1 << team);
        let in_play = |seat: u8| turned & (1 << team_of(seat)) == 0;
        let mut still: [u8; TEAMS] = [0; TEAMS];
        let mut n_still = 0;
        for u in 0..TEAMS as u8 {
            if turned & (1 << u) == 0 {
                still[n_still] = u;
                n_still += 1;
            }
        }
        // Rule 34: with one partnership left in play, it takes every trick.
        let only = (n_still == 1).then_some(still[0]);
        let k = self.hand.trick_winners.len();
        let mut passes = [(0usize, 0u8); TRICKS_PER_HAND as usize];
        let mut n_passes = 0;
        let mut scores = self.scores;
        scores[usize::from(team)] = self.hand.start_scores[usize::from(team)];
        for (i, d) in self.hand.done[..k].iter().enumerate() {
            if d.holder != team {
                continue;
            }
            let to = match only {
                Some(u) => u,
                None => match strongest_in_play(d.cards(), trump, in_play) {
                    Some(seat) => team_of(seat),
                    None => {
                        return refuse(TurnRefusal::NoStandingWinner { trick: i as u8 + 1 });
                    }
                },
            };
            passes[n_passes] = (i, to);
            n_passes += 1;
            scores[usize::from(to)] = scores[usize::from(to)].saturating_add(TRICK_POINTS);
        }
        let home = still[..n_still]
            .iter()
            .filter(|&&u| scores[usize::from(u)] >= WINNING_SCORE)
            .count();
        if home >= 2 {
            return refuse(TurnRefusal::SeveralReachFortyOne);
        }

        // Nothing is refused from here on.
        self.hand.turned_pairs = turned;
        self.turns.push((self.actions.len() as u32, team));
        let t = usize::from(team);
        self.scores[t] = self.hand.start_scores[t];
        self.bonus_taken[t] = self.hand.start_bonus[t];
        self.hand.tricks_won[t] = 0;
        self.emit_public(Event::PairTurned {
            team,
            score: self.scores[t],
        });
        for &(i, to) in &passes[..n_passes] {
            let u = usize::from(to);
            self.hand.done[i].holder = to;
            self.hand.tricks_won[u] += 1;
            self.scores[u] = self.scores[u].saturating_add(TRICK_POINTS);
            self.emit_public(Event::TrickPassed {
                trick: i as u8 + 1,
                team: to,
                points: TRICK_POINTS,
            });
        }
        // Rule 40: the turned cards leave the trick in progress (never its lead: refused above).
        self.hand.trick.retain(|&(s, _)| in_play(s));
        // Rule 39: only cards of partnerships still in play count for the best-trump bonus.
        self.recount_trumps(trump);
        // Rule 32: the game ends the moment a partnership reaches 41, a passed trick included.
        if let Some(&u) = still[..n_still]
            .iter()
            .find(|&&u| self.scores[usize::from(u)] >= WINNING_SCORE)
        {
            self.end_game(u, GameEndReason::FortyOneByPassedTrick);
            return Ok(());
        }
        // Rule 34: the partnership still in play takes every trick of the hand: a jink.
        if let Some(u) = only {
            self.end_game(u, GameEndReason::Jink);
            return Ok(());
        }
        // Rule 40: the play goes on among the seats still in play.
        if !in_play(to_act) {
            if self.hand.trick.is_empty() {
                // The seat to lead is turned: the lead passes to the next seat on its left still
                // in play.
                let mut l = left_of(to_act);
                while !in_play(l) {
                    l = left_of(l);
                }
                self.hand.leader = l;
                self.turn_to(l);
            } else {
                match self.next_in_trick(to_act) {
                    Some(next) => self.turn_to(next),
                    None => self.finish_trick(),
                }
            }
        }
        Ok(())
    }

    /// The best trump played this hand and whether the 5 of trumps is among the cards played,
    /// counting only the cards of partnerships still in play (rule 39), from the completed
    /// tricks and the trick in progress.
    fn recount_trumps(&mut self, trump: Suit) {
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
            if c == five_of(trump) {
                five = true;
            }
        }
        self.hand.best_trump = best;
        self.hand.five_played = five;
    }

    /// Partnerships turned in the current hand (all `false` unless [`Match::turn_team`] turned
    /// one).
    pub fn turned_teams(&self) -> [bool; TEAMS] {
        core::array::from_fn(|t| self.hand.turned_pairs & (1 << t) != 0)
    }

    /// Every turn applied in the match, in order: the number of actions applied before it, and
    /// the partnership. With [`Match::actions`] it is the match's record ([`Match::apply_record`]).
    pub fn turns(&self) -> &[(u32, u8)] {
        &self.turns
    }

    /// Rebuild a match from its seed and its record (actions and turns, as [`Match::actions`] and
    /// [`Match::turns`] give them). Fails on the first refused action or turn.
    pub fn replay_record(
        seed: u64,
        rules: Rules,
        actions: &[Action],
        turns: &[(u32, u8)],
    ) -> Result<Match, RecordError> {
        let mut m = Match::new(seed, rules);
        m.apply_record(actions, turns)?;
        Ok(m)
    }

    /// Apply a record to this match: the actions in order, each turn `(i, team)` applied once the
    /// match has `i` actions in all (counting from its start, as [`Match::turns`] records). Turns
    /// must be in order and none may point past the last action. Stops at the first refusal.
    pub fn apply_record(
        &mut self,
        actions: &[Action],
        turns: &[(u32, u8)],
    ) -> Result<(), RecordError> {
        let mut next = 0;
        for &a in actions {
            next = self.turns_due(turns, next)?;
            self.apply(a).map_err(RecordError::Action)?;
        }
        next = self.turns_due(turns, next)?;
        if next < turns.len() {
            return Err(RecordError::BadTurnIndex);
        }
        Ok(())
    }

    /// Apply the turns from `turns[next..]` that are due now; returns the index of the first one
    /// not yet due.
    fn turns_due(&mut self, turns: &[(u32, u8)], mut next: usize) -> Result<usize, RecordError> {
        while let Some(&(at, team)) = turns.get(next) {
            if at as usize != self.actions.len() {
                break;
            }
            self.turn_team(team).map_err(RecordError::Turn)?;
            next += 1;
        }
        Ok(next)
    }

    /// Why a card in hand may not be played now.
    fn illegal_play_reason(&self) -> &'static str {
        let led = self.hand.trick.first().map(|&(_, c)| c);
        match (led, self.hand.trump) {
            (Some(led), Some(trump)) => match effective_suit(led, trump) {
                EffSuit::Plain(_) => "you hold the suit led: follow suit or play a trump (rule 14)",
                EffSuit::Trump => {
                    "a trump was led and you hold a trump you may not hold back: play a trump (rules 16-17)"
                }
            },
            _ => "that card may not be played now",
        }
    }

    // ----- the state machine (docs/M0-SPEC.md §2) -----

    fn emit_public(&mut self, e: Event) {
        self.events.push((Visibility::Public, e));
    }

    fn emit_private(&mut self, seat: u8, e: Event) {
        self.events.push((Visibility::Private(seat), e));
    }

    /// The seat holding this card, if any.
    fn holder(&self, c: Card) -> Option<u8> {
        (0..SEATS as u8).find(|&s| has(self.truth.hands[usize::from(s)], c))
    }

    /// Steps 1 to 5: deal, turn, and handle the dealer's forced takes.
    fn start_hand(&mut self) {
        let deck = if let Some(d) = self.decks.get(self.next_deck) {
            self.next_deck += 1;
            *d
        } else if let Some(HandDecks(f)) = &self.hand_decks {
            let d = f(self.hand_index);
            assert!(
                is_full_deck(&d),
                "the block's deck for hand {} must hold each of the 53 cards exactly once",
                self.hand_index
            );
            d
        } else {
            let mut d = identity_deck();
            self.rng.shuffle(&mut d);
            d
        };
        let dealer = self.dealer;
        self.hand = HandState::new(dealer);
        self.hand.start_scores = self.scores;
        self.hand.start_bonus = self.bonus_taken;
        self.truth = Truth {
            deck,
            hands: [0; SEATS],
            burns: [0; SEATS],
            undealt: 0,
        };
        self.emit_public(Event::HandStart {
            hand_index: self.hand_index,
            game_index: self.game_index,
            dealer,
        });

        // Step 1: two cards each, then three, starting left of the dealer.
        let order: [u8; SEATS] = core::array::from_fn(|i| (dealer + 1 + i as u8) % SEATS as u8);
        for (i, &seat) in order.iter().enumerate() {
            let mut h = bit(deck[2 * i]) | bit(deck[2 * i + 1]);
            for &c in &deck[16 + 3 * i..16 + 3 * i + 3] {
                h |= bit(c);
            }
            self.truth.hands[usize::from(seat)] = h;
        }
        self.truth.undealt = deck[TURNED_POS + 1..].iter().fold(0, |a, &c| a | bit(c));
        for &seat in &order {
            let cards = self.truth.hands[usize::from(seat)];
            self.emit_private(seat, Event::Dealt { seat, cards });
        }
        let turned = deck[TURNED_POS];
        self.hand.turned.push((turned, None));
        self.hand.table_card = Some(turned);
        self.emit_public(Event::Turned { card: turned });

        if turned == JOKER {
            // Step 2a: the dealer takes the Joker and burns before trumps are known.
            self.hand.joker_turned = true;
            self.take(dealer, turned);
            self.await_burn(dealer, AfterBurn::TurnSecond);
            return;
        }
        let trump = Suit::from_index(turned / 13);
        self.hand.trump = Some(trump);
        if rank_of(turned) == ACE {
            // Step 3: a turned ace is robbed by the dealer at once.
            self.take(dealer, turned);
            self.await_burn(dealer, AfterBurn::StartTricks);
            return;
        }
        // Step 4: the card stays face up; the ace holder (if dealt) robs at their first turn.
        self.hand.pending_robber = self.holder(ace_of_trumps(trump));
        self.start_tricks();
    }

    /// Steps 2b–2e: after the dealer's Joker burn, turn the next card.
    fn turn_second(&mut self) {
        let dealer = self.dealer;
        let card = self.truth.deck[TURNED_POS + 1];
        self.truth.undealt &= !bit(card);
        self.hand.turned.push((card, None));
        self.hand.table_card = Some(card);
        self.emit_public(Event::Turned { card });
        // There is only one Joker, and it was the first turned card.
        let trump = Suit::from_index(card / 13);
        self.hand.trump = Some(trump);
        let ace = ace_of_trumps(trump);
        if rank_of(card) == ACE {
            // Step 2c: a turned ace is robbed by the dealer at once.
            self.take(dealer, card);
            self.await_burn(dealer, AfterBurn::StartTricks);
            return;
        }
        // Step 2e: the card stays face up; whoever holds the ace of trumps, the dealer included,
        // robs at their own first turn in trick 1 (H2, H4 as amended on 2026-10-14).
        self.hand.pending_robber = self.holder(ace);
        self.start_tricks();
    }

    /// The seat takes the turned card face up on the table.
    fn take(&mut self, seat: u8, card: Card) {
        self.truth.hands[usize::from(seat)] |= bit(card);
        self.hand.table_card = None;
        if let Some(last) = self.hand.turned.last_mut() {
            last.1 = Some(seat);
        }
        self.hand.took[usize::from(seat)] = true;
        if card != JOKER {
            self.hand.robber = Some(seat);
        }
        self.emit_public(Event::Took { seat, card });
    }

    fn await_burn(&mut self, seat: u8, after: AfterBurn) {
        self.hand.after_burn = after;
        self.phase = Phase::AwaitBurn(seat);
    }

    fn burn(&mut self, seat: u8, card: Card) {
        self.truth.hands[usize::from(seat)] &= !bit(card);
        self.truth.burns[usize::from(seat)] |= bit(card);
        let kind = match self.hand.after_burn {
            AfterBurn::TurnSecond => BurnKind::JokerTake,
            AfterBurn::StartTricks => BurnKind::AceRobAtOnce,
            AfterBurn::PlayFirst => BurnKind::RobAtTurn,
        };
        self.hand.burn_order.push((seat, kind));
        self.emit_private(seat, Event::Burned { seat, card });
        self.emit_public(Event::BurnDone { seat });
        match self.hand.after_burn {
            AfterBurn::TurnSecond => self.turn_second(),
            AfterBurn::StartTricks => self.start_tricks(),
            AfterBurn::PlayFirst => self.phase = Phase::AwaitPlay(seat),
        }
    }

    /// Step 5: trick 1 is led by the seat left of the dealer.
    fn start_tricks(&mut self) {
        let leader = left_of(self.dealer);
        self.hand.trick_no = 1;
        self.hand.leader = leader;
        self.turn_to(leader);
    }

    /// Is a rob still possible in this hand (trick 1, nobody robbed, the turned card on the table)?
    fn rob_open(&self) -> bool {
        self.hand.trick_no == 1 && self.hand.robber.is_none() && self.hand.table_card.is_some()
    }

    /// Step 6: the turn passes to `seat`; the pending robber takes the turned card first.
    fn turn_to(&mut self, seat: u8) {
        if self.rob_open()
            && self.hand.pending_robber == Some(seat)
            && let Some(card) = self.hand.table_card
        {
            self.take(seat, card);
            self.await_burn(seat, AfterBurn::PlayFirst);
            return;
        }
        self.phase = Phase::AwaitPlay(seat);
    }

    /// Step 7: a play.
    fn play(&mut self, seat: u8, card: Card) {
        let s = usize::from(seat);
        if self.rob_open() && !self.hand.proved_no_ace[s] {
            // Step 6: playing a first card without robbing proves the seat lacks the ace.
            self.hand.proved_no_ace[s] = true;
            self.emit_public(Event::ProvedNoAce { seat });
        }
        self.truth.hands[s] &= !bit(card);
        self.hand.played |= bit(card);
        self.hand.trick.push((seat, card));
        self.emit_public(Event::Played { seat, card });
        if let Some(trump) = self.hand.trump {
            let rank = trump_rank(card, trump);
            let better = match self.hand.best_trump {
                None => true,
                Some((best, _)) => rank > trump_rank(best, trump),
            };
            if rank > 0 && better {
                self.hand.best_trump = Some((card, seat));
            }
            if card == five_of(trump) {
                self.hand.five_played = true;
            }
        }
        match self.next_in_trick(left_of(seat)) {
            Some(next) => self.turn_to(next),
            None => self.finish_trick(),
        }
    }

    /// The next seat to play to the current trick, starting from `from` and going left: the first
    /// seat still in play (rule 40: a turned partnership plays no more) before the play comes
    /// back round to the leader; `None` when every seat still in play has played (the trick is
    /// complete). With nobody turned this is `from` until the eighth card.
    fn next_in_trick(&self, from: u8) -> Option<u8> {
        let mut s = from;
        while s != self.hand.leader {
            if self.hand.in_play(s) {
                return Some(s);
            }
            s = left_of(s);
        }
        None
    }

    /// Step 8: every seat still in play has played to the trick (the eighth card, unless a
    /// partnership is turned).
    fn finish_trick(&mut self) {
        let Some(trump) = self.hand.trump else {
            return;
        };
        let n = self.hand.trick.len();
        let mut trick_cards = [0 as Card; SEATS];
        for (slot, &(_, c)) in trick_cards.iter_mut().zip(&self.hand.trick) {
            *slot = c;
        }
        let trick_cards = &trick_cards[..n];
        let winner = self.hand.trick[trick_winner(trick_cards, trump)].0;
        let team = team_of(winner);
        let t = usize::from(team);
        let trick_no = self.hand.trick_no;
        let five = five_of(trump);
        let five_here = trick_cards.contains(&five);
        let k = self.hand.trick_winners.len();
        let done = &mut self.hand.done[k];
        done.cards[..n].copy_from_slice(&self.hand.trick);
        done.n = n as u8;
        done.holder = team;
        self.hand.trick.clear();
        self.hand.trick_winners.push(winner);
        self.hand.tricks_won[t] += 1;
        self.scores[t] = self.scores[t].saturating_add(TRICK_POINTS);
        self.emit_public(Event::TrickWon {
            trick: trick_no,
            seat: winner,
            team,
            points: TRICK_POINTS,
        });
        let before_bonus = self.scores[t];
        if five_here {
            // Rule 29: the 5 of trumps' trick takes the bonus as soon as it is won.
            self.award_bonus(team, BonusKind::Immediate, five);
        }
        if self.scores[t] >= WINNING_SCORE {
            let reason = if five_here && before_bonus < WINNING_SCORE {
                GameEndReason::FortyOneByImmediateBonus
            } else {
                GameEndReason::FortyOneByTrick
            };
            self.end_game(team, reason);
            return;
        }
        // Nobody robbed and all eight seats played to trick 1, each proving it lacks the ace of
        // trumps. (A seat turned before its first card proved nothing: its pair may hold it.)
        if trick_no == 1 && self.hand.robber.is_none() && self.hand.proved_no_ace.iter().all(|&p| p)
        {
            debug_assert!(has(
                self.truth.undealt | self.truth.burns[usize::from(self.dealer)],
                ace_of_trumps(trump)
            ));
            self.hand.ace_undealt = true;
            self.emit_public(Event::AceUndealt);
        }
        if trick_no < TRICKS_PER_HAND {
            self.hand.trick_no += 1;
            self.hand.leader = winner;
            self.turn_to(winner);
        } else {
            self.end_of_hand();
        }
    }

    fn award_bonus(&mut self, team: u8, kind: BonusKind, card: Card) {
        let t = usize::from(team);
        let points = if self.bonus_taken[t] {
            BONUS_LATER
        } else {
            BONUS_FIRST
        };
        self.bonus_taken[t] = true;
        self.scores[t] = self.scores[t].saturating_add(points);
        self.emit_public(Event::Bonus {
            team,
            points,
            kind,
            card,
        });
    }

    /// Steps 9 and 10: after trick 5.
    fn end_of_hand(&mut self) {
        if let Some(team) =
            (0..TEAMS as u8).find(|&t| self.hand.tricks_won[usize::from(t)] == TRICKS_PER_HAND)
        {
            self.end_game(team, GameEndReason::Jink);
            return;
        }
        // Rule 30; rule 31: no trump played, no bonus (unreachable in a completed 8-player hand).
        if !self.hand.five_played
            && let Some((card, seat)) = self.hand.best_trump
        {
            let team = team_of(seat);
            self.award_bonus(team, BonusKind::EndOfHand, card);
            if self.scores[usize::from(team)] >= WINNING_SCORE {
                self.end_game(team, GameEndReason::FortyOneByEndBonus);
                return;
            }
        }
        self.emit_public(Event::HandEnd {
            hand_index: self.hand_index,
            aborted: false,
        });
        self.dealer = left_of(self.dealer);
        self.hand_index += 1;
        self.start_hand();
    }

    /// Step 11: a team has won the game.
    fn end_game(&mut self, team: u8, reason: GameEndReason) {
        let t = usize::from(team);
        self.emit_public(Event::GameWon {
            team,
            reason,
            scores: self.scores,
        });
        self.games_won[t] += 1;
        self.emit_public(Event::HandEnd {
            hand_index: self.hand_index,
            aborted: self.hand.trick_winners.len() < usize::from(TRICKS_PER_HAND),
        });
        if self.games_won[t] >= GAMES_TO_WIN {
            let lose = self.rules.stake.saturating_mul(-2);
            let mut money = [lose; TEAMS];
            money[t] = self.rules.stake.saturating_mul(6);
            self.money = Some(money);
            self.emit_public(Event::MatchWon { team, money });
            self.phase = Phase::MatchOver;
            return;
        }
        self.game_index += 1;
        self.scores = [0; TEAMS];
        self.bonus_taken = [false; TEAMS];
        self.dealer = left_of(self.dealer);
        self.hand_index += 1;
        self.start_hand();
    }
}

/// The seat that played the strongest card of a completed trick among the seats `in_play`
/// accepts, judged against the suit led (rule 39): the highest trump, else the highest card of
/// the suit led (rule 19's order); `None` when none of those cards is a trump or of the suit led.
/// `cards` are in play order, the led card first (it may belong to a turned partnership).
fn strongest_in_play(
    cards: &[(u8, Card)],
    trump: Suit,
    in_play: impl Fn(u8) -> bool,
) -> Option<u8> {
    let led = effective_suit(cards.first()?.1, trump);
    let mut best: Option<(u8, u8)> = None;
    for &(seat, c) in cards {
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
        if strength > 0 && best.is_none_or(|(b, _)| strength > b) {
            best = Some((strength, seat));
        }
    }
    best.map(|(_, seat)| seat)
}

/// Build a deck in deal order (docs/M0-SPEC.md §1) that deals the given cards, for hand-made
/// deals. `hands[s]` holds up to five cards for seat `s`; `pack` lists the cards placed from deck
/// position 40 on (the turned card, then the second turned card or the top of the undealt pack).
/// Every other card fills the hands (seats in deal order from left of the dealer, lowest card
/// index first) and then the rest of the pack (lowest index first).
pub fn build_deck(
    dealer: u8,
    hands: &[Bits; SEATS],
    pack: &[Card],
) -> Result<[Card; DECK_SIZE], &'static str> {
    if usize::from(dealer) >= SEATS {
        return Err("no such dealer seat");
    }
    if pack.len() > DECK_SIZE - TURNED_POS {
        return Err("at most 13 cards follow the dealt ones");
    }
    let mut used: Bits = 0;
    for &h in hands {
        if h & !ALL_CARDS != 0 {
            return Err("no such card");
        }
        if count(h) > HAND_SIZE {
            return Err("a hand holds at most five cards");
        }
        if h & used != 0 {
            return Err("a card is given twice");
        }
        used |= h;
    }
    for &c in pack {
        if usize::from(c) >= DECK_SIZE {
            return Err("no such card");
        }
        if has(used, c) {
            return Err("a card is given twice");
        }
        used |= bit(c);
    }
    let mut spare = ALL_CARDS & !used;
    let mut full = *hands;
    let mut deck = [0 as Card; DECK_SIZE];
    for i in 0..SEATS {
        let seat = (usize::from(dealer) + 1 + i) % SEATS;
        while count(full[seat]) < HAND_SIZE {
            let c = spare.trailing_zeros() as Card;
            full[seat] |= bit(c);
            spare &= spare - 1;
        }
        let mut cs = cards(full[seat]);
        for pos in [2 * i, 2 * i + 1, 16 + 3 * i, 17 + 3 * i, 18 + 3 * i] {
            deck[pos] = cs.next().unwrap_or(0);
        }
    }
    for (pos, c) in (TURNED_POS..).zip(pack.iter().copied().chain(cards(spare))) {
        deck[pos] = c;
    }
    Ok(deck)
}

/// A card in words, such as "the 5 of hearts" or "the Joker".
pub fn card_words(c: Card) -> String {
    match suit_of(c) {
        None => "the Joker".to_string(),
        Some(s) => {
            let rank = match rank_of(c) {
                ACE => "ace".to_string(),
                JACK => "jack".to_string(),
                QUEEN => "queen".to_string(),
                KING => "king".to_string(),
                n => n.to_string(),
            };
            format!("the {rank} of {}", suit_words(s))
        }
    }
}

fn suit_words(s: Suit) -> &'static str {
    match s {
        Suit::Clubs => "clubs",
        Suit::Diamonds => "diamonds",
        Suit::Hearts => "hearts",
        Suit::Spades => "spades",
    }
}

/// Team letter: A = seats 0 and 4, B = 1 and 5, C = 2 and 6, D = 3 and 7.
fn team_letter(team: u8) -> char {
    char::from(b'A' + team % 4)
}

fn score_words(scores: &[u8; TEAMS]) -> String {
    format!(
        "A {}, B {}, C {}, D {}",
        scores[0], scores[1], scores[2], scores[3]
    )
}

/// The match's public event log in plain English, one line per event, for owner review.
/// Hand and game numbers count from 1 here (the canonical strings count from 0).
pub fn transcript(m: &Match) -> String {
    render(m, false)
}

/// Like [`transcript`], but also showing every seat's dealt cards and burned cards (the private
/// events), for owner review of hand-made deals.
pub fn transcript_full(m: &Match) -> String {
    render(m, true)
}

fn render(m: &Match, include_private: bool) -> String {
    let mut out = String::new();
    let mut scores = [0u8; TEAMS];
    let mut game = 0u8;
    for (vis, ev) in m.events() {
        if *vis != Visibility::Public && !include_private {
            continue;
        }
        let line = match *ev {
            Event::MatchStart { first_dealer } => format!(
                "The match starts; seat {first_dealer} deals first. Partners: A = seats 0 and 4, \
                 B = seats 1 and 5, C = seats 2 and 6, D = seats 3 and 7."
            ),
            Event::HandStart {
                hand_index,
                game_index,
                dealer,
            } => {
                game = game_index;
                format!(
                    "Hand {} (game {}): seat {dealer} deals.",
                    hand_index + 1,
                    game_index + 1
                )
            }
            Event::Dealt { seat, cards } => {
                format!("Seat {seat} is dealt {}.", names(cards))
            }
            Event::Turned { card } => match suit_of(card) {
                None => "The Joker is turned: the dealer must take it.".to_string(),
                Some(s) => format!(
                    "{} is turned: {} are trumps.",
                    capitalise(&card_words(card)),
                    suit_words(s)
                ),
            },
            Event::Took { seat, card } => format!("Seat {seat} takes {}.", card_words(card)),
            Event::Burned { seat, card } => {
                format!(
                    "Seat {seat} burns {} (seen only by seat {seat}).",
                    card_words(card)
                )
            }
            Event::BurnDone { seat } => format!("Seat {seat} burns a card face down."),
            Event::Played { seat, card } => format!("Seat {seat} plays {}.", card_words(card)),
            Event::ProvedNoAce { seat } => {
                format!("Seat {seat} has not robbed, so does not hold the ace of trumps.")
            }
            Event::AceUndealt => {
                "Nobody robbed: the ace of trumps is out of play (undealt or burned).".to_string()
            }
            Event::TrickWon {
                trick,
                seat,
                team,
                points,
            } => {
                scores[usize::from(team % 4)] =
                    scores[usize::from(team % 4)].saturating_add(points);
                format!(
                    "Trick {trick} goes to seat {seat} (team {}) for {points} points. Scores: {}.",
                    team_letter(team),
                    score_words(&scores)
                )
            }
            Event::Bonus {
                team,
                points,
                kind,
                card,
            } => {
                scores[usize::from(team % 4)] =
                    scores[usize::from(team % 4)].saturating_add(points);
                let when = match kind {
                    BonusKind::Immediate => "at once, for the trick with",
                    BonusKind::EndOfHand => "at the end of the hand, for the trick with",
                };
                format!(
                    "Best-trump bonus: {points} points to team {} {when} {}. Scores: {}.",
                    team_letter(team),
                    card_words(card),
                    score_words(&scores)
                )
            }
            Event::GameWon {
                team,
                reason,
                scores: final_scores,
            } => {
                scores = [0; TEAMS];
                let how = match reason {
                    GameEndReason::FortyOneByTrick => "by reaching 41 with a trick",
                    GameEndReason::FortyOneByImmediateBonus => {
                        "by reaching 41 with the bonus for the 5 of trumps"
                    }
                    GameEndReason::FortyOneByEndBonus => {
                        "by reaching 41 with the end-of-hand bonus"
                    }
                    GameEndReason::Jink => "with a jink (all five tricks)",
                    GameEndReason::FortyOneByPassedTrick => {
                        "by reaching 41 with tricks passed from a turned partnership"
                    }
                };
                format!(
                    "Team {} wins game {} {how}. Final scores: {}.",
                    team_letter(team),
                    game + 1,
                    score_words(&final_scores)
                )
            }
            Event::HandEnd {
                hand_index,
                aborted,
            } => {
                if aborted {
                    format!("Hand {} stops here: the game is over.", hand_index + 1)
                } else {
                    format!("Hand {} is over.", hand_index + 1)
                }
            }
            Event::PairTurned { team, score } => {
                scores[usize::from(team % 4)] = score;
                format!(
                    "Team {} is turned: its cards are turned face down, it plays no more this \
                     hand and scores nothing for it (back to {score}). Scores: {}.",
                    team_letter(team),
                    score_words(&scores)
                )
            }
            Event::TrickPassed {
                trick,
                team,
                points,
            } => {
                scores[usize::from(team % 4)] =
                    scores[usize::from(team % 4)].saturating_add(points);
                format!(
                    "Trick {trick} passes to team {} for {points} points. Scores: {}.",
                    team_letter(team),
                    score_words(&scores)
                )
            }
            Event::MatchWon { team, money } => format!(
                "Team {} wins the match. Money: A {:+}, B {:+}, C {:+}, D {:+}.",
                team_letter(team),
                money[0],
                money[1],
                money[2],
                money[3]
            ),
        };
        out.push_str(&line);
        out.push('\n');
    }
    out
}

fn capitalise(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

mod hand_start;
pub use hand_start::{HandStartState, StateError, current_hand_start_of};

#[cfg(test)]
mod tests;
