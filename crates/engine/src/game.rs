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
    EffSuit, ace_of_trumps, effective_suit, five_of, legal_burns, legal_plays, trick_winner,
    trump_rank,
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
    /// The team won all five tricks of a hand (rule 33).
    Jink,
}

impl GameEndReason {
    /// The word used in canonical event strings.
    pub const fn canonical(self) -> &'static str {
        match self {
            GameEndReason::FortyOneByTrick => "trick",
            GameEndReason::FortyOneByImmediateBonus => "immediate_bonus",
            GameEndReason::FortyOneByEndBonus => "end_bonus",
            GameEndReason::Jink => "jink",
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
        }
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
        if self.hand.trick.len() < SEATS {
            self.turn_to(left_of(seat));
        } else {
            self.finish_trick();
        }
    }

    /// Step 8: the eighth card of a trick has been played.
    fn finish_trick(&mut self) {
        let Some(trump) = self.hand.trump else {
            return;
        };
        let mut trick_cards = [0 as Card; SEATS];
        for (slot, &(_, c)) in trick_cards.iter_mut().zip(&self.hand.trick) {
            *slot = c;
        }
        let winner = self.hand.trick[trick_winner(&trick_cards, trump)].0;
        let team = team_of(winner);
        let t = usize::from(team);
        let trick_no = self.hand.trick_no;
        let five = five_of(trump);
        let five_here = trick_cards.contains(&five);
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
        if trick_no == 1 && self.hand.robber.is_none() {
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
