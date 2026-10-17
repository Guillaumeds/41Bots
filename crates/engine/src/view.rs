//! What the players can see (docs/M0-SPEC.md §3).
//!
//! [`PublicState`] is everything every seat knows: the table, the scores, the cards played and
//! the hard facts announced so far. [`SeatView`] adds one seat's private knowledge: its hand, its
//! own burned cards and the events only it saw. Both are built by
//! [`Match::public`](crate::game::Match::public) and [`Match::view`](crate::game::Match::view)
//! from the engine's state and the visibility tags on its event log, so a seat's view can never
//! contain another seat's hidden cards.

use crate::cards::{Bits, Card, Suit};
use crate::game::{Event, Phase};

/// Which kind of burn a seat made (M3-SPEC §2.2). The order is the derived order used by the
/// belief crate's locations.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BurnKind {
    /// The dealer's burn after taking a turned Joker, made before trumps are known (rule 24): it
    /// may hold any card, even the card that turns out to be the ace of trumps (H3).
    JokerTake,
    /// The dealer's burn after robbing a turned ace at once (rules 22 and 24, H4): never the ace
    /// of trumps.
    AceRobAtOnce,
    /// A robber's burn at their own first turn in trick 1 (rule 20, H2): never the ace of trumps.
    RobAtTurn,
}

/// Everything every seat knows at this moment.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicState {
    /// Who must act and how, or that the match is over.
    pub phase: Phase,
    /// The seat to act, `None` once the match is over.
    pub to_act: Option<u8>,
    /// The dealer of the current hand.
    pub dealer: u8,
    /// Hands dealt so far in the match, counting from 0.
    pub hand_index: u32,
    /// Games started so far in the match, counting from 0.
    pub game_index: u8,
    /// Each team's score in the current game.
    pub scores: [u8; 4],
    /// Games each team has won in the match.
    pub games_won: [u8; 4],
    /// Whether each team has already taken a best-trump bonus in the current game.
    pub bonus_taken: [bool; 4],
    /// The trump suit; `None` only during the dealer's burn after taking a turned Joker.
    pub trump: Option<Suit>,
    /// The turned card currently face up on the table and not taken.
    pub table_card: Option<Card>,
    /// Every card turned this hand, in order, and the seat that took it (if any).
    pub turned: Vec<(Card, Option<u8>)>,
    /// Whether the first turned card this hand was the Joker.
    pub joker_turned: bool,
    /// Cards in each seat's hand (6 while that seat must burn).
    pub hand_sizes: [u8; 8],
    /// Cards each seat has burned this hand.
    pub burn_counts: [u8; 8],
    /// Every burn made this hand, in order: the seat and which kind of burn it was (M3-SPEC §2.2).
    /// The cards themselves stay hidden; this tells a dealer's pre-trump burn after a turned Joker
    /// from a rob burn.
    pub burn_order: Vec<(u8, BurnKind)>,
    /// Hidden, out-of-play cards: the undealt pack plus every burned card (always 12 or 13).
    pub hidden_count: u8,
    /// Every card played to a trick this hand.
    pub played: Bits,
    /// The trick being played: 0 before trick 1 starts (during the dealer's burns), else 1..=5.
    pub trick_no: u8,
    /// The seat that leads (or led) the current trick.
    pub leader: u8,
    /// The cards played to the current trick so far, in play order.
    pub trick_cards: Vec<(u8, Card)>,
    /// The seat that won each completed trick of this hand.
    pub trick_winners: Vec<u8>,
    /// Tricks each team holds this hand: the tricks it won, plus those passed to it from a turned
    /// partnership (rule 39); 0 for a turned partnership.
    pub tricks_won: [u8; 4],
    /// The highest trump played so far this hand, and who played it. Once a partnership is
    /// turned, only cards of partnerships still in play count (rule 39).
    pub best_trump: Option<(Card, u8)>,
    /// Whether the 5 of trumps has been played this hand (by a partnership still in play, once
    /// one is turned: a turned partnership's 5 no longer counts, rule 39).
    pub five_played: bool,
    /// The seat that robbed (took a turned card other than the Joker), so holds or held the ace of
    /// trumps or took the turned ace.
    pub robber: Option<u8>,
    /// Seats that have shown they do not hold the ace of trumps this hand.
    pub proved_no_ace: [bool; 8],
    /// Whether everyone has played to trick 1 without a rob, proving the ace of trumps out of play:
    /// undealt, or in the dealer's burn after a turned Joker (made before trumps were known, H3).
    pub ace_undealt: bool,
    /// Partnerships turned this hand (rules 39-41 and 34; the app turns them through
    /// [`Match::turn_team`](crate::game::Match::turn_team)): they play no more cards this hand and
    /// their cards no longer count. A turned seat keeps its unplayed cards face down (still in
    /// `hand_sizes`), and its cards in the trick in progress leave `trick_cards`. All `false` in
    /// a hand where nobody is turned (and then left out of the JSON form, which is unchanged).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "no_team_turned")
    )]
    pub turned_teams: [bool; 4],
}

/// Is no partnership turned? (The JSON form leaves `turned_teams` out then.)
#[cfg(feature = "serde")]
fn no_team_turned(t: &[bool; 4]) -> bool {
    !t.iter().any(|&x| x)
}

/// One seat's view: the public state plus that seat's private knowledge.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeatView {
    /// The seat this view belongs to.
    pub seat: u8,
    /// The seat's current hand.
    pub hand: Bits,
    /// The cards this seat has burned this hand.
    pub own_burns: Bits,
    /// Everything every seat knows.
    pub public: PublicState,
    /// Every public event and every event private to this seat, in order.
    pub events: Vec<Event>,
}

impl SeatView {
    /// This seat's events as canonical strings (docs/M0-SPEC.md §5), in order.
    pub fn canonical_events(&self) -> Vec<String> {
        self.events.iter().map(Event::canonical).collect()
    }
}
