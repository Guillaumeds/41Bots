//! The match: hands, games and the money, following docs/M0-SPEC.md section 2
//! step by step, with the canonical event strings of section 5.
//!
//! The reference does not shuffle: each hand's deck is handed in from outside
//! with [`Match::start_hand`].

use crate::card::{Card, Suit};
use crate::order;

mod referee;
#[cfg(test)]
mod tests;

pub use referee::Lapse;
use referee::{Renege, standing_winner};

/// Points for a team's first best-trump bonus in a game (6: an 11-point trick).
const FIRST_BONUS: u8 = 6;
/// Points for every later bonus in the same game (house rule: 5, a 10-point trick).
const LATER_BONUS: u8 = 5;

/// Match settings (M0-SPEC section 1): only the stake varies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rules {
    /// The stake each player of a losing team pays.
    pub stake: i32,
}

impl Default for Rules {
    fn default() -> Self {
        Rules { stake: 5 }
    }
}

/// A player's action, carrying a shared card index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Burn(u8),
    Play(u8),
}

impl Action {
    pub fn card(self) -> u8 {
        match self {
            Action::Burn(c) | Action::Play(c) => c,
        }
    }
}

/// What the match is waiting for. `NeedsDeck` has no engine counterpart: the
/// reference waits for the next deck between hands (and before the first).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    AwaitBurn(u8),
    AwaitPlay(u8),
    MatchOver,
    NeedsDeck,
}

/// The public state, with the same fields as the engine's `PublicState`
/// (M0-SPEC section 3): card sets are sorted index lists, the trump suit is a
/// suit index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Public {
    pub phase: Phase,
    pub to_act: Option<u8>,
    pub dealer: u8,
    pub hand_index: u32,
    pub game_index: u8,
    pub scores: [u8; 4],
    pub games_won: [u8; 4],
    pub bonus_taken: [bool; 4],
    /// None only during the dealer's burn after taking a turned Joker.
    pub trump: Option<u8>,
    /// The turned card currently face up on the table and not taken.
    pub table_card: Option<u8>,
    /// Every turned card this hand, and who took it.
    pub turned: Vec<(u8, Option<u8>)>,
    pub joker_turned: bool,
    pub hand_sizes: [u8; 8],
    pub burn_counts: [u8; 8],
    /// Every burn of this hand in order: the seat and the kind of burn (`joker_take`: the
    /// dealer's burn after taking a turned Joker; `ace_rob_at_once`: the dealer's burn after
    /// robbing a turned ace; `rob_at_turn`: a robber's burn at their first turn in trick 1).
    pub burn_order: Vec<(u8, &'static str)>,
    /// Undealt cards plus burned cards.
    pub hidden_count: u8,
    /// Every card played to a trick this hand, sorted.
    pub played: Vec<u8>,
    /// 0 before trick 1 is set up, else 1..=5.
    pub trick_no: u8,
    pub leader: u8,
    /// The current trick so far, `(seat, card)` in play order.
    pub trick_cards: Vec<(u8, u8)>,
    /// The seat that won each finished trick of this hand.
    pub trick_winners: Vec<u8>,
    pub tricks_won: [u8; 4],
    /// The best trump played so far this hand, `(card, seat)`.
    pub best_trump: Option<(u8, u8)>,
    pub five_played: bool,
    /// The seat that robbed (took the trump-deciding turned card), once known.
    pub robber: Option<u8>,
    pub proved_no_ace: [bool; 8],
    pub ace_undealt: bool,
    /// The partnerships turned this hand (rules 39-41 and 34).
    pub turned_teams: [bool; 4],
}

/// Internal: what we are waiting for, and from whom.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Waiting {
    Deck,
    /// Step 2a: the dealer has taken a turned Joker and must burn.
    DealerJokerBurn,
    /// Steps 2c, 2d and 3: the dealer has robbed at once and must burn.
    DealerRobBurn,
    /// Step 6: this seat has robbed at its first turn in trick 1 and must burn.
    RobberBurn(u8),
    /// Step 7: this seat must play.
    Play(u8),
    MatchOver,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GameEnd {
    Trick,
    ImmediateBonus,
    EndBonus,
    Jink,
    /// Tricks passed from a turned partnership took a team to 41 (rules 32 and 39).
    PassedTrick,
}

impl GameEnd {
    fn word(self) -> &'static str {
        match self {
            GameEnd::Trick => "trick",
            GameEnd::ImmediateBonus => "immediate_bonus",
            GameEnd::EndBonus => "end_bonus",
            GameEnd::Jink => "jink",
            GameEnd::PassedTrick => "passed_trick",
        }
    }
}

fn team_of(seat: u8) -> usize {
    (seat % 4) as usize
}

fn left_of(seat: u8) -> u8 {
    (seat + 1) % 8
}

fn sorted_indices(cards: &[Card]) -> Vec<u8> {
    let mut v: Vec<u8> = cards.iter().map(|c| c.index()).collect();
    v.sort();
    v
}

fn names_joined(cards: &[Card]) -> String {
    let mut sorted = cards.to_vec();
    sorted.sort_by_key(|c| c.index());
    let names: Vec<String> = sorted.iter().map(|c| c.name()).collect();
    names.join(",")
}

fn four_joined<T: std::fmt::Display>(values: &[T; 4]) -> String {
    format!("{},{},{},{}", values[0], values[1], values[2], values[3])
}

/// A whole match, played until one partnership has won two games.
#[derive(Clone, Debug)]
pub struct Match {
    rules: Rules,
    waiting: Waiting,

    // Match and game state.
    dealer: u8,
    hand_index: u32,
    game_index: u8,
    scores: [u8; 4],
    games_won: [u8; 4],
    bonus_taken: [bool; 4],
    money: Option<[i32; 4]>,

    // Hand state.
    deck: Vec<Card>,
    hands: [Vec<Card>; 8],
    burns: [Vec<Card>; 8],
    /// Every burn of this hand in order: the seat and why it burned, as a word.
    burn_log: Vec<(u8, &'static str)>,
    undealt: Vec<Card>,
    trump: Option<Suit>,
    table_card: Option<Card>,
    turned: Vec<(Card, Option<u8>)>,
    joker_turned: bool,
    /// The seat holding the ace of trumps that has still to rob (hidden information).
    pending_robber: Option<u8>,
    robber: Option<u8>,
    took_a_card: [bool; 8],
    played: Vec<Card>,
    trick_no: u8,
    leader: u8,
    trick: Vec<(u8, Card)>,
    /// Finished tricks of this hand: the cards in play order and the winning seat.
    finished_tricks: Vec<(Vec<(u8, Card)>, u8)>,
    tricks_won: [u8; 4],
    best_trump: Option<(Card, u8)>,
    five_played: bool,
    proved_no_ace: [bool; 8],
    ace_undealt: bool,

    // Turned partnerships (rules 34 and 37-41, the app's referee).
    /// Which partnerships are turned this hand.
    teams_out: [bool; 4],
    /// The partnership holding each finished trick now (its winner's, or the one it passed to);
    /// `None` for a trick nobody still in play played in.
    holders: Vec<Option<usize>>,
    /// The scores and bonus flags when this hand started (rule 41 restores a turned pair's).
    start_scores: [u8; 4],
    start_bonus: [bool; 4],
    /// The reneges the app let through this hand, in order, and whether each is caught.
    reneges: Vec<Renege>,

    /// Every event as a canonical string; `Some(seat)` marks a private event.
    events: Vec<(Option<u8>, String)>,
}

impl Match {
    /// A new match. The first hand needs a deck (`start_hand`).
    pub fn new(first_dealer: u8, rules: Rules) -> Match {
        assert!(first_dealer < 8, "the first dealer must be a seat 0-7");
        let mut m = Match {
            rules,
            waiting: Waiting::Deck,
            dealer: first_dealer,
            hand_index: 0,
            game_index: 0,
            scores: [0; 4],
            games_won: [0; 4],
            bonus_taken: [false; 4],
            money: None,
            deck: Vec::new(),
            hands: Default::default(),
            burns: Default::default(),
            burn_log: Vec::new(),
            undealt: Vec::new(),
            trump: None,
            table_card: None,
            turned: Vec::new(),
            joker_turned: false,
            pending_robber: None,
            robber: None,
            took_a_card: [false; 8],
            played: Vec::new(),
            trick_no: 0,
            leader: left_of(first_dealer),
            trick: Vec::new(),
            finished_tricks: Vec::new(),
            tricks_won: [0; 4],
            best_trump: None,
            five_played: false,
            proved_no_ace: [false; 8],
            ace_undealt: false,
            teams_out: [false; 4],
            holders: Vec::new(),
            start_scores: [0; 4],
            start_bonus: [false; 4],
            reneges: Vec::new(),
            events: Vec::new(),
        };
        m.emit_public(format!("MatchStart first_dealer={first_dealer}"));
        m
    }

    // ----------------------------------------------------------------------
    // Queries
    // ----------------------------------------------------------------------

    pub fn rules(&self) -> Rules {
        self.rules
    }

    /// True between hands and before the first hand.
    pub fn needs_deck(&self) -> bool {
        self.waiting == Waiting::Deck
    }

    pub fn match_over(&self) -> bool {
        self.waiting == Waiting::MatchOver
    }

    pub fn phase(&self) -> Phase {
        match self.waiting {
            Waiting::Deck => Phase::NeedsDeck,
            Waiting::DealerJokerBurn | Waiting::DealerRobBurn => Phase::AwaitBurn(self.dealer),
            Waiting::RobberBurn(seat) => Phase::AwaitBurn(seat),
            Waiting::Play(seat) => Phase::AwaitPlay(seat),
            Waiting::MatchOver => Phase::MatchOver,
        }
    }

    pub fn to_act(&self) -> Option<u8> {
        match self.phase() {
            Phase::AwaitBurn(seat) | Phase::AwaitPlay(seat) => Some(seat),
            Phase::MatchOver | Phase::NeedsDeck => None,
        }
    }

    pub fn dealer(&self) -> u8 {
        self.dealer
    }

    pub fn hand_index(&self) -> u32 {
        self.hand_index
    }

    pub fn game_index(&self) -> u8 {
        self.game_index
    }

    pub fn scores(&self) -> [u8; 4] {
        self.scores
    }

    pub fn games_won(&self) -> [u8; 4] {
        self.games_won
    }

    pub fn bonus_taken(&self) -> [bool; 4] {
        self.bonus_taken
    }

    /// Each team's money once the match is over (winners `+6 x stake`, losers `-2 x stake`).
    pub fn money(&self) -> Option<[i32; 4]> {
        self.money
    }

    /// A seat's cards, as sorted indices.
    pub fn hand(&self, seat: u8) -> Vec<u8> {
        sorted_indices(&self.hands[seat as usize])
    }

    /// The cards a seat has burned this hand, as sorted indices.
    pub fn burns(&self, seat: u8) -> Vec<u8> {
        sorted_indices(&self.burns[seat as usize])
    }

    /// The undealt pack (still face down), as sorted indices.
    pub fn undealt(&self) -> Vec<u8> {
        sorted_indices(&self.undealt)
    }

    /// The canonical event strings visible to `seat`, in order: every public
    /// event plus that seat's private ones.
    pub fn events_for(&self, seat: u8) -> Vec<String> {
        let mut out = Vec::new();
        for (visibility, text) in &self.events {
            match visibility {
                None => out.push(text.clone()),
                Some(owner) if *owner == seat => out.push(text.clone()),
                Some(_) => {}
            }
        }
        out
    }

    /// The legal actions, sorted by card index. Empty while a deck is needed
    /// and once the match is over.
    pub fn legal_actions(&self) -> Vec<Action> {
        let mut actions = match self.waiting {
            Waiting::Deck | Waiting::MatchOver => Vec::new(),
            Waiting::DealerJokerBurn | Waiting::DealerRobBurn => {
                let hand = &self.hands[self.dealer as usize];
                order::legal_burns(hand, self.trump)
                    .into_iter()
                    .map(|c| Action::Burn(c.index()))
                    .collect()
            }
            Waiting::RobberBurn(seat) => {
                let hand = &self.hands[seat as usize];
                order::legal_burns(hand, self.trump)
                    .into_iter()
                    .map(|c| Action::Burn(c.index()))
                    .collect()
            }
            Waiting::Play(seat) => {
                let hand = &self.hands[seat as usize];
                let led = self.trick.first().map(|(_, card)| *card);
                let trump = self.trump.expect("trumps are known during play");
                order::legal_plays(hand, led, trump)
                    .into_iter()
                    .map(|c| Action::Play(c.index()))
                    .collect()
            }
        };
        actions.sort_by_key(|a| a.card());
        actions
    }

    pub fn public(&self) -> Public {
        let mut hand_sizes = [0u8; 8];
        let mut burn_counts = [0u8; 8];
        let mut burned_total = 0usize;
        for seat in 0..8 {
            hand_sizes[seat] = self.hands[seat].len() as u8;
            burn_counts[seat] = self.burns[seat].len() as u8;
            burned_total += self.burns[seat].len();
        }
        Public {
            phase: self.phase(),
            to_act: self.to_act(),
            dealer: self.dealer,
            hand_index: self.hand_index,
            game_index: self.game_index,
            scores: self.scores,
            games_won: self.games_won,
            bonus_taken: self.bonus_taken,
            trump: self.trump.map(|s| s.index()),
            table_card: self.table_card.map(|c| c.index()),
            turned: self
                .turned
                .iter()
                .map(|(card, who)| (card.index(), *who))
                .collect(),
            joker_turned: self.joker_turned,
            hand_sizes,
            burn_counts,
            burn_order: self.burn_log.clone(),
            hidden_count: (self.undealt.len() + burned_total) as u8,
            played: sorted_indices(&self.played),
            trick_no: self.trick_no,
            leader: self.leader,
            // Rule 40: a turned player's card no longer counts in the trick being played.
            trick_cards: self
                .trick
                .iter()
                .filter(|(seat, _)| !self.teams_out[team_of(*seat)])
                .map(|(seat, card)| (*seat, card.index()))
                .collect(),
            trick_winners: self.finished_tricks.iter().map(|(_, w)| *w).collect(),
            tricks_won: self.tricks_won,
            best_trump: self.best_trump.map(|(card, seat)| (card.index(), seat)),
            five_played: self.five_played,
            robber: self.robber,
            proved_no_ace: self.proved_no_ace,
            ace_undealt: self.ace_undealt,
            turned_teams: self.teams_out,
        }
    }

    // ----------------------------------------------------------------------
    // Events
    // ----------------------------------------------------------------------

    fn emit_public(&mut self, text: String) {
        self.events.push((None, text));
    }

    fn emit_private(&mut self, seat: u8, text: String) {
        self.events.push((Some(seat), text));
    }

    // ----------------------------------------------------------------------
    // Steps 1-5: the deal and the turned card
    // ----------------------------------------------------------------------

    /// Deals a hand from `deck` (53 card indices in order) and runs steps 1-5.
    /// Panics if no deck is needed or the deck is not a permutation of 0..53.
    pub fn start_hand(&mut self, deck: &[u8; 53]) {
        assert!(
            self.waiting == Waiting::Deck,
            "start_hand called while no deck is needed"
        );
        let mut seen = [false; 53];
        let mut cards = Vec::new();
        for &i in deck {
            let card = Card::from_index(i).expect("deck holds an index above 52");
            assert!(!seen[i as usize], "deck holds card {i} twice");
            seen[i as usize] = true;
            cards.push(card);
        }

        self.reset_hand(cards);
        let d = self.dealer;

        // Step 1: the deal.
        self.emit_public(format!(
            "HandStart hand={} game={} dealer={}",
            self.hand_index, self.game_index, d
        ));
        let mut next = 0usize;
        // Two cards each, starting left of the dealer...
        for i in 0..8u8 {
            let seat = (d + 1 + i) % 8;
            for _ in 0..2 {
                let card = self.deck[next];
                self.hands[seat as usize].push(card);
                next += 1;
            }
        }
        // ...then three cards each in the same order.
        for i in 0..8u8 {
            let seat = (d + 1 + i) % 8;
            for _ in 0..3 {
                let card = self.deck[next];
                self.hands[seat as usize].push(card);
                next += 1;
            }
        }
        assert_eq!(next, 40);
        for i in 0..8u8 {
            let seat = (d + 1 + i) % 8;
            let text = format!(
                "Dealt seat={} cards={}",
                seat,
                names_joined(&self.hands[seat as usize])
            );
            self.emit_private(seat, text);
        }
        // Everything after position 40 stays face down in the pack for now.
        self.undealt = self.deck[41..53].to_vec();
        let first_turned = self.deck[40];
        self.turn_up(first_turned);

        if first_turned == Card::Joker {
            // Step 2a: the dealer takes the Joker and burns before trumps are known.
            self.joker_turned = true;
            self.trump = None;
            self.take_table_card(d, false);
            self.waiting = Waiting::DealerJokerBurn;
        } else if first_turned.is_ace() {
            // Step 3: a turned ace is robbed by the dealer at once.
            self.trump = first_turned.suit();
            self.take_table_card(d, true);
            self.waiting = Waiting::DealerRobBurn;
        } else {
            // Step 4: an ordinary turned card stays on the table.
            self.trump = first_turned.suit();
            self.pending_robber = self.holder_of_ace_of_trumps();
            self.start_trick_one();
        }
    }

    fn reset_hand(&mut self, deck: Vec<Card>) {
        self.deck = deck;
        self.hands = Default::default();
        self.burns = Default::default();
        self.burn_log = Vec::new();
        self.undealt = Vec::new();
        self.trump = None;
        self.table_card = None;
        self.turned = Vec::new();
        self.joker_turned = false;
        self.pending_robber = None;
        self.robber = None;
        self.took_a_card = [false; 8];
        self.played = Vec::new();
        self.trick_no = 0;
        self.leader = left_of(self.dealer);
        self.trick = Vec::new();
        self.finished_tricks = Vec::new();
        self.tricks_won = [0; 4];
        self.best_trump = None;
        self.five_played = false;
        self.proved_no_ace = [false; 8];
        self.ace_undealt = false;
        self.teams_out = [false; 4];
        self.holders = Vec::new();
        self.reneges = Vec::new();
        self.start_scores = self.scores;
        self.start_bonus = self.bonus_taken;
    }

    /// Turns a card face up on the table.
    fn turn_up(&mut self, card: Card) {
        self.turned.push((card, None));
        self.table_card = Some(card);
        self.emit_public(format!("Turned card={}", card.name()));
    }

    /// `seat` takes the face-up turned card into their hand. `is_rob` is false
    /// only for the dealer taking a turned Joker.
    fn take_table_card(&mut self, seat: u8, is_rob: bool) {
        let card = self
            .table_card
            .take()
            .expect("a turned card is on the table");
        self.hands[seat as usize].push(card);
        let last = self.turned.len() - 1;
        self.turned[last].1 = Some(seat);
        self.took_a_card[seat as usize] = true;
        if is_rob {
            self.robber = Some(seat);
        }
        self.emit_public(format!("Took seat={} card={}", seat, card.name()));
    }

    fn holder_of_ace_of_trumps(&self) -> Option<u8> {
        let ace = order::ace_of_trumps(self.trump.expect("trumps are known"));
        (0..8u8).find(|&seat| self.hands[seat as usize].contains(&ace))
    }

    /// Step 5 (rule 40: the first player still in play on the dealer's left leads).
    fn start_trick_one(&mut self) {
        self.trick_no = 1;
        let mut leader = left_of(self.dealer);
        while self.teams_out[team_of(leader)] {
            leader = left_of(leader);
        }
        self.leader = leader;
        self.give_turn(leader);
    }

    /// Step 6, first half: the pending robber robs at their own first turn in
    /// trick 1 (H2), before playing.
    fn give_turn(&mut self, seat: u8) {
        let robbing = self.trick_no == 1
            && self.robber.is_none()
            && self.table_card.is_some()
            && self.pending_robber == Some(seat);
        if robbing {
            self.take_table_card(seat, true);
            self.waiting = Waiting::RobberBurn(seat);
        } else {
            self.waiting = Waiting::Play(seat);
        }
    }

    // ----------------------------------------------------------------------
    // Actions
    // ----------------------------------------------------------------------

    /// Applies an action. Illegal actions are rejected and change nothing.
    pub fn apply(&mut self, action: Action) -> Result<(), String> {
        match self.waiting {
            Waiting::Deck => {
                return Err(format!(
                    "{action:?} rejected: no hand in progress (a deck is needed)"
                ));
            }
            Waiting::MatchOver => {
                return Err(format!("{action:?} rejected: the match is over"));
            }
            _ => {}
        }
        let legal = self.legal_actions();
        if !legal.contains(&action) {
            return Err(format!(
                "{action:?} rejected in phase {:?}: {}",
                self.phase(),
                self.why_illegal(action)
            ));
        }
        let card = Card::from_index(action.card()).expect("legal actions hold real cards");
        match (self.waiting, action) {
            (Waiting::DealerJokerBurn, Action::Burn(_)) => {
                self.burn_log.push((self.dealer, "joker_take"));
                self.burn(self.dealer, card);
                self.after_joker_burn();
            }
            (Waiting::DealerRobBurn, Action::Burn(_)) => {
                self.burn_log.push((self.dealer, "ace_rob_at_once"));
                self.burn(self.dealer, card);
                self.start_trick_one();
            }
            (Waiting::RobberBurn(seat), Action::Burn(_)) => {
                self.burn_log.push((seat, "rob_at_turn"));
                self.burn(seat, card);
                self.waiting = Waiting::Play(seat);
            }
            (Waiting::Play(seat), Action::Play(_)) => {
                if self.reneges.iter().any(|r| !r.caught) {
                    // The card may prove a renege let through: an open question may refuse it.
                    let mut copy = self.clone();
                    if let Err(e) = copy.play(seat, card) {
                        return Err(format!("{action:?} rejected: {e}"));
                    }
                    *self = copy;
                } else {
                    self.play(seat, card)
                        .expect("nothing is turned by a play without a renege about");
                }
            }
            _ => return Err(format!("{action:?} rejected: wrong kind of action")),
        }
        Ok(())
    }

    fn why_illegal(&self, action: Action) -> &'static str {
        let seat = match self.to_act() {
            Some(seat) => seat,
            None => return "nobody is to act",
        };
        let in_hand = Card::from_index(action.card())
            .map(|c| self.hands[seat as usize].contains(&c))
            .unwrap_or(false);
        match (self.phase(), action) {
            (Phase::AwaitBurn(_), Action::Play(_)) => "a burn is required first",
            (Phase::AwaitPlay(_), Action::Burn(_)) => "no burn is due now",
            _ if !in_hand => "the card is not in the hand of the seat to act",
            (Phase::AwaitBurn(_), Action::Burn(_)) => {
                "the ace of trumps may not be burned (rule 21)"
            }
            _ => "the card does not follow suit or the trump-lead rules (rules 14-18)",
        }
    }

    fn burn(&mut self, seat: u8, card: Card) {
        let hand = &mut self.hands[seat as usize];
        let pos = hand
            .iter()
            .position(|c| *c == card)
            .expect("burned card is in hand");
        hand.remove(pos);
        self.burns[seat as usize].push(card);
        self.emit_private(seat, format!("Burned seat={} card={}", seat, card.name()));
        self.emit_public(format!("BurnDone seat={seat}"));
    }

    /// Steps 2b-2e: after the dealer burns for a turned Joker.
    fn after_joker_burn(&mut self) {
        let d = self.dealer;
        // 2b: the next card of the pack is turned for trumps.
        let second = self.undealt.remove(0);
        assert_eq!(second, self.deck[41]);
        self.turn_up(second);
        let trump = second.suit().expect("there is only one Joker");
        self.trump = Some(trump);

        if second.is_ace() && !self.teams_out[team_of(d)] {
            // 2c: a turned ace is robbed by the dealer at once. (A dealer whose partnership was
            // turned for leaving the Joker takes nothing: the ace stays on the pack.)
            self.take_table_card(d, true);
            self.waiting = Waiting::DealerRobBurn;
        } else {
            // 2e: the card stays up; whoever holds the ace robs at their own turn in
            // trick 1, the dealer included (H4 as amended on 2026-10-14: a dealer holding
            // the ace of trumps no longer robs at once).
            self.pending_robber = self.holder_of_ace_of_trumps();
            self.start_trick_one();
        }
    }

    /// Steps 6 (second half), 7 and 8. Refused (`Err`; the caller drops this copy of the match)
    /// only in the open case: a renege the card proves turns the partnership that led the trick
    /// being played.
    fn play(&mut self, seat: u8, card: Card) -> Result<(), String> {
        self.put_down(seat, card);
        if !self.reneges.is_empty() && self.check_reneges(seat, card) {
            return Ok(());
        }
        self.move_on(left_of(seat))
    }

    /// Steps 6 (second half) and 7: the card goes on the table.
    fn put_down(&mut self, seat: u8, card: Card) {
        let trump = self.trump.expect("trumps are known during play");

        // Step 6: playing a first card of trick 1 without robbing proves the
        // seat lacks the ace of trumps (as far as the table can tell).
        if self.trick_no == 1
            && self.robber.is_none()
            && self.table_card.is_some()
            && !self.proved_no_ace[seat as usize]
        {
            self.proved_no_ace[seat as usize] = true;
            self.emit_public(format!("ProvedNoAce seat={seat}"));
        }

        // Step 7.
        let hand = &mut self.hands[seat as usize];
        let pos = hand
            .iter()
            .position(|c| *c == card)
            .expect("played card is in hand");
        hand.remove(pos);
        self.played.push(card);
        self.trick.push((seat, card));
        self.emit_public(format!("Played seat={} card={}", seat, card.name()));

        if order::is_trump(card, trump) {
            let better = match self.best_trump {
                None => true,
                Some((best, _)) => order::trump_beats(card, best, trump),
            };
            if better {
                self.best_trump = Some((card, seat));
            }
        }
        if card == order::five_of_trumps(trump) {
            self.five_played = true;
        }
    }

    /// The play moves on from seat `from`: to the first player on the left (from `from` itself)
    /// still in play who has not played to this trick, or the trick is complete. A trick not
    /// yet led is led by the first player still in play from `from` (rule 40). With every
    /// partnership turned the trick, if begun, goes to its best card for nothing, and the hand
    /// ends.
    fn move_on(&mut self, from: u8) -> Result<(), String> {
        let everyone_out = self.teams_out.iter().all(|t| *t);
        if self.trick.is_empty() {
            if everyone_out {
                self.end_of_hand();
            } else {
                let leader = self.first_in_play_from(from);
                self.leader = leader;
                self.give_turn(leader);
            }
            return Ok(());
        }
        if everyone_out {
            self.finish_trick();
            return Ok(());
        }
        if self.teams_out[team_of(self.trick[0].0)] {
            return Err(
                "led_trick_in_progress: the partnership that led this trick is turned \
                 (open question for the owner)"
                    .to_string(),
            );
        }
        let mut seat = from;
        for _ in 0..8 {
            let played = self.trick.iter().any(|(s, _)| *s == seat);
            if !played && !self.teams_out[team_of(seat)] {
                self.give_turn(seat);
                return Ok(());
            }
            seat = left_of(seat);
        }
        self.finish_trick();
        Ok(())
    }

    /// The first seat at or to the left of `seat` whose partnership is not turned.
    fn first_in_play_from(&self, seat: u8) -> u8 {
        let mut s = seat;
        while self.teams_out[team_of(s)] {
            s = left_of(s);
        }
        s
    }

    /// Step 8: every player still in play has played to the trick.
    fn finish_trick(&mut self) {
        let trump = self.trump.expect("trumps are known during play");
        let cards = std::mem::take(&mut self.trick);
        let plain_winner = |cards: &[(u8, Card)]| {
            let seat = order::trick_winner(cards, trump);
            cards
                .iter()
                .position(|(s, _)| *s == seat)
                .expect("a seat of the trick")
        };
        // Rule 40: once a partnership is turned, only the cards of the others count; with none
        // of theirs in it (everyone turned), the best card takes it for nothing.
        let (at, counts) = if self.teams_out.iter().any(|t| *t) {
            match standing_winner(&cards, trump, &self.teams_out) {
                Some(i) => (i, true),
                None => (plain_winner(&cards), false),
            }
        } else {
            (plain_winner(&cards), true)
        };
        let winner = cards[at].0;
        let team = team_of(winner);
        let five = order::five_of_trumps(trump);
        let five_here = cards[at].1 == five;
        let five_points = match (five_here, self.bonus_taken[team]) {
            (false, _) => 0,
            (true, false) => FIRST_BONUS,
            (true, true) => LATER_BONUS,
        };
        self.holders.push(if counts { Some(team) } else { None });
        self.finished_tricks.push((cards, winner));

        if !counts || self.claim_caught(team, 5 + five_points) {
            // Nobody in play played to it, or its winner would claim the game with a hidden rob
            // without the ace or burned ace and is turned instead (rule 38): no points.
            if counts {
                self.tricks_won[team] += 1;
            }
            self.emit_public(format!(
                "TrickWon trick={} seat={} team={} points=0",
                self.trick_no, winner, team
            ));
            if counts && self.catch_aces(Some(team), "claim") {
                return;
            }
            self.after_trick(winner);
            return;
        }

        self.emit_public(format!(
            "TrickWon trick={} seat={} team={} points=5",
            self.trick_no, winner, team
        ));
        self.tricks_won[team] += 1;
        self.scores[team] += 5;
        let score_before_bonus = self.scores[team];

        // Rule 29: the trick holding the 5 of trumps takes the bonus as soon as it is won.
        if five_here {
            self.award_bonus(team, five, "immediate");
        }

        // Rule 32: the game ends the moment a team reaches 41.
        if self.scores[team] >= 41 {
            let reason = if score_before_bonus < 41 && five_here {
                GameEnd::ImmediateBonus
            } else {
                GameEnd::Trick
            };
            self.end_game(team, reason);
            return;
        }
        self.after_trick(winner);
    }

    /// After a trick, when the game goes on: the ace-of-trumps fact of trick 1, then the next
    /// trick (led by its winner, or by the first player on its left still in play) or the end
    /// of the hand.
    fn after_trick(&mut self, winner: u8) {
        let trump = self.trump.expect("trumps are known during play");
        // After trick 1, nobody robbing means the ace of trumps is out of play: undealt, or
        // burned by the dealer after a turned Joker, before trumps were known (nothing is
        // forbidden then, H3). Only once all eight have played to trick 1 without robbing
        // (RULES, "Ace-of-trumps facts"): a player turned before its first card proved nothing.
        if self.trick_no == 1 && self.robber.is_none() && self.proved_no_ace.iter().all(|p| *p) {
            let ace = order::ace_of_trumps(trump);
            let dealer = self.dealer;
            // (A play without robbing let through hides the ace in a hand: the table cannot
            // tell.)
            assert!(
                !self.reneges.is_empty()
                    || self.undealt.contains(&ace)
                    || self.burns[dealer as usize].contains(&ace),
                "unrobbed ace must be undealt or in the dealer's burn"
            );
            self.ace_undealt = true;
            self.emit_public("AceUndealt".to_string());
        }

        if self.teams_out.iter().all(|t| *t) {
            self.end_of_hand();
        } else if self.trick_no < 5 {
            self.trick_no += 1;
            let leader = self.first_in_play_from(winner);
            self.leader = leader;
            self.give_turn(leader);
        } else {
            self.end_of_hand();
        }
    }

    fn award_bonus(&mut self, team: usize, card: Card, kind: &str) {
        let points = if self.bonus_taken[team] {
            LATER_BONUS
        } else {
            FIRST_BONUS
        };
        self.bonus_taken[team] = true;
        self.scores[team] += points;
        self.emit_public(format!(
            "Bonus team={} points={} kind={} card={}",
            team,
            points,
            kind,
            card.name()
        ));
    }

    /// Step 9: after trick 5 (or once every partnership is turned).
    fn end_of_hand(&mut self) {
        if self.teams_out.iter().any(|t| *t) || !self.reneges.is_empty() {
            self.end_of_refereed_hand();
            return;
        }
        // Rule 33: a jink wins the game at once.
        for team in 0..4 {
            if self.holders.iter().filter(|h| **h == Some(team)).count() == 5 {
                self.end_game(team, GameEnd::Jink);
                return;
            }
        }

        // Rule 30: without the 5, the bonus goes to the trick that held the
        // highest trump played. Rule 31: no trump played, no bonus.
        if let Some((card, team)) = self.end_bonus() {
            self.award_bonus(team, card, "end");
            if self.scores[team] >= 41 {
                self.end_game(team, GameEnd::EndBonus);
                return;
            }
        }
        self.close_hand();
    }

    /// The end-of-hand bonus, if one is due: no 5 of trumps counting in the hand, and the
    /// highest trump that counts, with the team holding its trick now. The cards of a turned
    /// partnership do not count (rule 39).
    fn end_bonus(&self) -> Option<(Card, usize)> {
        let trump = self.trump.expect("trumps are known during play");
        let five = order::five_of_trumps(trump);
        let five_counts = self.finished_tricks.iter().any(|(cards, _)| {
            cards
                .iter()
                .any(|(seat, card)| *card == five && !self.teams_out[team_of(*seat)])
        });
        if five_counts {
            return None;
        }
        let mut best: Option<(Card, usize)> = None; // (trump, index of its trick)
        for (i, (cards, _)) in self.finished_tricks.iter().enumerate() {
            for (seat, card) in cards {
                if !order::is_trump(*card, trump) || self.teams_out[team_of(*seat)] {
                    continue;
                }
                let better = match best {
                    None => true,
                    Some((b, _)) => order::trump_beats(*card, b, trump),
                };
                if better {
                    best = Some((*card, i));
                }
            }
        }
        let (card, i) = best?;
        // The bonus goes to whoever holds that trick now.
        let team = self.holders[i].expect("the best trump in play holds its trick");
        Some((card, team))
    }

    /// Step 10: the hand is over and the game goes on.
    fn close_hand(&mut self) {
        let aborted = self.finished_tricks.len() < 5;
        self.emit_public(format!(
            "HandEnd hand={} aborted={}",
            self.hand_index, aborted
        ));
        self.dealer = left_of(self.dealer);
        self.hand_index += 1;
        self.waiting = Waiting::Deck;
    }

    /// Step 11.
    fn end_game(&mut self, team: usize, reason: GameEnd) {
        self.emit_public(format!(
            "GameWon team={} reason={} scores={}",
            team,
            reason.word(),
            four_joined(&self.scores)
        ));
        self.games_won[team] += 1;
        let aborted = self.finished_tricks.len() < 5;
        self.emit_public(format!(
            "HandEnd hand={} aborted={}",
            self.hand_index, aborted
        ));

        if self.games_won[team] == 2 {
            // Rule 35 and H6: the money.
            let stake = self.rules.stake;
            let mut money = [-2 * stake; 4];
            money[team] = 6 * stake;
            self.money = Some(money);
            self.emit_public(format!(
                "MatchWon team={} money={}",
                team,
                four_joined(&money)
            ));
            self.waiting = Waiting::MatchOver;
        } else {
            // Rule 36: a new game, scores back to nought; H5: the deal moves left.
            self.game_index += 1;
            self.scores = [0; 4];
            self.bonus_taken = [false; 4];
            self.dealer = left_of(self.dealer);
            self.hand_index += 1;
            self.waiting = Waiting::Deck;
        }
    }
}
