//! `rules-v1`: the published 25/45-family heuristics as an exact, deterministic decision list
//! (docs/M1-SPEC.md §4; their origin is docs/REVIEW-2026-10-13.md part 1 §3.15).
//!
//! The first rule that applies decides:
//!
//! 1. **Burn** ([`Rule::Burn`]): burn a plain (non-trump) card if possible: the lowest plain rank,
//!    then the suit the hand holds fewest plain cards of, then the lowest card index. With no
//!    plain card to burn, burn the lowest trump. A trump this seat took this hand (the robbed
//!    card, or the Joker the dealer took) is never burned unless nothing else may be. Any card is
//!    a legal burn (house rule H3, amended 2026-10-14), so aces are eligible by plain rank order
//!    like any other card (a red plain ace is the lowest card of its suit). While trumps are
//!    unknown (the dealer's burn after a turned Joker), only the Joker and the ace of hearts
//!    count as trumps.
//! 2. **Clinch defence** ([`Rule::ClinchDefence`], following only): if another team T would
//!    reach 41 or more by winning this trick (`score + 5 + bonus ≥ 41`, the bonus counting only
//!    if the trick already holds the 5 of trumps, at T's first or later rate), T can still take
//!    the trick (its seat is winning it now, or one of its seats still to play could hold a card
//!    that beats the current winner), and a legal card of mine beats the current winner: play
//!    the lowest such card, top trumps included.
//! 3. **Partner is winning** ([`Rule::PartnerWinning`], following only): if my partner's card is
//!    winning and no seat still to play after me could hold a card that beats it (judged with
//!    the tracker's facts), play the lowest legal card that does not beat it (the lowest legal
//!    card if every legal card beats it).
//! 4. **Leading** (leading only): if the 5 and the J of trumps are both known out of play and I
//!    hold the best trump not known out of play, lead it ([`Rule::LeadBestTrump`]). Otherwise on
//!    tricks 1–3 lead the lowest plain card of my longest plain suit (equal lengths: the suit
//!    holding the lower plain rank, then the lower suit index), or my lowest trump if I hold no
//!    plain card ([`Rule::LeadLow`]); on tricks 4 and 5 lead my strongest card
//!    ([`Rule::LeadStrongest`]; on trick 5 it is my only card).
//! 5. **Win cheaply** ([`Rule::WinCheaply`], following only): play the lowest legal card that
//!    beats the current winner (a partner's card included, when rule 3 did not apply), but on
//!    tricks 1–3 never a top trump (5, J, Joker, A♥) unless every legal card is a top trump.
//! 6. **Play low** ([`Rule::PlayLow`]): play the lowest legal card. This always keeps the
//!    "saver" (the highest trump) while any other legal card exists.
//!
//! **Order of cards.** "Lowest" and "highest" use [`strength`]: every trump is above every plain
//! card; trumps compare by the engine's `trump_rank`, plain cards by `plain_rank` (a card's rank
//! within its own suit, so plain cards of different suits with equal rank are equal). Equal
//! strengths are broken by the lower card index (clubs, diamonds, hearts, spades; ace to king),
//! for "lowest" and "highest" alike.
//!
//! **"Beats the current winner"** is the engine's `beats` against the card winning the trick as
//! it stands. **"Known out of play"** and **"could hold"** come from the [`Tracker`].
//!
//! The bot uses no randomness, so its seed is ignored; it reads only its `SeatView`.

use bot::Bot;
use engine::cards::{Bits, Card, Suit, bit, cards, count, has, suit_of};
use engine::game::{Action, BONUS_FIRST, BONUS_LATER, Rules, TRICK_POINTS, WINNING_SCORE};
use engine::rules::{
    ACE_HEARTS, EffSuit, beats, effective_suit, five_of, is_top_trump, jack_of, plain_rank,
    trick_winner, trump_mask, trump_rank,
};
use engine::view::SeatView;
use engine::{SEATS, TEAMS, team_of};

use crate::RULES_V1;
use crate::tracker::Tracker;

/// Which rule of the decision list chose an action (for tests and diagnostics).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// Rule 1: the burn.
    Burn,
    /// Rule 2: stop another team reaching 41 with this trick.
    ClinchDefence,
    /// Rule 3: the partner is winning safely; play low without beating them.
    PartnerWinning,
    /// Rule 4: lead the best remaining trump (the 5 and the J are gone).
    LeadBestTrump,
    /// Rule 4: tricks 1–3, lead low from the longest plain suit.
    LeadLow,
    /// Rule 4: tricks 4–5, lead the strongest card.
    LeadStrongest,
    /// Rule 5: win the trick with the cheapest winning card.
    WinCheaply,
    /// Rule 6: play the lowest legal card.
    PlayLow,
}

/// The rules bot. Deterministic; it keeps only the rule switches between decisions.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RulesBot {
    rules: Rules,
}

impl RulesBot {
    /// A rules bot; the rule switches come with [`Bot::start_match`].
    pub fn new() -> Self {
        Self::default()
    }
}

impl Bot for RulesBot {
    fn name(&self) -> &str {
        RULES_V1
    }

    fn start_match(&mut self, _seat: u8, rules: Rules, _seed: u64) {
        self.rules = rules;
    }

    fn act(&mut self, view: &SeatView, legal: &[Action]) -> Action {
        decide(view, &self.rules, legal).0
    }
}

/// The card order behind every "lowest" and "highest": 101–115 for trumps (by `trump_rank`),
/// 1–13 for plain cards (by `plain_rank`).
pub fn strength(c: Card, trump: Suit) -> u8 {
    let t = trump_rank(c, trump);
    if t > 0 { 100 + t } else { plain_rank(c) }
}

/// The lowest card of a set by [`strength`], ties to the lower card index.
pub fn lowest(set: Bits, trump: Suit) -> Option<Card> {
    cards(set).min_by_key(|&c| (strength(c, trump), c))
}

/// The highest card of a set by [`strength`], ties to the lower card index.
pub fn highest(set: Bits, trump: Suit) -> Option<Card> {
    cards(set).max_by_key(|&c| (strength(c, trump), core::cmp::Reverse(c)))
}

/// The four top trumps: the 5 and J of trumps, the Joker and the ace of hearts.
fn top_trumps(trump: Suit) -> Bits {
    cards(trump_mask(trump))
        .filter(|&c| is_top_trump(c, trump))
        .fold(0, |a, c| a | bit(c))
}

/// The score `team` would have if it won the current trick as it stands: its score, plus 5,
/// plus its best-trump bonus (first or later rate) if the trick already holds the 5 of trumps.
/// This is the engine's arithmetic for a trick (rules 27–29).
pub fn score_if_team_wins_trick(view: &SeatView, team: u8) -> u16 {
    let p = &view.public;
    let t = usize::from(team % TEAMS as u8);
    let five_in = p
        .trump
        .is_some_and(|tr| p.trick_cards.iter().any(|&(_, c)| c == five_of(tr)));
    let bonus = match (five_in, p.bonus_taken[t]) {
        (false, _) => 0,
        (true, false) => BONUS_FIRST,
        (true, true) => BONUS_LATER,
    };
    u16::from(p.scores[t]) + u16::from(TRICK_POINTS) + u16::from(bonus)
}

/// Choose an action for the seat to act, and say which rule chose it.
///
/// # Panics
/// If `legal` is empty (the table never asks a bot to act without a legal action).
pub fn decide(view: &SeatView, _rules: &Rules, legal: &[Action]) -> (Action, Rule) {
    let Some(&first) = legal.first() else {
        panic!("rules-v1 was asked to act with no legal action");
    };
    let mask = legal.iter().fold(0, |a: Bits, x| a | bit(x.card()));
    let t = Tracker::from_view(view);
    match first {
        Action::Burn(_) => (Action::Burn(choose_burn(&t, mask)), Rule::Burn),
        Action::Play(_) => {
            let Some(trump) = t.trump else {
                // Unreachable: the trump suit is always known when a card is to be played.
                return (first, Rule::PlayLow);
            };
            let choice = if view.public.trick_cards.is_empty() {
                lead(&t, trump, view.public.trick_no, mask)
            } else {
                follow(view, &t, trump, mask)
            };
            match choice {
                Some((c, rule)) if has(mask, c) => (Action::Play(c), rule),
                _ => (first, Rule::PlayLow),
            }
        }
    }
}

// ----- rule 1: the burn -----

/// Plain cards of the suit of `c` in the hand (the trumps set aside).
fn plain_suit_length(hand: Bits, c: Card, trumps: Bits) -> u8 {
    suit_of(c).map_or(0, |s| count(hand & s.mask() & !trumps))
}

/// The trump order when the trump suit may still be unknown (only the Joker and the A♥).
fn trump_order(c: Card, trump: Option<Suit>) -> u8 {
    match trump {
        Some(t) => trump_rank(c, t),
        None if c == engine::cards::JOKER => 13,
        None if c == ACE_HEARTS => 12,
        None => 0,
    }
}

fn choose_burn(t: &Tracker, legal: Bits) -> Card {
    let protected = t.my_taken_trumps();
    let pool = if legal & !protected != 0 {
        legal & !protected
    } else {
        legal
    };
    let trumps = t.trumps();
    let plain = pool & !trumps;
    let pick = if plain != 0 {
        cards(plain).min_by_key(|&c| (plain_rank(c), plain_suit_length(t.hand, c, trumps), c))
    } else {
        cards(pool).min_by_key(|&c| (trump_order(c, t.trump), c))
    };
    // `legal` is never empty, so neither is `pool`.
    pick.unwrap_or(legal.trailing_zeros() as Card)
}

// ----- rule 4: leading -----

/// The longest plain suit in `plain`; ties go to the suit whose lowest card has the lower plain
/// rank, then to the lower suit index (clubs, diamonds, hearts, spades).
fn longest_plain_suit(plain: Bits) -> Option<Suit> {
    Suit::ALL
        .into_iter()
        .filter(|s| plain & s.mask() != 0)
        .min_by_key(|&s| {
            let cs = plain & s.mask();
            let low = cards(cs).map(plain_rank).min().unwrap_or(u8::MAX);
            (core::cmp::Reverse(count(cs)), low, s as u8)
        })
}

fn lead(t: &Tracker, trump: Suit, trick_no: u8, legal: Bits) -> Option<(Card, Rule)> {
    let out = t.known_out();
    if has(out, five_of(trump)) && has(out, jack_of(trump)) {
        let best_remaining = cards(trump_mask(trump) & !out).max_by_key(|&c| trump_rank(c, trump));
        if let Some(best) = best_remaining
            && has(legal, best)
        {
            return Some((best, Rule::LeadBestTrump));
        }
    }
    if trick_no <= 3 {
        let plain = legal & !trump_mask(trump);
        let card = match longest_plain_suit(plain) {
            Some(s) => cards(plain & s.mask()).min_by_key(|&c| (plain_rank(c), c)),
            None => lowest(legal, trump),
        };
        return card.map(|c| (c, Rule::LeadLow));
    }
    highest(legal, trump).map(|c| (c, Rule::LeadStrongest))
}

// ----- rules 2, 3, 5 and 6: following -----

/// The trick as it stands, from the point of view of the seat to play.
struct TrickState {
    led: EffSuit,
    winner_seat: u8,
    winner_card: Card,
    /// The seats still to play after this one, in order.
    after: Vec<u8>,
}

impl TrickState {
    fn new(view: &SeatView, trump: Suit) -> Option<TrickState> {
        let tc = &view.public.trick_cards;
        let &(_, led) = tc.first()?;
        let plays: Vec<Card> = tc.iter().map(|&(_, c)| c).collect();
        let &(winner_seat, winner_card) = tc.get(trick_winner(&plays, trump))?;
        let n_after = SEATS.saturating_sub(tc.len() + 1);
        let after = (1..=n_after)
            .map(|i| ((usize::from(view.seat) + i) % SEATS) as u8)
            .collect();
        Some(TrickState {
            led: effective_suit(led, trump),
            winner_seat,
            winner_card,
            after,
        })
    }

    /// The cards of `set` that beat the card winning the trick now.
    fn beating(&self, set: Bits, trump: Suit) -> Bits {
        cards(set)
            .filter(|&c| beats(c, self.winner_card, self.led, trump))
            .fold(0, |a, c| a | bit(c))
    }

    /// Could a seat still to play after me, among those `which` accepts, hold a card that beats
    /// the card winning the trick now?
    fn beatable_after(&self, t: &Tracker, trump: Suit, which: impl Fn(u8) -> bool) -> bool {
        cards(self.beating(t.unseen(), trump))
            .any(|c| self.after.iter().any(|&s| which(s) && t.could_hold(s, c)))
    }
}

/// Rule 2's test: is there another team that would reach 41 by winning this trick and can still
/// take it?
fn clinch_threat(view: &SeatView, t: &Tracker, trump: Suit, trick: &TrickState) -> bool {
    let mine = team_of(view.seat);
    (0..TEAMS as u8).filter(|&tm| tm != mine).any(|tm| {
        score_if_team_wins_trick(view, tm) >= u16::from(WINNING_SCORE)
            && (team_of(trick.winner_seat) == tm
                || trick.beatable_after(t, trump, |s| team_of(s) == tm))
    })
}

fn follow(view: &SeatView, t: &Tracker, trump: Suit, legal: Bits) -> Option<(Card, Rule)> {
    let trick = TrickState::new(view, trump)?;
    let winners = trick.beating(legal, trump);

    // Rule 2: clinch defence.
    if winners != 0 && clinch_threat(view, t, trump, &trick) {
        return lowest(winners, trump).map(|c| (c, Rule::ClinchDefence));
    }

    // Rule 3: the partner is winning and nobody after me can beat it.
    let partner = (view.seat + 4) % SEATS as u8;
    if trick.winner_seat == partner && !trick.beatable_after(t, trump, |_| true) {
        let safe = legal & !winners;
        let pool = if safe != 0 { safe } else { legal };
        return lowest(pool, trump).map(|c| (c, Rule::PartnerWinning));
    }

    // Rule 5: win cheaply, sparing the top trumps before trick 4.
    if winners != 0 {
        let top = top_trumps(trump);
        let pool = if view.public.trick_no < 4 && legal & !top != 0 {
            winners & !top
        } else {
            winners
        };
        if let Some(c) = lowest(pool, trump) {
            return Some((c, Rule::WinCheaply));
        }
    }

    // Rule 6: play low, keeping the saver.
    lowest(legal, trump).map(|c| (c, Rule::PlayLow))
}
