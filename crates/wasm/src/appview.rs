//! The app's seat view rebuilt as the engine's own (docs/WASM-API.md §3).
//!
//! A request carries the app's `MatchState` as one seat may see it (`maskMatch(m, seat)`). The
//! search needs the engine's [`SeatView`] of the same moment: the same public events, the seat's
//! own cards and burns. A seat's view depends only on what the seat saw, so it is made by playing
//! the hand in the engine from *any* deal consistent with what the seat saw:
//!
//! 1. read what the seat saw: its cards, the turned cards and who took them, every play in order,
//!    the burns, the pairs turned and the reneges caught (nothing a seat may not see is read:
//!    another hand only for its size, the discards only at the seat's own burns);
//! 2. find one consistent deal: every hidden card is placed (the other hands, the other seats'
//!    burns, the undealt pack) so that every play the table saw is legal unless the cards seen
//!    give it away as a renege, the robber holds the ace of trumps, and nobody who played in
//!    trick 1 while the turned card lay on the table holds it (a small matching problem, solved
//!    exactly);
//! 3. replay the hand in the engine from that deal, adding the app's referee steps where the app
//!    made them: a pair turned before a play (Guide's play without robbing, a refused ace, a left
//!    Joker) and the lapses the app let through that are now known (a caught false rob, burned
//!    ace or play without robbing, a card the seen cards prove illegal). The engine catches
//!    reneges itself exactly as the app does (contract §7.1);
//! 4. check the replay against the app's own state (the seat to act, the phase, the pairs
//!    turned, every hand's size, the seat's own cards) and take the engine's view of the seat.
//!
//! The deal is only a vehicle: the search draws its own deals from the view with the card
//! tracker, and any consistent deal gives the same view (tested on random matches, with and
//! without turned pairs, against the engine's own view of the seat).

use engine::cards::{ALL_CARDS, Bits, Card, JOKER, Suit, bit, cards, count, has};
use engine::game::{Action, HandStartState, Lapse, Match, Phase, Rules, TurnRefusal, build_deck};
use engine::rules::{ace_of_trumps, legal_plays};
use engine::team_of;
use engine::view::SeatView;
use serde_json::Value;

use crate::cards::{engine_card, engine_suit};

/// A refused request: a stable code (docs/WASM-API.md §5) and why, in plain English.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// The code.
    pub code: &'static str,
    /// Why.
    pub reason: String,
}

impl Refusal {
    /// A refusal.
    pub fn new(code: &'static str, reason: impl Into<String>) -> Refusal {
        Refusal {
            code,
            reason: reason.into(),
        }
    }
}

fn bad(reason: impl Into<String>) -> Refusal {
    Refusal::new("bad_request", reason)
}

fn inconsistent(reason: impl Into<String>) -> Refusal {
    Refusal::new("inconsistent_view", reason)
}

// ----- reading JSON -----

/// A field of an object.
pub(crate) fn field<'a>(v: &'a Value, name: &str, at: &str) -> Result<&'a Value, Refusal> {
    v.get(name)
        .ok_or_else(|| bad(format!("{at}.{name} is missing")))
}

/// A whole number field within `lo..=hi`.
pub(crate) fn int(v: &Value, name: &str, at: &str, lo: i64, hi: i64) -> Result<i64, Refusal> {
    let x = field(v, name, at)?;
    int_value(x, &format!("{at}.{name}"), lo, hi)
}

fn int_value(x: &Value, what: &str, lo: i64, hi: i64) -> Result<i64, Refusal> {
    let n = x
        .as_i64()
        .ok_or_else(|| bad(format!("{what} must be a whole number")))?;
    if n < lo || n > hi {
        return Err(bad(format!("{what} must be {lo}..{hi}, not {n}")));
    }
    Ok(n)
}

fn array<'a>(v: &'a Value, name: &str, at: &str) -> Result<&'a Vec<Value>, Refusal> {
    field(v, name, at)?
        .as_array()
        .ok_or_else(|| bad(format!("{at}.{name} must be an array")))
}

fn ints<const N: usize>(
    v: &Value,
    name: &str,
    at: &str,
    lo: i64,
    hi: i64,
) -> Result<[i64; N], Refusal> {
    let a = array(v, name, at)?;
    if a.len() != N {
        return Err(bad(format!("{at}.{name} must hold {N} numbers")));
    }
    let mut out = [0; N];
    for (i, x) in a.iter().enumerate() {
        out[i] = int_value(x, &format!("{at}.{name}[{i}]"), lo, hi)?;
    }
    Ok(out)
}

fn card_value(x: &Value, what: &str) -> Result<Card, Refusal> {
    let n = int_value(x, what, 0, 53)?;
    engine_card(n).ok_or_else(|| Refusal::new("two_jokers", format!("{what}: the second Joker")))
}

/// A card or -1 (hidden or none).
fn opt_card(x: &Value, what: &str) -> Result<Option<Card>, Refusal> {
    if x.as_i64() == Some(-1) {
        Ok(None)
    } else {
        card_value(x, what).map(Some)
    }
}

// ----- the app's view -----

/// What the app's hand is waiting for (`HandState.phase`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppPhase {
    /// The dealer must take a turned Joker and burn a card (`steal`).
    JokerSteal,
    /// The dealer robs a turned ace (`rob`).
    DealerRob,
    /// Tricks (a robber's `rob` at its first turn included).
    Play,
}

/// A burn of the hand, in the order the app made them (its `discards`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BurnKind {
    /// The dealer's after taking a turned Joker (before trumps are known).
    JokerTake,
    /// The dealer's after robbing a turned ace at once.
    AceAtOnce,
    /// A robber's at its turn in trick 1 (a false rob's too).
    AtTurn,
}

/// The app's seat view, read.
#[derive(Clone, Debug)]
pub struct AppTable {
    /// The viewer.
    pub seat: u8,
    /// The seat to act.
    pub to_act: u8,
    /// The phase.
    pub phase: AppPhase,
    /// The hand's start.
    pub start: HandStartState,
    /// The viewer's cards (engine numbering).
    own: Bits,
    /// Every hand's size as the app shows it (a turned seat: 0).
    sizes: [usize; 8],
    /// A Joker was turned first this hand (stolen, left, or waiting to be stolen).
    joker_turned: bool,
    /// The dealer took it.
    joker_stolen: bool,
    /// The dealer left it on the deck (rule 24, turned).
    joker_left: bool,
    /// The card that fixed trumps (the Joker itself while it waits to be stolen).
    turned: Card,
    /// The seat that took `turned` (the dealer robbing a turned ace, or the robber in trick 1).
    taken_by: Option<u8>,
    /// Every trick's plays in order, the trick in progress last.
    tricks: Vec<Vec<(u8, Card)>>,
    /// Seats whose cards are turned.
    penalised: [bool; 8],
    /// Seats with a caught renege, by kind.
    caught_rob: [bool; 8],
    caught_false_rob: [bool; 8],
    caught_burned_ace: [bool; 8],
    /// The app's `discards`, in order (`None`: face down).
    discards: Vec<Option<Card>>,
    /// The viewer's burns from the request's `ownBurns`.
    own_burns: Option<Vec<Card>>,
}

/// The house rules' values of the app's `MatchConfig`, by field: a field that is present must
/// have this value.
const HOUSE_CONFIG: [(&str, HouseValue); 8] = [
    ("target", HouseValue::Int(41)),
    ("gamesToWin", HouseValue::Int(2)),
    ("trickPoints", HouseValue::Int(5)),
    ("bonusExtra", HouseValue::Int(6)),
    ("jinkWins", HouseValue::Bool(true)),
    ("allowDiscardTrumpAce", HouseValue::Bool(false)),
    ("partnerPlaysOn", HouseValue::Bool(false)),
    ("mustTrumpWhenVoid", HouseValue::Bool(false)),
];

#[derive(Clone, Copy)]
enum HouseValue {
    Int(i64),
    Bool(bool),
}

impl AppTable {
    /// Read the app's `MatchState` (`m`) as `seat` sees it. `own_burns` is the request's
    /// `ownBurns` (app numbering), when given.
    pub fn read(m: &Value, seat: u8, own_burns: Option<Vec<Card>>) -> Result<AppTable, Refusal> {
        if !m.is_object() {
            return Err(bad("match must be the app's MatchState object"));
        }
        let config = field(m, "config", "match")?;
        if int(config, "n", "match.config", 0, 1000)? != 8 {
            return Err(Refusal::new(
                "not_8_seats",
                "the engine plays the 8-seat game only",
            ));
        }
        if let Some(j) = config.get("jokers")
            && j.as_i64() != Some(1)
        {
            return Err(Refusal::new(
                "two_jokers",
                "the engine plays with one Joker (config.jokers 1)",
            ));
        }
        for (name, want) in HOUSE_CONFIG {
            let Some(x) = config.get(name) else { continue };
            let ok = match want {
                HouseValue::Int(n) => x.as_i64() == Some(n),
                HouseValue::Bool(b) => x.as_bool() == Some(b),
            };
            if !ok {
                return Err(Refusal::new(
                    "not_house_rules",
                    format!("config.{name} is not the 41 house rules' value"),
                ));
            }
        }
        match field(m, "status", "match")?.as_str() {
            Some("playing") => {}
            Some(_) => return Err(Refusal::new("not_playing", "the match is not playing")),
            None => return Err(bad("match.status must be a string")),
        }
        let scores = ints::<4>(m, "gameScores", "match", 0, 40)?;
        let games = ints::<4>(m, "gamesWon", "match", 0, 1)?;
        let game_index = int(m, "gameIndex", "match", 0, 4)?;
        let hand_index = int(m, "handIndex", "match", 0, i64::from(u32::MAX))?;
        let dealer = int(m, "dealer", "match", 0, 7)? as u8;
        let start = HandStartState {
            dealer,
            scores: scores.map(|s| s as u8),
            bonus_taken: scores.map(|s| s % 5 == 1),
            games_won: games.map(|g| g as u8),
            hand_index: hand_index as u32,
            game_index: game_index as u8,
        };
        start
            .validate()
            .map_err(|e| bad(format!("match: {e} (gameScores, gamesWon, gameIndex)")))?;

        let h = field(m, "hand", "match")?;
        if !h.is_object() {
            return Err(bad("match.hand must be an object"));
        }
        if let Some(d) = h.get("dealer")
            && d.as_i64() != Some(i64::from(dealer))
        {
            return Err(bad("match.hand.dealer differs from match.dealer"));
        }
        let phase = match field(h, "phase", "hand")?.as_str() {
            Some("jokerSteal") => AppPhase::JokerSteal,
            Some("dealerRob") => AppPhase::DealerRob,
            Some("play") => AppPhase::Play,
            Some("over") => return Err(Refusal::new("not_playing", "the hand is over")),
            _ => {
                return Err(bad(
                    "hand.phase must be jokerSteal, dealerRob, play or over",
                ));
            }
        };
        let to_act = match field(h, "toAct", "hand")?.as_i64() {
            Some(s @ 0..=7) => s as u8,
            Some(-1) => return Err(Refusal::new("not_playing", "nobody is to act")),
            _ => return Err(bad("hand.toAct must be a seat")),
        };
        if to_act != seat {
            return Err(Refusal::new(
                "not_to_act",
                format!(
                    "seat {seat} is not to act (seat {to_act} is); a view from another seat is not built yet"
                ),
            ));
        }
        let turned = card_value(field(h, "turnedCard", "hand")?, "hand.turnedCard")?;
        let on_table = opt_card(field(h, "turned", "hand")?, "hand.turned")?;
        let stolen = array(h, "jokerStolen", "hand")?;
        let left = array(h, "jokerLeft", "hand")?;
        for (i, x) in stolen.iter().chain(left).enumerate() {
            if card_value(x, &format!("hand.jokerStolen/jokerLeft[{i}]"))? != JOKER {
                return Err(bad("hand.jokerStolen and hand.jokerLeft hold Jokers only"));
            }
        }
        if stolen.len() + left.len() > 1 {
            return Err(Refusal::new(
                "two_jokers",
                "one Joker can be turned only once",
            ));
        }
        let joker_stolen = !stolen.is_empty();
        let joker_left = !left.is_empty();
        let joker_turned = joker_stolen || joker_left || phase == AppPhase::JokerSteal;
        if (phase == AppPhase::JokerSteal) != (turned == JOKER) {
            return Err(inconsistent(
                "the turned card is the Joker exactly while the dealer must take it",
            ));
        }
        let taken_by = match field(h, "turnedTakenBy", "hand")?.as_i64() {
            Some(-1) => None,
            Some(s @ 0..=7) => Some(s as u8),
            _ => return Err(bad("hand.turnedTakenBy must be a seat or -1")),
        };
        if taken_by.is_some() == on_table.is_some() {
            return Err(inconsistent(
                "the turned card is either on the table or taken (hand.turned, hand.turnedTakenBy)",
            ));
        }
        if let Some(c) = on_table
            && c != turned
        {
            return Err(inconsistent("hand.turned is not hand.turnedCard"));
        }
        if let Some(t) = h.get("trump")
            && t.as_i64() != Some(-1)
        {
            let s = t.as_i64().and_then(engine_suit);
            if s.is_none() || s != engine::cards::suit_of(turned) {
                return Err(inconsistent("hand.trump is not the turned card's suit"));
            }
        }

        let hands = array(h, "hands", "hand")?;
        if hands.len() != 8 {
            return Err(bad("hand.hands must hold 8 hands"));
        }
        let mut sizes = [0usize; 8];
        let mut own: Bits = 0;
        for (s, x) in hands.iter().enumerate() {
            let a = x
                .as_array()
                .ok_or_else(|| bad(format!("hand.hands[{s}] must be an array")))?;
            sizes[s] = a.len();
            if s == usize::from(seat) {
                for (i, c) in a.iter().enumerate() {
                    let c = card_value(c, &format!("hand.hands[{s}][{i}] (the seat's own card)"))?;
                    if has(own, c) {
                        return Err(bad("the seat holds a card twice"));
                    }
                    own |= bit(c);
                }
            }
        }

        let mut tricks = Vec::new();
        for (t, x) in array(h, "tricks", "hand")?.iter().enumerate() {
            tricks.push(plays(x, &format!("hand.tricks[{t}]"))?);
        }
        if tricks.iter().any(|t| t.is_empty()) {
            return Err(inconsistent("a finished trick has no cards"));
        }
        match h.get("current") {
            None | Some(Value::Null) => {}
            Some(x) => tricks.push(plays(x, "hand.current")?),
        }
        if tricks.len() > 5 {
            return Err(inconsistent("a hand has five tricks"));
        }

        let mut penalised = [false; 8];
        if let Some(p) = h.get("penalised") {
            let a = p
                .as_array()
                .filter(|a| a.len() == 8)
                .ok_or_else(|| bad("hand.penalised must hold 8 booleans"))?;
            for (s, x) in a.iter().enumerate() {
                penalised[s] = x
                    .as_bool()
                    .ok_or_else(|| bad("hand.penalised must hold 8 booleans"))?;
            }
        }
        for s in 0..8 {
            if penalised[s] != penalised[(s + 4) % 8] {
                return Err(inconsistent("partners are turned together"));
            }
        }

        let mut caught_rob = [false; 8];
        let mut caught_false_rob = [false; 8];
        let mut caught_burned_ace = [false; 8];
        if let Some(r) = h.get("reneges") {
            let a = r
                .as_array()
                .ok_or_else(|| bad("hand.reneges must be an array"))?;
            for (i, x) in a.iter().enumerate() {
                let at = format!("hand.reneges[{i}]");
                let s = int(x, "seat", &at, 0, 7)? as usize;
                let caught = int(x, "caughtAt", &at, -1, 5)? >= 0;
                let kind = field(x, "kind", &at)?
                    .as_str()
                    .ok_or_else(|| bad(format!("{at}.kind must be a string")))?;
                if !caught {
                    if s == usize::from(seat) {
                        return Err(inconsistent(
                            "the seat has a renege of its own nobody has caught",
                        ));
                    }
                    // Another seat's renege nobody has caught is not the seat's to know.
                    continue;
                }
                match kind {
                    "rob" => caught_rob[s] = true,
                    "falseRob" => caught_false_rob[s] = true,
                    "burnedAce" => caught_burned_ace[s] = true,
                    "follow" | "trumpLead" | "joker" => {}
                    "trumpWhenVoid" => {
                        return Err(Refusal::new(
                            "not_house_rules",
                            "a trumpWhenVoid renege needs the must-trump rule",
                        ));
                    }
                    _ => return Err(bad(format!("{at}.kind {kind} is unknown"))),
                }
            }
        }

        let mut discards = Vec::new();
        for (i, x) in array(h, "discards", "hand")?.iter().enumerate() {
            discards.push(opt_card(x, &format!("hand.discards[{i}]"))?);
        }

        Ok(AppTable {
            seat,
            to_act,
            phase,
            start,
            own,
            sizes,
            joker_turned,
            joker_stolen,
            joker_left,
            turned,
            taken_by,
            tricks,
            penalised,
            caught_rob,
            caught_false_rob,
            caught_burned_ace,
            discards,
            own_burns,
        })
    }

    /// The trump suit, once known.
    fn trump(&self) -> Option<Suit> {
        if self.phase == AppPhase::JokerSteal {
            None
        } else {
            engine::cards::suit_of(self.turned)
        }
    }

    /// The burns of the hand, in order: the seat and the kind.
    fn burn_seq(&self) -> Vec<(u8, BurnKind)> {
        let d = self.start.dealer;
        let mut out = Vec::new();
        if self.joker_stolen {
            out.push((d, BurnKind::JokerTake));
        }
        let ace_turned = engine::cards::rank_of(self.turned) == engine::cards::ACE;
        match self.taken_by {
            Some(s) if ace_turned && s == d => out.push((d, BurnKind::AceAtOnce)),
            Some(s) => out.push((s, BurnKind::AtTurn)),
            None => {}
        }
        out
    }
}

/// A trick's plays.
fn plays(x: &Value, at: &str) -> Result<Vec<(u8, Card)>, Refusal> {
    let mut out = Vec::new();
    for (i, p) in array(x, "plays", at)?.iter().enumerate() {
        let pat = format!("{at}.plays[{i}]");
        let seat = int(p, "seat", &pat, 0, 7)? as u8;
        let c = card_value(field(p, "card", &pat)?, &format!("{pat}.card"))?;
        out.push((seat, c));
    }
    Ok(out)
}

// ----- one consistent deal -----

/// A place hidden cards can go: how many, and which cards may.
#[derive(Clone, Copy, Debug)]
struct Place {
    cap: usize,
    allowed: Bits,
}

/// Place every card of `unknown` (ascending) in `places`, within their capacities and allowed
/// sets: an exact bipartite matching by augmenting paths (cards in index order, places in their
/// order), so the answer is a pure function of the problem. `None` if no placement exists.
fn place_cards(unknown: Bits, places: &[Place]) -> Option<Vec<Bits>> {
    let list: Vec<Card> = cards(unknown).collect();
    let mut held: Vec<Vec<Card>> = vec![Vec::new(); places.len()];
    fn augment(c: Card, places: &[Place], held: &mut [Vec<Card>], seen: &mut [bool]) -> bool {
        for p in 0..places.len() {
            if seen[p] || !has(places[p].allowed, c) {
                continue;
            }
            seen[p] = true;
            if held[p].len() < places[p].cap {
                held[p].push(c);
                return true;
            }
            for k in 0..held[p].len() {
                let other = held[p][k];
                if augment(other, places, held, seen) {
                    held[p][k] = c;
                    return true;
                }
            }
        }
        false
    }
    for &c in &list {
        let mut seen = vec![false; places.len()];
        if !augment(c, places, &mut held, &mut seen) {
            return None;
        }
    }
    if held.iter().zip(places).any(|(h, p)| h.len() != p.cap) {
        return None;
    }
    Some(
        held.iter()
            .map(|h| h.iter().fold(0, |a, &c| a | bit(c)))
            .collect(),
    )
}

/// The cards `c` could not have been played beside, as a play against `led` (any one of them in
/// the hand makes `c` illegal), by the engine's own legality rule.
fn forbidden_beside(c: Card, led: Card, trump: Suit) -> Bits {
    cards(ALL_CARDS & !bit(c))
        .filter(|&f| !has(legal_plays(bit(c) | bit(f), Some(led), trump), c))
        .fold(0, |a, f| a | bit(f))
}

/// The seat's view rebuilt by the engine, and the replayed match it came from.
pub struct Rebuilt {
    /// The match at the decision (from the hand's start).
    pub m: Match,
    /// The seat's view.
    pub view: SeatView,
}

/// Rebuild the app's seat view in the engine (the module notes).
pub fn rebuild(t: &AppTable) -> Result<Rebuilt, Refusal> {
    let me = t.seat;
    let dealer = t.start.dealer;
    let trump = t.trump();
    let ace = trump.map(ace_of_trumps);
    let ace_turned = Some(t.turned) == ace;

    // The cards turned, in order, and who took what.
    let mut turned_seq = Vec::new();
    if t.joker_turned {
        turned_seq.push(JOKER);
    }
    if t.phase != AppPhase::JokerSteal {
        if t.turned == JOKER {
            return Err(inconsistent("the Joker cannot fix trumps"));
        }
        turned_seq.push(t.turned);
    }
    let mut taken = [0 as Bits; 8];
    if t.joker_stolen {
        taken[usize::from(dealer)] |= bit(JOKER);
    }
    if let Some(s) = t.taken_by {
        taken[usize::from(s)] |= bit(t.turned);
    }

    // Plays in order, and each seat's.
    let all_plays: Vec<(usize, u8, Card)> = t
        .tricks
        .iter()
        .enumerate()
        .flat_map(|(k, tr)| tr.iter().map(move |&(s, c)| (k, s, c)))
        .collect();
    let mut played: Bits = 0;
    let mut played_by = [0 as Bits; 8];
    for &(_, s, c) in &all_plays {
        if has(played, c) {
            return Err(inconsistent("a card is played twice"));
        }
        played |= bit(c);
        played_by[usize::from(s)] |= bit(c);
    }
    if trump.is_none() && !all_plays.is_empty() {
        return Err(inconsistent("cards are played before trumps are known"));
    }

    // The burns, and the seat's own.
    let seq = t.burn_seq();
    if t.discards.len() != seq.len() {
        return Err(inconsistent(format!(
            "{} discards for {} burns (hand.discards)",
            t.discards.len(),
            seq.len()
        )));
    }
    let mut own_burns: Vec<Card> = Vec::new();
    let mine: Vec<usize> = (0..seq.len()).filter(|&i| seq[i].0 == me).collect();
    if !mine.is_empty() {
        let from_discards: Option<Vec<Card>> = mine.iter().map(|&i| t.discards[i]).collect();
        own_burns = match (&t.own_burns, from_discards) {
            (Some(b), _) => b.clone(),
            (None, Some(b)) => b,
            (None, None) => {
                return Err(Refusal::new(
                    "own_burns_hidden",
                    "the seat burned a card: give ownBurns or the seat's own discards",
                ));
            }
        };
        if own_burns.len() != mine.len() {
            return Err(inconsistent(format!(
                "the seat made {} burns, ownBurns lists {}",
                mine.len(),
                own_burns.len()
            )));
        }
    } else if t.own_burns.as_ref().is_some_and(|b| !b.is_empty()) {
        return Err(inconsistent(
            "ownBurns given for a seat that burned nothing",
        ));
    }
    let own_burn_bits = own_burns.iter().fold(0, |a: Bits, &c| a | bit(c));

    // Each seat's hand size now (a turned seat keeps its cards face down).
    let mut burns_by = [0usize; 8];
    for &(s, _) in &seq {
        burns_by[usize::from(s)] += 1;
    }
    let mut size_now = [0usize; 8];
    for s in 0..8 {
        let n = 5 + count(taken[s]) as usize;
        let out = burns_by[s] + count(played_by[s]) as usize;
        if out > n {
            return Err(inconsistent(format!(
                "seat {s} played more cards than it held"
            )));
        }
        size_now[s] = n - out;
        if s != usize::from(me) && !t.penalised[s] && t.sizes[s] != size_now[s] {
            return Err(inconsistent(format!(
                "seat {s} holds {} cards, not {} (hand.hands)",
                t.sizes[s], size_now[s]
            )));
        }
    }
    let m_idx = usize::from(me);
    // The seat to act holds the ace of trumps at its first turn in trick 1 with the turned card
    // still on the table: its rob is due (the engine takes the card for it and awaits its burn).
    let pending_rob = t.phase == AppPhase::Play
        && t.to_act == me
        && t.taken_by.is_none()
        && !ace_turned
        && ace.is_some_and(|a| has(t.own, a))
        && t.tricks.len() <= 1
        && !all_plays.iter().any(|&(_, s, _)| s == me);
    if count(t.own) as usize != size_now[m_idx] {
        return Err(inconsistent(format!(
            "the seat holds {} cards, the plays say {}",
            count(t.own),
            size_now[m_idx]
        )));
    }

    // Every card the seat knows the place of.
    let public_held: [Bits; 8] = core::array::from_fn(|s| taken[s] & !played_by[s]);
    let turned_bits = turned_seq.iter().fold(0, |a: Bits, &c| a | bit(c));
    let mut known = t.own | own_burn_bits | played | turned_bits;
    if t.own & (own_burn_bits | played | turned_bits) & !public_held[m_idx] != 0
        || own_burn_bits & (played | turned_bits) != 0
    {
        return Err(inconsistent("a card of the seat's is also elsewhere"));
    }
    let mut hidden_cap = [0usize; 8];
    for s in 0..8 {
        if s == m_idx {
            continue;
        }
        let pubn = count(public_held[s]) as usize;
        if pubn > size_now[s] {
            return Err(inconsistent("a taken card is missing"));
        }
        hidden_cap[s] = size_now[s] - pubn;
    }

    // The ace of trumps: who must, and who cannot, hold it.
    let mut forced_hand = [0 as Bits; 8];
    let mut excluded = [0 as Bits; 8];
    let mut forced_burn: Option<usize> = None; // index in `seq`
    if let Some(ace) = ace
        && !ace_turned
        && !has(known, ace)
    {
        let robber_at_turn = t.taken_by;
        // Trick-1 plays made while the turned card lay on the table.
        let first = t.tricks.first().map(Vec::as_slice).unwrap_or(&[]);
        let rob_at = robber_at_turn.and_then(|r| first.iter().position(|&(s, _)| s == r));
        let on_table = &first[..rob_at.unwrap_or(first.len())];
        for &(s, _) in on_table {
            if !t.caught_rob[usize::from(s)] {
                excluded[usize::from(s)] |= bit(ace);
            }
        }
        match robber_at_turn {
            Some(r) if t.caught_burned_ace[usize::from(r)] => {
                forced_burn = seq
                    .iter()
                    .position(|&(s, k)| s == r && k == BurnKind::AtTurn);
            }
            Some(r) if t.caught_false_rob[usize::from(r)] => {
                excluded[usize::from(r)] |= bit(ace);
            }
            // A robber with no hidden card left that never showed the ace of trumps lapsed (the
            // app catches it at the end of the hand): it burned the ace, or robbed without it.
            // Either fits what the seat saw; the deal takes the burned ace.
            Some(r) if hidden_cap[usize::from(r)] == 0 => {
                forced_burn = seq
                    .iter()
                    .position(|&(s, k)| s == r && k == BurnKind::AtTurn);
            }
            Some(r) => forced_hand[usize::from(r)] |= bit(ace),
            None => {
                // A pair turned at its robber's turn in trick 1 (Guide's play without robbing):
                // its first seat in trick-1 order without a trick-1 play held the ace.
                if let Some(x) = no_rob_seat(t, first) {
                    forced_hand[usize::from(x)] |= bit(ace);
                    excluded[usize::from(x)] &= !bit(ace);
                }
            }
        }
    }
    for s in 0..8 {
        known |= forced_hand[s];
        if count(forced_hand[s]) as usize > hidden_cap[s] {
            return Err(inconsistent("the robber has no room for the ace of trumps"));
        }
    }
    if let Some(i) = forced_burn {
        known |= bit(ace.expect("trumps known"));
        if seq[i].0 == me {
            return Err(inconsistent("the seat's own burn is not hidden"));
        }
    }

    // Show-outs: a play the table saw as legal tells which cards its seat did not hold.
    if let Some(trump) = trump {
        let mut g = 0; // the play's index in `all_plays`
        for tr in &t.tricks {
            let Some(&(_, led)) = tr.first() else {
                continue;
            };
            for (i, &(s, c)) in tr.iter().enumerate() {
                g += 1;
                if i == 0 || s == me {
                    continue;
                }
                let f = forbidden_beside(c, led, trump);
                // Cards it is seen to have held then: its later plays and the cards it took.
                let later = all_plays[g..]
                    .iter()
                    .filter(|&&(_, ss, _)| ss == s)
                    .fold(taken[usize::from(s)], |a, &(_, _, cc)| a | bit(cc));
                if f & later != 0 {
                    continue; // a renege the cards seen give away: not a show-out
                }
                excluded[usize::from(s)] |= f;
            }
        }
    }

    // Place the hidden cards: the other hands, the other seats' burns, the undealt pack.
    let unknown = ALL_CARDS & !known;
    let mut places = Vec::new();
    let mut hand_place = [usize::MAX; 8];
    for s in 0..8 {
        if s == m_idx {
            continue;
        }
        hand_place[s] = places.len();
        places.push(Place {
            cap: hidden_cap[s] - count(forced_hand[s]) as usize,
            allowed: unknown & !excluded[s],
        });
    }
    let mut burn_place = vec![usize::MAX; seq.len()];
    for (i, &(s, kind)) in seq.iter().enumerate() {
        if s == me || Some(i) == forced_burn {
            continue;
        }
        let mut allowed = unknown;
        if kind != BurnKind::JokerTake
            && let Some(a) = ace
        {
            allowed &= !bit(a);
        }
        burn_place[i] = places.len();
        places.push(Place { cap: 1, allowed });
    }
    let undealt_place = places.len();
    let undealt_cap = 13usize
        .checked_sub(turned_seq.len())
        .ok_or_else(|| inconsistent("too many turned cards"))?;
    places.push(Place {
        cap: undealt_cap,
        allowed: unknown,
    });
    let need: usize = places.iter().map(|p| p.cap).sum();
    if need != count(unknown) as usize {
        return Err(inconsistent(format!(
            "{} hidden cards for {need} hidden places",
            count(unknown)
        )));
    }
    let placed = place_cards(unknown, &places)
        .ok_or_else(|| inconsistent("no deal of the hidden cards fits what the seat saw"))?;

    // The deal: each seat's five cards, then the pack from the turned card on.
    let mut burn_cards: Vec<Card> = Vec::with_capacity(seq.len());
    let mut own_iter = own_burns.iter();
    for (i, &(s, _)) in seq.iter().enumerate() {
        let c = if s == me {
            *own_iter.next().expect("counted above")
        } else if Some(i) == forced_burn {
            ace.expect("trumps known")
        } else {
            cards(placed[burn_place[i]]).next().expect("one card")
        };
        burn_cards.push(c);
    }
    let mut orig = [0 as Bits; 8];
    for s in 0..8 {
        let hand_now = if s == m_idx {
            t.own
        } else {
            placed[hand_place[s]] | forced_hand[s] | public_held[s]
        };
        let burned = seq
            .iter()
            .zip(&burn_cards)
            .filter(|&(&(bs, _), _)| usize::from(bs) == s)
            .fold(0, |a: Bits, (_, &c)| a | bit(c));
        orig[s] = (hand_now | played_by[s] | burned) & !taken[s];
        if count(orig[s]) != 5 {
            return Err(inconsistent(format!(
                "seat {s} would have been dealt {} cards",
                count(orig[s])
            )));
        }
    }
    let mut pack = turned_seq.clone();
    pack.extend(cards(placed[undealt_place]));
    let deck = build_deck(dealer, &orig, &pack).map_err(inconsistent)?;
    let mut m = Match::from_hand_start(t.start, deck, 0, Rules::HOUSE)
        .map_err(|e| bad(format!("match: {e}")))?;

    replay(t, &mut m, &all_plays, &seq, &burn_cards, pending_rob)?;

    // The replay must be the app's table.
    let turned_now = m.turned_teams();
    for (team, (&engine_turned, &app_turned)) in turned_now.iter().zip(&t.penalised).enumerate() {
        if engine_turned != app_turned {
            return Err(inconsistent(format!(
                "pair {team} is {}turned in the app but {}in the engine",
                if app_turned { "" } else { "not " },
                if engine_turned { "" } else { "not " }
            )));
        }
    }
    let view = m.view(me);
    // A take whose burn is due has not happened in the app yet: the engine holds that card too.
    let mut want_hand = t.own;
    if let Phase::AwaitBurn(s) = m.phase()
        && s == me
        && let Some(&(c, Some(by))) = view.public.turned.last()
        && by == me
    {
        want_hand |= bit(c);
    }
    if view.hand != want_hand {
        return Err(inconsistent("the seat's cards differ after the replay"));
    }
    Ok(Rebuilt { m, view })
}

/// The seat that did not rob at its turn in trick 1 (Guide: its pair was turned then), if a pair
/// was turned with no renege of its own caught and not for the dealer's refused ace or left
/// Joker: the first seat of that pair, in trick-1 order from the dealer's left, with no trick-1
/// play.
fn no_rob_seat(t: &AppTable, first: &[(u8, Card)]) -> Option<u8> {
    let d = t.start.dealer;
    let ace_turned = engine::cards::rank_of(t.turned) == engine::cards::ACE;
    let dealer_pre = t.joker_left || (ace_turned && t.taken_by != Some(d));
    for team in 0..4u8 {
        let seats = [team, team + 4];
        if !t.penalised[usize::from(team)] {
            continue;
        }
        if seats.iter().any(|&s| {
            let s = usize::from(s);
            t.caught_rob[s] || t.caught_false_rob[s] || t.caught_burned_ace[s]
        }) {
            continue;
        }
        if dealer_pre && team == team_of(d) {
            continue;
        }
        for i in 1..=8u8 {
            let s = (d + i) % 8;
            if team_of(s) == team && !first.iter().any(|&(x, _)| x == s) {
                return Some(s);
            }
        }
    }
    None
}

/// Replay the hand: the burns, the plays and the app's referee steps, in order, up to the
/// decision.
fn replay(
    t: &AppTable,
    m: &mut Match,
    plays: &[(usize, u8, Card)],
    seq: &[(u8, BurnKind)],
    burn_cards: &[Card],
    pending_rob: bool,
) -> Result<(), Refusal> {
    let hand = m.hand_index();
    let led_refusal = || {
        Refusal::new(
            "led_trick_in_progress",
            TurnRefusal::LedTrickInProgress.reason(),
        )
    };
    let mut i = 0; // next play
    let mut b = 0; // next burn
    for _ in 0..200 {
        if m.hand_index() != hand || m.phase() == Phase::MatchOver {
            return Err(inconsistent("the hand ended before the decision"));
        }
        let next_play = plays.get(i).map(|&(_, s, c)| (s, c));
        match m.phase() {
            Phase::AwaitBurn(s) => {
                if b < seq.len() && seq[b].0 == s {
                    let card = burn_cards[b];
                    let ace = m.public().trump.map(ace_of_trumps);
                    if seq[b].1 == BurnKind::AtTurn && Some(card) == ace {
                        m.let_through(Lapse::BurnAce)
                            .map_err(|e| inconsistent(format!("a burned ace: {e}")))?;
                    } else {
                        m.apply(Action::Burn(card))
                            .map_err(|e| inconsistent(format!("a burn: {e}")))?;
                    }
                    b += 1;
                } else if let Some((ps, pc)) = next_play
                    && ps == s
                    && t.caught_rob[usize::from(s)]
                {
                    m.let_through(Lapse::PlayWithoutRob(pc)).map_err(|e| {
                        if e.why == engine::game::LapseRefusal::LedTrickInProgress {
                            led_refusal()
                        } else {
                            inconsistent(format!("a play without robbing: {e}"))
                        }
                    })?;
                    i += 1;
                } else if next_play.is_none() && s == t.to_act {
                    let want = match t.phase {
                        AppPhase::JokerSteal => m.public().trump.is_none(),
                        AppPhase::DealerRob => {
                            m.public().trump.is_some() && m.public().trick_no == 0
                        }
                        AppPhase::Play => pending_rob && m.public().trick_no == 1,
                    };
                    if !want {
                        return Err(inconsistent(
                            "the engine awaits a burn the app does not show",
                        ));
                    }
                    return Ok(());
                } else if t.penalised[usize::from(s)] {
                    m.turn_team(team_of(s)).map_err(|e| turn_refusal(e.why))?;
                } else {
                    return Err(inconsistent(format!(
                        "the engine awaits seat {s}'s burn, the app shows otherwise"
                    )));
                }
            }
            Phase::AwaitPlay(s) => {
                match next_play {
                    Some((ps, pc)) if ps == s => {
                        if t.taken_by == Some(s) && b < seq.len() && seq[b] == (s, BurnKind::AtTurn)
                        {
                            // The app's rob by a seat the deal gives no ace of trumps: a false
                            // rob (caught, or the turned ace a dealer refused).
                            m.let_through(Lapse::FalseRob(burn_cards[b]))
                                .map_err(|e| inconsistent(format!("a false rob: {e}")))?;
                            b += 1;
                            continue;
                        }
                        if let Err(e) = m.apply(Action::Play(pc)) {
                            if e.reason == TurnRefusal::LedTrickInProgress.reason() {
                                return Err(led_refusal());
                            }
                            m.let_through(Lapse::Play(pc)).map_err(|e| {
                                if e.why == engine::game::LapseRefusal::LedTrickInProgress {
                                    led_refusal()
                                } else {
                                    inconsistent(format!("a play: {e}"))
                                }
                            })?;
                        }
                        i += 1;
                    }
                    None if s == t.to_act => {
                        if t.phase != AppPhase::Play || pending_rob {
                            return Err(inconsistent("the engine awaits a play the app does not"));
                        }
                        return Ok(());
                    }
                    _ if t.penalised[usize::from(s)] => {
                        m.turn_team(team_of(s)).map_err(|e| turn_refusal(e.why))?;
                    }
                    _ => {
                        return Err(inconsistent(format!(
                            "the engine awaits seat {s}'s card, the app shows {}",
                            match next_play {
                                Some((ps, _)) => format!("seat {ps}'s"),
                                None => format!("seat {} to act", t.to_act),
                            }
                        )));
                    }
                }
            }
            Phase::MatchOver => unreachable!("checked above"),
        }
    }
    Err(inconsistent("the hand does not reach the decision"))
}

fn turn_refusal(why: TurnRefusal) -> Refusal {
    if why == TurnRefusal::LedTrickInProgress {
        Refusal::new("led_trick_in_progress", why.reason())
    } else {
        inconsistent(format!("a pair cannot be turned there: {}", why.reason()))
    }
}
