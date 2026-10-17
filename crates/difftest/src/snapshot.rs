//! The canonical snapshot: everything the engine and the reference expose about a match, in one
//! shared form, so the two can be compared with `==` and rendered as text when they differ.
//!
//! Card sets (hands, burns, the undealt pack, the cards played) are sorted lists of card indices;
//! the legal actions are sorted by card index; each seat's events are the canonical strings of
//! docs/M0-SPEC.md section 5.

use engine::cards::{Bits, cards, name};
use engine::game::{Action as EAction, Match as EMatch, Phase as EPhase};
use engine::view::{BurnKind, PublicState, SeatView};

/// Seats at the table.
pub const SEATS: usize = 8;

/// What the match is waiting for, in the shared form. `NeedsDeck` exists only in the reference
/// (between hands); the driver always deals before comparing, so it never appears in a comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapPhase {
    /// This seat must burn.
    AwaitBurn(u8),
    /// This seat must play.
    AwaitPlay(u8),
    /// The match is over.
    MatchOver,
    /// The reference is waiting for the next deck.
    NeedsDeck,
}

/// A legal action, in the shared form (card index).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SnapAction {
    /// Burn this card.
    Burn(u8),
    /// Play this card.
    Play(u8),
}

impl SnapAction {
    /// The card the action names.
    pub fn card(self) -> u8 {
        match self {
            SnapAction::Burn(c) | SnapAction::Play(c) => c,
        }
    }
}

impl From<EAction> for SnapAction {
    fn from(a: EAction) -> Self {
        match a {
            EAction::Burn(c) => SnapAction::Burn(c),
            EAction::Play(c) => SnapAction::Play(c),
        }
    }
}

impl From<reference::Action> for SnapAction {
    fn from(a: reference::Action) -> Self {
        match a {
            reference::Action::Burn(c) => SnapAction::Burn(c),
            reference::Action::Play(c) => SnapAction::Play(c),
        }
    }
}

/// Every field of the public state (engine `PublicState`, reference `Public`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicSnap {
    pub phase: SnapPhase,
    pub to_act: Option<u8>,
    pub dealer: u8,
    pub hand_index: u32,
    pub game_index: u8,
    pub scores: [u8; 4],
    pub games_won: [u8; 4],
    pub bonus_taken: [bool; 4],
    /// Suit index (clubs 0, diamonds 1, hearts 2, spades 3).
    pub trump: Option<u8>,
    pub table_card: Option<u8>,
    pub turned: Vec<(u8, Option<u8>)>,
    pub joker_turned: bool,
    pub hand_sizes: [u8; 8],
    pub burn_counts: [u8; 8],
    /// Seat and kind word (`joker_take`, `ace_rob_at_once`, `rob_at_turn`) of every burn.
    pub burn_order: Vec<(u8, &'static str)>,
    pub hidden_count: u8,
    /// Sorted.
    pub played: Vec<u8>,
    pub trick_no: u8,
    pub leader: u8,
    pub trick_cards: Vec<(u8, u8)>,
    pub trick_winners: Vec<u8>,
    pub tricks_won: [u8; 4],
    pub best_trump: Option<(u8, u8)>,
    pub five_played: bool,
    pub robber: Option<u8>,
    pub proved_no_ace: [bool; 8],
    pub ace_undealt: bool,
    /// Partnerships turned this hand (rules 39-41, 34).
    pub turned_teams: [bool; 4],
}

/// Everything in a snapshot except the event lists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    /// `phase()`.
    pub phase: SnapPhase,
    /// `to_act()`.
    pub to_act: Option<u8>,
    /// `legal_actions()`, sorted by card index.
    pub legal: Vec<SnapAction>,
    /// `dealer()`.
    pub dealer: u8,
    /// `hand_index()`.
    pub hand_index: u32,
    /// `game_index()`.
    pub game_index: u8,
    /// `scores()`.
    pub scores: [u8; 4],
    /// `games_won()`.
    pub games_won: [u8; 4],
    /// `bonus_taken()`.
    pub bonus_taken: [bool; 4],
    /// `money()`.
    pub money: Option<[i32; 4]>,
    /// `public()`.
    pub public: PublicSnap,
    /// Each seat's hand, sorted.
    pub hands: [Vec<u8>; SEATS],
    /// Each seat's burned cards this hand, sorted.
    pub burns: [Vec<u8>; SEATS],
    /// The undealt pack, sorted.
    pub undealt: Vec<u8>,
    /// The reneges the app let through this hand (hidden): seat, kind word, caught.
    pub reneges: Vec<(u8, String, bool)>,
}

/// A full snapshot: the state plus each seat's canonical event list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    /// Everything but the events.
    pub state: State,
    /// Each seat's canonical events (public ones plus its private ones), in order.
    pub events: [Vec<String>; SEATS],
}

fn bits_sorted(set: Bits) -> Vec<u8> {
    cards(set).collect()
}

fn sorted(mut v: Vec<u8>) -> Vec<u8> {
    v.sort_unstable();
    v
}

fn engine_phase(p: EPhase) -> SnapPhase {
    match p {
        EPhase::AwaitBurn(s) => SnapPhase::AwaitBurn(s),
        EPhase::AwaitPlay(s) => SnapPhase::AwaitPlay(s),
        EPhase::MatchOver => SnapPhase::MatchOver,
    }
}

fn reference_phase(p: reference::Phase) -> SnapPhase {
    match p {
        reference::Phase::AwaitBurn(s) => SnapPhase::AwaitBurn(s),
        reference::Phase::AwaitPlay(s) => SnapPhase::AwaitPlay(s),
        reference::Phase::MatchOver => SnapPhase::MatchOver,
        reference::Phase::NeedsDeck => SnapPhase::NeedsDeck,
    }
}

fn burn_kind_word(k: BurnKind) -> &'static str {
    match k {
        BurnKind::JokerTake => "joker_take",
        BurnKind::AceRobAtOnce => "ace_rob_at_once",
        BurnKind::RobAtTurn => "rob_at_turn",
    }
}

impl PublicSnap {
    fn from_engine(p: &PublicState) -> PublicSnap {
        PublicSnap {
            phase: engine_phase(p.phase),
            to_act: p.to_act,
            dealer: p.dealer,
            hand_index: p.hand_index,
            game_index: p.game_index,
            scores: p.scores,
            games_won: p.games_won,
            bonus_taken: p.bonus_taken,
            trump: p.trump.map(|s| s as u8),
            table_card: p.table_card,
            turned: p.turned.clone(),
            joker_turned: p.joker_turned,
            hand_sizes: p.hand_sizes,
            burn_counts: p.burn_counts,
            burn_order: p
                .burn_order
                .iter()
                .map(|&(s, k)| (s, burn_kind_word(k)))
                .collect(),
            hidden_count: p.hidden_count,
            played: bits_sorted(p.played),
            trick_no: p.trick_no,
            leader: p.leader,
            trick_cards: p.trick_cards.clone(),
            trick_winners: p.trick_winners.clone(),
            tricks_won: p.tricks_won,
            best_trump: p.best_trump,
            five_played: p.five_played,
            robber: p.robber,
            proved_no_ace: p.proved_no_ace,
            ace_undealt: p.ace_undealt,
            turned_teams: p.turned_teams,
        }
    }

    fn from_reference(p: reference::Public) -> PublicSnap {
        PublicSnap {
            phase: reference_phase(p.phase),
            to_act: p.to_act,
            dealer: p.dealer,
            hand_index: p.hand_index,
            game_index: p.game_index,
            scores: p.scores,
            games_won: p.games_won,
            bonus_taken: p.bonus_taken,
            trump: p.trump,
            table_card: p.table_card,
            turned: p.turned,
            joker_turned: p.joker_turned,
            hand_sizes: p.hand_sizes,
            burn_counts: p.burn_counts,
            burn_order: p.burn_order,
            hidden_count: p.hidden_count,
            played: sorted(p.played),
            trick_no: p.trick_no,
            leader: p.leader,
            trick_cards: p.trick_cards,
            trick_winners: p.trick_winners,
            tricks_won: p.tricks_won,
            best_trump: p.best_trump,
            five_played: p.five_played,
            robber: p.robber,
            proved_no_ace: p.proved_no_ace,
            ace_undealt: p.ace_undealt,
            turned_teams: p.turned_teams,
        }
    }
}

/// The eight seat views of an engine match.
pub fn engine_views(m: &EMatch) -> [SeatView; SEATS] {
    core::array::from_fn(|s| m.view(s as u8))
}

impl State {
    /// The engine's state, with hands and burns from the seat views (what bots see).
    pub fn from_engine(m: &EMatch, views: &[SeatView; SEATS]) -> State {
        let hands: [Bits; SEATS] = core::array::from_fn(|s| views[s].hand);
        let burns: [Bits; SEATS] = core::array::from_fn(|s| views[s].own_burns);
        State::from_engine_parts(m, &hands, &burns)
    }

    /// The engine's state, with hands and burns from the hidden truth (no seat views built).
    pub fn from_engine_truth(m: &EMatch) -> State {
        let t = m.truth();
        State::from_engine_parts(m, &t.hands, &t.burns)
    }

    fn from_engine_parts(m: &EMatch, hands: &[Bits; SEATS], burns: &[Bits; SEATS]) -> State {
        let mut legal: Vec<SnapAction> = m.legal_actions().into_iter().map(Into::into).collect();
        legal.sort_unstable_by_key(|a| a.card());
        State {
            phase: engine_phase(m.phase()),
            to_act: m.to_act(),
            legal,
            dealer: m.dealer(),
            hand_index: m.hand_index(),
            game_index: m.game_index(),
            scores: m.scores(),
            games_won: m.games_won(),
            bonus_taken: m.bonus_taken(),
            money: m.money(),
            public: PublicSnap::from_engine(&m.public()),
            hands: core::array::from_fn(|s| bits_sorted(hands[s])),
            burns: core::array::from_fn(|s| bits_sorted(burns[s])),
            undealt: bits_sorted(m.truth().undealt),
            reneges: m
                .reneges()
                .iter()
                .map(|r| (r.seat, r.kind.canonical(), r.caught))
                .collect(),
        }
    }

    /// The reference's state.
    pub fn from_reference(r: &reference::Match) -> State {
        let mut legal: Vec<SnapAction> = r.legal_actions().into_iter().map(Into::into).collect();
        legal.sort_unstable_by_key(|a| a.card());
        State {
            phase: reference_phase(r.phase()),
            to_act: r.to_act(),
            legal,
            dealer: r.dealer(),
            hand_index: r.hand_index(),
            game_index: r.game_index(),
            scores: r.scores(),
            games_won: r.games_won(),
            bonus_taken: r.bonus_taken(),
            money: r.money(),
            public: PublicSnap::from_reference(r.public()),
            hands: core::array::from_fn(|s| sorted(r.hand(s as u8))),
            burns: core::array::from_fn(|s| sorted(r.burns(s as u8))),
            undealt: sorted(r.undealt()),
            reneges: r.reneges(),
        }
    }

    /// Every field as a `(name, value)` line, in a fixed order. Values render injectively, so two
    /// states are equal exactly when all their lines are equal.
    pub fn fields(&self) -> Vec<(String, String)> {
        let p = &self.public;
        let mut f: Vec<(String, String)> = vec![
            ("phase".into(), phase_text(self.phase)),
            ("to_act".into(), opt_text(self.to_act, |s| s.to_string())),
            ("legal".into(), legal_text(&self.legal)),
            ("dealer".into(), self.dealer.to_string()),
            ("hand_index".into(), self.hand_index.to_string()),
            ("game_index".into(), self.game_index.to_string()),
            ("scores".into(), list_text(&self.scores)),
            ("games_won".into(), list_text(&self.games_won)),
            ("bonus_taken".into(), list_text(&self.bonus_taken)),
            ("money".into(), opt_text(self.money, |m| list_text(&m))),
            ("public.phase".into(), phase_text(p.phase)),
            (
                "public.to_act".into(),
                opt_text(p.to_act, |s| s.to_string()),
            ),
            ("public.dealer".into(), p.dealer.to_string()),
            ("public.hand_index".into(), p.hand_index.to_string()),
            ("public.game_index".into(), p.game_index.to_string()),
            ("public.scores".into(), list_text(&p.scores)),
            ("public.games_won".into(), list_text(&p.games_won)),
            ("public.bonus_taken".into(), list_text(&p.bonus_taken)),
            ("public.trump".into(), opt_text(p.trump, suit_text)),
            (
                "public.table_card".into(),
                opt_text(p.table_card, card_text),
            ),
            ("public.turned".into(), turned_text(&p.turned)),
            ("public.joker_turned".into(), p.joker_turned.to_string()),
            ("public.hand_sizes".into(), list_text(&p.hand_sizes)),
            ("public.burn_counts".into(), list_text(&p.burn_counts)),
            (
                "public.burn_order".into(),
                p.burn_order
                    .iter()
                    .map(|(s, k)| format!("{s}:{k}"))
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            ("public.hidden_count".into(), p.hidden_count.to_string()),
            ("public.played".into(), cards_text(&p.played)),
            ("public.trick_no".into(), p.trick_no.to_string()),
            ("public.leader".into(), p.leader.to_string()),
            ("public.trick_cards".into(), seat_cards_text(&p.trick_cards)),
            ("public.trick_winners".into(), list_text(&p.trick_winners)),
            ("public.tricks_won".into(), list_text(&p.tricks_won)),
            (
                "public.best_trump".into(),
                opt_text(p.best_trump, |(c, s)| {
                    format!("{} by seat {s}", card_text(c))
                }),
            ),
            ("public.five_played".into(), p.five_played.to_string()),
            (
                "public.robber".into(),
                opt_text(p.robber, |s| s.to_string()),
            ),
            ("public.proved_no_ace".into(), seats_text(&p.proved_no_ace)),
            ("public.ace_undealt".into(), p.ace_undealt.to_string()),
            ("public.turned_teams".into(), list_text(&p.turned_teams)),
        ];
        for (s, h) in self.hands.iter().enumerate() {
            f.push((format!("hand[{s}]"), cards_text(h)));
        }
        for (s, b) in self.burns.iter().enumerate() {
            f.push((format!("burns[{s}]"), cards_text(b)));
        }
        f.push(("undealt".into(), cards_text(&self.undealt)));
        let reneges: Vec<String> = self
            .reneges
            .iter()
            .map(|(s, k, c)| format!("{s}:{k}:{}", if *c { "caught" } else { "hidden" }))
            .collect();
        f.push(("reneges".into(), reneges.join(",")));
        f
    }
}

impl Snapshot {
    /// A full snapshot of an engine match, built from scratch through its public API.
    pub fn from_engine(m: &EMatch) -> Snapshot {
        let views = engine_views(m);
        let state = State::from_engine(m, &views);
        Snapshot {
            state,
            events: core::array::from_fn(|s| views[s].canonical_events()),
        }
    }

    /// A full snapshot of a reference match.
    pub fn from_reference(r: &reference::Match) -> Snapshot {
        Snapshot {
            state: State::from_reference(r),
            events: core::array::from_fn(|s| r.events_for(s as u8)),
        }
    }

    /// The snapshot as text: every state field, then each seat's events of the current hand and
    /// the last few before it (the full count is shown; [`Snapshot::differences`] pinpoints any
    /// earlier difference).
    pub fn render(&self) -> String {
        const BEFORE: usize = 12;
        let mut out = String::new();
        for (k, v) in self.state.fields() {
            out.push_str(&format!("  {k:<22} {v}\n"));
        }
        for (s, ev) in self.events.iter().enumerate() {
            let hand = ev
                .iter()
                .rposition(|e| e.starts_with("HandStart "))
                .unwrap_or(0);
            let from = hand.saturating_sub(BEFORE);
            out.push_str(&format!(
                "  events seat {s}: {} in all; from #{from} (this hand starts at #{hand}):\n",
                ev.len()
            ));
            for (i, e) in ev.iter().enumerate().skip(from) {
                out.push_str(&format!("    #{i:<4} {e}\n"));
            }
        }
        out
    }

    /// Plain-text lines naming every field where `self` (called `mine`) and `other` (called
    /// `theirs`) differ, with both values. Empty exactly when the snapshots are equal.
    pub fn differences(&self, other: &Snapshot, mine: &str, theirs: &str) -> Vec<String> {
        let mut out = Vec::new();
        let a = self.state.fields();
        let b = other.state.fields();
        for ((ka, va), (_, vb)) in a.iter().zip(&b) {
            if va != vb {
                out.push(format!("{ka}: {mine} `{va}` | {theirs} `{vb}`"));
            }
        }
        for s in 0..SEATS {
            let (ea, eb) = (&self.events[s], &other.events[s]);
            if ea == eb {
                continue;
            }
            let i = ea.iter().zip(eb).take_while(|(x, y)| x == y).count();
            let at = |v: &Vec<String>| v.get(i).cloned().unwrap_or_else(|| "(none)".into());
            out.push(format!(
                "events seat {s}: {mine} has {}, {theirs} has {}; first difference at #{i}: \
                 {mine} `{}` | {theirs} `{}`",
                ea.len(),
                eb.len(),
                at(ea),
                at(eb)
            ));
        }
        if out.is_empty() && self != other {
            out.push("the snapshots differ but render identically (a difftest bug)".into());
        }
        out
    }
}

/// A card's canonical name, or `#n` for an index that is not a card.
pub fn card_text(c: u8) -> String {
    if c <= engine::cards::JOKER {
        name(c)
    } else {
        format!("#{c}")
    }
}

fn cards_text(cs: &[u8]) -> String {
    if cs.is_empty() {
        "-".into()
    } else {
        cs.iter()
            .map(|&c| card_text(c))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn suit_text(s: u8) -> String {
    match s {
        0 => "C".into(),
        1 => "D".into(),
        2 => "H".into(),
        3 => "S".into(),
        n => format!("#{n}"),
    }
}

fn phase_text(p: SnapPhase) -> String {
    match p {
        SnapPhase::AwaitBurn(s) => format!("AwaitBurn {s}"),
        SnapPhase::AwaitPlay(s) => format!("AwaitPlay {s}"),
        SnapPhase::MatchOver => "MatchOver".into(),
        SnapPhase::NeedsDeck => "NeedsDeck".into(),
    }
}

/// An action as text, such as `Play 5H`.
pub fn action_text(a: SnapAction) -> String {
    match a {
        SnapAction::Burn(c) => format!("Burn {}", card_text(c)),
        SnapAction::Play(c) => format!("Play {}", card_text(c)),
    }
}

fn legal_text(l: &[SnapAction]) -> String {
    if l.is_empty() {
        "-".into()
    } else {
        l.iter()
            .map(|&a| action_text(a))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn opt_text<T>(o: Option<T>, f: impl FnOnce(T) -> String) -> String {
    match o {
        None => "none".into(),
        Some(v) => f(v),
    }
}

fn list_text<T: ToString>(v: &[T]) -> String {
    if v.is_empty() {
        "-".into()
    } else {
        v.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn seats_text(flags: &[bool; SEATS]) -> String {
    let seats: Vec<String> = (0..SEATS)
        .filter(|&s| flags[s])
        .map(|s| s.to_string())
        .collect();
    if seats.is_empty() {
        "no seats".into()
    } else {
        format!("seats {}", seats.join(","))
    }
}

fn seat_cards_text(v: &[(u8, u8)]) -> String {
    if v.is_empty() {
        "-".into()
    } else {
        v.iter()
            .map(|&(s, c)| format!("{s}:{}", card_text(c)))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn turned_text(v: &[(u8, Option<u8>)]) -> String {
    if v.is_empty() {
        "-".into()
    } else {
        v.iter()
            .map(|&(c, who)| match who {
                Some(s) => format!("{} taken by {s}", card_text(c)),
                None => format!("{} not taken", card_text(c)),
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}
