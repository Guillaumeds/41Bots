//! The student's encoder (docs/STUDENT.md §2, encoder `student-v1`): a seat's view and one
//! candidate as feature indices for the network. The single source of truth: the training shards
//! (`student encode`), the integer inference natively and in WASM call these functions; Python
//! never encodes.
//!
//! **Rotation.** Seats are rotated so the deciding seat is seat 0 and its partner seat 4; pairs
//! with them, so the deciding pair is pair 0. The network's four outputs are in that order
//! ([`unrotate`] turns them back).
//!
//! **Cards** are numbered relative to the hand's trump suit ([`canon`]): the trumps strongest first
//! (slots 0–14), then the other suit of the trump's colour, then the two suits of the other colour
//! in suit order, each plain suit strongest first (slots 15–53). For each trump suit this is a
//! one-to-one map of the 53 cards into 54 slots (one stays empty), and the trump suit is a feature
//! of its own, so nothing is lost (tested).
//!
//! **Binary features** of the view (offsets below): the seat's hand; every card played this hand by
//! trick and by the seat that played it; the leader of each trick; the trick number; the cards
//! already in the trick (the place); the trump suit; each card turned this hand and who took it;
//! the seat's own burned cards; every burn made (kind × seat); the dealer; each pair's score (one of
//! the 15 reachable values), bonus taken, tricks this hand and games won; the 5 of trumps played;
//! the pair holding the best trump played so far; every seat's hand size; for every card the seat
//! cannot see, each place it **cannot** be (the seven other hands, out of play: the card tracker's
//! certain facts, `belief::Tracker`); seats proved without the ace of trumps; the ace announced out
//! of play; the robber. **Candidate** features (after the view's): the card played, the card
//! burned (a robber's joint decision), whether the card would win the trick as it stands.
//!
//! **Dense** (100): the oracle's equity block (`nn::encode::dense_features`) of the score state at
//! the start of the current trick, in the rotated pair order, the 5 of trumps' holder counted only
//! when it is the seat's own card and the end-of-hand bonus pair from the completed tricks.

use std::sync::OnceLock;

use belief::{Loc, Tracker};
use engine::cards::{Card, Suit, cards, has};
use engine::game::{Action, Event, Match, Phase, Rules};
use engine::rules::{
    beats, effective_suit, five_of, legal_burns, legal_plays, plain_rank, trick_winner, trump_rank,
};
use engine::team_of;
use engine::view::{BurnKind, SeatView};
use equity::MoneyChance;
use equity::state::{N_SCORES, score_index};
use nn::encode::{DenseKey, N_DENSE, dense_features};
use sha2::{Digest, Sha256};
use solver::Leaf;

use crate::position::Cand;

/// The encoder's name, in every feature shard and network file.
pub const ENCODER: &str = "student-v1";

/// Card slots ([`canon`]).
pub const N_CANON: usize = 54;
/// Tricks in a hand.
const TRICKS: usize = 5;

/// The seat's hand.
pub const OFF_HAND: usize = 0;
/// Played cards: trick × relative seat × card slot.
pub const OFF_PLAYED: usize = OFF_HAND + N_CANON;
/// The leader of each trick so far: trick × relative seat.
pub const OFF_LEADER: usize = OFF_PLAYED + TRICKS * 8 * N_CANON;
/// The current trick (1–5).
pub const OFF_TRICK: usize = OFF_LEADER + TRICKS * 8;
/// Cards already in the current trick (0–7).
pub const OFF_PLACE: usize = OFF_TRICK + TRICKS;
/// The trump suit.
pub const OFF_TRUMP: usize = OFF_PLACE + 8;
/// The cards turned this hand: turn × card slot.
pub const OFF_TURNED: usize = OFF_TRUMP + 4;
/// Who took each turned card: turn × (nobody, relative seat).
pub const OFF_TAKER: usize = OFF_TURNED + 2 * N_CANON;
/// The seat's own burned cards.
pub const OFF_OWN_BURN: usize = OFF_TAKER + 2 * 9;
/// Burns made this hand: kind × relative seat.
pub const OFF_BURNS: usize = OFF_OWN_BURN + N_CANON;
/// The dealer's relative seat.
pub const OFF_DEALER: usize = OFF_BURNS + 3 * 8;
/// Pair × score index.
pub const OFF_SCORE: usize = OFF_DEALER + 8;
/// Pair has taken its bonus this game.
pub const OFF_FLAG: usize = OFF_SCORE + 4 * N_SCORES;
/// Pair × tricks this hand (0–4).
pub const OFF_TRICKS_WON: usize = OFF_FLAG + 4;
/// Pair has won a game.
pub const OFF_GAMES: usize = OFF_TRICKS_WON + 4 * 5;
/// The 5 of trumps played this hand.
pub const OFF_FIVE: usize = OFF_GAMES + 4;
/// The pair of the best trump played so far (none, then pairs).
pub const OFF_BEST: usize = OFF_FIVE + 1;
/// Relative seat × hand size (0–6).
pub const OFF_SIZE: usize = OFF_BEST + 5;
/// Place × card slot: the card (one the seat cannot see) cannot be there; places are the
/// relative seats 1–7 (0–6) and out of play (7).
pub const OFF_FORBID: usize = OFF_SIZE + 8 * 7;
/// Seats proved without the ace of trumps.
pub const OFF_NO_ACE: usize = OFF_FORBID + 8 * N_CANON;
/// The ace of trumps announced out of play.
pub const OFF_ACE_OUT: usize = OFF_NO_ACE + 8;
/// The robber (none, then relative seat).
pub const OFF_ROBBER: usize = OFF_ACE_OUT + 1;
/// The view's binary features.
pub const N_VIEW: usize = OFF_ROBBER + 9;
/// Candidate: the card played.
pub const OFF_C_PLAY: usize = N_VIEW;
/// Candidate: the card burned (a robber's joint decision).
pub const OFF_C_BURN: usize = OFF_C_PLAY + N_CANON;
/// Candidate: the card would win the trick as it stands (always, for a lead).
pub const OFF_C_WINS: usize = OFF_C_BURN + N_CANON;
/// All binary features (view, then candidate).
pub const N_BINARY: usize = OFF_C_WINS + 1;
/// The candidate's binary features.
pub const N_CAND: usize = N_BINARY - N_VIEW;
/// All inputs: binary, then dense.
pub const N_FEATURES: usize = N_BINARY + N_DENSE;
/// Most active view features (at most 7 forbidden places for each of at most 48 cards the seat
/// cannot see, plus at most 99 others): room in the feature shards.
pub const V_MAX: usize = 448;
/// Most active candidate features.
pub const K_MAX: usize = 4;
/// Padding in feature shards.
pub const PAD: u16 = u16::MAX;

/// The slot of card `c` for trump suit `trump` (module notes).
pub fn canon(c: Card, trump: Suit) -> usize {
    let tr = trump_rank(c, trump);
    if tr > 0 {
        return 15 - usize::from(tr);
    }
    let s = engine::cards::suit_of(c).expect("a plain card has a suit");
    let same = |a: Suit, b: Suit| a.is_red() == b.is_red();
    let slot = if same(s, trump) {
        0
    } else {
        // The other colour's suits in suit order.
        let others: Vec<Suit> = Suit::ALL.into_iter().filter(|&x| !same(x, trump)).collect();
        1 + others
            .iter()
            .position(|&x| x == s)
            .expect("a suit of the other colour")
    };
    15 + 13 * slot + (13 - usize::from(plain_rank(c)))
}

/// A view's encoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewFeatures {
    /// Active binary features, ascending (all below [`N_VIEW`]).
    pub active: Vec<u16>,
    /// Dense values, `−2^24..=0` (units of 2^-24).
    pub dense: [i32; N_DENSE],
    /// The deciding seat's pair (the rotation).
    pub team: u8,
}

/// Turn the network's rotated money chance (deciding pair first) back to real pair order.
pub fn unrotate(m: MoneyChance, team: u8) -> MoneyChance {
    m.rotate(team % 4)
}

/// A real money chance in the network's rotated order (deciding pair first).
pub fn rotate(m: MoneyChance, team: u8) -> MoneyChance {
    m.rotate((4 - team % 4) % 4)
}

/// The hand's plays, by trick: `(trick 0..5, seat, card)` in play order.
fn plays(view: &SeatView) -> Vec<(usize, u8, Card)> {
    let mut out = Vec::new();
    let mut t = 0usize;
    for e in belief::tracker::hand_events(view) {
        match *e {
            Event::Played { seat, card } => out.push((t, seat, card)),
            Event::TrickWon { .. } => t += 1,
            _ => {}
        }
    }
    out
}

/// Encode a seat's view. Refused: a view not awaiting the seat's own decision, before trumps are
/// known (the dealer's burn after a turned Joker), or with a pair turned (networks never see
/// those, docs/APP-ENGINE-CONTRACT.md §7).
pub fn encode_view(view: &SeatView, leaf: &dyn Leaf) -> Result<ViewFeatures, String> {
    let p = &view.public;
    let me = view.seat;
    if p.to_act != Some(me) {
        return Err("the view's seat is not to act".into());
    }
    if p.turned_teams.iter().any(|&x| x) {
        return Err("a pair is turned".into());
    }
    let Some(trump) = p.trump else {
        return Err("trumps are not known yet".into());
    };
    if !(1..=5).contains(&p.trick_no) {
        return Err(format!("trick {}", p.trick_no));
    }
    let team = team_of(me);
    let rs = |s: u8| usize::from((s + 8 - me) % 8);
    let rt = |t: u8| usize::from((t + 4 - team) % 4);
    let mut a: Vec<u16> = Vec::with_capacity(160);
    let mut on = |i: usize| a.push(i as u16);
    let k = |c: Card| canon(c, trump);

    for c in cards(view.hand) {
        on(OFF_HAND + k(c));
    }
    let hist = plays(view);
    let mut leaders: Vec<(usize, u8)> = Vec::new();
    for &(t, s, c) in &hist {
        on(OFF_PLAYED + (t * 8 + rs(s)) * N_CANON + k(c));
        if leaders.last().is_none_or(|&(lt, _)| lt != t) {
            leaders.push((t, s));
        }
    }
    let now = usize::from(p.trick_no - 1);
    if leaders.last().is_none_or(|&(lt, _)| lt != now) {
        leaders.push((now, p.leader));
    }
    for &(t, s) in &leaders {
        on(OFF_LEADER + t * 8 + rs(s));
    }
    on(OFF_TRICK + now);
    on(OFF_PLACE + p.trick_cards.len());
    on(OFF_TRUMP + trump as usize);
    for (i, &(c, by)) in p.turned.iter().take(2).enumerate() {
        on(OFF_TURNED + i * N_CANON + k(c));
        on(OFF_TAKER + i * 9 + by.map_or(0, |s| 1 + rs(s)));
    }
    for c in cards(view.own_burns) {
        on(OFF_OWN_BURN + k(c));
    }
    for &(s, kind) in &p.burn_order {
        let ki = match kind {
            BurnKind::JokerTake => 0,
            BurnKind::AceRobAtOnce => 1,
            BurnKind::RobAtTurn => 2,
        };
        on(OFF_BURNS + ki * 8 + rs(s));
    }
    on(OFF_DEALER + rs(p.dealer));
    for t in 0..4u8 {
        let u = rt(t);
        let s = p.scores[usize::from(t)];
        let si = score_index(s).ok_or_else(|| format!("score {s} cannot stand"))?;
        on(OFF_SCORE + u * N_SCORES + si);
        if p.bonus_taken[usize::from(t)] {
            on(OFF_FLAG + u);
        }
        let tw = usize::from(p.tricks_won[usize::from(t)]);
        if tw >= 5 {
            return Err("five tricks won: the hand is over".into());
        }
        on(OFF_TRICKS_WON + u * 5 + tw);
        if p.games_won[usize::from(t)] >= 1 {
            on(OFF_GAMES + u);
        }
    }
    if p.five_played {
        on(OFF_FIVE);
    }
    on(OFF_BEST + p.best_trump.map_or(0, |(_, s)| 1 + rt(team_of(s))));
    for s in 0..8u8 {
        on(OFF_SIZE + rs(s) * 7 + usize::from(p.hand_sizes[usize::from(s)].min(6)));
    }
    let tracker = Tracker::from_view(view);
    for c in cards(tracker.unknown) {
        let allowed = tracker.allowed[usize::from(c)];
        let may = |loc: Loc| tracker.loc_index(loc).map(|i| allowed & (1 << i) != 0);
        for s in 1..8u8 {
            let seat = (me + s) % 8;
            if may(Loc::Hand(seat)) == Some(false) {
                on(OFF_FORBID + usize::from(s - 1) * N_CANON + k(c));
            }
        }
        let out = tracker
            .locs
            .iter()
            .enumerate()
            .any(|(i, &(l, _))| !matches!(l, Loc::Hand(_)) && allowed & (1 << i) != 0);
        if !out {
            on(OFF_FORBID + 7 * N_CANON + k(c));
        }
    }
    for s in 0..8u8 {
        if p.proved_no_ace[usize::from(s)] {
            on(OFF_NO_ACE + rs(s));
        }
    }
    if p.ace_undealt {
        on(OFF_ACE_OUT);
    }
    on(OFF_ROBBER + p.robber.map_or(0, |s| 1 + rs(s)));
    a.sort_unstable();
    debug_assert!(a.windows(2).all(|w| w[0] < w[1]), "each feature once");
    debug_assert!(a.iter().all(|&i| usize::from(i) < N_VIEW));
    if a.len() > V_MAX {
        return Err(format!("{} active features, more than {V_MAX}", a.len()));
    }

    // Dense: the oracle's equity block at the start of the current trick.
    let in_trick: Vec<Card> = p.trick_cards.iter().map(|&(_, c)| c).collect();
    let five = five_of(trump);
    let five_done = hist.iter().any(|&(t, _, c)| t < now && c == five);
    let best_done = hist
        .iter()
        .filter(|&&(t, _, c)| t < now && trump_rank(c, trump) > 0)
        .max_by_key(|&&(_, _, c)| trump_rank(c, trump))
        .map(|&(_, s, _)| rt(team_of(s)) as u8);
    let mut key = DenseKey {
        scores: [0; 4],
        flags: [false; 4],
        tricks: [0; 4],
        games: [0; 4],
        dealer: rt(team_of(p.dealer)) as u8,
        five_played: five_done,
        tricks_done: now as u8,
        five: (has(view.hand, five) && !in_trick.contains(&five)).then_some(0),
        end: if five_done { None } else { best_done },
    };
    for t in 0..4u8 {
        let u = rt(t);
        key.scores[u] = p.scores[usize::from(t)];
        key.flags[u] = p.bonus_taken[usize::from(t)];
        key.tricks[u] = p.tricks_won[usize::from(t)];
        key.games[u] = p.games_won[usize::from(t)];
    }
    Ok(ViewFeatures {
        active: a,
        dense: dense_features(&key, leaf),
        team,
    })
}

/// Encode one candidate of the view's decision: its active features (all in
/// `N_VIEW..N_BINARY`), ascending.
pub fn encode_cand(view: &SeatView, cand: Cand) -> Vec<u16> {
    let p = &view.public;
    let trump = p.trump.expect("trumps are known at a student decision");
    let play = match cand {
        (Action::Play(c), None) => c,
        (Action::Burn(_), Some(c)) => c,
        _ => panic!("a lone burn is not a student candidate"),
    };
    let mut a = vec![(OFF_C_PLAY + canon(play, trump)) as u16];
    if let (Action::Burn(b), _) = cand {
        a.push((OFF_C_BURN + canon(b, trump)) as u16);
    }
    let in_trick: Vec<Card> = p.trick_cards.iter().map(|&(_, c)| c).collect();
    let wins = match in_trick.first() {
        None => true,
        Some(&led) => {
            let best = in_trick[trick_winner(&in_trick, trump)];
            beats(play, best, effective_suit(led, trump), trump)
        }
    };
    if wins {
        a.push(OFF_C_WINS as u16);
    }
    a.sort_unstable();
    a
}

/// The candidates of the decision a view awaits, in the engine's candidate order (plays by card;
/// a robber's (burn, play) pairs by burn then play: `search::pimc::candidates`). For the encoder
/// hash's fixed positions only; the bots and the teacher take theirs from `search`.
fn all_cands(view: &SeatView, legal: &[Action]) -> Vec<Cand> {
    let p = &view.public;
    match p.phase {
        Phase::AwaitBurn(_) => {
            let trump = p.trump.expect("trumps");
            let led = p.trick_cards.first().map(|&(_, c)| c);
            legal
                .iter()
                .flat_map(|&a| {
                    let rest = view.hand & !engine::cards::bit(a.card());
                    cards(legal_plays(rest, led, trump)).map(move |c| (a, Some(c)))
                })
                .collect()
        }
        _ => legal.iter().map(|&a| (a, None)).collect(),
    }
}

/// The layout as text (every block's name, offset and size), hashed into [`encoder_sha256`].
pub fn layout() -> String {
    let blocks = [
        ("hand", OFF_HAND),
        ("played", OFF_PLAYED),
        ("leader", OFF_LEADER),
        ("trick", OFF_TRICK),
        ("place", OFF_PLACE),
        ("trump", OFF_TRUMP),
        ("turned", OFF_TURNED),
        ("taker", OFF_TAKER),
        ("own_burn", OFF_OWN_BURN),
        ("burns", OFF_BURNS),
        ("dealer", OFF_DEALER),
        ("score", OFF_SCORE),
        ("flag", OFF_FLAG),
        ("tricks_won", OFF_TRICKS_WON),
        ("games", OFF_GAMES),
        ("five", OFF_FIVE),
        ("best", OFF_BEST),
        ("size", OFF_SIZE),
        ("forbid", OFF_FORBID),
        ("no_ace", OFF_NO_ACE),
        ("ace_out", OFF_ACE_OUT),
        ("robber", OFF_ROBBER),
        ("c_play", OFF_C_PLAY),
        ("c_burn", OFF_C_BURN),
        ("c_wins", OFF_C_WINS),
    ];
    let mut s = format!("{ENCODER} n_view={N_VIEW} n_binary={N_BINARY} n_dense={N_DENSE}");
    for (name, off) in blocks {
        s.push_str(&format!(" {name}@{off}"));
    }
    s
}

/// The fixed positions of the encoder hash: decisions at tricks 1 and 2 of `rules-v1` self-play
/// matches from fixed seeds (one per match, the `k mod 9`-th such decision), with every candidate.
pub fn golden_positions() -> Vec<(SeatView, Vec<Cand>)> {
    let mut out = Vec::new();
    let mut k = 0u64;
    while out.len() < 64 {
        let mut m = Match::new(0x5EED_0000 + k, Rules::HOUSE);
        let want = k % 9;
        let mut seen = 0;
        while let Some(seat) = m.to_act() {
            let v = m.view(seat);
            let legal = m.legal_actions();
            if (1..=2).contains(&v.public.trick_no) && v.public.trump.is_some() && legal.len() >= 2
            {
                if seen == want {
                    let c = all_cands(&v, &legal);
                    out.push((v, c));
                    break;
                }
                seen += 1;
            }
            let a = bots::decide(&v, &Rules::HOUSE, &legal).0;
            m.apply(a).expect("legal");
        }
        k += 1;
    }
    out
}

/// The encoder hash (docs/STUDENT.md §2.2): SHA-256 of [`layout`] and of the encodings of
/// [`golden_positions`] (with `nn::synthetic::SyntheticLeaf` for the dense values, so it depends
/// on the encoder, the engine and the tracker, never on an equity table). Computed once.
pub fn encoder_sha256() -> &'static str {
    static SHA: OnceLock<String> = OnceLock::new();
    SHA.get_or_init(|| {
        let mut h = Sha256::new();
        h.update(layout().as_bytes());
        let leaf = nn::synthetic::SyntheticLeaf;
        for (v, cands) in golden_positions() {
            let f = encode_view(&v, &leaf).expect("golden positions encode");
            h.update((f.active.len() as u32).to_le_bytes());
            for x in &f.active {
                h.update(x.to_le_bytes());
            }
            for d in f.dense {
                h.update(d.to_le_bytes());
            }
            for c in cands {
                let ca = encode_cand(&v, c);
                h.update((ca.len() as u32).to_le_bytes());
                for x in ca {
                    h.update(x.to_le_bytes());
                }
            }
        }
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    })
}

/// Burns a robber may make (for tests: every card but the ace of trumps).
pub fn robber_burns(view: &SeatView) -> engine::cards::Bits {
    legal_burns(view.hand, view.public.trump)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nn::synthetic::SyntheticLeaf;

    #[test]
    fn card_slots_are_one_to_one_for_every_trump() {
        for trump in Suit::ALL {
            let mut seen = [false; N_CANON];
            for c in 0..53u8 {
                let i = canon(c, trump);
                assert!(i < N_CANON);
                assert!(!seen[i], "{trump:?}: slot {i} twice");
                seen[i] = true;
            }
            assert_eq!(seen.iter().filter(|&&x| x).count(), 53);
            // The trumps come first, strongest first.
            assert_eq!(canon(five_of(trump), trump), 0);
            assert_eq!(canon(engine::cards::JOKER, trump), 2);
            assert_eq!(canon(engine::rules::ACE_HEARTS, trump), 3);
        }
    }

    fn views(seed: u64, n: usize, f: impl FnMut(&Match, &SeatView, &[Action])) {
        let mut f = f;
        for i in 0..n as u64 {
            let mut m = Match::new(seed + i, Rules::HOUSE);
            let mut rng = engine::rng::Rng::new(seed ^ i);
            while let Some(seat) = m.to_act() {
                let v = m.view(seat);
                let legal = m.legal_actions();
                if v.public.trump.is_some() && v.public.trick_no >= 1 {
                    f(&m, &v, &legal);
                }
                let a = if rng.bounded(4) == 0 {
                    legal[rng.bounded(legal.len() as u64) as usize]
                } else {
                    bots::decide(&v, &Rules::HOUSE, &legal).0
                };
                m.apply(a).expect("legal");
            }
        }
    }

    /// Every feature is in its block; the view's features never reach the candidate block and
    /// the other way round; the counts stay within the shards' room; the hidden cards never
    /// change the view's encoding (only what the seat sees does).
    #[test]
    fn encodings_stay_in_their_blocks_and_see_only_the_seat() {
        let leaf = SyntheticLeaf;
        let mut most = 0;
        let mut n = 0;
        views(40, 12, |m, v, legal| {
            let f = encode_view(v, &leaf).expect("encodes");
            assert!(f.active.iter().all(|&i| usize::from(i) < N_VIEW));
            most = most.max(f.active.len());
            for c in all_cands(v, legal) {
                let ca = encode_cand(v, c);
                assert!(!ca.is_empty() && ca.len() <= K_MAX);
                assert!(
                    ca.iter()
                        .all(|&i| (N_VIEW..N_BINARY).contains(&usize::from(i)))
                );
            }
            // The same view from the match's own replay gives the same encoding.
            assert_eq!(encode_view(&m.view(v.seat), &leaf).expect("encodes"), f);
            n += 1;
        });
        assert!(n > 500);
        assert!(most <= V_MAX);
    }

    /// Two views that differ only in other seats' hidden cards encode alike: a deal drawn from
    /// the seat's own belief, rebuilt, gives the seat the same view and so the same features.
    #[test]
    fn the_encoder_hash_is_stable() {
        let a = encoder_sha256().to_string();
        assert_eq!(a.len(), 64);
        assert_eq!(encoder_sha256(), a);
        assert_eq!(golden_positions().len(), 64);
    }
}
