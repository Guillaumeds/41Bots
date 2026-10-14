//! One test per numbered rule of `docs/RULES.md` (rules 1–41), one per house rule (H1–H7) and one
//! per owner-confirmed decision or implementer default, all through the public engine API.
//!
//! Deals are hand-made with `build_deck` and `Match::with_decks`. Three builders keep the tests
//! short:
//! - `duel`: one real trick in which only the first two cards can win, to compare two cards
//!   (rules 6–11 and 19);
//! - `scripted`: a whole hand written trick by trick (which seat wins each trick), from which every
//!   seat's five cards are derived, for scores built up over several hands (rules 27–36);
//! - `random_match`: a whole match of random legal actions, for the rules that H1 makes impossible
//!   to break (rules 25, 26, 34, 37–41).
//!
//! Rules 25, 26, 34 and 37–41 are penalties for illegal play. House rule H1 makes illegal play
//! impossible, so their tests assert that the engine refuses the illegal action and leaves the
//! match unchanged, and that it performs every forced action (robbing) itself.

use engine::cards::{
    ACE, ALL_CARDS, Bits, Card, DECK_SIZE, JACK, JOKER, KING, QUEEN, Suit, bit, card, cards, count,
    has, name, names, parse, suit_of,
};
use engine::game::build_deck;
use engine::game::{Action, Event, Match, Phase, RULES_VERSION, Rules, TRICK_POINTS, Visibility};
use engine::rng::Rng;
use engine::rules::trump_mask;
use engine::{SEATS, TEAMS, team_of};

// ===== helpers: cards and deals =====

fn c(n: &str) -> Card {
    parse(n).unwrap_or_else(|| panic!("bad card name {n}"))
}

fn set(s: &str) -> Bits {
    s.split_whitespace().fold(0, |a, n| a | bit(c(n)))
}

/// Every trump when `t` is the trump suit: its 13 cards, the Joker and the A♥ (rules 6–8).
fn trumps_of(t: Suit) -> Bits {
    t.mask() | bit(JOKER) | bit(c("AH"))
}

/// The pack 0, 1, …, 52: each card's index is its position in the deal.
fn identity_deck() -> [Card; DECK_SIZE] {
    core::array::from_fn(|i| i as Card)
}

/// A deal: `hands[s]` lists up to five cards for seat `s`, `pack` the cards from position 40 on
/// (the turned card first). `build_deck` fills the rest, lowest card index first.
fn deck(dealer: u8, hands: [&str; 8], pack: &str) -> [Card; DECK_SIZE] {
    let hb: [Bits; SEATS] = core::array::from_fn(|s| set(hands[s]));
    let pack: Vec<Card> = pack.split_whitespace().map(c).collect();
    build_deck(dealer, &hb, &pack).expect("a valid hand-made deal")
}

fn game(dealer: u8, hands: [&str; 8], pack: &str) -> Match {
    Match::with_decks(dealer, vec![deck(dealer, hands, pack)], 1, Rules::default())
}

/// Like `game`, but every trump not named in `hands` goes into the undealt pack under the turned
/// card, so the cards added to fill the hands are never trumps (and nobody robs unless a hand
/// names the ace of trumps).
fn quiet(dealer: u8, hands: [&str; 8], turned: &str) -> Match {
    let t = c(turned);
    let named = hands.iter().fold(0, |a, h| a | set(h));
    let rest = trumps_of(suit_of(t).expect("an ordinary turned card")) & !named & !bit(t);
    let mut pack = vec![turned.to_string()];
    pack.extend(cards(rest).map(name));
    game(dealer, hands, &pack.join(" "))
}

fn hand(m: &Match, seat: u8) -> Bits {
    m.truth().hands[usize::from(seat)]
}

// ===== helpers: acting =====

fn act(m: &mut Match, a: Action) {
    if let Err(e) = m.apply(a) {
        panic!("{e} (phase {:?})", m.phase());
    }
}

fn play(m: &mut Match, n: &str) {
    act(m, Action::Play(c(n)));
}

fn burn(m: &mut Match, n: &str) {
    act(m, Action::Burn(c(n)));
}

/// Apply the lowest legal action.
fn lowest(m: &mut Match) {
    let a = m.legal_actions()[0];
    act(m, a);
}

/// Apply the lowest legal action until the current hand is over (or the match is).
fn finish_hand(m: &mut Match) {
    let h = m.hand_index();
    while m.phase() != Phase::MatchOver && m.hand_index() == h {
        lowest(m);
    }
}

/// The cards named by the legal actions.
fn legal(m: &Match) -> Bits {
    m.legal_actions().iter().fold(0, |a, x| a | bit(x.card()))
}

fn assert_legal(m: &Match, expected: &str) {
    assert_eq!(
        names(legal(m)),
        names(set(expected)),
        "legal cards for seat {:?}",
        m.to_act()
    );
}

/// For each step, check the exact legal cards of the seat to act, then play the given card.
fn walk(m: &mut Match, steps: &[(&str, &str)]) {
    for &(allowed, card) in steps {
        assert_legal(m, allowed);
        play(m, card);
    }
}

/// Apply an action that must be refused; check that the refusal changed nothing at all, and
/// return the reason.
fn refused(m: &mut Match, a: Action) -> &'static str {
    let before = m.clone();
    let events = m.events().len();
    let public = m.public();
    let truth = m.truth().clone();
    let actions = m.actions().len();
    let err = m.apply(a).expect_err("the engine must refuse this action");
    assert_eq!(err.action, a);
    assert_eq!(m.events().len(), events, "a refused action added an event");
    assert_eq!(
        m.public(),
        public,
        "a refused action changed the public state"
    );
    assert_eq!(
        *m.truth(),
        truth,
        "a refused action changed the hidden cards"
    );
    assert_eq!(m.actions().len(), actions, "a refused action was recorded");
    assert_eq!(*m, before, "a refused action changed the match");
    err.reason
}

// ===== helpers: the event log =====

fn log_from(m: &Match, from: usize) -> Vec<String> {
    m.events()[from..]
        .iter()
        .map(|(_, e)| e.canonical())
        .collect()
}

fn log(m: &Match) -> Vec<String> {
    log_from(m, 0)
}

fn last_event(m: &Match) -> String {
    m.events()
        .last()
        .map(|(_, e)| e.canonical())
        .unwrap_or_default()
}

fn count_events(m: &Match, prefix: &str) -> usize {
    log(m).iter().filter(|l| l.starts_with(prefix)).count()
}

/// The `n` events that follow the first event at or after index `from` whose canonical line is
/// `line`.
fn after(m: &Match, from: usize, line: &str, n: usize) -> Vec<String> {
    let l = log_from(m, from);
    let i = l
        .iter()
        .position(|x| x == line)
        .unwrap_or_else(|| panic!("no event {line}"));
    l[i + 1..].iter().take(n).cloned().collect()
}

/// The `TrickWon` lines from index `from` on.
fn tricks_won(m: &Match, from: usize) -> Vec<String> {
    log_from(m, from)
        .into_iter()
        .filter(|l| l.starts_with("TrickWon"))
        .collect()
}

/// (hand index, game index, dealer) of every `HandStart`.
fn hand_starts(m: &Match) -> Vec<(u32, u8, u8)> {
    m.events()
        .iter()
        .filter_map(|(_, e)| match *e {
            Event::HandStart {
                hand_index,
                game_index,
                dealer,
            } => Some((hand_index, game_index, dealer)),
            _ => None,
        })
        .collect()
}

/// The seat that won the first trick completed at or after event index `from`.
fn winner_from(m: &Match, from: usize) -> u8 {
    m.events()[from..]
        .iter()
        .find_map(|(_, e)| match *e {
            Event::TrickWon { seat, .. } => Some(seat),
            _ => None,
        })
        .expect("a completed trick")
}

// ===== builder: duels (two cards compared in a real trick) =====

/// Take the `n` lowest cards of `pool` that are still `free`.
fn take(free: &mut Bits, pool: Bits, n: usize) -> Bits {
    let got = cards(*free & pool).take(n).fold(0, |a, x| a | bit(x));
    *free &= !got;
    got
}

/// A one-trick table for comparing two cards. Seat 7 deals, so seat 0 leads `led` and seat 1
/// plays `follow`. Seats 2–7 hold no card of the suit led and, when a trump is led, no trump, so
/// they throw away plain cards of other suits, which never win (rule 19). Seat 1 holds more of
/// the suit led whenever it may (to show that following or trumping is allowed while holding the
/// suit), and otherwise nothing that would oblige it to play another card. The turned card is a
/// low card of `trump`; the ace of trumps is undealt unless it is one of the two cards, whose
/// holder then robs and burns the card just taken. Returns the match with seat 1 to play, and
/// the throw-away cards.
fn duel_before(trump: Suit, led: &str, follow: &str) -> (Match, Bits) {
    let (a, b) = (c(led), c(follow));
    let trumps = trumps_of(trump);
    let ace = card(trump, ACE);
    let turned = [9u8, 8, 7, 6, 10, 4, 3, 2]
        .into_iter()
        .map(|r| card(trump, r))
        .find(|&t| t != a && t != b)
        .expect("a low turned card");
    let trump_led = has(trumps, a);
    let led_suit = if trump_led {
        trumps
    } else {
        suit_of(a).expect("a plain card").mask() & !trumps
    };
    let junk = ALL_CARDS & !trumps & !led_suit;
    let mut free = ALL_CARDS & !(bit(a) | bit(b) | bit(turned) | bit(ace));
    let mut hands = [0 as Bits; SEATS];
    hands[0] = bit(a);
    hands[1] = bit(b);
    let short = count(free & junk) < 30;
    for h in hands.iter_mut().skip(2) {
        *h = take(&mut free, junk, if short { 4 } else { 5 });
        if short {
            // A plain card was led: a spare trump does not oblige a void player to anything.
            *h |= take(&mut free, trumps, 1);
        }
    }
    let (first, second) = if !trump_led && (has(led_suit, b) || has(trumps, b)) {
        (led_suit, ALL_CARDS)
    } else if !trump_led {
        (junk, trumps)
    } else if has(trumps, b) {
        (junk, ALL_CARDS)
    } else {
        (junk, 0)
    };
    hands[1] |= take(&mut free, first, 4);
    let missing = 5 - usize::from(count(hands[1]));
    hands[1] |= take(&mut free, second, missing);
    hands[0] |= take(&mut free, ALL_CARDS, 4);
    assert!(
        hands.iter().all(|&h| count(h) == 5),
        "duel deal {led} {follow}"
    );
    let d = build_deck(7, &hands, &[turned]).expect("a valid duel deal");
    let mut m = Match::with_decks(7, vec![d], 1, Rules::default());
    if m.phase() == Phase::AwaitBurn(0) {
        act(&mut m, Action::Burn(turned));
    }
    act(&mut m, Action::Play(a));
    if m.phase() == Phase::AwaitBurn(1) {
        act(&mut m, Action::Burn(turned));
    }
    assert_eq!(m.phase(), Phase::AwaitPlay(1));
    (m, junk)
}

/// Seat 0 leads `led`, seat 1 plays `follow`, the other six throw away: which seat wins?
fn duel(trump: Suit, led: &str, follow: &str) -> u8 {
    let (mut m, junk) = duel_before(trump, led, follow);
    play(&mut m, follow);
    for seat in 2..8u8 {
        assert_eq!(m.to_act(), Some(seat));
        let throw = cards(hand(&m, seat) & junk)
            .next()
            .expect("a card to throw away");
        act(&mut m, Action::Play(throw));
    }
    winner_from(&m, 0)
}

/// `best_first` lists cards from best to worst. Every pair meets in a real trick twice, each card
/// once led and once played second, and the better card must win both times.
fn assert_order(trump: Suit, best_first: &str) {
    let order: Vec<&str> = best_first.split_whitespace().collect();
    for (i, &hi) in order.iter().enumerate() {
        for &lo in &order[i + 1..] {
            assert_eq!(
                duel(trump, lo, hi),
                1,
                "{lo} led and {hi} played on it ({trump:?} trumps): the {hi} must win"
            );
            assert_eq!(
                duel(trump, hi, lo),
                0,
                "{hi} led and {lo} played on it ({trump:?} trumps): the {hi} must win"
            );
        }
    }
}

// ===== builder: scripted hands (for scores over several hands) =====

/// The kind of each of tricks 1–4 of a scripted hand (trick 5 plays everyone's last card).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Eight clubs; the king wins.
    Clubs,
    /// Eight diamonds; the king wins.
    Diamonds,
    /// Eight plain hearts; the king wins.
    Hearts,
    /// Eight trumps; the J of spades wins: the best trump of the hand, bonus at the end.
    Trumps,
    /// Eight trumps; the 5 of spades wins: bonus at once (rule 29).
    FiveOfTrumps,
}
use Kind::{Clubs, Diamonds, FiveOfTrumps, Hearts, Trumps};

/// A whole hand written trick by trick.
struct Script {
    dealer: u8,
    deck: [Card; DECK_SIZE],
    /// Each trick's eight (seat, card) plays, in play order.
    tricks: [[(u8, Card); 8]; 5],
}

/// A hand in which `winners[k]` wins trick k+1. Spades are trumps: the 9S is turned and the AS,
/// KS, 10S, Joker and A♥ (and the 5S or JS not used) are undealt, so nobody robs. Tricks 1–4 are
/// one each of `kinds`: a full trick of clubs, of diamonds, of plain hearts and of trumps; the
/// winner plays its best card (the king, or the J or 5 of spades). Trick 5 plays everyone's last
/// card, and the QC wins it. Every play is legal by construction: in a trick of one suit everyone
/// follows, and a last card is always legal. Each trick is led by the previous trick's winner.
fn scripted(dealer: u8, kinds: [Kind; 4], winners: [u8; 5]) -> Script {
    for k in [Clubs, Diamonds, Hearts] {
        assert_eq!(
            kinds.iter().filter(|&&x| x == k).count(),
            1,
            "one {k:?} trick"
        );
    }
    assert_eq!(
        kinds
            .iter()
            .filter(|&&x| matches!(x, Trumps | FiveOfTrumps))
            .count(),
        1,
        "one trick of trumps"
    );
    assert!(winners.iter().all(|&w| w < 8));
    let mut tricks = [[(0u8, 0 as Card); 8]; 5];
    let mut hands = [0 as Bits; SEATS];
    let mut leader = (dealer + 1) % 8;
    for (k, &kind) in kinds.iter().enumerate() {
        let (top, low) = match kind {
            Clubs => ("KC", "10C 9C 8C 7C 6C 5C 4C"),
            Diamonds => ("KD", "AD 2D 3D 4D 5D 6D 7D"),
            Hearts => ("KH", "2H 3H 4H 5H 6H 7H 8H"),
            Trumps => ("JS", "QS 2S 3S 4S 6S 7S 8S"),
            FiveOfTrumps => ("5S", "QS 2S 3S 4S 6S 7S 8S"),
        };
        let mut low = low.split_whitespace().map(c);
        for (j, slot) in tricks[k].iter_mut().enumerate() {
            let seat = (leader + j as u8) % 8;
            let card = if seat == winners[k] {
                c(top)
            } else {
                low.next().expect("seven low cards")
            };
            *slot = (seat, card);
            hands[usize::from(seat)] |= bit(card);
        }
        leader = winners[k];
    }
    let w = winners[4];
    let rest = if leader == w {
        "3C JC AC 2C QD JD 10D"
    } else {
        "JC AC 2C QD JD 10D"
    };
    let mut rest = rest.split_whitespace().map(c);
    for (j, slot) in tricks[4].iter_mut().enumerate() {
        let seat = (leader + j as u8) % 8;
        let card = if seat == w {
            c("QC")
        } else if j == 0 {
            c("3C")
        } else {
            rest.next().expect("a last card")
        };
        *slot = (seat, card);
        hands[usize::from(seat)] |= bit(card);
    }
    assert!(hands.iter().all(|&h| count(h) == 5));
    let deck = build_deck(dealer, &hands, &[c("9S")]).expect("a full deal");
    Script {
        dealer,
        deck,
        tricks,
    }
}

/// A match whose first hands are the scripted ones (each dealt one seat left of the one before).
fn scripted_match(rules: Rules, deals: &[&Script]) -> Match {
    for (i, s) in deals.iter().enumerate() {
        assert_eq!(s.dealer, (deals[0].dealer + i as u8) % 8, "deal {i}");
    }
    Match::with_decks(
        deals[0].dealer,
        deals.iter().map(|s| s.deck).collect(),
        1,
        rules,
    )
}

/// Play trick `k` (0-based) of the script.
fn play_trick(m: &mut Match, s: &Script, k: usize) {
    assert_eq!(
        m.dealer(),
        s.dealer,
        "the script dealt by seat {}",
        s.dealer
    );
    for &(seat, card) in &s.tricks[k] {
        assert_eq!(
            m.to_act(),
            Some(seat),
            "trick {} of the hand dealt by seat {}",
            k + 1,
            s.dealer
        );
        act(m, Action::Play(card));
    }
}

/// Play the first `n` tricks of the script.
fn play_tricks(m: &mut Match, s: &Script, n: usize) {
    for k in 0..n {
        play_trick(m, s, k);
    }
}

/// Dealer 7. Team 0 takes tricks 1–4, the fourth with the J of spades (the best trump: 11 at the
/// end of the hand); seat 1 takes trick 5. Scores 26, 5, 0, 0; team 0 has had its first bonus.
fn a0() -> Script {
    scripted(7, [Clubs, Diamonds, Hearts, Trumps], [0, 0, 0, 4, 1])
}

/// Dealer 0, after `a0`: team 0 (on 26) takes tricks 1–3, no trump played: 41 at trick 3.
fn a1_plain_41() -> Script {
    scripted(0, [Clubs, Diamonds, Hearts, Trumps], [4, 4, 0, 1, 2])
}

/// Dealer 0, after `a0`: seat 2 (team 2) wins trick 1 with the J of spades, the best trump of the
/// hand; team 0 (on 26) takes tricks 2–4 and reaches 41 at trick 4.
fn a1_pending_lost() -> Script {
    scripted(0, [Trumps, Clubs, Diamonds, Hearts], [2, 4, 4, 0, 1])
}

/// Dealer 0, after `a0`: team 0 (on 26) takes trick 1 (31), then trick 2 with the 5 of spades:
/// 36 for the trick, and its second bonus (5) at once makes 41.
fn a1_immediate_41() -> Script {
    scripted(0, [Clubs, FiveOfTrumps, Diamonds, Hearts], [4, 0, 1, 2, 3])
}

/// Dealer 0, after `a0`: team 0 (on 26) takes tricks 1 and 2 (36), then trick 3 with the 5 of
/// spades: 41 for the trick, then its second bonus at once: 46 (47 under the flat rule).
fn a1_five_from_36() -> Script {
    scripted(0, [Clubs, Diamonds, FiveOfTrumps, Hearts], [4, 4, 0, 1, 2])
}

/// Dealer 0, after `a0`: teams 1, 2, 3 take tricks 1–3; team 0 takes trick 4 with the J of
/// spades (31) and trick 5 (36); its second bonus (5) at the end of the hand makes 41.
fn a1_end_bonus_41() -> Script {
    scripted(0, [Clubs, Diamonds, Hearts, Trumps], [1, 2, 3, 4, 0])
}

/// Dealer 0, after `a0`: team 0 takes two tricks (36); team 1 takes three, one with the J of
/// spades, and its first bonus: 26. Scores 36, 26, 0, 0.
fn a1_two_teams() -> Script {
    scripted(0, [Clubs, Diamonds, Hearts, Trumps], [1, 0, 5, 1, 4])
}

/// Dealer 1, after `a0` and `a1_two_teams`: team 1 takes tricks 1 (with the J of spades) and 2:
/// 36, its bonus pending; teams 2 and 3 take tricks 3 and 4; team 0 (on 36) takes trick 5: 41.
fn a2_two_teams() -> Script {
    scripted(1, [Trumps, Clubs, Diamonds, Hearts], [5, 1, 2, 3, 0])
}

/// Dealer 7. Team 0 takes four tricks without the bonus (20); seat 1 takes the J of spades
/// trick (11). Scores 20, 11, 0, 0.
fn b0() -> Script {
    scripted(7, [Clubs, Diamonds, Hearts, Trumps], [0, 0, 4, 1, 0])
}

/// Dealer 0, after `b0`: team 0 takes four tricks again (40); seat 5 takes the J of spades
/// trick, team 1's second bonus (10). Scores 40, 21, 0, 0.
fn b1() -> Script {
    scripted(0, [Clubs, Diamonds, Hearts, Trumps], [4, 0, 4, 5, 0])
}

/// Dealer 1, after `b0` and `b1`: team 0 (on 40) takes trick 1: 45.
fn b2() -> Script {
    scripted(1, [Clubs, Diamonds, Hearts, Trumps], [4, 2, 3, 1, 2])
}

/// Dealer 0, after `b0`: team 0 (on 20) takes all five tricks: 40 after trick 4, 45 after trick 5.
fn b1_all_five() -> Script {
    scripted(0, [Clubs, Diamonds, Hearts, Trumps], [0, 4, 0, 4, 0])
}

/// One seat takes all five tricks: a jink for its team, 25 points.
fn jink(dealer: u8, seat: u8) -> Script {
    scripted(dealer, [Clubs, Diamonds, Hearts, Trumps], [seat; 5])
}

/// Dealer 7. Tricks go to seats 0, 4, 1, 5 (the J of spades) and 3. Scores 10, 16, 0, 5.
fn hand_a() -> Script {
    scripted(7, [Clubs, Diamonds, Hearts, Trumps], [0, 4, 1, 5, 3])
}

// ===== helpers: whole random matches =====

/// Play a whole match from `seed` choosing uniformly among the legal actions, and call `check`
/// before every action and once at the end.
fn random_match(seed: u64, rules: Rules, mut check: impl FnMut(&Match)) -> Match {
    let mut m = Match::new(seed, rules);
    let mut rng = Rng::derive(seed, &[41]);
    loop {
        check(&m);
        if m.phase() == Phase::MatchOver {
            return m;
        }
        let legal = m.legal_actions();
        assert!(
            !legal.is_empty(),
            "no legal action before the match is over"
        );
        let a = legal[rng.bounded(legal.len() as u64) as usize];
        act(&mut m, a);
    }
}

/// The trumps when `t` is the trump suit, best first, written straight from rules 6–8.
fn trump_order(t: Suit) -> Vec<Card> {
    let mut v = vec![card(t, 5), card(t, JACK), JOKER, c("AH")];
    if t != Suit::Hearts {
        v.push(card(t, ACE));
    }
    v.extend([card(t, KING), card(t, QUEEN)]);
    let numbers: [u8; 8] = if t.is_red() {
        [10, 9, 8, 7, 6, 4, 3, 2]
    } else {
        [2, 3, 4, 6, 7, 8, 9, 10]
    };
    v.extend(numbers.map(|n| card(t, n)));
    v
}

/// The cards `held` may play, written straight from rules 13–18, independently of the engine.
fn rulebook_legal(held: Bits, led: Option<Card>, t: Suit) -> Bits {
    let order = trump_order(t);
    let rank = |x: Card| order.iter().position(|&y| y == x); // 0 is the best trump
    let my_trumps = cards(held)
        .filter(|&x| rank(x).is_some())
        .fold(0, |a, x| a | bit(x));
    // Rule 13: any card may be led.
    let Some(led) = led else { return held };
    match rank(led) {
        Some(led_rank) => {
            // Rules 16–18: only a top-four trump better than the one led may be held back.
            let must = cards(my_trumps).any(|x| {
                let r = rank(x).unwrap_or(usize::MAX);
                !(r < 4 && r < led_rank)
            });
            if must { my_trumps } else { held }
        }
        None => {
            // Rules 14–15: follow the plain suit led or trump; void, anything.
            let s = suit_of(led);
            let suit = cards(held)
                .filter(|&x| suit_of(x) == s && rank(x).is_none())
                .fold(0, |a, x| a | bit(x));
            if suit == 0 { held } else { suit | my_trumps }
        }
    }
}

/// Check the structure of every trick in the log: eight cards from eight seats in turn, the first
/// led by the seat left of the dealer (trick 1) or by the previous trick's winner; the winner is
/// one of the eight and scores for its own team.
fn check_tricks(m: &Match) -> usize {
    let mut trick: Vec<u8> = Vec::new();
    let mut next_leader = None;
    let mut done = 0;
    for (_, e) in m.events() {
        match *e {
            Event::HandStart { dealer, .. } => {
                assert!(trick.is_empty(), "a hand ended in the middle of a trick");
                next_leader = Some((dealer + 1) % 8);
            }
            Event::Played { seat, .. } => {
                match trick.last() {
                    None => assert_eq!(Some(seat), next_leader, "the wrong seat led"),
                    Some(&prev) => assert_eq!(seat, (prev + 1) % 8, "a seat was skipped"),
                }
                trick.push(seat);
            }
            Event::TrickWon {
                seat, team, points, ..
            } => {
                assert_eq!(trick.len(), 8, "a trick without all eight players");
                assert!(trick.contains(&seat));
                assert_eq!(team, team_of(seat));
                assert_eq!(points, TRICK_POINTS);
                for (j, &s) in trick.iter().enumerate() {
                    assert_eq!(team_of(s), (team_of(trick[0]) + j as u8) % 4);
                }
                next_leader = Some(seat);
                trick.clear();
                done += 1;
            }
            _ => {}
        }
    }
    done
}

// ===================================================================================
// The table (rules 1–5)
// ===================================================================================

/// Rule 1: "Eight players play in pairs. Partners sit opposite each other."
///
/// Seats 0–7 sit clockwise and seat s partners seat s + 4, the seat opposite, so the four teams
/// play in turn, twice round, in every trick. Partners pool their tricks: seats 0 and 4 both win
/// tricks for team 0 below, and seats 1 and 5 for team 1.
#[test]
fn rule_01_eight_players_in_pairs() {
    assert_eq!((SEATS, TEAMS), (8, 4));
    for s in 0..8u8 {
        assert_eq!(
            team_of(s),
            team_of((s + 4) % 8),
            "seat {s} partners the seat opposite"
        );
        for d in 1..4u8 {
            assert_ne!(
                team_of(s),
                team_of((s + d) % 8),
                "seats {s} and {} are opponents",
                (s + d) % 8
            );
        }
    }
    let h = hand_a();
    let mut m = scripted_match(Rules::default(), &[&h]);
    play_tricks(&mut m, &h, 4);
    assert_eq!(
        tricks_won(&m, 0),
        [
            "TrickWon trick=1 seat=0 team=0 points=5",
            "TrickWon trick=2 seat=4 team=0 points=5",
            "TrickWon trick=3 seat=1 team=1 points=5",
            "TrickWon trick=4 seat=5 team=1 points=5",
        ]
    );
    assert_eq!(m.scores(), [10, 10, 0, 0]);
    assert_eq!(m.public().tricks_won, [2, 2, 0, 0]);
    assert_eq!(check_tricks(&m), 4);
}

/// Rule 2: "One pack of 53 cards: the 52 ordinary cards and one Joker."
///
/// Every deal uses all 53 cards exactly once: 40 in the hands, 1 turned and 12 undealt.
#[test]
fn rule_02_one_pack_of_53_cards() {
    assert_eq!(DECK_SIZE, 53);
    assert_eq!(count(ALL_CARDS), 53);
    let ordinary = Suit::ALL.iter().fold(0, |a, s| {
        assert_eq!(count(s.mask()), 13);
        a | s.mask()
    });
    assert_eq!(count(ordinary), 52);
    assert_eq!(ALL_CARDS & !ordinary, bit(JOKER));
    assert_eq!(suit_of(JOKER), None);
    assert_eq!(name(JOKER), "JK");
    for seed in [1u64, 2, 3, 4] {
        let m = Match::new(seed, Rules::default());
        let d = m.truth().deck;
        let mut sorted = d;
        sorted.sort_unstable();
        assert_eq!(sorted, identity_deck(), "the deck is all 53 cards");
        let mut dealt = 0;
        for (_, e) in m.events() {
            if let Event::Dealt { cards, .. } = *e {
                assert_eq!(count(cards), 5);
                assert_eq!(dealt & cards, 0);
                dealt |= cards;
            }
        }
        let undealt = d[41..].iter().fold(0, |a, &x| a | bit(x));
        assert_eq!(count(dealt), 40);
        assert_eq!(count(undealt), 12);
        assert_eq!(m.truth().undealt, undealt);
        assert_eq!(dealt | undealt | bit(d[40]), ALL_CARDS);
    }
    // A hand-made deal may not give a card twice, nor six cards to a seat.
    assert!(build_deck(0, &[set("2C"), set("2C"), 0, 0, 0, 0, 0, 0], &[]).is_err());
    assert!(build_deck(0, &[set("2C 3C 4C 5C 6C 7C"), 0, 0, 0, 0, 0, 0, 0], &[]).is_err());
    assert!(build_deck(0, &[set("2C"), 0, 0, 0, 0, 0, 0, 0], &[c("2C")]).is_err());
}

/// Rule 3: "The first dealer is chosen at random. After every hand the deal moves one seat to the
/// left."
///
/// The first dealer comes from the match's seeded generator (so it varies with the seed), and
/// the deal moves one seat left after a hand that goes on, after a game that ends in the middle
/// of a hand, and after a game that ends with the hand (here a jink).
#[test]
fn rule_03_first_dealer_random_then_deal_moves_left() {
    let mut seen = [false; 8];
    for seed in 0..64u64 {
        let m = Match::new(seed, Rules::default());
        let d = m.dealer();
        assert!(d < 8);
        assert_eq!(m.events()[0].1, Event::MatchStart { first_dealer: d });
        assert_eq!(hand_starts(&m), [(0, 0, d)]);
        assert_eq!(
            Match::new(seed, Rules::default()).dealer(),
            d,
            "same seed, same dealer"
        );
        seen[usize::from(d)] = true;
    }
    assert_eq!(seen, [true; 8], "every seat can be the first dealer");

    // Hand 0 goes on to hand 1; game 1 ends at trick 3 of hand 1; game 2 ends with a jink.
    let (h0, h1, h2) = (a0(), a1_plain_41(), jink(1, 2));
    let mut m = scripted_match(Rules::default(), &[&h0, &h1, &h2]);
    play_tricks(&mut m, &h0, 5);
    play_tricks(&mut m, &h1, 3);
    assert_eq!(m.games_won(), [1, 0, 0, 0]);
    play_tricks(&mut m, &h2, 5);
    assert_eq!(m.games_won(), [1, 0, 1, 0]);
    assert_eq!(
        hand_starts(&m),
        [(0, 0, 7), (1, 0, 0), (2, 1, 1), (3, 2, 2)]
    );
    assert_eq!(m.dealer(), 2);
}

/// Rule 4: "The dealer gives each player two cards and then three, starting with the player on
/// the left. Other bundles are fine, as long as everyone gets five cards in two rounds and
/// nobody gets all five at once."
///
/// With the pack 0, 1, …, 52 each card's index is its position, so the deal shows the bundles:
/// the seat i places left of the dealer gets positions 2i and 2i+1 in the first round and
/// 16+3i … 18+3i in the second; the deal starts on the dealer's left.
#[test]
fn rule_04_deal_two_then_three_from_dealers_left() {
    for dealer in [5u8, 0, 7] {
        let m = Match::with_decks(dealer, vec![identity_deck()], 1, Rules::default());
        let dealt: Vec<&(Visibility, Event)> = m
            .events()
            .iter()
            .filter(|(_, e)| matches!(e, Event::Dealt { .. }))
            .collect();
        assert_eq!(dealt.len(), 8);
        for (i, &(vis, e)) in dealt.iter().enumerate() {
            let i = i as u8;
            let seat = (dealer + 1 + i) % 8;
            let first_round = bit(2 * i) | bit(2 * i + 1);
            let second_round = bit(16 + 3 * i) | bit(17 + 3 * i) | bit(18 + 3 * i);
            assert_eq!(
                *vis,
                Visibility::Private(seat),
                "only the seat sees its cards"
            );
            assert_eq!(
                *e,
                Event::Dealt {
                    seat,
                    cards: first_round | second_round
                }
            );
            assert_eq!(hand(&m, seat), first_round | second_round);
        }
        // The first round (positions 0–15) is complete before the second (16–39) starts.
        assert_eq!(
            dealt[0].1,
            Event::Dealt {
                seat: (dealer + 1) % 8,
                cards: set("AC 2C 4D 5D 6D")
            }
        );
    }
}

/// Rule 5: "The next card of the pack is turned face up. Its suit is trumps for the hand. The
/// rest of the pack stays face down and out of play."
///
/// Position 40 is turned and names trumps; positions 41–52 are the undealt pack, which never
/// changes and never reaches a trick. The 40 cards played are exactly the 40 dealt.
#[test]
fn rule_05_turned_card_sets_trumps_rest_out_of_play() {
    // Diamonds trumps with the AD undealt, so nobody robs the turned 7D.
    let mut m = game(7, [""; 8], "7D AD");
    let d = m.truth().deck;
    assert_eq!(d[40], c("7D"));
    assert_eq!(last_event(&m), "Turned card=7D");
    let p = m.public();
    assert_eq!(p.trump, Some(Suit::Diamonds));
    assert_eq!(p.table_card, Some(c("7D")));
    assert_eq!(p.hidden_count, 12);
    let undealt = d[41..].iter().fold(0, |a, &x| a | bit(x));
    assert_eq!(count(undealt), 12);
    assert_eq!(m.truth().undealt, undealt);
    let dealt = m.truth().hands.iter().fold(0, |a, &h| a | h);
    let start = m.events().len();
    while m.hand_index() == 0 {
        assert_eq!(
            m.truth().undealt,
            undealt,
            "the undealt pack stays face down"
        );
        assert_eq!(m.public().played & (undealt | bit(c("7D"))), 0);
        lowest(&mut m);
    }
    let mut played = 0;
    for (_, e) in &m.events()[start..] {
        match *e {
            Event::Played { card, .. } => played |= bit(card),
            Event::HandEnd { .. } => break,
            _ => {}
        }
    }
    assert_eq!(played, dealt, "exactly the dealt cards are played");
}

// ===================================================================================
// Which cards beat which (rules 6–11)
// ===================================================================================

/// Rule 6: "Trumps from best to worst: the 5 of trumps, the J of trumps, the Joker, the ace of
/// hearts, the ace of trumps, king, queen, then the numbers. In a red trump suit the numbers run
/// 10 down to 2. In a black trump suit they run 2 up to 10, so the 2 beats the 10."
///
/// Every pair of the 15 trumps meets in a real trick, each card once led and once played second;
/// the better card wins both times. Two black suits and one red suit (hearts: rule 8).
#[test]
fn rule_06_trump_order_red_and_black() {
    assert_order(Suit::Clubs, "5C JC JK AH AC KC QC 2C 3C 4C 6C 7C 8C 9C 10C");
    assert_order(
        Suit::Spades,
        "5S JS JK AH AS KS QS 2S 3S 4S 6S 7S 8S 9S 10S",
    );
    assert_order(
        Suit::Diamonds,
        "5D JD JK AH AD KD QD 10D 9D 8D 7D 6D 4D 3D 2D",
    );
    assert_eq!(
        duel(Suit::Spades, "10S", "2S"),
        1,
        "black: the 2 beats the 10"
    );
    assert_eq!(
        duel(Suit::Spades, "2S", "10S"),
        0,
        "black: the 2 beats the 10"
    );
    assert_eq!(
        duel(Suit::Diamonds, "2D", "10D"),
        1,
        "red: the 10 beats the 2"
    );
}

/// Rule 7: "The Joker and the ace of hearts are trumps in every hand, whatever suit is turned."
///
/// For each of the four trump suits: a player holding the plain suit led may still play the
/// Joker or the A♥ (they are trumps), either one beats the king of the suit led, and either one
/// beats the king of trumps.
#[test]
fn rule_07_joker_and_ace_of_hearts_always_trumps() {
    for t in Suit::ALL {
        let x = [Suit::Clubs, Suit::Diamonds, Suit::Spades]
            .into_iter()
            .find(|&s| s != t)
            .expect("a plain suit other than hearts");
        let plain_king = name(card(x, KING));
        let trump_king = name(card(t, KING));
        for special in ["JK", "AH"] {
            let (m, _) = duel_before(t, &plain_king, special);
            let l = legal(&m);
            assert!(
                has(l, c(special)),
                "the {special} is a trump with {t:?} trumps"
            );
            assert_ne!(hand(&m, 1) & x.mask(), 0, "seat 1 holds the suit led");
            assert_eq!(
                duel(t, &plain_king, special),
                1,
                "the {special} trumps the {plain_king} when {t:?} are trumps"
            );
            assert_eq!(
                duel(t, special, &trump_king),
                0,
                "the {special} beats the {trump_king}"
            );
        }
    }
}

/// Rule 8: "When hearts are trumps, the ace of hearts is fourth best and there are 14 trumps
/// instead of 15. In any other trump suit, the ace of that suit comes straight after the ace of
/// hearts."
///
/// The 14 heart trumps meet pairwise in real tricks; with hearts trumps the A♥ is the ace of
/// trumps (its holder robs); in every other suit the A♥ beats the ace of trumps, which beats the
/// king (nothing ranks between them: rule 6 checks every pair).
#[test]
fn rule_08_hearts_trumps_fourteen_ace_of_hearts_fourth() {
    assert_order(Suit::Hearts, "5H JH JK AH KH QH 10H 9H 8H 7H 6H 4H 3H 2H");
    assert_eq!(count(trump_mask(Suit::Hearts)), 14);
    for t in [Suit::Clubs, Suit::Diamonds, Suit::Spades] {
        assert_eq!(count(trump_mask(t)), 15);
        let (ace, king) = (name(card(t, ACE)), name(card(t, KING)));
        assert_eq!(duel(t, &ace, "AH"), 1, "the A♥ beats the {ace}");
        assert_eq!(duel(t, "AH", &ace), 0, "the A♥ beats the {ace}");
        assert_eq!(duel(t, &king, &ace), 1, "the {ace} beats the {king}");
    }
    // With hearts trumps the holder of the A♥ robs the turned card (it is the ace of trumps).
    let mut m = quiet(7, ["2H", "", "AH 2C 3C 4C 5C", "", "", "", "", ""], "7H");
    lowest(&mut m);
    lowest(&mut m);
    assert_eq!(last_event(&m), "Took seat=2 card=7H");
    assert_eq!(m.phase(), Phase::AwaitBurn(2));
    // With spades trumps the holder of the A♥ does not rob: the AS is the ace of trumps.
    let mut m = quiet(7, ["AH 2S", "", "", "", "", "", "", ""], "7S");
    play(&mut m, "AH");
    assert_eq!(count_events(&m, "Took"), 0);
    assert_eq!(last_event(&m), "Played seat=0 card=AH");
}

/// Rule 9: "In a red suit that is not trumps, the order is king, queen, jack, 10 down to 2, then
/// the ace. The ace is the lowest red card."
///
/// Every pair of plain diamonds (spades trumps) and of plain hearts (clubs trumps, so no A♥:
/// rule 11) meets in a real trick.
#[test]
fn rule_09_red_plain_order_ace_lowest() {
    assert_order(Suit::Spades, "KD QD JD 10D 9D 8D 7D 6D 5D 4D 3D 2D AD");
    assert_order(Suit::Clubs, "KH QH JH 10H 9H 8H 7H 6H 5H 4H 3H 2H");
    assert_eq!(
        duel(Suit::Spades, "AD", "2D"),
        1,
        "the ace is the lowest red card"
    );
}

/// Rule 10: "In a black suit that is not trumps, the order is king, queen, jack, ace, then 2 up
/// to 10. The ace beats the 2, and the 10 is the lowest black card."
///
/// Every pair of plain clubs (diamonds trumps) and of plain spades (hearts trumps) meets in a
/// real trick.
#[test]
fn rule_10_black_plain_order_ace_after_jack() {
    assert_order(Suit::Diamonds, "KC QC JC AC 2C 3C 4C 5C 6C 7C 8C 9C 10C");
    assert_order(Suit::Hearts, "KS QS JS AS 2S 3S 4S 5S 6S 7S 8S 9S 10S");
    assert_eq!(duel(Suit::Diamonds, "2C", "AC"), 1, "the ace beats the 2");
    assert_eq!(duel(Suit::Diamonds, "10C", "9C"), 1, "the 10 is the lowest");
}

/// Rule 11: "The ace of hearts never belongs to the plain hearts, because it is always a trump."
///
/// With clubs trumps: a player whose only heart is the A♥ is void when hearts are led and may
/// play anything; the A♥ played on a heart lead wins as a trump (as the red plain ace it would
/// be the lowest heart); and the A♥ led is a trump lead, not a heart lead.
#[test]
fn rule_11_ace_of_hearts_never_a_plain_heart() {
    let mut m = quiet(
        7,
        [
            "KH 2D 3D 4D 5D",
            "AH 6D 7D 8D 9D",
            "2H 10D JD QD KD",
            "3H 2C 6S 7S 8S",
            "",
            "",
            "",
            "",
        ],
        "9C",
    );
    play(&mut m, "KH");
    walk(
        &mut m,
        &[
            ("AH 6D 7D 8D 9D", "6D"), // only the A♥: void in plain hearts
            ("2H", "2H"),
            ("3H 2C", "3H"),
        ],
    );
    for t in [Suit::Clubs, Suit::Diamonds, Suit::Spades] {
        assert_eq!(
            duel(t, "2H", "AH"),
            1,
            "the A♥ beats the lowest heart: it is a trump"
        );
        assert_eq!(duel(t, "KH", "AH"), 1, "the A♥ trumps the king of hearts");
    }
    // The A♥ led with clubs trumps: a player holding hearts but no trump may play anything.
    let mut m = quiet(7, ["AH 2C", "7H 8H 9H 2D 3D", "", "", "", "", "", ""], "9C");
    play(&mut m, "AH");
    assert_legal(&m, "7H 8H 9H 2D 3D");
}

// ===================================================================================
// Playing a trick (rules 12–19)
// ===================================================================================

/// Rule 12: "The player left of the dealer leads the first trick. After that, the winner of each
/// trick leads the next."
#[test]
fn rule_12_left_of_dealer_leads_then_trick_winner() {
    for dealer in 0..8u8 {
        let m = game(dealer, [""; 8], "7D AD");
        let leader = (dealer + 1) % 8;
        assert_eq!(m.phase(), Phase::AwaitPlay(leader), "dealer {dealer}");
        assert_eq!(m.public().leader, leader);
        assert_eq!(m.public().trick_no, 1);
    }
    let h = hand_a();
    let mut m = scripted_match(Rules::default(), &[&h]);
    let mut leader = 0;
    for k in 0..5 {
        assert_eq!(m.to_act(), Some(leader), "trick {}", k + 1);
        assert_eq!(m.public().leader, leader);
        let start = m.events().len();
        play_trick(&mut m, &h, k);
        leader = winner_from(&m, start);
    }
    // And in a whole random match: every trick is led by the previous trick's winner.
    let m = random_match(12, Rules::default(), |_| {});
    assert!(check_tricks(&m) >= 25);
}

/// Rule 13: "Any card may be led, trumps included."
#[test]
fn rule_13_any_card_may_be_led() {
    let m = quiet(7, ["5S JK AH 2C KD", "", "", "", "", "", "", ""], "9S");
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    assert_legal(&m, "5S JK AH 2C KD");
    for x in cards(hand(&m, 0)) {
        let mut t = m.clone();
        act(&mut t, Action::Play(x));
        assert_eq!(t.public().trick_cards, vec![(0, x)]);
    }
    // The leader of every later trick may lead any card it holds too.
    let h = hand_a();
    let mut m = scripted_match(Rules::default(), &[&h]);
    for k in 0..5 {
        let leader = m.to_act().expect("a leader");
        assert_eq!(legal(&m), hand(&m, leader), "trick {}", k + 1);
        play_trick(&mut m, &h, k);
    }
}

/// Spades trumps; seat 0 leads the 9D. Used by rules 14, 15 and 37.
fn rule14_deal() -> Match {
    quiet(
        7,
        [
            "9D 2C 3C 4C 5C",
            "3D KD 2H 5S JK",
            "4D 6S 3H 4H 5H",
            "6C 7C 8C 9H 7S",
            "5D 10H JH QH KH",
            "9C 10C JC QC KC",
            "AD 6D 7D 8D 10D",
            "",
        ],
        "9S",
    )
}

/// Rule 14: "When a plain suit is led, you must play that suit or a trump. You may play a card of
/// another suit only when you have none of the suit led."
#[test]
fn rule_14_plain_lead_follow_suit_or_trump() {
    let mut m = rule14_deal();
    play(&mut m, "9D");
    // Seat 1 holds diamonds: diamonds or trumps (5S, Joker), never the 2H.
    assert_legal(&m, "3D KD 5S JK");
    assert_eq!(
        refused(&mut m, Action::Play(c("2H"))),
        "you hold the suit led: follow suit or play a trump (rule 14)"
    );
    play(&mut m, "3D");
    // Seat 2 holds a diamond and trumps with the 6S anyway.
    assert_legal(&m, "4D 6S");
    play(&mut m, "6S");
    // Seat 3 has no diamond: it may play another suit.
    assert_legal(&m, "6C 7C 8C 9H 7S");
    play(&mut m, "6C");
    // Seat 4: one diamond and no trump: only the diamond.
    assert_legal(&m, "5D");
    assert_eq!(
        refused(&mut m, Action::Play(c("KH"))),
        "you hold the suit led: follow suit or play a trump (rule 14)"
    );
    play(&mut m, "5D");
}

/// Rule 15: "If you have none of the suit led, you may play any card at all. It can be a trump,
/// but it does not have to be."
#[test]
fn rule_15_void_may_play_any_card() {
    let mut m = rule14_deal();
    plays_all(&mut m, "9D 3D 4D");
    // Seat 3: no diamond, one trump: any of its five cards.
    assert_eq!(m.to_act(), Some(3));
    assert_legal(&m, "6C 7C 8C 9H 7S");
    let mut trumped = m.clone();
    play(&mut trumped, "7S");
    assert_eq!(trumped.public().best_trump, Some((c("7S"), 3)));
    let mut discarded = m.clone();
    play(&mut discarded, "9H");
    assert_eq!(discarded.public().best_trump, None);
    // Seat 5: no diamond and no trump: anything.
    plays_all(&mut m, "6C 5D");
    assert_legal(&m, "9C 10C JC QC KC");
}

fn plays_all(m: &mut Match, s: &str) {
    for n in s.split_whitespace() {
        play(m, n);
    }
}

/// Clubs trumps; seat 0 leads the 10C, the lowest trump. Used by rules 16 and 37.
fn rule16_deal() -> Match {
    quiet(
        7,
        [
            "10C",
            "2C 2D 3D 4D 5D",
            "5C JK 6D 7D 8D",
            "JC 3C 9D 10D JD",
            "QD KD 2H 3H 4H",
            "AH 5H 6H 7H 8H",
            "KC 9H 10H JH QH",
            "",
        ],
        "9C",
    )
}

/// Rule 16: "When a trump is led, you must play a trump if you have one. The exception: the four
/// top trumps, the 5, the J, the Joker and the ace of hearts, may be held back when a lower trump
/// is led. This is reneging."
#[test]
fn rule_16_trump_lead_must_trump_unless_top_four() {
    let mut m = rule16_deal();
    play(&mut m, "10C");
    // Seat 1: an ordinary trump: it must be played.
    assert_legal(&m, "2C");
    assert_eq!(
        refused(&mut m, Action::Play(c("2D"))),
        "a trump was led and you hold a trump you may not hold back: play a trump (rules 16-17)"
    );
    play(&mut m, "2C");
    // Seat 2: only the 5 and the Joker, both above the 10C: it may renege.
    walk(
        &mut m,
        &[
            ("5C JK 6D 7D 8D", "6D"),
            // Seat 3: the J may be held back but the 3C may not: a trump, either one.
            ("JC 3C", "JC"),
            // Seat 4: no trump at all: anything.
            ("QD KD 2H 3H 4H", "2H"),
            // Seat 5: the A♥ above the 10C: it may renege.
            ("AH 5H 6H 7H 8H", "5H"),
            // Seat 6: the king is not one of the four top trumps.
            ("KC", "KC"),
        ],
    );
    lowest(&mut m);
    assert_eq!(winner_from(&m, 0), 3, "the J of clubs wins");
}

/// Rule 17: "A top trump can be held back only against a lower trump. If a higher trump is led,
/// it must be played like any other trump. So the 5 led forces out every trump. The J led forces
/// out the Joker and the ace of hearts. The Joker led forces out the ace of hearts."
///
/// Clubs trumps; each seat holds one trump among plain cards. The fourth variant, the A♥ led
/// when hearts are not trumps (RULES "Trump leads"), forces the ace of trumps and below.
#[test]
fn rule_17_higher_trump_led_forces_top_trumps() {
    // The 5 led forces out every trump.
    let mut m = quiet(
        7,
        [
            "5C",
            "JC 2D 3D 4D 5D",
            "JK 6D 7D 8D 9D",
            "AH 10D JD QD KD",
            "KC 2H 3H 4H 5H",
            "6H 7H 8H 9H 10H",
            "",
            "",
        ],
        "9C",
    );
    play(&mut m, "5C");
    walk(
        &mut m,
        &[
            ("JC", "JC"),
            ("JK", "JK"),
            ("AH", "AH"),
            ("KC", "KC"),
            ("6H 7H 8H 9H 10H", "6H"),
        ],
    );
    // The J led forces out the Joker and the A♥; the 5 may still be held back.
    let mut m = quiet(
        7,
        [
            "JC",
            "JK 2D 3D 4D 5D",
            "AH 6D 7D 8D 9D",
            "5C 10D JD QD KD",
            "QC 2H 3H 4H 5H",
            "",
            "",
            "",
        ],
        "9C",
    );
    play(&mut m, "JC");
    walk(
        &mut m,
        &[
            ("JK", "JK"),
            ("AH", "AH"),
            ("5C 10D JD QD KD", "10D"),
            ("QC", "QC"),
        ],
    );
    // The Joker led forces out the A♥; the 5 and the J may be held back.
    let mut m = quiet(
        7,
        [
            "JK",
            "AH 2D 3D 4D 5D",
            "5C 6D 7D 8D 9D",
            "JC 10D JD QD KD",
            "KC 2H 3H 4H 5H",
            "",
            "",
            "",
        ],
        "9C",
    );
    play(&mut m, "JK");
    walk(
        &mut m,
        &[
            ("AH", "AH"),
            ("5C 6D 7D 8D 9D", "6D"),
            ("JC 10D JD QD KD", "10D"),
            ("KC", "KC"),
        ],
    );
    // The A♥ led forces out the ace of trumps (its holder robs first) and every lower trump; the
    // 5, the J and the Joker may be held back.
    let mut m = quiet(
        7,
        [
            "AH",
            "AC 2D 3D 4D 5D",
            "5C 6D 7D 8D 9D",
            "JC 10D JD QD KD",
            "JK 2H 3H 4H 5H",
            "KC 6H 7H 8H 9H",
            "",
            "",
        ],
        "9C",
    );
    play(&mut m, "AH");
    assert_eq!(m.phase(), Phase::AwaitBurn(1));
    burn(&mut m, "9C");
    walk(
        &mut m,
        &[
            ("AC", "AC"),
            ("5C 6D 7D 8D 9D", "6D"),
            ("JC 10D JD QD KD", "10D"),
            ("JK 2H 3H 4H 5H", "2H"),
            ("KC", "KC"),
        ],
    );
}

/// Rule 18: "If every trump you hold is one you may hold back, you may play any card, even a
/// plain one."
///
/// Both branches: four top trumps all above the king led (any card, here a plain one); and a 5
/// with a Joker against the J led (the Joker is below the J, so may not be held back, so a trump
/// must be played, and the 5 is allowed too).
#[test]
fn rule_18_only_holdable_trumps_may_play_anything() {
    let mut m = quiet(
        7,
        ["KC", "5C JC JK AH 7D", "3C 2H 3H 4H 5H", "", "", "", "", ""],
        "9C",
    );
    play(&mut m, "KC");
    walk(&mut m, &[("5C JC JK AH 7D", "7D"), ("3C", "3C")]);
    let mut m = quiet(
        7,
        ["JC", "5C JK 2D 3D 4D", "AH 6D 7D 8D 9D", "", "", "", "", ""],
        "9C",
    );
    play(&mut m, "JC");
    walk(&mut m, &[("5C JK", "5C"), ("AH", "AH")]);
}

/// Rule 19: "The trick goes to the highest trump in it. If no trump is played, it goes to the
/// highest card of the suit led. A card of any other suit never wins."
#[test]
fn rule_19_highest_trump_wins_else_highest_of_suit_led() {
    // Clubs trumps: three trumps in one trick; the 2C (best black number) beats the 3C and 4C,
    // and an off-suit king of spades wins nothing.
    let mut m = quiet(
        7,
        [
            "2D 5H 6H 7H 8H",
            "KD 9H 10H JH QH",
            "3C 4D 2S 3S 4S",
            "KS 2H 3H 4H KH",
            "2C 5D 5S 6S 7S",
            "QD 6D 8S 9S 10S",
            "4C 7D JS QS AS",
            "AD 8D 9D 10D JD",
        ],
        "9C",
    );
    plays_all(&mut m, "2D KD 3C KS 2C QD 4C AD");
    assert_eq!(
        last_event(&m),
        "AceUndealt",
        "nobody robbed: the AC is undealt"
    );
    assert_eq!(winner_from(&m, 0), 4, "the highest trump, the 2C, wins");
    assert_eq!(m.public().best_trump, Some((c("2C"), 4)));
    // No trump: the highest card of the suit led wins; another suit never does.
    assert_eq!(
        duel(Suit::Clubs, "2D", "KS"),
        0,
        "the KS cannot beat the 2D led"
    );
    assert_eq!(
        duel(Suit::Clubs, "9H", "KD"),
        0,
        "the KD cannot beat the 9H led"
    );
    assert_eq!(duel(Suit::Clubs, "2D", "3D"), 1);
    // Any trump beats every plain card: the lowest club beats the king of the suit led.
    assert_eq!(duel(Suit::Clubs, "KD", "10C"), 1);
}

// ===================================================================================
// Robbing the turned card (rules 20–26)
// ===================================================================================

/// Spades trumps (7S turned); seat 2 holds the ace of spades. Used by rules 20, 21, 25, H2 and
/// the decision on burn-then-play.
fn robber_deal() -> Match {
    game(7, ["", "", "AS 2D 3D 4D 5D", "", "", "", "", ""], "7S")
}

/// Rule 20: "The player holding the ace of trumps must rob the turned card before playing their
/// first card. To rob it, they burn one card from their hand: they put it face down on the
/// table, where it stays hidden for the whole hand."
///
/// At seat 2's first turn the engine gives it the turned card (Took), asks for a burn, and only
/// then for its first card; the burned card stays hidden and out of play all hand.
#[test]
fn rule_20_ace_holder_robs_before_first_card() {
    let mut m = robber_deal();
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    let start = m.events().len();
    lowest(&mut m);
    lowest(&mut m);
    let tail = log_from(&m, start);
    assert_eq!(tail.len(), 5);
    assert_eq!(tail[4], "Took seat=2 card=7S");
    assert_eq!(m.phase(), Phase::AwaitBurn(2));
    assert_eq!(hand(&m, 2), set("AS 2D 3D 4D 5D 7S"));
    let p = m.public();
    assert_eq!(p.hand_sizes[2], 6);
    assert_eq!(p.table_card, None);
    assert_eq!(p.robber, Some(2));
    assert_eq!(p.turned, vec![(c("7S"), Some(2))]);
    let start = m.events().len();
    burn(&mut m, "2D");
    assert_eq!(
        m.events()[start..],
        [
            (
                Visibility::Private(2),
                Event::Burned {
                    seat: 2,
                    card: c("2D")
                }
            ),
            (Visibility::Public, Event::BurnDone { seat: 2 }),
        ]
    );
    assert_eq!(m.phase(), Phase::AwaitPlay(2), "now the first card");
    let p = m.public();
    assert_eq!(
        (p.hand_sizes[2], p.burn_counts[2], p.hidden_count),
        (5, 1, 13)
    );
    // The burned card stays face down for the whole hand and is never played.
    while m.hand_index() == 0 {
        assert_eq!(m.truth().burns[2], set("2D"));
        assert!(!has(m.public().played, c("2D")));
        lowest(&mut m);
    }
}

/// Rule 21: "The ace of trumps itself may not be burned. Nobody sees the burned card, but
/// everyone now knows that player holds the ace of trumps."
///
/// The burn may be any card but an ace (H3 tightens rule 21 to every ace); only the burner's
/// view shows the burned card; every view shows who robbed.
#[test]
fn rule_21_ace_of_trumps_never_burned_rob_is_public() {
    let mut m = robber_deal();
    lowest(&mut m);
    lowest(&mut m);
    assert_legal(&m, "2D 3D 4D 5D 7S");
    assert_eq!(
        refused(&mut m, Action::Burn(c("AS"))),
        "an ace may never be burned (house rule H3)"
    );
    burn(&mut m, "3D");
    for seat in 0..8u8 {
        let v = m.view(seat);
        let burned: Vec<Card> = v
            .events
            .iter()
            .filter_map(|e| match *e {
                Event::Burned { card, .. } => Some(card),
                _ => None,
            })
            .collect();
        if seat == 2 {
            assert_eq!(burned, [c("3D")]);
            assert_eq!(v.own_burns, set("3D"));
        } else {
            assert!(
                burned.is_empty(),
                "seat {seat} must not see the burned card"
            );
            assert_eq!(v.own_burns, 0);
        }
        assert!(v.events.contains(&Event::BurnDone { seat: 2 }));
        assert!(v.events.contains(&Event::Took {
            seat: 2,
            card: c("7S")
        }));
        assert_eq!(
            v.public.robber,
            Some(2),
            "everyone knows seat 2 holds the AS"
        );
        assert_eq!(v.public.burn_counts[2], 1);
    }
}

/// Rule 22: "If the turned card is an ace, the dealer must rob it at once, before the first card
/// is led, and burns a card the same way."
#[test]
fn rule_22_turned_ace_dealer_robs_at_once() {
    let mut m = game(7, ["", "", "", "", "", "", "", "2D 3D 4D 5D 6D"], "AC");
    assert!(m.actions().is_empty(), "the rob needs no action");
    let l = log(&m);
    assert_eq!(l[l.len() - 2..], ["Turned card=AC", "Took seat=7 card=AC"]);
    assert_eq!(m.phase(), Phase::AwaitBurn(7));
    let p = m.public();
    assert_eq!(p.trick_no, 0, "before the first card is led");
    assert!(p.trick_cards.is_empty());
    assert_eq!(p.trump, Some(Suit::Clubs));
    assert_eq!(p.robber, Some(7));
    assert_legal(&m, "2D 3D 4D 5D 6D");
    assert_eq!(
        refused(&mut m, Action::Burn(c("AC"))),
        "an ace may never be burned (house rule H3)"
    );
    burn(&mut m, "6D");
    assert_eq!(
        log_from(&m, l.len()),
        ["Burned seat=7 card=6D", "BurnDone seat=7"]
    );
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    assert_eq!(m.public().trick_no, 1);
    assert_eq!(m.public().hidden_count, 13);
}

/// Rule 23: "If the ace of trumps is in the undealt pack, nobody robs. The turned card stays face
/// up on the table for the whole hand and is not played."
#[test]
fn rule_23_ace_undealt_nobody_robs_card_stays() {
    let mut m = game(7, [""; 8], "7S AS");
    for _ in 0..8 {
        assert!(!m.public().ace_undealt);
        lowest(&mut m);
    }
    assert_eq!(last_event(&m), "AceUndealt");
    assert!(m.public().ace_undealt);
    while m.hand_index() == 0 {
        let p = m.public();
        assert_eq!(p.table_card, Some(c("7S")), "the turned card stays face up");
        assert_eq!(p.turned, vec![(c("7S"), None)]);
        assert!(!has(p.played, c("7S")));
        assert_eq!(p.hidden_count, 12);
        assert!(p.burn_counts.iter().all(|&b| b == 0));
        lowest(&mut m);
    }
    assert_eq!(count_events(&m, "Took"), 0, "nobody robs");
    assert_eq!(count_events(&m, "AceUndealt"), 1);
}

/// Rule 24: "If a Joker is turned, the dealer must take it, burning a card; there is no choice,
/// and a dealer who does not take it is turned. The next card of the pack is then turned for
/// trumps, and rules 20 to 23 apply to that card. If that card is an ace of trumps, the dealer
/// must rob it as well and burn another card, else the hand is turned. A dealer who holds the
/// ace of trumps must likewise rob that newly turned card, burning another card. The dealer may
/// not burn the ace of trumps anywhere."
#[test]
fn rule_24_turned_joker_dealer_takes_and_burns() {
    // The dealer takes the Joker at once (no action, no choice) and burns before trumps exist.
    let mut m = game(
        7,
        ["", "", "", "", "", "", "", "AC AD 2C 3C 4C"],
        "JK 7S AS",
    );
    assert!(m.actions().is_empty());
    let l = log(&m);
    assert_eq!(l[l.len() - 2..], ["Turned card=JK", "Took seat=7 card=JK"]);
    assert_eq!(m.phase(), Phase::AwaitBurn(7));
    let p = m.public();
    assert_eq!((p.trump, p.trick_no, p.joker_turned), (None, 0, true));
    assert_eq!(p.robber, None, "taking the Joker is not a rob");
    assert_eq!(p.hand_sizes[7], 6);
    assert_legal(&m, "2C 3C 4C JK");
    assert_eq!(
        refused(&mut m, Action::Play(c("2C"))),
        "a burn is due: the player who took the turned card must burn first"
    );
    burn(&mut m, "2C");
    assert_eq!(
        log_from(&m, l.len()),
        [
            "Burned seat=7 card=2C",
            "BurnDone seat=7",
            "Turned card=7S",
            "ProvedNoAce seat=7",
        ]
    );
    let p = m.public();
    assert_eq!(p.trump, Some(Suit::Spades), "the next card names trumps");
    assert_eq!(p.table_card, Some(c("7S")));
    assert_eq!(p.turned, vec![(JOKER, Some(7)), (c("7S"), None)]);
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    // Rule 23 applies to the new card: its ace is undealt, so nobody robs.
    for _ in 0..8 {
        lowest(&mut m);
    }
    assert_eq!(last_event(&m), "AceUndealt");

    // The new card is an ace: the dealer robs it too and burns another card.
    let mut m = game(7, ["", "", "", "", "", "", "", "2C 3C 4C 5C 6C"], "JK AD");
    burn(&mut m, "JK");
    let l = log(&m);
    assert_eq!(l[l.len() - 2..], ["Turned card=AD", "Took seat=7 card=AD"]);
    assert_eq!(m.phase(), Phase::AwaitBurn(7));
    assert_eq!(m.public().trick_no, 0);
    assert_legal(&m, "2C 3C 4C 5C 6C");
    burn(&mut m, "6C");
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    assert_eq!(m.truth().burns[7], set("JK 6C"));
    assert_eq!(m.public().robber, Some(7));
    assert_eq!(count(m.truth().undealt), 11);

    // The dealer holds the ace of the new trump suit: it robs the new card at once.
    let mut m = game(7, ["", "", "", "", "", "", "", "AS 2C 3C 4C 5C"], "JK 7S");
    burn(&mut m, "2C");
    let l = log(&m);
    assert_eq!(l[l.len() - 2..], ["Turned card=7S", "Took seat=7 card=7S"]);
    assert_eq!(m.phase(), Phase::AwaitBurn(7));
    assert_legal(&m, "3C 4C 5C 7S JK");
    assert_eq!(
        refused(&mut m, Action::Burn(c("AS"))),
        "an ace may never be burned (house rule H3)"
    );
    burn(&mut m, "JK");
    assert_eq!(m.phase(), Phase::AwaitPlay(0));

    // Another seat holds the new ace: rule 20 applies, it robs at its first turn.
    let mut m = game(7, ["", "", "AS", "", "", "", "", "2C 3C 4C 5C 6C"], "JK 7S");
    burn(&mut m, "2C");
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    lowest(&mut m);
    lowest(&mut m);
    assert_eq!(last_event(&m), "Took seat=2 card=7S");
    assert_eq!(m.phase(), Phase::AwaitBurn(2));
}

/// Rule 25: "A player who holds the ace of trumps and plays a card without robbing has their
/// cards turned face down, and so does their partner. The pair plays no more cards that hand and
/// scores nothing."
///
/// Impossible under H1: the engine robs for the ace holder, and refuses every play (by anyone)
/// until the burn is made. The pair then plays the whole hand as usual.
#[test]
fn rule_25_cannot_play_without_robbing() {
    let mut m = robber_deal();
    lowest(&mut m);
    lowest(&mut m);
    assert_eq!(m.phase(), Phase::AwaitBurn(2));
    assert!(
        m.legal_actions()
            .iter()
            .all(|a| matches!(a, Action::Burn(_)))
    );
    for x in cards(hand(&m, 2)).chain(cards(hand(&m, 3))) {
        assert_eq!(
            refused(&mut m, Action::Play(x)),
            "a burn is due: the player who took the turned card must burn first"
        );
    }
    burn(&mut m, "5D");
    finish_hand(&mut m);
    let mut cards_played = [0; 8];
    for (_, e) in m.events() {
        if let Event::Played { seat, .. } = *e {
            cards_played[usize::from(seat)] += 1;
        }
    }
    assert_eq!(
        cards_played, [5; 8],
        "the robber and its partner play on as usual"
    );
}

/// Rule 26: "A dealer who refuses or forgets to rob a turned ace is punished the same way."
///
/// Impossible under H1: the dealer's take is automatic and no card can be played until the
/// dealer has burned, whether the ace is turned first or after a Joker.
#[test]
fn rule_26_dealer_cannot_refuse_a_turned_ace() {
    for pack in ["AC", "JK AC"] {
        let mut m = game(7, ["", "", "", "", "", "", "", "2D 3D 4D 5D 6D"], pack);
        if pack.starts_with("JK") {
            burn(&mut m, "JK");
        }
        assert!(log(&m).contains(&"Took seat=7 card=AC".to_string()));
        assert_eq!(m.phase(), Phase::AwaitBurn(7));
        assert!(!m.legal_actions().is_empty());
        assert!(
            m.legal_actions()
                .iter()
                .all(|a| matches!(a, Action::Burn(_)))
        );
        for x in cards(hand(&m, 7)).chain(cards(hand(&m, 0))) {
            refused(&mut m, Action::Play(x));
        }
        refused(&mut m, Action::Burn(c("AC")));
        burn(&mut m, "2D");
        assert_eq!(m.public().robber, Some(7));
        assert_eq!(m.phase(), Phase::AwaitPlay(0));
    }
}

// ===================================================================================
// Scoring (rules 27–36)
// ===================================================================================

/// Rule 27: "Each trick is worth 5 points."
#[test]
fn rule_27_each_trick_worth_five() {
    assert_eq!(TRICK_POINTS, 5);
    let h = hand_a();
    let mut m = scripted_match(Rules::default(), &[&h]);
    let expected = [[5, 0, 0, 0], [10, 0, 0, 0], [10, 5, 0, 0], [10, 10, 0, 0]];
    for (k, scores) in expected.iter().enumerate() {
        play_trick(&mut m, &h, k);
        assert_eq!(m.scores(), *scores, "after trick {}", k + 1);
    }
    let start = m.events().len();
    play_trick(&mut m, &h, 4);
    assert_eq!(
        log_from(&m, start)[8..11],
        [
            "TrickWon trick=5 seat=3 team=3 points=5",
            "Bonus team=1 points=6 kind=end card=JS",
            "HandEnd hand=0 aborted=false",
        ]
    );
    assert_eq!(
        m.scores(),
        [10, 16, 0, 5],
        "25 for the tricks and 6 for the bonus"
    );
}

/// Rule 28: "The trick holding the best trump played in the hand is worth 11 points, 5 for the
/// trick and 6 extra, the first time in a game that a partnership takes it. Every later time in
/// the same game it is worth 10 to that partnership, 5 and 5, while each other partnership still
/// gets 11 for its own first one."
///
/// Three hands of one game: team 1 takes its first best-trump trick (11), then team 2 its first
/// (still 11), then team 1 its second (10).
#[test]
fn rule_28_best_trump_trick_eleven_first_then_ten() {
    let (r0, r1, r2) = rule28_hands();
    let mut m = scripted_match(Rules::default(), &[&r0, &r1, &r2]);
    play_tricks(&mut m, &r0, 5);
    assert_eq!(
        count_events(&m, "Bonus team=1 points=6 kind=end card=JS"),
        1
    );
    assert_eq!(m.scores(), [10, 11, 5, 5]);
    assert_eq!(m.bonus_taken(), [false, true, false, false]);
    let start = m.events().len();
    play_tricks(&mut m, &r1, 5);
    assert_eq!(
        after(&m, start, "TrickWon trick=5 seat=3 team=3 points=5", 1),
        ["Bonus team=2 points=6 kind=end card=JS"],
        "another team's first bonus is still 6"
    );
    assert_eq!(m.scores(), [15, 16, 16, 15]);
    assert_eq!(m.bonus_taken(), [false, true, true, false]);
    let start = m.events().len();
    play_tricks(&mut m, &r2, 5);
    assert_eq!(
        after(&m, start, "TrickWon trick=5 seat=0 team=0 points=5", 1),
        ["Bonus team=1 points=5 kind=end card=JS"],
        "a team's second bonus in a game is 5: a 10-point trick"
    );
    assert_eq!(m.scores(), [25, 26, 21, 20]);
    assert_eq!(m.game_index(), 0);
}

/// The three hands of rule 28: dealers 7, 0, 1; the J of spades trick goes to seats 1, 2 and 5.
fn rule28_hands() -> (Script, Script, Script) {
    (
        scripted(7, [Clubs, Diamonds, Hearts, Trumps], [0, 2, 3, 1, 0]),
        scripted(0, [Clubs, Diamonds, Hearts, Trumps], [1, 3, 0, 2, 3]),
        scripted(1, [Clubs, Diamonds, Hearts, Trumps], [2, 0, 3, 5, 0]),
    )
}

/// Rule 29: "If the 5 of trumps is played, its trick is the best-trump trick, and the points
/// count as soon as that trick is won."
///
/// The bonus is in the scores straight after the trick, before the next card, and there is no
/// second bonus at the end of the hand. Counting at once can end a game in the middle of a hand.
#[test]
fn rule_29_five_of_trumps_bonus_counts_at_once() {
    let h = scripted(7, [FiveOfTrumps, Clubs, Diamonds, Hearts], [0, 0, 2, 3, 1]);
    let mut m = scripted_match(Rules::default(), &[&h]);
    play_trick(&mut m, &h, 0);
    assert_eq!(
        after(&m, 0, "Played seat=7 card=8S", 3),
        [
            "TrickWon trick=1 seat=0 team=0 points=5",
            "Bonus team=0 points=6 kind=immediate card=5S",
            "AceUndealt",
        ]
    );
    assert_eq!(m.scores(), [11, 0, 0, 0], "counted before the next card");
    assert!(m.public().five_played);
    assert_eq!(m.bonus_taken(), [true, false, false, false]);
    play_tricks_from(&mut m, &h, 1);
    assert_eq!(count_events(&m, "Bonus"), 1, "no second bonus at the end");
    assert_eq!(m.scores(), [16, 5, 5, 5]);

    // Team 0 on 31 wins the 5 of spades trick: 36, and the bonus at once makes 41.
    let (h0, h1) = (a0(), a1_immediate_41());
    let mut m = scripted_match(Rules::default(), &[&h0, &h1]);
    play_tricks(&mut m, &h0, 5);
    let start = m.events().len();
    play_tricks(&mut m, &h1, 2);
    assert_eq!(
        after(&m, start, "TrickWon trick=2 seat=0 team=0 points=5", 3),
        [
            "Bonus team=0 points=5 kind=immediate card=5S",
            "GameWon team=0 reason=immediate_bonus scores=41,5,0,0",
            "HandEnd hand=1 aborted=true",
        ]
    );
}

/// Play the script's tricks from `k` to the end.
fn play_tricks_from(m: &mut Match, s: &Script, k: usize) {
    for t in k..5 {
        play_trick(m, s, t);
    }
}

/// Rule 30: "If the 5 is never played, the bonus goes at the end of the hand to the trick that
/// held the highest trump played, whatever that trump is."
///
/// Only two trumps are dealt (the fewest possible): the 9S and the 10S, the two lowest black
/// trumps. The 9S wins trick 4 and the 10S trick 5; the bonus goes, at the end of the hand, to
/// trick 4, whose 9S is the highest trump played, not to the last trick won by a trump.
#[test]
fn rule_30_end_of_hand_bonus_to_highest_trump_played() {
    let mut m = game(
        7,
        [
            "KC KD 2H 9H QH",
            "10C 2D 3H 10H JH",
            "9C 3D 4H 9S 8D",
            "8C 4D KH QC QD",
            "7C 5D 5H JC JD",
            "6C 6D 6H AC 10S",
            "5C 7D 7H 2C 10D",
            "4C AD 8H 3C 9D",
        ],
        "3S 5S JS JK AH AS KS QS 2S 4S 6S 7S 8S",
    );
    plays_all(&mut m, "KC 10C 9C 8C 7C 6C 5C 4C");
    plays_all(&mut m, "KD 2D 3D 4D 5D 6D 7D AD");
    plays_all(&mut m, "2H 3H 4H KH 5H 6H 7H 8H");
    plays_all(&mut m, "QC JC AC 2C 3C 9H 10H 9S");
    assert_eq!(winner_from(&m, m.events().len() - 2), 2);
    assert_eq!(m.scores(), [10, 0, 5, 5], "no bonus yet");
    assert_eq!(m.public().best_trump, Some((c("9S"), 2)));
    let start = m.events().len();
    plays_all(&mut m, "8D QD JD 10S 10D 9D QH JH");
    assert_eq!(
        log_from(&m, start)[8..11],
        [
            "TrickWon trick=5 seat=5 team=1 points=5",
            "Bonus team=2 points=6 kind=end card=9S",
            "HandEnd hand=0 aborted=false",
        ]
    );
    assert_eq!(m.scores(), [10, 5, 11, 5]);
}

/// Rule 31: "A hand in which no trump at all is played has no best-trump trick, and thus no
/// bonus: just five tricks of 5 each."
///
/// RULES.md notes this cannot happen in a completed 8-player hand (at least 14 trumps, at most
/// 13 cards out of play), so it is tested on a partial hand: team 0, on 26, wins tricks 1–3 of
/// the next hand with no trump played and reaches 41; no bonus is awarded. The decision test
/// `decision_rule_31_cannot_occur_in_a_completed_hand` covers completed hands.
#[test]
fn rule_31_no_trump_played_no_bonus() {
    let (h0, h1) = (a0(), a1_plain_41());
    let mut m = scripted_match(Rules::default(), &[&h0, &h1]);
    play_tricks(&mut m, &h0, 5);
    let start = m.events().len();
    play_tricks(&mut m, &h1, 2);
    for &(seat, x) in &h1.tricks[2][..7] {
        assert_eq!(m.to_act(), Some(seat));
        act(&mut m, Action::Play(x));
    }
    let p = m.public();
    assert_eq!(p.best_trump, None, "no trump played so far");
    assert_eq!(p.played & trumps_of(Suit::Spades), 0);
    let (seat, x) = h1.tricks[2][7];
    assert_eq!(m.to_act(), Some(seat));
    act(&mut m, Action::Play(x));
    assert_eq!(
        after(&m, start, "TrickWon trick=3 seat=0 team=0 points=5", 1),
        ["GameWon team=0 reason=trick scores=41,5,0,0"]
    );
    assert_eq!(
        log_from(&m, start)
            .iter()
            .filter(|l| l.starts_with("Bonus"))
            .count(),
        0,
        "no bonus"
    );
}

/// Rule 32: "The game is won by the first pair to reach 41. The game ends the moment a pair gets
/// there, even in the middle of a hand, and the rest of the hand is not played."
#[test]
fn rule_32_game_ends_at_41_even_mid_hand() {
    let (h0, h1) = (a0(), a1_plain_41());
    let mut m = scripted_match(Rules::default(), &[&h0, &h1]);
    play_tricks(&mut m, &h0, 5);
    assert_eq!(m.scores(), [26, 5, 0, 0]);
    let start = m.events().len();
    play_tricks(&mut m, &h1, 2);
    assert_eq!(m.scores(), [36, 5, 0, 0]);
    assert_eq!(m.games_won(), [0; 4]);
    play_trick(&mut m, &h1, 2);
    assert_eq!(
        after(&m, start, "TrickWon trick=3 seat=0 team=0 points=5", 3),
        [
            "GameWon team=0 reason=trick scores=41,5,0,0",
            "HandEnd hand=1 aborted=true",
            "HandStart hand=2 game=1 dealer=1",
        ]
    );
    assert_eq!(m.games_won(), [1, 0, 0, 0]);
    // Tricks 4 and 5 of that hand were never played: a new hand is already dealt.
    let played = log_from(&m, start)
        .iter()
        .take_while(|l| !l.starts_with("HandEnd"))
        .filter(|l| l.starts_with("Played"))
        .count();
    assert_eq!(played, 24);
    assert_eq!(m.hand_index(), 2);
    assert_eq!(m.public().trick_no, 1);
    assert!(m.public().trick_winners.is_empty());
}

/// Rule 33: "A pair that wins all five tricks of a hand makes a jink and wins the game at once,
/// whatever the score."
///
/// Team 1, on 5, takes all five tricks: 30 points, while team 0 has 26. It wins the game at once;
/// no end-of-hand bonus is awarded (the jink is checked first).
#[test]
fn rule_33_jink_wins_the_game() {
    let (h0, h1) = (a0(), jink(0, 5));
    let mut m = scripted_match(Rules::default(), &[&h0, &h1]);
    play_tricks(&mut m, &h0, 5);
    let start = m.events().len();
    play_tricks(&mut m, &h1, 5);
    assert_eq!(
        after(&m, start, "TrickWon trick=5 seat=5 team=1 points=5", 3),
        [
            "GameWon team=1 reason=jink scores=26,30,0,0",
            "HandEnd hand=1 aborted=false",
            "HandStart hand=2 game=1 dealer=1",
        ]
    );
    assert_eq!(m.games_won(), [0, 1, 0, 0]);
    assert_eq!(
        log_from(&m, start)
            .iter()
            .filter(|l| l.starts_with("Bonus"))
            .count(),
        0
    );
}

/// Rule 34: "If every other partnership has been turned, the partnership still in play takes
/// every trick of the hand, the turned partnerships' tricks included under rule 39, and that is
/// a jink: it wins the game."
///
/// Impossible under H1: no partnership is ever turned. In a whole random match every trick is
/// played by all eight seats in turn, and every jink is a team winning the five tricks itself.
#[test]
fn rule_34_no_partnership_is_ever_turned() {
    let m = random_match(34, Rules::default(), |_| {});
    assert!(check_tricks(&m) >= 25);
    let mut won = [0u8; 4];
    for (_, e) in m.events() {
        match *e {
            Event::HandStart { .. } => won = [0; 4],
            Event::TrickWon { team, .. } => won[usize::from(team)] += 1,
            Event::GameWon {
                team,
                reason: engine::game::GameEndReason::Jink,
                ..
            } => assert_eq!(won[usize::from(team)], 5, "a jink is five tricks won"),
            _ => {}
        }
    }
}

/// Rule 35: "The money is the match: the first pair to win two games."
///
/// Team 0 wins game 1, team 1 game 2 and team 0 game 3 (three jinks): the match ends at once
/// with the money for team 0; nothing more can be played.
#[test]
fn rule_35_match_is_first_to_two_games() {
    let (h0, h1, h2) = (jink(7, 0), jink(0, 1), jink(1, 4));
    let mut m = scripted_match(Rules::default(), &[&h0, &h1, &h2]);
    play_tricks(&mut m, &h0, 5);
    assert_eq!(m.games_won(), [1, 0, 0, 0]);
    assert_eq!((m.money(), m.game_index()), (None, 1));
    assert_ne!(m.phase(), Phase::MatchOver);
    play_tricks(&mut m, &h1, 5);
    assert_eq!(m.games_won(), [1, 1, 0, 0]);
    assert_eq!((m.money(), m.game_index()), (None, 2));
    play_tricks(&mut m, &h2, 5);
    let l = log(&m);
    assert_eq!(
        l[l.len() - 4..],
        [
            "TrickWon trick=5 seat=4 team=0 points=5",
            "GameWon team=0 reason=jink scores=25,0,0,0",
            "HandEnd hand=2 aborted=false",
            "MatchWon team=0 money=30,-10,-10,-10",
        ]
    );
    assert_eq!(m.games_won(), [2, 1, 0, 0]);
    assert_eq!(m.phase(), Phase::MatchOver);
    assert_eq!(m.to_act(), None);
    assert!(m.legal_actions().is_empty());
    assert_eq!(refused(&mut m, Action::Play(c("2C"))), "the match is over");
}

/// Rule 36: "In each new game the scores go back to nought. The games each partnership has won
/// are kept."
///
/// Team 0 wins game 1 with its bonus flag set; the new game starts from 0, 0, 0, 0 with every
/// flag cleared and the game kept, so team 0's next best-trump trick is a first one again (11).
#[test]
fn rule_36_scores_reset_each_game_games_kept() {
    let (h0, h1) = (a0(), a1_plain_41());
    let h2 = scripted(1, [Clubs, Diamonds, Hearts, Trumps], [2, 0, 3, 4, 1]);
    let mut m = scripted_match(Rules::default(), &[&h0, &h1, &h2]);
    play_tricks(&mut m, &h0, 5);
    assert_eq!(m.bonus_taken(), [true, false, false, false]);
    play_tricks(&mut m, &h1, 3);
    assert_eq!(m.scores(), [0; 4]);
    assert_eq!(m.bonus_taken(), [false; 4]);
    assert_eq!(m.games_won(), [1, 0, 0, 0]);
    assert_eq!(m.game_index(), 1);
    let p = m.public();
    assert_eq!(
        (p.scores, p.games_won, p.game_index),
        ([0; 4], [1, 0, 0, 0], 1)
    );
    let start = m.events().len();
    play_tricks(&mut m, &h2, 5);
    assert_eq!(
        after(&m, start, "TrickWon trick=5 seat=1 team=1 points=5", 1),
        ["Bonus team=0 points=6 kind=end card=JS"],
        "team 0's first bonus of the new game"
    );
    assert_eq!(m.scores(), [16, 5, 5, 5]);
    assert_eq!(m.games_won(), [1, 0, 0, 0]);
}

// ===================================================================================
// Penalties (rules 37–41): impossible under H1
// ===================================================================================

/// Rule 37: "An illegal card is not always caught when it is played. The partnership's cards are
/// turned the moment a later card proves it: a card of the suit the player claimed not to hold
/// (played off-suit when it was led), a trump the player should have played, or the ace the
/// player failed to rob with."
///
/// Impossible under H1: an illegal card is refused when it is played. Rule 37's three examples
/// are refused (a revoke, a renege, and a play before robbing is shown in rule 25), and at every
/// play of a whole random match the engine offers exactly the cards that rules 13–18 allow
/// (checked against a separate reading of those rules) and refuses every other card in hand.
#[test]
fn rule_37_illegal_card_never_offered_or_accepted() {
    let mut m = rule14_deal();
    play(&mut m, "9D");
    assert_eq!(
        refused(&mut m, Action::Play(c("2H"))),
        "you hold the suit led: follow suit or play a trump (rule 14)"
    );
    let mut m = rule16_deal();
    play(&mut m, "10C");
    assert_eq!(
        refused(&mut m, Action::Play(c("2D"))),
        "a trump was led and you hold a trump you may not hold back: play a trump (rules 16-17)"
    );
    let mut decisions = 0;
    random_match(37, Rules::default(), |m| {
        let Phase::AwaitPlay(seat) = m.phase() else {
            return;
        };
        let p = m.public();
        let held = hand(m, seat);
        let led = p.trick_cards.first().map(|&(_, x)| x);
        let allowed = rulebook_legal(held, led, p.trump.expect("trumps are known"));
        assert_eq!(names(legal(m)), names(allowed), "seat {seat}");
        for x in cards(held & !allowed) {
            refused(&mut m.clone(), Action::Play(x));
        }
        decisions += 1;
    });
    assert!(decisions >= 200, "{decisions} plays checked");
}

/// Rule 38: "Anyone can break the rules on purpose, for example by taking the turned card
/// without the ace. The partnership's cards are turned as soon as the cards shown at the table
/// prove it: the ace of trumps appearing in another hand, the hand ending without it, or the
/// pair trying to claim the game."
///
/// Impossible under H1: there is no "take" action. A burn is asked only of the seat that has just
/// been given a turned card (the dealer with a Joker or an ace, or the holder of the ace of
/// trumps), and a burn by anyone else is refused.
#[test]
fn rule_38_rules_cannot_be_broken_on_purpose() {
    let mut m = game(7, [""; 8], "7S AS");
    let first = cards(hand(&m, 0)).next().expect("a card");
    assert_eq!(
        refused(&mut m, Action::Burn(first)),
        "a card must be played: nobody has a burn to make"
    );
    let mut burns = 0;
    random_match(38, Rules::default(), |m| match m.phase() {
        Phase::AwaitBurn(seat) => {
            let took = matches!(
                m.events().last(),
                Some((_, Event::Took { seat: s, .. })) if *s == seat
            );
            assert!(took, "a burn without a take");
            if let Some(t) = m.public().trump {
                assert!(
                    has(hand(m, seat), card(t, ACE)),
                    "a turned card taken without the ace of trumps"
                );
            }
            burns += 1;
        }
        Phase::AwaitPlay(seat) => {
            assert!(
                m.legal_actions()
                    .iter()
                    .all(|a| matches!(a, Action::Play(_)))
            );
            let x = cards(hand(m, seat)).next().expect("a card");
            refused(&mut m.clone(), Action::Burn(x));
        }
        Phase::MatchOver => {}
    });
    assert!(burns > 0);
}

/// Rule 39: "Once a partnership is turned, none of its cards counts any more. Every trick it had
/// won in the hand passes, with its points, to the partnership still in play that played the
/// strongest card in that trick. If the highest trump of the hand was one of the turned cards,
/// the bonus goes to the highest trump played in the hand by a partnership that is not turned,
/// and counts for that partnership as rule 28 says."
///
/// Impossible under H1: no trick ever passes. Over a whole random match each team's score is
/// exactly 5 per trick it won plus its own bonuses, after every action.
#[test]
fn rule_39_tricks_never_pass_to_another_team() {
    random_match(39, Rules::default(), |m| {
        let mut scores = [0u8; 4];
        for (_, e) in m.events() {
            match *e {
                Event::TrickWon {
                    seat, team, points, ..
                } => {
                    assert_eq!(team, team_of(seat));
                    scores[usize::from(team)] += points;
                }
                Event::Bonus { team, points, .. } => scores[usize::from(team)] += points,
                Event::GameWon { scores: s, .. } => {
                    assert_eq!(s, scores);
                    scores = [0; 4];
                }
                _ => {}
            }
        }
        if m.phase() != Phase::MatchOver {
            assert_eq!(m.scores(), scores);
        }
    });
}

/// Rule 40: "When a partnership is turned in the middle of a trick, the trick goes on among the
/// other players without the turned cards. If the player who was to lead is turned, the lead
/// passes to the next player on the left who is still in."
///
/// Impossible under H1: every trick of a whole random match has all eight cards, in seat order,
/// led by the seat left of the dealer or by the previous winner, and no hand ends in the middle
/// of a trick.
#[test]
fn rule_40_every_trick_completes_with_all_eight() {
    let mut longest = 0;
    let m = random_match(40, Rules::default(), |m| {
        longest = longest.max(m.public().trick_cards.len());
        assert!(m.public().trick_cards.len() < 8);
    });
    assert_eq!(longest, 7);
    assert!(check_tricks(&m) >= 25);
}

/// Rule 41: "The turned partnership scores nothing for that hand."
///
/// Impossible under H1: an illegal action is refused and leaves the match exactly as it was
/// (events, public state, hidden cards, recorded actions), so nobody is ever turned; and within a
/// game no team's score ever goes down.
#[test]
fn rule_41_turned_pair_scores_nothing() {
    let mut last = (0u8, [0u8; 4]);
    random_match(41, Rules::default(), |m| {
        let mut t = m.clone();
        refused(&mut t, Action::Play(53));
        refused(&mut t, Action::Burn(255));
        match m.phase() {
            Phase::AwaitPlay(seat) | Phase::AwaitBurn(seat) => {
                let missing = cards(ALL_CARDS & !hand(m, seat)).next().expect("a card");
                refused(&mut t, Action::Play(missing));
                refused(&mut t, Action::Burn(missing));
                let x = cards(hand(m, seat)).next().expect("a card");
                if let Phase::AwaitPlay(_) = m.phase() {
                    refused(&mut t, Action::Burn(x));
                } else {
                    refused(&mut t, Action::Play(x));
                }
            }
            Phase::MatchOver => {
                refused(&mut t, Action::Play(0));
            }
        }
        if m.game_index() == last.0 {
            for team in 0..4 {
                assert!(m.scores()[team] >= last.1[team], "a score went down");
            }
        }
        last = (m.game_index(), m.scores());
    });
}

// ===================================================================================
// House rules (H1–H7)
// ===================================================================================

/// Each of the 512 possible actions (burn or play of any byte) is accepted exactly when it is
/// offered, and a refused one leaves the match unchanged.
fn assert_only_offered_accepted(m: &Match) {
    let offered = m.legal_actions();
    for x in 0..=255u8 {
        for a in [Action::Burn(x), Action::Play(x)] {
            let mut t = m.clone();
            if offered.contains(&a) {
                act(&mut t, a);
            } else {
                refused(&mut t, a);
            }
        }
    }
}

/// H1: "Illegal plays are impossible. The engine offers only legal actions, and all forced
/// actions (robbing) happen automatically. So rules 25, 26, 34 and 37–41 can never trigger and
/// are not implemented as gameplay; tests assert that illegal actions are rejected."
#[test]
fn house_h1_illegal_actions_impossible() {
    // The dealer's Joker burn, before trumps are known.
    let m = game(
        7,
        ["", "", "", "", "", "", "", "AC AD 2C 3C 4C"],
        "JK 7S AS",
    );
    assert_only_offered_accepted(&m);
    // A robber's burn: the rob itself happened without any action.
    let mut m = game(7, ["", "", "AS AC AD AH 9C", "", "", "", "", ""], "7S");
    lowest(&mut m);
    lowest(&mut m);
    assert_eq!(m.actions().len(), 2);
    assert_eq!(last_event(&m), "Took seat=2 card=7S");
    assert_only_offered_accepted(&m);
    // A plain lead and a trump lead, mid-trick.
    let mut m = rule14_deal();
    play(&mut m, "9D");
    assert_only_offered_accepted(&m);
    let mut m = rule16_deal();
    play(&mut m, "10C");
    assert_only_offered_accepted(&m);
    // The match is over.
    let (h0, h1) = (jink(7, 0), jink(0, 4));
    let mut m = scripted_match(Rules::default(), &[&h0, &h1]);
    play_tricks(&mut m, &h0, 5);
    play_tricks(&mut m, &h1, 5);
    assert_eq!(m.phase(), Phase::MatchOver);
    assert_only_offered_accepted(&m);
}

/// H2: "Robbing timing (rule 20). The ace-of-trumps holder robs at their own first turn in trick
/// 1, immediately before playing their first card. If they lead trick 1, they rob before leading.
/// They choose the burn after seeing any trick-1 cards already played."
#[test]
fn house_h2_rob_at_own_first_turn_in_trick_one() {
    // Third to play: the rob comes after seeing the first two cards of trick 1.
    let mut m = robber_deal();
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    lowest(&mut m);
    assert_eq!(m.phase(), Phase::AwaitPlay(1));
    assert_eq!(m.public().robber, None, "not before its turn");
    lowest(&mut m);
    assert_eq!(m.phase(), Phase::AwaitBurn(2));
    let p = m.public();
    assert_eq!(p.trick_no, 1);
    assert_eq!(
        p.trick_cards.len(),
        2,
        "the burn is chosen after seeing these"
    );
    burn(&mut m, "4D");
    assert_eq!(m.phase(), Phase::AwaitPlay(2));
    // Leading trick 1: the rob comes before the lead.
    let mut m = game(7, ["AS 2D 3D 4D 5D", "", "", "", "", "", "", ""], "7S");
    assert_eq!(m.phase(), Phase::AwaitBurn(0));
    assert_eq!(last_event(&m), "Took seat=0 card=7S");
    let p = m.public();
    assert_eq!((p.trick_no, p.leader), (1, 0));
    assert!(p.trick_cards.is_empty());
    burn(&mut m, "2D");
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    play(&mut m, "AS");
    assert_eq!(m.public().trick_cards, vec![(0, c("AS"))]);
}

/// H3: "Burn choice (tightened by the owner on 2026-10-13). Any card may be burned except an ace
/// of any suit (the ace of hearts included). This applies to every burn: the dealer's after
/// taking a turned Joker (made before the next card is turned, so before trumps are known), the
/// dealer's after robbing a turned ace, and the robber's. The card just taken (the turned card,
/// or the Joker for the dealer) may be burned. A legal burn always exists: at most four aces can
/// sit among the six cards held at a burn, so at least two cards are burnable."
///
/// Each kind of burn with all four aces in hand: exactly the two other cards may be burned.
#[test]
fn house_h3_no_ace_is_ever_burned() {
    let aces = ["AC", "AD", "AH", "AS"];
    // The dealer's burn after a turned Joker, then (holding the A♥ when hearts turn up) the
    // dealer's rob of the new card.
    let mut m = game(7, ["", "", "", "", "", "", "", "AC AD AH AS 2C"], "JK 7H");
    assert_eq!(m.public().trump, None);
    assert_legal(&m, "2C JK");
    for a in aces {
        refused(&mut m, Action::Burn(c(a)));
    }
    burn(&mut m, "2C");
    assert_eq!(last_event(&m), "Took seat=7 card=7H");
    assert_eq!(m.phase(), Phase::AwaitBurn(7));
    assert_legal(&m, "7H JK");
    for a in aces {
        refused(&mut m, Action::Burn(c(a)));
    }
    burn(&mut m, "7H");
    assert_eq!(m.truth().burns[7], set("2C 7H"));
    // The dealer's burn after robbing a turned ace.
    let mut m = game(7, ["", "", "", "", "", "", "", "AD AH AS 2C 3C"], "AC");
    assert_legal(&m, "2C 3C");
    for a in aces {
        refused(&mut m, Action::Burn(c(a)));
    }
    // The robber's burn: the turned card just taken may be burned.
    let mut m = game(7, ["", "", "AS AC AD AH 9C", "", "", "", "", ""], "7S");
    lowest(&mut m);
    lowest(&mut m);
    assert_legal(&m, "9C 7S");
    for a in aces {
        assert_eq!(
            refused(&mut m, Action::Burn(c(a))),
            "an ace may never be burned (house rule H3)"
        );
    }
    burn(&mut m, "7S");
    assert_eq!(hand(&m, 2), set("AS AC AD AH 9C"));
}

/// H4: "Dealer robs at once (rules 22 and 24). A turned ace is robbed by the dealer at once,
/// before the first lead. With a turned Joker, the dealer takes it and burns at once, then the
/// next card is turned. If that card is an ace, or if the dealer holds the ace of trumps, the
/// dealer robs the new card at once, always, burning another card. Otherwise rule 20 / H2 applies
/// to whoever holds the ace of trumps, or rule 23 if it is undealt."
#[test]
fn house_h4_dealer_robs_at_once() {
    let dealer = "2C 3C 4C 5C 6C";
    // At once: no action and no card played before the dealer's burn; trick 1 not started.
    let at_once = |m: &Match, took: &str| {
        assert_eq!(last_event(m), took);
        assert_eq!(m.phase(), Phase::AwaitBurn(7));
        assert_eq!(m.public().trick_no, 0);
        assert_eq!(count_events(m, "Played"), 0);
        assert_eq!(m.public().robber, Some(7));
    };
    let m = game(7, ["", "", "", "", "", "", "", dealer], "AD");
    at_once(&m, "Took seat=7 card=AD");
    // A turned A♥ makes hearts trumps and is the ace of trumps.
    let m = game(7, ["", "", "", "", "", "", "", dealer], "AH");
    at_once(&m, "Took seat=7 card=AH");
    assert_eq!(m.public().trump, Some(Suit::Hearts));
    // A turned Joker: taken and burned at once; then an ace is robbed at once.
    let mut m = game(7, ["", "", "", "", "", "", "", dealer], "JK AS");
    assert_eq!(m.phase(), Phase::AwaitBurn(7));
    assert_eq!(m.public().trick_no, 0);
    burn(&mut m, "2C");
    at_once(&m, "Took seat=7 card=AS");
    // A turned Joker with the dealer holding the new ace of trumps: robbed at once.
    let mut m = game(7, ["", "", "", "", "", "", "", "AD 2C 3C 4C 5C"], "JK 7D");
    burn(&mut m, "2C");
    at_once(&m, "Took seat=7 card=7D");
    // Otherwise H2: the holder robs at its own first turn in trick 1.
    let mut m = game(7, ["", "", "", "AS", "", "", "", dealer], "JK 7S");
    burn(&mut m, "2C");
    for _ in 0..3 {
        lowest(&mut m);
    }
    assert_eq!(last_event(&m), "Took seat=3 card=7S");
    assert_eq!(m.public().trick_no, 1);
    // Or rule 23: nobody robs.
    let mut m = game(7, ["", "", "", "", "", "", "", dealer], "JK 7S AS");
    burn(&mut m, "2C");
    for _ in 0..8 {
        lowest(&mut m);
    }
    assert_eq!(last_event(&m), "AceUndealt");
    assert_eq!(count_events(&m, "Took"), 1, "only the Joker was taken");
}

/// H5: "Next dealer after a game ends mid-hand: the deal always moves one seat left, exactly as
/// after any hand."
#[test]
fn house_h5_deal_moves_left_after_a_game_ends_mid_hand() {
    let (h0, h1) = (a0(), a1_plain_41());
    let mut m = scripted_match(Rules::default(), &[&h0, &h1]);
    play_tricks(&mut m, &h0, 5);
    assert_eq!(m.dealer(), 0);
    play_tricks(&mut m, &h1, 3);
    assert_eq!(
        last_hand_line(&m),
        "HandStart hand=2 game=1 dealer=1",
        "seat 0 dealt the hand that ended at trick 3; seat 1 deals next"
    );
    assert_eq!(m.dealer(), 1);
    assert_eq!(m.phase(), Phase::AwaitPlay(2));
}

/// The last `HandStart` line of the log.
fn last_hand_line(m: &Match) -> String {
    log(m)
        .into_iter()
        .rev()
        .find(|l| l.starts_with("HandStart"))
        .expect("a hand")
}

/// H6: "The money. A match is won by the first partnership to win 2 games. Each player of every
/// losing partnership pays the agreed stake to the winning partnership. The pub stake is €5:
/// each winner receives €15 and each loser pays €5. The stake is configurable."
///
/// Money is per team: the winners (two players) get 6 stakes, each losing team (two players)
/// pays 2 stakes.
#[test]
fn house_h6_money_settlement() {
    for (stake, team, expected) in [
        (5, 0u8, [30, -10, -10, -10]),
        (10, 2, [-20, -20, 60, -20]),
        (1, 3, [-2, -2, -2, 6]),
    ] {
        let rules = Rules {
            stake,
            ..Rules::default()
        };
        let (h0, h1) = (jink(7, team), jink(0, team + 4));
        let mut m = scripted_match(rules, &[&h0, &h1]);
        play_tricks(&mut m, &h0, 5);
        play_tricks(&mut m, &h1, 5);
        assert_eq!(m.money(), Some(expected), "stake {stake}");
        let per_player = expected.map(|t| t / 2);
        assert_eq!(
            per_player[usize::from(team)],
            3 * stake,
            "each winner gets three stakes"
        );
        for (t, &p) in per_player.iter().enumerate() {
            if t != usize::from(team) {
                assert_eq!(p, -stake, "each loser pays the stake");
            }
        }
        assert_eq!(expected.iter().sum::<i32>(), 0);
    }
    assert_eq!(Rules::default().stake, 5, "the pub stake");
}

/// H7: "What bots play for. Each partnership maximises its expected money, which equals
/// maximising its chance of winning the match. Points, tricks and games have no value of their
/// own; they matter only through their effect on the chance of the money. The chance of the money
/// is re-evaluated after every trick (see docs/PLAN.md §5.3)."
///
/// The engine's part: the money is the only settled result. It is unknown until the match ends,
/// then equals the `MatchWon` event's money, sums to zero, and depends only on who won the match:
/// two matches won by team 0 with very different points give the same money.
#[test]
fn house_h7_money_is_the_only_ground_truth() {
    let m = random_match(7, Rules::default(), |m| {
        if m.phase() != Phase::MatchOver {
            assert_eq!(m.money(), None);
        }
    });
    let Some((_, Event::MatchWon { team, money })) = m.events().last() else {
        panic!("the match ends with the money");
    };
    assert_eq!(m.money(), Some(*money));
    assert_eq!(money.iter().sum::<i32>(), 0);
    assert_eq!(m.games_won()[usize::from(*team)], 2);
    for (t, &amount) in money.iter().enumerate() {
        assert_eq!(amount > 0, t == usize::from(*team), "only the winners gain");
    }
    // Two jinks (25 points each) or 41 points then a jink: the same money.
    let (j0, j1) = (jink(7, 0), jink(0, 4));
    let mut quick = scripted_match(Rules::default(), &[&j0, &j1]);
    play_tricks(&mut quick, &j0, 5);
    play_tricks(&mut quick, &j1, 5);
    let (h0, h1, h2) = (a0(), a1_plain_41(), jink(1, 4));
    let mut slow = scripted_match(Rules::default(), &[&h0, &h1, &h2]);
    play_tricks(&mut slow, &h0, 5);
    play_tricks(&mut slow, &h1, 3);
    play_tricks(&mut slow, &h2, 5);
    assert_eq!(quick.money(), Some([30, -10, -10, -10]));
    assert_eq!(slow.money(), quick.money());
}

// ===================================================================================
// Owner-confirmed decisions and implementer defaults (RULES.md, 2026-10-13)
// ===================================================================================

/// Decision (default): "Reachable scores. A team's first best-trump trick scores 11 and every
/// later one 10, so a team's score is always 5k or 5k+6 with k ≥ 1 in the second case … A score
/// of 6 is unreachable. A team has taken its first bonus this game exactly when score mod 5 == 1;
/// the engine still keeps the flag explicitly and a test asserts the two agree."
///
/// After every action of whole random matches. (Under the flat rule the shortcut fails: see
/// `decision_bonus_later_rule_switch`.)
#[test]
fn decision_reachable_scores_and_the_bonus_flag() {
    for seed in [101u64, 102, 103] {
        let m = random_match(seed, Rules::default(), |m| {
            for t in 0..4 {
                let s = m.scores()[t];
                assert!(s % 5 <= 1, "score {s} is neither 5k nor 5k+6");
                assert_ne!(s, 6);
                assert_eq!(s % 5 == 1, m.bonus_taken()[t], "flag and score of team {t}");
                assert_eq!(m.public().bonus_taken[t], m.bonus_taken()[t]);
            }
        });
        for (_, e) in m.events() {
            if let Event::GameWon { scores, .. } = *e {
                assert!(scores.iter().all(|&s| s % 5 <= 1 && s != 6), "{scores:?}");
            }
        }
    }
}

/// Decision (owner, rule 32): "'Reach 41' means 41 or more: a team on 40 taking a plain trick
/// (45) or a team on 36 taking the best-trump trick (46, since a team on 36 has already had its
/// first bonus; 47 under the flat rule) wins the game."
#[test]
fn decision_reaching_41_means_41_or_more() {
    let (h0, h1, h2) = (b0(), b1(), b2());
    let mut m = scripted_match(Rules::default(), &[&h0, &h1, &h2]);
    play_tricks(&mut m, &h0, 5);
    assert_eq!(m.scores(), [20, 11, 0, 0]);
    play_tricks(&mut m, &h1, 5);
    assert_eq!(m.scores(), [40, 21, 0, 0], "40 does not win");
    assert_eq!(m.game_index(), 0);
    let start = m.events().len();
    play_trick(&mut m, &h2, 0);
    assert_eq!(
        after(&m, start, "TrickWon trick=1 seat=4 team=0 points=5", 1),
        ["GameWon team=0 reason=trick scores=45,21,0,0"]
    );
    for (rules, total) in [(Rules::HOUSE, 46), (Rules::FLAT, 47)] {
        let (h0, h1) = (a0(), a1_five_from_36());
        let mut m = scripted_match(rules, &[&h0, &h1]);
        play_tricks(&mut m, &h0, 5);
        play_tricks(&mut m, &h1, 2);
        assert_eq!(m.scores(), [36, 5, 0, 0]);
        assert!(m.bonus_taken()[0]);
        let start = m.events().len();
        play_trick(&mut m, &h1, 2);
        assert_eq!(
            after(&m, start, "TrickWon trick=3 seat=0 team=0 points=5", 2),
            [
                format!("Bonus team=0 points={} kind=immediate card=5S", total - 41),
                format!("GameWon team=0 reason=trick scores={total},5,0,0"),
            ]
        );
    }
}

/// Decision (default): "Order of checks after tricks 1–4: award the 5 points for the trick → if
/// the 5 of trumps was in it, award its bonus at once (rule 29) → check 41 → check whether the
/// hand is over. After trick 5 the order in 'End checks' above applies; the jink check comes
/// before the end-of-hand bonus, so a jink and 41 in one hand count as one game."
///
/// End checks after trick 5: "Award the trick first, then check for 41. Then check for a jink.
/// Then award the end-of-hand bonus and check for 41 again."
#[test]
fn decision_order_of_checks_after_each_trick() {
    // Trick 2: 5 points (36), the 5's bonus at once (41), then the 41 check ends the game.
    let (h0, h1) = (a0(), a1_immediate_41());
    let mut m = scripted_match(Rules::default(), &[&h0, &h1]);
    play_tricks(&mut m, &h0, 5);
    let start = m.events().len();
    play_tricks(&mut m, &h1, 2);
    assert_eq!(
        after(&m, start, "Played seat=3 card=8S", 4),
        [
            "TrickWon trick=2 seat=0 team=0 points=5",
            "Bonus team=0 points=5 kind=immediate card=5S",
            "GameWon team=0 reason=immediate_bonus scores=41,5,0,0",
            "HandEnd hand=1 aborted=true",
        ]
    );
    // After trick 5: the jink is checked before the end-of-hand bonus (no bonus for the JS).
    let h = jink(7, 0);
    let mut m = scripted_match(Rules::default(), &[&h]);
    play_tricks(&mut m, &h, 5);
    assert_eq!(
        after(&m, 0, "TrickWon trick=5 seat=0 team=0 points=5", 2),
        [
            "GameWon team=0 reason=jink scores=25,0,0,0",
            "HandEnd hand=0 aborted=false",
        ]
    );
    // A jink that also reaches 41: the 41 check comes first, and it is one game.
    let (h0, h1) = (b0(), b1_all_five());
    let mut m = scripted_match(Rules::default(), &[&h0, &h1]);
    play_tricks(&mut m, &h0, 5);
    let start = m.events().len();
    play_tricks(&mut m, &h1, 5);
    assert_eq!(
        after(&m, start, "TrickWon trick=5 seat=0 team=0 points=5", 2),
        [
            "GameWon team=0 reason=trick scores=45,11,0,0",
            "HandEnd hand=1 aborted=false",
        ]
    );
    assert_eq!(count_events(&m, "GameWon"), 1);
    assert_eq!(m.games_won(), [1, 0, 0, 0]);
    // No jink: the end-of-hand bonus, then 41 again (36 + 5).
    let (h0, h1) = (a0(), a1_end_bonus_41());
    let mut m = scripted_match(Rules::default(), &[&h0, &h1]);
    play_tricks(&mut m, &h0, 5);
    let start = m.events().len();
    play_tricks(&mut m, &h1, 5);
    assert_eq!(
        after(&m, start, "TrickWon trick=5 seat=0 team=0 points=5", 3),
        [
            "Bonus team=0 points=5 kind=end card=JS",
            "GameWon team=0 reason=end_bonus scores=41,10,5,5",
            "HandEnd hand=1 aborted=false",
        ]
    );
}

/// Decision (default): "Pending bonus lost: if a game ends at an earlier trick, a best-trump
/// bonus that would have been awarded at the end of the hand is never awarded (rule 32: the rest
/// of the hand is not played)."
///
/// Team 2 wins trick 1 with the J of spades, the best trump of the hand; team 0 reaches 41 at
/// trick 4; team 2 never gets its bonus.
#[test]
fn decision_pending_bonus_lost_when_the_game_ends_early() {
    let (h0, h1) = (a0(), a1_pending_lost());
    let mut m = scripted_match(Rules::default(), &[&h0, &h1]);
    play_tricks(&mut m, &h0, 5);
    let start = m.events().len();
    play_tricks(&mut m, &h1, 3);
    assert_eq!(m.public().best_trump, Some((c("JS"), 2)));
    assert_eq!(m.scores(), [36, 5, 5, 0]);
    play_trick(&mut m, &h1, 3);
    assert_eq!(
        after(&m, start, "TrickWon trick=4 seat=0 team=0 points=5", 3),
        [
            "GameWon team=0 reason=trick scores=41,5,5,0",
            "HandEnd hand=1 aborted=true",
            "HandStart hand=2 game=1 dealer=1",
        ]
    );
    assert_eq!(
        log_from(&m, start)
            .iter()
            .filter(|l| l.starts_with("Bonus"))
            .count(),
        0
    );
}

/// Decision (default): "Two teams at 41 after the last trick: the trick's points are scored
/// first, so the trick winner wins the game; a pending end-of-hand bonus for another team is
/// lost."
///
/// Team 0 on 36 wins trick 5 (41); team 1 on 36 holds the hand's best trump and its bonus (5)
/// would also make 41, but is never awarded.
#[test]
fn decision_two_teams_over_41_after_trick_five_the_trick_winner_wins() {
    let (h0, h1, h2) = (a0(), a1_two_teams(), a2_two_teams());
    let mut m = scripted_match(Rules::default(), &[&h0, &h1, &h2]);
    play_tricks(&mut m, &h0, 5);
    play_tricks(&mut m, &h1, 5);
    assert_eq!(m.scores(), [36, 26, 0, 0]);
    assert_eq!(m.bonus_taken(), [true, true, false, false]);
    let start = m.events().len();
    play_tricks(&mut m, &h2, 4);
    assert_eq!(m.scores(), [36, 36, 5, 5]);
    assert_eq!(m.public().best_trump, Some((c("JS"), 5)));
    play_trick(&mut m, &h2, 4);
    assert_eq!(
        after(&m, start, "TrickWon trick=5 seat=0 team=0 points=5", 2),
        [
            "GameWon team=0 reason=trick scores=41,36,5,5",
            "HandEnd hand=2 aborted=false",
        ]
    );
    assert_eq!(
        log_from(&m, start)
            .iter()
            .filter(|l| l.starts_with("Bonus"))
            .count(),
        0,
        "team 1's bonus is lost"
    );
    assert_eq!(m.games_won(), [1, 0, 0, 0]);
}

/// Decision (owner, H2): "Dealer holding the ace of trumps with an ordinary turned card: the
/// dealer robs at their own first turn, as the eighth card of trick 1, after seeing the other
/// seven. Only the turned-ace and turned-Joker cases are 'at once' (H4)."
#[test]
fn decision_dealer_holding_the_ace_robs_as_the_eighth_card() {
    let mut m = game(7, ["", "", "", "", "", "", "", "AS 2D 3D 4D 5D"], "7S");
    assert_eq!(m.phase(), Phase::AwaitPlay(0), "not at once");
    for seat in 0..7u8 {
        assert_eq!(m.phase(), Phase::AwaitPlay(seat));
        let start = m.events().len();
        lowest(&mut m);
        let l = log_from(&m, start);
        assert_eq!(l[0], format!("ProvedNoAce seat={seat}"));
        assert!(l[1].starts_with(&format!("Played seat={seat} ")));
    }
    assert_eq!(last_event(&m), "Took seat=7 card=7S");
    assert_eq!(m.phase(), Phase::AwaitBurn(7));
    assert_eq!(
        m.public().trick_cards.len(),
        7,
        "the dealer has seen the other seven"
    );
    burn(&mut m, "2D");
    lowest(&mut m);
    assert!(log(&m).iter().any(|l| l.starts_with("TrickWon trick=1")));
    assert_eq!(m.public().robber, Some(7));
}

/// Decision (owner): "Burn then first card: nothing happens between the robber's burn and their
/// first card, so a bot may choose them as one joint decision internally; they are always
/// recorded and shown as two actions."
#[test]
fn decision_burn_then_first_card_are_two_actions() {
    let mut m = robber_deal();
    lowest(&mut m);
    lowest(&mut m);
    let start = m.events().len();
    burn(&mut m, "2D");
    assert_eq!(m.phase(), Phase::AwaitPlay(2), "the same seat plays next");
    play(&mut m, "3D");
    assert_eq!(
        log_from(&m, start),
        [
            "Burned seat=2 card=2D",
            "BurnDone seat=2",
            "Played seat=2 card=3D",
        ]
    );
    let a = m.actions();
    assert_eq!(
        a[a.len() - 2..],
        [Action::Burn(c("2D")), Action::Play(c("3D"))]
    );
}

/// Decision (default, rules 16–17): "Trump leads: the Joker led is a trump lead. The ace of
/// hearts led when hearts are not trumps is a trump lead too: it forces the ace of trumps and
/// every lower trump, while the 5, J and Joker may be held back."
#[test]
fn decision_joker_and_ace_of_hearts_leads_are_trump_leads() {
    // The Joker led: holders of a lower trump must play one; others may play anything.
    let mut m = quiet(
        7,
        [
            "JK",
            "2C 2D 3D 4D 5D",
            "2H 3H 4H 5H 6H",
            "5C 6D 7D 8D 9D",
            "AH 10D JD QD KD",
            "",
            "",
            "",
        ],
        "9C",
    );
    play(&mut m, "JK");
    walk(
        &mut m,
        &[
            ("2C", "2C"),
            ("2H 3H 4H 5H 6H", "2H"),
            ("5C 6D 7D 8D 9D", "6D"),
            ("AH", "AH"),
        ],
    );
    // The A♥ led with clubs trumps: a trump lead, not a heart lead.
    let mut m = quiet(
        7,
        [
            "AH",
            "3C 7H 8H 9H 10H",
            "JH QH KH 6D 7D",
            "5C 2D 3D 4D 5D",
            "JC 8D 9D 10D JD",
            "JK QD KD 2S 3S",
            "AC 4S 5S 6S 7S",
            "KC 8S 9S 10S JS",
        ],
        "9C",
    );
    play(&mut m, "AH");
    walk(
        &mut m,
        &[
            ("3C", "3C"),              // must trump, though it holds hearts
            ("JH QH KH 6D 7D", "JH"),  // hearts, no trump: anything
            ("5C 2D 3D 4D 5D", "2D"),  // the 5 may be held back
            ("JC 8D 9D 10D JD", "8D"), // the J may be held back
            ("JK QD KD 2S 3S", "QD"),  // the Joker may be held back
        ],
    );
    assert_eq!(
        m.phase(),
        Phase::AwaitBurn(6),
        "seat 6 holds the AC and robs first"
    );
    burn(&mut m, "9C");
    walk(&mut m, &[("AC", "AC"), ("KC", "KC")]);
    assert_eq!(winner_from(&m, 0), 0, "the A♥ beats the AC and the KC");
}

/// Decision (default, rule 24): "Two takes in one hand: when the dealer takes a turned Joker and
/// the next turned card is an ordinary card, the holder of the ace of trumps (if dealt) still
/// robs that card at their own first turn. There are then two burns in the hand, the dealer's and
/// the robber's, each a separate hidden location."
#[test]
fn decision_two_takes_in_one_hand() {
    let mut m = game(7, ["", "", "AS", "", "", "", "", "2C 3C 4C 5C 6C"], "JK 7S");
    burn(&mut m, "2C");
    lowest(&mut m);
    lowest(&mut m);
    assert_eq!(last_event(&m), "Took seat=2 card=7S");
    let x = cards(legal(&m)).next().expect("a burn");
    act(&mut m, Action::Burn(x));
    let p = m.public();
    assert_eq!(p.burn_counts, [0, 0, 1, 0, 0, 0, 0, 1]);
    assert_eq!(p.hidden_count, 13);
    assert_eq!(p.turned, vec![(JOKER, Some(7)), (c("7S"), Some(2))]);
    assert_eq!(p.robber, Some(2));
    assert_eq!(m.truth().burns[7], set("2C"));
    assert_eq!(m.truth().burns[2], bit(x));
    assert_eq!(count(m.truth().undealt), 11);
    assert_eq!(m.view(7).own_burns, set("2C"));
    assert_eq!(m.view(2).own_burns, bit(x));
}

/// Decision (default): "Ace-of-trumps facts become known incrementally: the dealer holds it the
/// moment they rob at once; every other player proves they lack it the moment they play their
/// first card without robbing; only when all eight have played to trick 1 without a rob is it
/// certain to be undealt."
#[test]
fn decision_ace_of_trumps_facts_known_incrementally() {
    // The dealer robs at once: known before anyone plays.
    let m = game(7, ["", "", "", "", "", "", "", "2C 3C 4C 5C 6C"], "AD");
    assert_eq!(m.public().robber, Some(7));
    assert_eq!(m.public().proved_no_ace, [false; 8]);
    // Undealt: one seat at a time, each just before its card; certain only after trick 1.
    let mut m = game(7, [""; 8], "7S AS");
    for k in 0..8usize {
        let p = m.public();
        assert_eq!(p.proved_no_ace.iter().filter(|&&b| b).count(), k);
        assert!(!p.ace_undealt);
        let start = m.events().len();
        lowest(&mut m);
        let l = log_from(&m, start);
        assert_eq!(l[0], format!("ProvedNoAce seat={k}"));
        assert!(l[1].starts_with(&format!("Played seat={k} ")));
    }
    assert_eq!(last_event(&m), "AceUndealt");
    assert!(m.public().ace_undealt);
    // A robber in trick 1: the seats before it prove, the seats after it need not.
    let mut m = game(7, ["", "", "", "AS", "", "", "", ""], "7S");
    finish_hand(&mut m);
    let proved: Vec<u8> = m
        .events()
        .iter()
        .filter_map(|(_, e)| match *e {
            Event::ProvedNoAce { seat } => Some(seat),
            _ => None,
        })
        .collect();
    assert_eq!(proved, [0, 1, 2]);
    assert_eq!(count_events(&m, "AceUndealt"), 0);
}

/// Decision (default, rules 3–4): "First dealer and deal order: the first dealer is drawn once
/// per match from the seeded random-number generator and the deal then moves one seat left after
/// every hand, across games and after a game that ends mid-hand (H5). The engine deals two cards
/// then three, starting left of the dealer, from a freshly shuffled 53-card pack every hand; the
/// next card is turned."
#[test]
fn decision_first_dealer_drawn_once_and_fresh_deck_every_hand() {
    let seed = 2026;
    let mut m = Match::new(seed, Rules::default());
    let mut rng = Rng::new(seed);
    let dealer = rng.bounded(8) as u8;
    assert_eq!(m.dealer(), dealer);
    for hand_no in 0..3u32 {
        let mut d = identity_deck();
        rng.shuffle(&mut d);
        assert_eq!(m.hand_index(), hand_no);
        assert_eq!(m.truth().deck, d, "hand {hand_no} is a fresh shuffle");
        let start = m
            .events()
            .iter()
            .rposition(|(_, e)| matches!(e, Event::HandStart { .. }))
            .expect("a hand");
        let dealer = m.dealer();
        for (i, (_, e)) in m.events()[start + 1..start + 9].iter().enumerate() {
            let i8 = i as u8;
            let want = [2 * i, 2 * i + 1, 16 + 3 * i, 17 + 3 * i, 18 + 3 * i]
                .iter()
                .fold(0, |a, &p| a | bit(d[p]));
            assert_eq!(
                *e,
                Event::Dealt {
                    seat: (dealer + 1 + i8) % 8,
                    cards: want
                }
            );
        }
        assert_eq!(m.events()[start + 9].1, Event::Turned { card: d[40] });
        finish_hand(&mut m);
    }
    // A whole match: the deal moves one seat left at every hand, across games.
    let m = random_match(seed, Rules::default(), |_| {});
    let starts = hand_starts(&m);
    assert_eq!(starts[0].2, dealer);
    for w in starts.windows(2) {
        assert_eq!(w[1].0, w[0].0 + 1);
        assert_eq!(w[1].2, (w[0].2 + 1) % 8);
    }
    assert!(starts.last().expect("hands").1 >= 1, "more than one game");
}

/// Decision (owner): "Rule switch bonus_later: the house rule is bonus_later = 5 (first bonus 6,
/// later ones 5). The Irish 25 app and every other written source use a flat 6 every time;
/// bonus_later = 6 reproduces that. With the flat rule the 'bonus flag ⟺ score mod 5 == 1'
/// shortcut no longer holds."
#[test]
fn decision_bonus_later_rule_switch() {
    assert_eq!(Rules::default(), Rules::HOUSE);
    assert_eq!(
        (
            Rules::HOUSE.bonus_first,
            Rules::HOUSE.bonus_later,
            Rules::HOUSE.stake
        ),
        (6, 5, 5)
    );
    assert_eq!((Rules::FLAT.bonus_first, Rules::FLAT.bonus_later), (6, 6));
    for (rules, second, team1) in [(Rules::HOUSE, 5, 26), (Rules::FLAT, 6, 27)] {
        let (r0, r1, r2) = rule28_hands();
        let mut m = scripted_match(rules, &[&r0, &r1, &r2]);
        play_tricks(&mut m, &r0, 5);
        play_tricks(&mut m, &r1, 5);
        assert_eq!(
            m.scores(),
            [15, 16, 16, 15],
            "first bonuses are 6 either way"
        );
        let start = m.events().len();
        play_tricks(&mut m, &r2, 5);
        assert_eq!(
            after(&m, start, "TrickWon trick=5 seat=0 team=0 points=5", 1),
            [format!("Bonus team=1 points={second} kind=end card=JS")]
        );
        assert_eq!(m.scores(), [25, team1, 21, 20]);
        assert!(m.bonus_taken()[1]);
    }
    // Under the flat rule team 1's 27 is 2 more than a multiple of 5: the shortcut fails.
    assert_eq!(27 % 5, 2);
}

/// Decision (owner): "Table size: 8 players in 4 partnerships is the only configuration the bots
/// are built for."
#[test]
fn decision_table_of_eight_in_four_partnerships() {
    assert_eq!((SEATS, TEAMS), (8, 4));
    let m = Match::new(13, Rules::default());
    let p = m.public();
    assert_eq!(
        (
            p.hand_sizes.len(),
            p.burn_counts.len(),
            p.proved_no_ace.len()
        ),
        (8, 8, 8)
    );
    assert_eq!(
        (p.scores.len(), p.games_won.len(), p.tricks_won.len()),
        (4, 4, 4)
    );
    let dealt: Vec<u8> = m
        .events()
        .iter()
        .filter_map(|(_, e)| match *e {
            Event::Dealt { seat, .. } => Some(seat),
            _ => None,
        })
        .collect();
    let mut sorted = dealt.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        (0..8).collect::<Vec<u8>>(),
        "eight hands, one per seat"
    );
    assert_eq!((0..8u8).map(team_of).max(), Some(3));
}

/// Decision (default): "Records: every game record carries a rules_version, bumped whenever an
/// owner answer changes the engine."
///
/// M0 has no record type yet; the engine carries the version constant, and a record of seed,
/// rules and actions replays the match exactly (CLAUDE.md: replays re-apply actions).
#[test]
fn decision_records_carry_rules_version_and_replay() {
    assert_eq!(RULES_VERSION, 1);
    let m = random_match(14, Rules::default(), |_| {});
    let again = Match::replay(m.seed(), m.rules(), m.actions()).expect("a valid record");
    assert_eq!(again.events(), m.events());
    assert_eq!(again.money(), m.money());
    // A corrupted record is refused rather than replayed differently (an ace is never burnable).
    let mut bad = m.actions().to_vec();
    bad[0] = Action::Burn(c("AC"));
    assert!(Match::replay(m.seed(), m.rules(), &bad).is_err());
}

/// Decision (default): "Rule 31 cannot occur at an 8-player table (follows from the counts): 40
/// cards are in play and at most 13 are out of play, while there are 14 or 15 trumps, so every
/// completed hand contains at least one trump. The rule is implemented (no bonus when no trump
/// was played) and tested on partial hands only; a property test asserts it never fires in a
/// completed hand."
#[test]
fn decision_rule_31_cannot_occur_in_a_completed_hand() {
    // The counts: 12 undealt + 1 turned or burned (+1 more burn after a Joker) ≤ 13 < 14 trumps.
    assert_eq!(40 + 12 + 1, DECK_SIZE);
    assert!(count(trump_mask(Suit::Hearts)) > 13);
    for seed in [31u64, 32, 33] {
        let m = random_match(seed, Rules::default(), |_| {});
        let mut trump = None;
        let mut played: Bits = 0;
        let mut tricks = 0;
        let mut completed = 0;
        for (_, e) in m.events() {
            match *e {
                Event::HandStart { .. } => {
                    (trump, played, tricks) = (None, 0, 0);
                }
                Event::Turned { card } if card != JOKER => trump = suit_of(card),
                Event::Played { card, .. } => played |= bit(card),
                Event::TrickWon { .. } => tricks += 1,
                Event::HandEnd { .. } if tricks == 5 => {
                    let t = trump.expect("trumps are known");
                    assert_ne!(played & trumps_of(t), 0, "a completed hand with no trump");
                    completed += 1;
                }
                _ => {}
            }
        }
        assert!(completed > 0);
    }
}
