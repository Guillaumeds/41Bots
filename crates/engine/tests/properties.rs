//! Property tests over random seeds (docs/STATUS.md task 5) and replay determinism.
//!
//! Each case plays a whole match with uniformly random legal actions and, after every action,
//! checks the invariants against an independent bookkeeper that reads only the event log, plus
//! the hidden truth. Only the public engine API is used.

use engine::cards::{ALL_CARDS, Bits, Card, JOKER, Suit, bit, cards, count, has, suit_of};
use engine::game::{
    Action, BonusKind, Event, GameEndReason, Match, Phase, Rules, Visibility, transcript,
};
use engine::rng::Rng;
use engine::rules::{
    ACES, EffSuit, ace_of_trumps, effective_suit, five_of, is_top_trump, is_trump, plain_mask,
    plain_rank, trump_mask, trump_rank,
};
use proptest::prelude::*;

/// Rebuilds the match from its event log and checks every invariant against the engine.
struct Checker {
    rules: Rules,
    seen: usize,
    checks: u64,
    // Match.
    first_dealer: u8,
    next_hand_index: u32,
    dealer: Option<u8>,
    games_won: [u8; 4],
    game_index: u8,
    games_ended: u32,
    hands_in_game: u32,
    reset_pending: bool,
    match_won: bool,
    // Game.
    scores: [u8; 4],
    bonus_flag: [bool; 4],
    // Hand.
    trump: Option<Suit>,
    dealt_order: Vec<u8>,
    dealt: Bits,
    turned: Vec<Card>,
    takes: u32,
    trick: Vec<(u8, Card)>,
    leader: u8,
    played: Vec<(u8, Card)>,
    tricks_by_team: [u8; 4],
    tricks_done: u8,
    five_played: bool,
    last_trick_had_five: bool,
    score_after_trick: u8,
    game_ended_this_hand: bool,
    prev: Option<Event>,
    // Hard facts a card tracker may deduce from the public events.
    void: [[bool; 4]; 8],
    trump_cap: [u8; 8],
    proved: [bool; 8],
    robber: Option<u8>,
    ace_undealt: bool,
}

impl Checker {
    fn new(rules: Rules) -> Self {
        Checker {
            rules,
            seen: 0,
            checks: 0,
            first_dealer: 0,
            next_hand_index: 0,
            dealer: None,
            games_won: [0; 4],
            game_index: 0,
            games_ended: 0,
            hands_in_game: 0,
            reset_pending: false,
            match_won: false,
            scores: [0; 4],
            bonus_flag: [false; 4],
            trump: None,
            dealt_order: Vec::new(),
            dealt: 0,
            turned: Vec::new(),
            takes: 0,
            trick: Vec::new(),
            leader: 0,
            played: Vec::new(),
            tricks_by_team: [0; 4],
            tricks_done: 0,
            five_played: false,
            last_trick_had_five: false,
            score_after_trick: 0,
            game_ended_this_hand: false,
            prev: None,
            void: [[false; 4]; 8],
            trump_cap: [0; 8],
            proved: [false; 8],
            robber: None,
            ace_undealt: false,
        }
    }

    fn house_rule(&self) -> bool {
        self.rules.bonus_first == 6 && self.rules.bonus_later == 5
    }

    fn check(&mut self, m: &Match) {
        let events = m.events();
        for (vis, e) in &events[self.seen..] {
            assert!(!self.match_won, "an event after MatchWon");
            match *e {
                Event::Dealt { seat, .. } | Event::Burned { seat, .. } => {
                    assert_eq!(*vis, Visibility::Private(seat), "{e:?} must be private")
                }
                _ => assert_eq!(*vis, Visibility::Public, "{e:?} must be public"),
            }
            self.event(e);
            self.prev = Some(e.clone());
        }
        self.seen = events.len();
        self.state(m);
        self.checks += 1;
    }

    fn event(&mut self, e: &Event) {
        match *e {
            Event::MatchStart { first_dealer } => {
                assert!(self.prev.is_none());
                assert!(first_dealer < 8);
                self.first_dealer = first_dealer;
            }
            Event::HandStart {
                hand_index,
                game_index,
                dealer,
            } => {
                assert_eq!(hand_index, self.next_hand_index);
                self.next_hand_index += 1;
                let want = self.dealer.map_or(self.first_dealer, |d| (d + 1) % 8);
                assert_eq!(dealer, want, "the deal moves one seat left every hand (H5)");
                self.dealer = Some(dealer);
                if self.reset_pending {
                    self.reset_pending = false;
                    self.game_index += 1;
                    self.scores = [0; 4];
                    self.bonus_flag = [false; 4];
                    self.hands_in_game = 0;
                }
                assert_eq!(game_index, self.game_index);
                self.hands_in_game += 1;
                assert!(self.hands_in_game <= 7, "a game lasts at most 7 hands");
                self.trump = None;
                self.dealt_order.clear();
                self.dealt = 0;
                self.turned.clear();
                self.takes = 0;
                self.trick.clear();
                self.leader = (dealer + 1) % 8;
                self.played.clear();
                self.tricks_by_team = [0; 4];
                self.tricks_done = 0;
                self.five_played = false;
                self.last_trick_had_five = false;
                self.game_ended_this_hand = false;
                self.void = [[false; 4]; 8];
                self.trump_cap = [0; 8];
                self.proved = [false; 8];
                self.robber = None;
                self.ace_undealt = false;
            }
            Event::Dealt { seat, cards } => {
                let dealer = self.dealer.unwrap();
                let k = self.dealt_order.len() as u8;
                assert_eq!(
                    seat,
                    (dealer + 1 + k) % 8,
                    "deal order starts left of the dealer"
                );
                self.dealt_order.push(seat);
                assert_eq!(count(cards), 5);
                assert_eq!(self.dealt & cards, 0);
                self.dealt |= cards;
            }
            Event::Turned { card } => {
                assert_eq!(self.dealt_order.len(), 8);
                assert!(self.turned.len() < 2);
                if self.turned.len() == 1 {
                    assert_eq!(
                        self.turned[0], JOKER,
                        "a second card is turned only after a Joker"
                    );
                }
                self.turned.push(card);
                if card != JOKER {
                    self.trump = suit_of(card);
                }
            }
            Event::Took { seat, card } => {
                assert_eq!(Some(&card), self.turned.last());
                if card == JOKER {
                    assert_eq!(Some(seat), self.dealer);
                } else {
                    self.takes += 1;
                    assert!(self.robber.is_none());
                    assert!(!self.proved[usize::from(seat)]);
                    self.robber = Some(seat);
                }
            }
            Event::Burned { card, .. } => {
                assert!(!has(ACES, card), "an ace was burned");
            }
            Event::BurnDone { seat } => {
                assert!(matches!(self.prev, Some(Event::Burned { seat: s, .. }) if s == seat));
            }
            Event::Played { seat, card } => {
                let trump = self.trump.expect("a card played before trumps are known");
                match self.trick.last() {
                    None => assert_eq!(seat, self.leader, "the trick winner leads"),
                    Some(&(last, _)) => assert_eq!(seat, (last + 1) % 8, "play goes left"),
                }
                if let Some(&(_, led)) = self.trick.first() {
                    let s = usize::from(seat);
                    match effective_suit(led, trump) {
                        EffSuit::Plain(suit) => {
                            if !is_trump(card, trump)
                                && effective_suit(card, trump) != EffSuit::Plain(suit)
                            {
                                self.void[s][suit as usize] = true;
                            }
                        }
                        EffSuit::Trump => {
                            if !is_trump(card, trump) {
                                self.trump_cap[s] = self.trump_cap[s].max(trump_rank(led, trump));
                            }
                        }
                    }
                }
                if card == five_of(trump) {
                    self.five_played = true;
                }
                self.trick.push((seat, card));
                self.played.push((seat, card));
            }
            Event::ProvedNoAce { seat } => {
                assert_ne!(self.robber, Some(seat));
                assert!(self.tricks_done == 0);
                self.proved[usize::from(seat)] = true;
            }
            Event::AceUndealt => {
                assert_eq!(self.tricks_done, 1);
                let ace = ace_of_trumps(self.trump.unwrap());
                assert!(!has(self.dealt, ace) && !self.turned.contains(&ace));
                assert!(self.robber.is_none());
                self.ace_undealt = true;
            }
            Event::TrickWon {
                trick,
                seat,
                team,
                points,
            } => {
                let trump = self.trump.unwrap();
                assert_eq!(self.trick.len(), 8);
                assert_eq!(trick, self.tricks_done + 1);
                assert_eq!(team, seat % 4);
                assert_eq!(points, 5);
                for (i, &(s, _)) in self.trick.iter().enumerate() {
                    assert_eq!(
                        s % 4,
                        (self.trick[0].0 + i as u8) % 4,
                        "teams cycle A B C D"
                    );
                }
                assert_eq!(seat, naive_winner(&self.trick, trump));
                self.tricks_done += 1;
                if self.tricks_done == 5 {
                    // Rule 31 cannot occur at 8 players: 40 cards in play, at most 13 out, 14-15 trumps.
                    assert!(
                        self.played.iter().any(|&(_, c)| is_trump(c, trump)),
                        "a completed hand with no trump played"
                    );
                }
                self.tricks_by_team[usize::from(team)] += 1;
                self.scores[usize::from(team)] += 5;
                self.score_after_trick = self.scores[usize::from(team)];
                self.last_trick_had_five = self.trick.iter().any(|&(_, c)| c == five_of(trump));
                self.leader = seat;
                self.trick.clear();
            }
            Event::Bonus {
                team,
                points,
                kind,
                card,
            } => {
                let trump = self.trump.unwrap();
                let t = usize::from(team);
                let want = if self.bonus_flag[t] {
                    self.rules.bonus_later
                } else {
                    self.rules.bonus_first
                };
                assert_eq!(points, want);
                match kind {
                    BonusKind::Immediate => {
                        assert!(
                            matches!(self.prev, Some(Event::TrickWon { team: w, .. }) if w == team)
                        );
                        assert!(self.last_trick_had_five);
                        assert_eq!(card, five_of(trump));
                    }
                    BonusKind::EndOfHand => {
                        assert!(matches!(self.prev, Some(Event::TrickWon { trick: 5, .. })));
                        assert!(!self.five_played);
                        assert!(
                            self.tricks_by_team.iter().all(|&n| n < 5),
                            "a jink comes first"
                        );
                        let best = self
                            .played
                            .iter()
                            .filter(|&&(_, c)| is_trump(c, trump))
                            .max_by_key(|&&(_, c)| trump_rank(c, trump))
                            .copied()
                            .expect("an end bonus needs a trump played");
                        assert_eq!(card, best.1);
                        assert_eq!(team, best.0 % 4);
                    }
                }
                self.bonus_flag[t] = true;
                self.scores[t] += points;
            }
            Event::GameWon {
                team,
                reason,
                scores,
            } => {
                let t = usize::from(team);
                assert_eq!(scores, self.scores);
                assert!(!self.game_ended_this_hand, "exactly one game end per game");
                for (i, &s) in scores.iter().enumerate() {
                    if i != t {
                        assert!(s <= 40, "only the winner can be over 40");
                    }
                }
                match reason {
                    GameEndReason::Jink => {
                        assert_eq!(self.tricks_by_team[t], 5);
                        assert!(scores[t] < 41);
                    }
                    GameEndReason::FortyOneByTrick => {
                        assert!(self.score_after_trick >= 41);
                        assert!(self.score_after_trick - 5 < 41);
                    }
                    GameEndReason::FortyOneByImmediateBonus => {
                        assert!(matches!(
                            self.prev,
                            Some(Event::Bonus {
                                kind: BonusKind::Immediate,
                                ..
                            })
                        ));
                        assert!(self.score_after_trick < 41 && scores[t] >= 41);
                    }
                    GameEndReason::FortyOneByEndBonus => {
                        let Some(Event::Bonus {
                            kind: BonusKind::EndOfHand,
                            points,
                            team: bt,
                            ..
                        }) = self.prev
                        else {
                            panic!("an end-bonus win must follow the end bonus");
                        };
                        assert_eq!(bt, team);
                        assert!(scores[t] >= 41 && scores[t] - points < 41);
                    }
                }
                if reason != GameEndReason::Jink {
                    assert!(scores[t] >= 41);
                }
                self.games_won[t] += 1;
                self.games_ended += 1;
                assert_eq!(self.games_ended, u32::from(self.game_index) + 1);
                self.game_ended_this_hand = true;
                self.reset_pending = true;
            }
            Event::HandEnd {
                hand_index,
                aborted,
            } => {
                assert_eq!(hand_index + 1, self.next_hand_index);
                assert_eq!(aborted, self.tricks_done < 5);
                if !self.game_ended_this_hand {
                    assert_eq!(self.tricks_done, 5);
                }
                let ace = ace_of_trumps(self.trump.unwrap());
                let dealt_or_turned = has(self.dealt, ace) || self.turned.contains(&ace);
                assert_eq!(
                    self.takes,
                    u32::from(dealt_or_turned),
                    "the ace of trumps is robbed exactly once when dealt or turned"
                );
            }
            Event::MatchWon { team, money } => {
                let t = usize::from(team);
                assert!(matches!(self.prev, Some(Event::HandEnd { .. })));
                assert_eq!(self.games_won[t], 2);
                assert!(
                    self.games_won
                        .iter()
                        .enumerate()
                        .all(|(i, &g)| i == t || g < 2)
                );
                assert!(self.games_ended <= 5, "a match lasts at most five games");
                assert_eq!(money.iter().sum::<i32>(), 0);
                for (i, &v) in money.iter().enumerate() {
                    let want = if i == t { 6 } else { -2 } * self.rules.stake;
                    assert_eq!(v, want);
                }
                self.match_won = true;
            }
        }
    }

    fn state(&mut self, m: &Match) {
        let truth = m.truth();
        let p = m.public();
        let over = m.phase() == Phase::MatchOver;
        assert_eq!(over, self.match_won);
        assert_eq!(m.games_won(), self.games_won);
        if over {
            return;
        }
        assert!(!m.legal_actions().is_empty(), "no legal action");
        assert_eq!(m.scores(), self.scores);
        assert_eq!(m.bonus_taken(), self.bonus_flag);
        for (t, &s) in self.scores.iter().enumerate() {
            assert!(s <= 40, "a non-terminal score exceeds 40");
            if self.house_rule() {
                assert_eq!(self.bonus_flag[t], s % 5 == 1, "bonus flag vs score mod 5");
            }
        }

        // All 53 cards accounted for, each in exactly one place.
        let mut places: Vec<Bits> = truth.hands.to_vec();
        places.extend_from_slice(&truth.burns);
        places.push(truth.undealt);
        places.push(p.played);
        places.push(p.table_card.map_or(0, bit));
        let mut union = 0;
        for &b in &places {
            assert_eq!(union & b, 0, "a card is in two places");
            union |= b;
        }
        assert_eq!(union, ALL_CARDS, "a card is missing");
        assert_eq!(
            p.played,
            self.played.iter().fold(0, |a, &(_, c)| a | bit(c))
        );

        // Hidden cards: the undealt pack plus the burns.
        let burns: u8 = truth.burns.iter().map(|&b| count(b)).sum();
        assert!(matches!(count(truth.undealt), 11 | 12));
        assert_eq!(p.hidden_count, count(truth.undealt) + burns);
        assert!(
            matches!(p.hidden_count, 12 | 13),
            "hidden cards {}",
            p.hidden_count
        );
        let dealer = m.dealer();
        for (s, &b) in truth.burns.iter().enumerate() {
            assert_eq!(b & ACES, 0, "a burned ace");
            assert!(count(b) <= if s as u8 == dealer { 2 } else { 1 });
        }

        // Hand sizes: 5 at the start of trick 1 after any take and burn; 6 while burning.
        let in_trick: Bits = self.trick.iter().fold(0, |a, &(s, _)| a | bit(s));
        for s in 0..8u8 {
            let mut want = 5 - self.tricks_done - u8::from(has(in_trick, s));
            if m.phase() == Phase::AwaitBurn(s) {
                want += 1;
            }
            assert_eq!(
                count(truth.hands[usize::from(s)]),
                want,
                "seat {s} hand size"
            );
        }
        match m.phase() {
            Phase::AwaitPlay(s) => assert!(m.legal_mask() & !truth.hands[usize::from(s)] == 0),
            Phase::AwaitBurn(s) => {
                assert_eq!(m.legal_mask(), truth.hands[usize::from(s)] & !ACES);
            }
            Phase::MatchOver => unreachable!(),
        }

        // The public state agrees with the event log.
        assert_eq!(p.trick_cards, self.trick);
        assert_eq!(p.tricks_won, self.tricks_by_team);
        assert_eq!(p.proved_no_ace, self.proved);
        assert_eq!(p.robber, self.robber);
        assert_eq!(p.ace_undealt, self.ace_undealt);
        assert_eq!(p.trump, self.trump);
        assert_eq!(
            p.turned.iter().map(|&(c, _)| c).collect::<Vec<_>>(),
            self.turned
        );

        // The hard facts never exclude the true deal.
        if let Some(trump) = self.trump {
            let ace = ace_of_trumps(trump);
            for s in 0..8 {
                let hand = truth.hands[s];
                for suit in Suit::ALL {
                    if self.void[s][suit as usize] {
                        assert_eq!(hand & plain_mask(suit, trump), 0, "void fact broken");
                    }
                }
                if self.trump_cap[s] > 0 {
                    for t in cards(hand & trump_mask(trump)) {
                        assert!(is_top_trump(t, trump) && trump_rank(t, trump) > self.trump_cap[s]);
                    }
                }
                if self.proved[s] {
                    assert!(
                        !has(hand | truth.burns[s], ace),
                        "ProvedNoAce on the ace holder"
                    );
                }
            }
            if let Some(r) = self.robber {
                let played_it = self.played.contains(&(r, ace));
                assert!(
                    has(truth.hands[usize::from(r)], ace) || played_it,
                    "robber fact broken"
                );
            }
            if self.ace_undealt {
                assert!(has(truth.undealt, ace));
            }
        }

        // Seat views: now and then, check the visibility filter and the hidden information.
        if self.checks.is_multiple_of(29) {
            for seat in 0..8u8 {
                let v = m.view(seat);
                assert_eq!(v.public, p);
                assert_eq!(v.hand, truth.hands[usize::from(seat)]);
                assert_eq!(v.own_burns, truth.burns[usize::from(seat)]);
                let want: Vec<Event> = m
                    .events()
                    .iter()
                    .filter(|(vis, _)| vis.visible_to(seat))
                    .map(|(_, e)| e.clone())
                    .collect();
                assert_eq!(v.events, want);
                for e in &v.events {
                    if let Event::Dealt { seat: s, .. } | Event::Burned { seat: s, .. } = *e {
                        assert_eq!(s, seat, "a seat saw another seat's hidden cards");
                    }
                }
            }
        }
    }
}

/// The winner of a complete trick, written independently of `rules::trick_winner`.
fn naive_winner(trick: &[(u8, Card)], trump: Suit) -> u8 {
    if let Some(&(s, _)) = trick
        .iter()
        .filter(|&&(_, c)| is_trump(c, trump))
        .max_by_key(|&&(_, c)| trump_rank(c, trump))
    {
        return s;
    }
    let led = effective_suit(trick[0].1, trump);
    trick
        .iter()
        .filter(|&&(_, c)| effective_suit(c, trump) == led)
        .max_by_key(|&&(_, c)| plain_rank(c))
        .map(|&(s, _)| s)
        .unwrap()
}

/// Plays a whole match with random legal actions, checking everything after every action, and
/// probing an illegal action at every step.
fn play_checked(seed: u64, rules: Rules) -> Match {
    let mut m = Match::new(seed, rules);
    let mut rng = Rng::derive(seed, &[0x5EED]);
    let mut ck = Checker::new(rules);
    ck.check(&m);
    let mut steps = 0u32;
    while m.phase() != Phase::MatchOver {
        let legal = m.legal_actions();
        let probe = rng.bounded(64) as Card;
        for a in [Action::Burn(probe), Action::Play(probe)] {
            if !legal.contains(&a) {
                let (n_events, n_actions, phase) = (m.events().len(), m.actions().len(), m.phase());
                assert!(m.apply(a).is_err(), "an illegal action was accepted: {a:?}");
                assert_eq!(
                    (m.events().len(), m.actions().len(), m.phase()),
                    (n_events, n_actions, phase)
                );
            }
        }
        let a = legal[rng.bounded(legal.len() as u64) as usize];
        m.apply(a).expect("an offered action was refused");
        ck.check(&m);
        steps += 1;
        assert!(steps < 20_000, "the match never ended");
    }
    assert_eq!(ck.seen, m.events().len());
    assert!(ck.match_won);
    m
}

fn canonical_log(m: &Match) -> String {
    let mut s = String::new();
    for (vis, e) in m.events() {
        s.push_str(&format!("{vis:?} {}\n", e.canonical()));
    }
    s
}

fn assert_replays(m: &Match) {
    let r = Match::replay(m.seed(), m.rules(), m.actions()).expect("replay refused an action");
    assert_eq!(canonical_log(&r).as_bytes(), canonical_log(m).as_bytes());
    assert_eq!(r.events(), m.events());
    assert_eq!(&r, m);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn invariants_hold_house_rules(seed in any::<u64>()) {
        let m = play_checked(seed, Rules::HOUSE);
        assert_replays(&m);
    }

    #[test]
    fn invariants_hold_flat_rule(seed in any::<u64>(), stake in 1i32..100) {
        let m = play_checked(seed, Rules { stake, ..Rules::FLAT });
        assert_replays(&m);
    }

    #[test]
    fn replay_of_a_prefix_gives_the_same_state(seed in any::<u64>(), cut in 0usize..400) {
        let m = play_checked(seed, Rules::HOUSE);
        let cut = cut.min(m.actions().len());
        let mut a = Match::replay(seed, Rules::HOUSE, &m.actions()[..cut]).unwrap();
        let b = Match::replay(seed, Rules::HOUSE, &m.actions()[..cut]).unwrap();
        prop_assert_eq!(&a, &b);
        for &act in &m.actions()[cut..] {
            a.apply(act).unwrap();
        }
        prop_assert_eq!(&a, &m);
    }
}

#[test]
fn random_matches_replay_byte_identically() {
    for seed in 0..40u64 {
        let rules = if seed % 2 == 0 {
            Rules::HOUSE
        } else {
            Rules::FLAT
        };
        let m = play_checked(seed, rules);
        assert_replays(&m);
        assert_eq!(
            transcript(&m).lines().count(),
            m.events()
                .iter()
                .filter(|(v, _)| *v == Visibility::Public)
                .count()
        );
    }
}

#[test]
fn replay_rejects_an_illegal_action() {
    let m = Match::new(3, Rules::HOUSE);
    let legal = m.legal_actions();
    let bad = (0..53u8)
        .map(Action::Play)
        .chain((0..53u8).map(Action::Burn))
        .find(|a| !legal.contains(a))
        .unwrap();
    let err = Match::replay(3, Rules::HOUSE, &[bad]).unwrap_err();
    assert_eq!(err.action, bad);
}

/// Counts of rule branches over many random matches; the common ones must all occur.
#[test]
fn rule_branches_are_reached() {
    let mut joker = 0;
    let mut joker_then_ace = 0;
    let mut joker_dealer_holds_ace = 0;
    let mut turned_ace = 0;
    let mut dealer_robbed_at_once = 0;
    let mut non_dealer_robbed = 0;
    let mut nobody_robbed = 0;
    let mut jink = 0;
    let mut mid_hand = 0;
    let mut immediate = 0;
    let mut end_bonus = 0;
    let mut by_immediate = 0;
    let mut by_end = 0;
    let mut five_games = 0;
    for seed in 1000..1300u64 {
        let mut m = Match::new(seed, Rules::HOUSE);
        let mut rng = Rng::derive(seed, &[7]);
        while m.phase() != Phase::MatchOver {
            let legal = m.legal_actions();
            m.apply(legal[rng.bounded(legal.len() as u64) as usize])
                .unwrap();
        }
        let mut dealer = 0;
        let mut turned: Vec<Card> = Vec::new();
        for (_, e) in m.events() {
            match *e {
                Event::HandStart { dealer: d, .. } => {
                    dealer = d;
                    turned.clear();
                }
                Event::Turned { card } => {
                    turned.push(card);
                    if card == JOKER {
                        joker += 1;
                    } else if engine::cards::rank_of(card) == engine::cards::ACE {
                        turned_ace += 1;
                        if turned.len() == 2 {
                            joker_then_ace += 1;
                        }
                    }
                }
                Event::Took { seat, card } if card != JOKER => {
                    if seat != dealer {
                        non_dealer_robbed += 1;
                    } else if turned.len() == 2
                        && engine::cards::rank_of(card) != engine::cards::ACE
                    {
                        joker_dealer_holds_ace += 1;
                        dealer_robbed_at_once += 1;
                    } else if engine::cards::rank_of(card) == engine::cards::ACE {
                        dealer_robbed_at_once += 1;
                    }
                }
                Event::AceUndealt => nobody_robbed += 1,
                Event::Bonus {
                    kind: BonusKind::Immediate,
                    ..
                } => immediate += 1,
                Event::Bonus {
                    kind: BonusKind::EndOfHand,
                    ..
                } => end_bonus += 1,
                Event::GameWon { reason, .. } => match reason {
                    GameEndReason::Jink => jink += 1,
                    GameEndReason::FortyOneByImmediateBonus => by_immediate += 1,
                    GameEndReason::FortyOneByEndBonus => by_end += 1,
                    GameEndReason::FortyOneByTrick => {}
                },
                Event::HandEnd { aborted: true, .. } => mid_hand += 1,
                _ => {}
            }
        }
        if m.games_won().iter().sum::<u8>() == 5 {
            five_games += 1;
        }
    }
    let counts = [
        ("turned Joker", joker),
        ("Joker then ace", joker_then_ace),
        ("Joker, dealer holds the ace", joker_dealer_holds_ace),
        ("turned ace", turned_ace),
        ("dealer robbed at once", dealer_robbed_at_once),
        ("non-dealer robbed", non_dealer_robbed),
        ("nobody robbed", nobody_robbed),
        ("jink", jink),
        ("game ended mid-hand", mid_hand),
        ("immediate bonus", immediate),
        ("end bonus", end_bonus),
        ("41 by immediate bonus", by_immediate),
        ("41 by end bonus", by_end),
        ("five-game match", five_games),
    ];
    println!("{counts:#?}");
    for (what, n) in [
        ("turned Joker", joker),
        ("turned ace", turned_ace),
        ("dealer robbed at once", dealer_robbed_at_once),
        ("non-dealer robbed", non_dealer_robbed),
        ("nobody robbed", nobody_robbed),
        ("game ended mid-hand", mid_hand),
        ("immediate bonus", immediate),
        ("end bonus", end_bonus),
        ("41 by end bonus", by_end),
    ] {
        assert!(n > 0, "branch never reached: {what}");
    }
}

/// A longer run of the full checker, for manual use:
/// `ENGINE_STRESS_MATCHES=20000 cargo test --release -p engine --test properties -- --ignored`.
#[test]
#[ignore = "long; run by hand"]
fn stress_many_matches() {
    let n: u64 = std::env::var("ENGINE_STRESS_MATCHES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let (mut actions, mut hands) = (0usize, 0u64);
    for seed in 0..n {
        let rules = if seed % 3 == 0 {
            Rules::FLAT
        } else {
            Rules::HOUSE
        };
        let m = play_checked(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15), rules);
        assert_replays(&m);
        actions += m.actions().len();
        hands += u64::from(m.hand_index()) + 1;
    }
    println!("{n} matches, {hands} hands, {actions} actions: all invariants held");
}
