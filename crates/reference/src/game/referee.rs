//! Turned partnerships and the app's referee (rules 32-34 and 37-41; the owner's rulings of
//! 2026-10-17 in docs/APP-ENGINE-CONTRACT.md §7), written after the app's own code
//! (fivefingers `src/engine/game.ts`: `penalise`, `credit`, `catchHiddenAces`, `resolveTrick`,
//! `advance`, `finishHand`), separately from the engine.
//!
//! The app may turn a partnership (`turn_team`), turn one right after a card
//! (`play_then_turn`), or let a renege through (`let_through`), which is caught later: by a card
//! that proves it, by the ace of trumps showing from another hand, when the pair would claim the
//! game, or at the end of the hand. A turned partnership scores nothing for the hand; each trick
//! it holds passes, one at a time in trick order, to the partnership still in play whose card
//! beats the others' against the card led (the first of their cards when none does), and the
//! first partnership to reach 41 that way wins the game. The last partnership in play plays the
//! hand out alone. A partnership turned while the trick it led is being played is refused: the
//! owner has asked the pros.

use super::*;

/// What a renege let through was.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RenegeKind {
    /// Held this suit (led, plain) and played another plain card.
    Follow(Suit),
    /// This trump was led and the player, holding a trump it could not hold back, played a plain
    /// card.
    TrumpLead(Card),
    /// Held the ace of trumps and played without robbing.
    NoRob,
    /// Took the turned card without the ace of trumps.
    FalseRob,
    /// Robbed and burned the ace of trumps.
    BurnedAce,
}

impl RenegeKind {
    fn word(self) -> String {
        match self {
            RenegeKind::Follow(s) => format!("follow:{}", s.letter()),
            RenegeKind::TrumpLead(c) => format!("trump_lead:{}", c.name()),
            RenegeKind::NoRob => "no_rob".to_string(),
            RenegeKind::FalseRob => "false_rob".to_string(),
            RenegeKind::BurnedAce => "burned_ace".to_string(),
        }
    }

    /// Only the table's own cards can give these away (the ace elsewhere, a claim, the end).
    fn hidden_ace(self) -> bool {
        matches!(self, RenegeKind::FalseRob | RenegeKind::BurnedAce)
    }

    /// Does this card, played later by the same player, prove the renege?
    fn proved_by(self, card: Card, trump: Suit) -> bool {
        match self {
            RenegeKind::Follow(s) => order::plain_suit(card, trump) == Some(s),
            RenegeKind::TrumpLead(led) => {
                order::is_trump(card, trump)
                    && !(order::top_four(trump).contains(&card)
                        && order::trump_beats(card, led, trump))
            }
            RenegeKind::NoRob => card == order::ace_of_trumps(trump),
            RenegeKind::FalseRob | RenegeKind::BurnedAce => false,
        }
    }
}

/// A renege let through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Renege {
    seat: u8,
    kind: RenegeKind,
    pub(super) caught: bool,
}

/// What the app lets through (card indices as in [`Action`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lapse {
    /// The player to act plays this card although it is not legal.
    Play(u8),
    /// The robber waiting to burn plays this card without robbing.
    PlayWithoutRob(u8),
    /// The player to act in trick 1, without the ace, takes the turned card and burns this one
    /// (any card it then holds but the ace of trumps, H3).
    FalseRob(u8),
    /// The robber burns the ace of trumps.
    BurnAce,
}

/// The first card among the players still in play, replaced by any later one of theirs that
/// beats it against the card led (the app's `standingWinnerIndex` and `beats`): a position in
/// `cards`, or `None` when none of them played in it.
pub(super) fn standing_winner(cards: &[(u8, Card)], trump: Suit, out: &[bool; 4]) -> Option<usize> {
    let led = cards.first()?.1;
    let mut best: Option<usize> = None;
    for (i, &(seat, card)) in cards.iter().enumerate() {
        if out[team_of(seat)] {
            continue;
        }
        match best {
            None => best = Some(i),
            Some(b) => {
                if beats(card, cards[b].1, led, trump) {
                    best = Some(i);
                }
            }
        }
    }
    best
}

/// Does `card` beat `best` in a trick led with `led`?
fn beats(card: Card, best: Card, led: Card, trump: Suit) -> bool {
    let (ct, bt) = (order::is_trump(card, trump), order::is_trump(best, trump));
    if ct || bt {
        return ct && (!bt || order::trump_beats(card, best, trump));
    }
    let led_suit = order::plain_suit(led, trump);
    if order::plain_suit(card, trump) != led_suit {
        return false;
    }
    if order::plain_suit(best, trump) != led_suit {
        return true;
    }
    let ranking = order::plain_order(card.suit().expect("a plain card"), trump);
    let pos = |c: Card| ranking.iter().position(|o| *o == c);
    pos(card) < pos(best)
}

impl Match {
    /// The partnerships turned this hand.
    pub fn turned_teams(&self) -> [bool; 4] {
        self.teams_out
    }

    /// The reneges let through this hand: seat, kind word and whether caught.
    pub fn reneges(&self) -> Vec<(u8, String, bool)> {
        self.reneges
            .iter()
            .map(|r| (r.seat, r.kind.word(), r.caught))
            .collect()
    }

    /// Turns partnership `team` now: before the play the match waits for, or (its own) while
    /// its rob is due, which undoes the take (`no_rob`, the dealer's `declined_rob` of a turned
    /// ace, `left_joker`). Refused (nothing changes) with a message starting with a code word:
    /// `no_such_team`, `not_before_a_play`, `already_turned`, `led_trick_in_progress`.
    pub fn turn_team(&mut self, team: u8) -> Result<(), String> {
        if team >= 4 {
            return Err("no_such_team: teams are 0 to 3".to_string());
        }
        let team = team as usize;
        let allowed = match self.waiting {
            Waiting::Play(_) => true,
            Waiting::RobberBurn(seat) => team_of(seat) == team,
            Waiting::DealerRobBurn | Waiting::DealerJokerBurn => team_of(self.dealer) == team,
            Waiting::Deck | Waiting::MatchOver => false,
        };
        if !allowed {
            return Err("not_before_a_play: no play or own rob is awaited".to_string());
        }
        if self.teams_out[team] {
            return Err("already_turned: that team is already turned".to_string());
        }
        let mut m = self.clone();
        match m.waiting {
            Waiting::Play(seat) => {
                if !m.penalise(team, "before_play") {
                    m.move_on(seat)?;
                }
            }
            Waiting::RobberBurn(seat) => {
                m.untake(seat);
                if !m.penalise(team, "no_rob") {
                    // The robber does not play: the next player in play does.
                    m.move_on(seat)?;
                }
            }
            Waiting::DealerRobBurn => {
                m.untake(m.dealer);
                if !m.penalise(team, "declined_rob") {
                    m.start_trick_one();
                }
            }
            Waiting::DealerJokerBurn => {
                m.untake(m.dealer);
                if !m.penalise(team, "left_joker") {
                    m.after_joker_burn();
                }
            }
            Waiting::Deck | Waiting::MatchOver => unreachable!("checked above"),
        }
        *self = m;
        Ok(())
    }

    /// The player to act plays `card` (a legal card), then partnership `team` is turned before
    /// the trick goes on or is won. Refused like `turn_team`, and with `illegal_card`.
    pub fn play_then_turn(&mut self, card: u8, team: u8) -> Result<(), String> {
        if team >= 4 {
            return Err("no_such_team: teams are 0 to 3".to_string());
        }
        let team = team as usize;
        let Waiting::Play(seat) = self.waiting else {
            return Err("not_before_a_play: no play is awaited".to_string());
        };
        if self.teams_out[team] {
            return Err("already_turned: that team is already turned".to_string());
        }
        if !self.legal_actions().contains(&Action::Play(card)) {
            return Err("illegal_card: not a legal play of the player to act".to_string());
        }
        let card = Card::from_index(card).expect("a legal card");
        let mut m = self.clone();
        m.put_down(seat, card);
        let over = (!m.reneges.is_empty() && m.check_reneges(seat, card))
            || (!m.teams_out[team] && m.penalise(team, "with_play"));
        if !over {
            m.move_on(left_of(seat))?;
        }
        *self = m;
        Ok(())
    }

    /// Something the app's referee let through. Refused (nothing changes) with a message
    /// starting with a code word: `not_now`, `not_in_hand`, `legal`, `led_trick_in_progress`.
    pub fn let_through(&mut self, lapse: Lapse) -> Result<(), String> {
        let Some(trump) = self.trump else {
            return Err("not_now: trumps are not known yet".to_string());
        };
        let in_hand = |m: &Match, seat: u8, c: u8| {
            Card::from_index(c).filter(|card| m.hands[seat as usize].contains(card))
        };
        let mut m = self.clone();
        match (lapse, self.waiting) {
            (Lapse::Play(c), Waiting::Play(seat)) => {
                let Some(card) = in_hand(self, seat, c) else {
                    return Err("not_in_hand: the player does not hold that card".to_string());
                };
                if self.legal_actions().contains(&Action::Play(c)) {
                    return Err("legal: that card may be played".to_string());
                }
                m.note_illegal(seat, trump);
                m.play(seat, card)?;
            }
            (Lapse::PlayWithoutRob(c), Waiting::RobberBurn(seat)) => {
                let taken = self.turned.last().map(|(card, _)| *card);
                let Some(card) = in_hand(self, seat, c).filter(|card| Some(*card) != taken) else {
                    return Err("not_in_hand: the player does not hold that card".to_string());
                };
                m.untake(seat);
                m.reneges.push(Renege {
                    seat,
                    kind: RenegeKind::NoRob,
                    caught: false,
                });
                if !m.legal_actions().contains(&Action::Play(c)) {
                    m.note_illegal(seat, trump);
                }
                m.play(seat, card)?;
            }
            (Lapse::FalseRob(b), Waiting::Play(seat)) => {
                if self.table_card.is_none()
                    || self.trick_no != 1
                    || self.robber.is_some()
                    || self.hands[seat as usize].contains(&order::ace_of_trumps(trump))
                {
                    return Err("not_now: no turned card to take".to_string());
                }
                // H3: any card but the ace of trumps, the card just taken included.
                let Some(burn) = in_hand(self, seat, b)
                    .or(self.table_card.filter(|c| c.index() == b))
                    .filter(|c| *c != order::ace_of_trumps(trump))
                else {
                    return Err("not_in_hand: the player does not hold that card".to_string());
                };
                m.reneges.push(Renege {
                    seat,
                    kind: RenegeKind::FalseRob,
                    caught: false,
                });
                m.take_table_card(seat, true);
                m.burn_log.push((seat, "rob_at_turn"));
                m.burn(seat, burn);
                m.waiting = Waiting::Play(seat);
            }
            (Lapse::BurnAce, Waiting::RobberBurn(seat))
                if self.hands[seat as usize].contains(&order::ace_of_trumps(trump)) =>
            {
                m.reneges.push(Renege {
                    seat,
                    kind: RenegeKind::BurnedAce,
                    caught: false,
                });
                m.burn_log.push((seat, "rob_at_turn"));
                m.burn(seat, order::ace_of_trumps(trump));
                m.waiting = Waiting::Play(seat);
            }
            _ => return Err("not_now: that cannot happen now".to_string()),
        }
        *self = m;
        Ok(())
    }

    /// The illegal card's renege, as the app files it: to a trump lead, a trump held back; to a
    /// plain lead, the suit held.
    fn note_illegal(&mut self, seat: u8, trump: Suit) {
        let led = self.trick.first().expect("a lead is always legal").1;
        let kind = match order::plain_suit(led, trump) {
            None => RenegeKind::TrumpLead(led),
            Some(s) => RenegeKind::Follow(s),
        };
        self.reneges.push(Renege {
            seat,
            kind,
            caught: false,
        });
    }

    /// Take back the turned card the engine-like flow gave `seat` (its rob or the dealer's
    /// take): it was never taken at the app's table. The `Took` event is the last one.
    fn untake(&mut self, seat: u8) {
        let (_, text) = self.events.pop().expect("the take was the last event");
        assert!(text.starts_with(&format!("Took seat={seat} ")), "{text}");
        let last = self.turned.len() - 1;
        let card = self.turned[last].0;
        let hand = &mut self.hands[seat as usize];
        let pos = hand
            .iter()
            .position(|c| *c == card)
            .expect("the taken card");
        hand.remove(pos);
        self.turned[last].1 = None;
        self.table_card = Some(card);
        self.took_a_card[seat as usize] = false;
        if card != Card::Joker {
            self.robber = None;
        }
        self.waiting = Waiting::Play(seat);
    }

    /// After `seat` played `card` with reneges about: a renege of its own the card proves turns
    /// its partnership (all its reneges then count as caught); the ace of trumps gives away every
    /// other player who took the turned card without it. True when the game ended.
    pub(super) fn check_reneges(&mut self, seat: u8, card: Card) -> bool {
        let trump = self.trump.expect("trumps are known during play");
        let proved = self
            .reneges
            .iter()
            .any(|r| r.seat == seat && !r.caught && r.kind.proved_by(card, trump));
        if proved {
            for r in self.reneges.iter_mut() {
                if r.seat == seat {
                    r.caught = true;
                }
            }
            let team = team_of(seat);
            if !self.teams_out[team] && self.penalise(team, "proved") {
                return true;
            }
        }
        if card == order::ace_of_trumps(trump) {
            for i in 0..self.reneges.len() {
                let r = self.reneges[i];
                if r.kind != RenegeKind::FalseRob || r.seat == seat || r.caught {
                    continue;
                }
                self.reneges[i].caught = true;
                let team = team_of(r.seat);
                if !self.teams_out[team] && self.penalise(team, "ace_shown") {
                    return true;
                }
            }
        }
        false
    }

    /// Would `points` more take `team` to 41 with a rob without the ace or a burned ace of its
    /// still hidden? (Rule 38: it is turned instead.)
    pub(super) fn claim_caught(&self, team: usize, points: u8) -> bool {
        self.scores[team] + points >= 41
            && self
                .reneges
                .iter()
                .any(|r| !r.caught && r.kind.hidden_ace() && team_of(r.seat) == team)
    }

    /// Catch the hidden robs without the ace and burned aces (of `only`'s team, or everyone's),
    /// in order. True when the game ended.
    pub(super) fn catch_aces(&mut self, only: Option<usize>, cause: &str) -> bool {
        for i in 0..self.reneges.len() {
            let r = self.reneges[i];
            if r.caught || !r.kind.hidden_ace() {
                continue;
            }
            let team = team_of(r.seat);
            if only.is_some_and(|o| o != team) {
                continue;
            }
            self.reneges[i].caught = true;
            if !self.teams_out[team] && self.penalise(team, cause) {
                return true;
            }
        }
        false
    }

    /// Rules 39-41: `team` is turned. It goes back to its score and bonus flag of the start of
    /// the hand, and each trick it holds passes, in trick order, credited at once (rule 32: the
    /// first team to 41 wins). True when the game ended.
    pub(super) fn penalise(&mut self, team: usize, cause: &str) -> bool {
        self.teams_out[team] = true;
        self.scores[team] = self.start_scores[team];
        self.bonus_taken[team] = self.start_bonus[team];
        self.tricks_won[team] = 0;
        self.emit_public(format!(
            "PairTurned team={} score={} cause={}",
            team, self.scores[team], cause
        ));
        let Some(trump) = self.trump else {
            return false;
        };
        for i in 0..self.finished_tricks.len() {
            if self.holders[i] != Some(team) {
                continue;
            }
            let taker = standing_winner(&self.finished_tricks[i].0, trump, &self.teams_out)
                .map(|at| team_of(self.finished_tricks[i].0[at].0));
            self.holders[i] = taker;
            let Some(to) = taker else { continue };
            self.tricks_won[to] += 1;
            if self.claim_caught(to, 5) {
                self.emit_public(format!("TrickPassed trick={} team={} points=0", i + 1, to));
                if self.catch_aces(Some(to), "claim") {
                    return true;
                }
                continue;
            }
            self.scores[to] += 5;
            self.emit_public(format!("TrickPassed trick={} team={} points=5", i + 1, to));
            if self.scores[to] >= 41 {
                self.end_game(to, GameEnd::PassedTrick);
                return true;
            }
        }
        // Rule 39: its cards no longer count for the best trump or the 5.
        let five = order::five_of_trumps(trump);
        let mut best: Option<(Card, u8)> = None;
        let mut five_played = false;
        let everything = self
            .finished_tricks
            .iter()
            .flat_map(|(cards, _)| cards.iter())
            .chain(self.trick.iter());
        for &(seat, card) in everything {
            if self.teams_out[team_of(seat)] {
                continue;
            }
            if card == five {
                five_played = true;
            }
            if order::is_trump(card, trump)
                && best.is_none_or(|(b, _)| order::trump_beats(card, b, trump))
            {
                best = Some((card, seat));
            }
        }
        self.best_trump = best;
        self.five_played = five_played;
        false
    }

    /// The end of a hand with a turned partnership or a renege let through, in the app's order:
    /// the hidden aces are caught, then the end bonus, then the jink; every credit can still be
    /// caught at the claim.
    pub(super) fn end_of_refereed_hand(&mut self) {
        if self.catch_aces(None, "hand_end") {
            return;
        }
        if let Some((card, team)) = self.end_bonus() {
            let points = if self.bonus_taken[team] {
                LATER_BONUS
            } else {
                FIRST_BONUS
            };
            if self.claim_caught(team, points) {
                if self.catch_aces(Some(team), "claim") {
                    return;
                }
            } else {
                self.award_bonus(team, card, "end");
                if self.scores[team] >= 41 {
                    self.end_game(team, GameEnd::EndBonus);
                    return;
                }
            }
        }
        if self.finished_tricks.len() == 5
            && let Some(j) = self.holders[0]
            && !self.teams_out[j]
            && self.holders.iter().all(|h| *h == Some(j))
        {
            // Rules 33-34: a jink is worth what the team still needs, at least 10.
            let need = 41u8.saturating_sub(self.scores[j]).max(10);
            if self.claim_caught(j, need) {
                if self.catch_aces(Some(j), "claim") {
                    return;
                }
            } else {
                self.end_game(j, GameEnd::Jink);
                return;
            }
        }
        self.close_hand();
    }
}
