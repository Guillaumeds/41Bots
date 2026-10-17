//! Positions for the API's checks and measurements, found by playing seeded random matches
//! (every action drawn from the legal ones), optionally with the app's referee steps (a pair
//! turned at its rob, a refused ace or a left Joker; lapses let through), so the same positions
//! come out on every machine.

use engine::cards::{cards, has};
use engine::game::{Lapse, Match, Phase, Rules};
use engine::rng::Rng;
use engine::rules::ace_of_trumps;
use engine::team_of;

use crate::appjson::{Mask, app_request};

/// One referee step the app could make now, at random (about one chance in `rate` at each
/// opportunity); refused steps are skipped. Returns whether one was made.
pub fn referee_step(m: &mut Match, rng: &mut Rng, rate: u64) -> bool {
    let p = m.public();
    let Some(seat) = m.to_act() else { return false };
    let pick = |rng: &mut Rng| rng.bounded(rate) == 0;
    match p.phase {
        // The dealer refuses a turned ace or leaves a turned Joker; the robber does not rob
        // (Guide) or plays without robbing, or burns the ace (Real Life).
        Phase::AwaitBurn(s) if p.trick_no == 0 => pick(rng) && m.turn_team(team_of(s)).is_ok(),
        Phase::AwaitBurn(s) => {
            if !pick(rng) {
                return false;
            }
            match rng.bounded(3) {
                0 => m.turn_team(team_of(s)).is_ok(),
                1 => m.let_through(Lapse::BurnAce).is_ok(),
                _ => {
                    let taken = p.turned.last().map(|&(c, _)| c);
                    let hand: Vec<_> = cards(m.truth().hands[usize::from(s)])
                        .filter(|&c| Some(c) != taken)
                        .collect();
                    let c = hand[rng.bounded(hand.len() as u64) as usize];
                    m.let_through(Lapse::PlayWithoutRob(c)).is_ok()
                }
            }
        }
        Phase::AwaitPlay(s) => {
            let hand = m.truth().hands[usize::from(s)];
            let legal = m.legal_mask();
            let illegal: Vec<_> = cards(hand & !legal).collect();
            if !illegal.is_empty() && pick(rng) {
                let c = illegal[rng.bounded(illegal.len() as u64) as usize];
                return m.let_through(Lapse::Play(c)).is_ok();
            }
            // A false rob: the seat takes the turned card without the ace of trumps.
            if p.trick_no == 1
                && p.table_card.is_some()
                && p.robber.is_none()
                && p.trump.is_some_and(|t| !has(hand, ace_of_trumps(t)))
                && pick(rng)
            {
                let ace = ace_of_trumps(p.trump.expect("checked"));
                let burnable: Vec<_> = cards(hand).filter(|&c| c != ace).collect();
                let c = burnable[rng.bounded(burnable.len() as u64) as usize];
                return m.let_through(Lapse::FalseRob(c)).is_ok();
            }
            let _ = seat;
            false
        }
        Phase::MatchOver => false,
    }
}

/// The legal actions the app can make: every legal action but a burn of the card just taken
/// (the app's rob and steal burn from the hand before the take; contract §7.1, H3).
pub fn app_actions(m: &Match) -> Vec<engine::game::Action> {
    let p = m.public();
    let taken = match p.phase {
        Phase::AwaitBurn(s) => p
            .turned
            .last()
            .filter(|&&(_, by)| by == Some(s))
            .map(|&(c, _)| c),
        _ => None,
    };
    m.legal_actions()
        .into_iter()
        .filter(|a| !matches!(a, engine::game::Action::Burn(c) if Some(*c) == taken))
        .collect()
}

/// The first position satisfying `want` met while playing random matches from `seed` (match `i`
/// is `Match::new(seed + i)`, its actions and referee steps drawn from `Rng::derive(seed, [i])`;
/// referee steps about one chance in `referee` at each opportunity, none for 0).
///
/// # Panics
/// If no position is found in 20,000 matches.
pub fn find(seed: u64, referee: u64, want: &dyn Fn(&Match) -> bool) -> Match {
    for i in 0..20_000u64 {
        let mut m = Match::new(seed.wrapping_add(i), Rules::HOUSE);
        let mut rng = Rng::derive(seed, &[i]);
        while m.to_act().is_some() {
            if want(&m) {
                return m;
            }
            if referee > 0 && referee_step(&mut m, &mut rng, referee) {
                continue;
            }
            // A legal card may be refused after a lapse (the open rules question, contract
            // §7.1): try the others; a seat all of whose cards are refused ends the match.
            let mut legal = app_actions(&m);
            let mut played = false;
            while !legal.is_empty() {
                let a = legal.remove(rng.bounded(legal.len() as u64) as usize);
                if m.apply(a).is_ok() {
                    played = true;
                    break;
                }
            }
            if !played {
                break;
            }
        }
    }
    panic!("no position found from seed {seed}");
}

/// Does the seat to act decide something (more than one legal action), awaiting a play?
pub fn playing(m: &Match, trick: u8) -> bool {
    let p = m.public();
    matches!(p.phase, Phase::AwaitPlay(_)) && p.trick_no == trick && m.legal_actions().len() > 1
}

/// Is any partnership turned?
pub fn any_turned(m: &Match) -> bool {
    m.turned_teams().iter().any(|&t| t)
}

/// The check's requests (`tools/wasm_check.sh`), by name: one per kind of decision the API
/// meets, with few deals so the check stays short.
pub fn requests() -> Vec<(String, String)> {
    let plain = |m: &Match| !any_turned(m) && m.reneges().is_empty();
    let mut out: Vec<(String, Match, u32)> = vec![
        (
            "t0-steal".into(),
            find(11, 0, &|m| {
                let p = m.public();
                matches!(p.phase, Phase::AwaitBurn(_)) && p.trump.is_none()
            }),
            4,
        ),
        (
            "t1-rob".into(),
            find(12, 0, &|m| {
                let p = m.public();
                matches!(p.phase, Phase::AwaitBurn(_)) && p.trick_no == 1
            }),
            4,
        ),
        (
            "t1-play".into(),
            find(13, 0, &|m| plain(m) && playing(m, 1)),
            4,
        ),
        (
            "t2-hybrid".into(),
            find(14, 0, &|m| plain(m) && playing(m, 2)),
            4,
        ),
        (
            "t2-hybrid-b".into(),
            find(15, 0, &|m| plain(m) && playing(m, 2)),
            3,
        ),
        (
            "t3-exact".into(),
            find(16, 0, &|m| plain(m) && playing(m, 3)),
            16,
        ),
        (
            "t4-exact".into(),
            find(17, 0, &|m| plain(m) && playing(m, 4)),
            32,
        ),
        (
            "t2-forced".into(),
            find(18, 0, &|m| {
                let p = m.public();
                matches!(p.phase, Phase::AwaitPlay(_))
                    && p.trick_no == 2
                    && m.legal_actions().len() == 1
            }),
            8,
        ),
        (
            "t5-forced".into(),
            find(19, 0, &|m| m.public().trick_no == 5),
            8,
        ),
        (
            "t2-turned".into(),
            find(20, 4, &|m| {
                any_turned(m) && playing(m, 2) && m.reneges().iter().all(|r| r.caught)
            }),
            4,
        ),
        (
            "t3-turned".into(),
            find(21, 4, &|m| {
                any_turned(m) && playing(m, 3) && m.reneges().iter().all(|r| r.caught)
            }),
            16,
        ),
    ];
    // The coach's fields (`lines`) on three of them.
    let with_lines: Vec<(String, Match, u32)> = out
        .iter()
        .filter(|(n, _, _)| ["t2-hybrid", "t3-exact", "t3-turned"].contains(&n.as_str()))
        .map(|(n, m, d)| (format!("{n}-lines"), m.clone(), (*d).min(8)))
        .collect();
    out.extend(with_lines);
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out.into_iter()
        .map(|(name, m, deals)| {
            let mut req = app_request(&m, deals, 20_261_017, Mask::SeatOwnBurns);
            if name.ends_with("-lines") {
                req["lines"] = serde_json::Value::Bool(true);
            }
            (name, req.to_string())
        })
        .collect()
}

/// The speed measurement's requests: at tricks 2, 3 and 4 (no pair turned), `per_place`
/// positions at each place in the trick (the leader, the second to play, … the eighth), with
/// `deals` deals at trick 2 and four times as many at tricks 3 and 4.
pub fn bench_requests(deals: u32, per_place: u64) -> Vec<(String, String)> {
    let plain = |m: &Match| !any_turned(m) && m.reneges().is_empty();
    let mut out = Vec::new();
    for (trick, n) in [(2u8, deals), (3, 4 * deals), (4, 4 * deals)] {
        for place in 0..8usize {
            for k in 0..per_place {
                let seed = 100_000 * u64::from(trick) + 1000 * place as u64 + k;
                let m = find(seed, 0, &|m| {
                    plain(m) && playing(m, trick) && m.public().trick_cards.len() == place
                });
                let req = app_request(&m, n, seed, Mask::SeatOwnBurns);
                out.push((
                    format!("bench-t{trick}-p{}-{k}", place + 1),
                    req.to_string(),
                ));
            }
        }
    }
    out
}
