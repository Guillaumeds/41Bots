//! The match state between hands, and the table's canonical index (docs/M1-SPEC.md §6.2).
//!
//! A team's score is always one of 15 values: `5k` (0 to 40) or `5k+6` (11 to 36), because its
//! first best-trump trick scores 11 and every later one 10 (RULES, "Reachable scores"). Under the
//! house rule a team has taken its first bonus this game exactly when `score mod 5 == 1`, so the
//! bonus flags are not part of the state. Only the dealer's *team* matters (partners are a
//! symmetry), and rotating the four teams so the dealer's team is team 0 gives the canonical form.

use core::fmt;
use engine::view::PublicState;

use crate::money::MoneyChance;

/// The 15 reachable scores of a team between tricks (scores of 41 or more end the game), in
/// increasing order. A score's position in this list is its table index.
pub const SCORE_VALUES: [u8; 15] = [0, 5, 10, 11, 15, 16, 20, 21, 25, 26, 30, 31, 35, 36, 40];

/// Number of reachable scores per team.
pub const N_SCORES: usize = SCORE_VALUES.len();

/// Score combinations per games-won pattern: 15^4.
pub const SCORE_COMBOS: usize = N_SCORES * N_SCORES * N_SCORES * N_SCORES;

/// Games-won patterns of a non-terminal match: each team has won 0 or 1 games (4 bits).
pub const GAMES_PATTERNS: usize = 16;

/// Entries in the table: 16 × 15^4 = 810,000 canonical states.
pub const TABLE_STATES: usize = GAMES_PATTERNS * SCORE_COMBOS;

/// The `.npy` shape of the table: games bits, the four score indices, then the team.
pub const TABLE_SHAPE: [usize; 6] = [GAMES_PATTERNS, N_SCORES, N_SCORES, N_SCORES, N_SCORES, 4];

const fn build_score_index() -> [u8; 64] {
    let mut idx = [u8::MAX; 64];
    let mut i = 0;
    while i < N_SCORES {
        idx[SCORE_VALUES[i] as usize] = i as u8;
        i += 1;
    }
    idx
}

const SCORE_INDEX: [u8; 64] = build_score_index();

/// The table index (0..15) of a reachable score, or `None` (for example 6, or 41 and over).
pub const fn score_index(score: u8) -> Option<usize> {
    if score as usize >= SCORE_INDEX.len() {
        return None;
    }
    let i = SCORE_INDEX[score as usize];
    if i == u8::MAX { None } else { Some(i as usize) }
}

/// Has a team on this score already taken its first best-trump bonus this game? (house rule:
/// exactly when `score mod 5 == 1`)
pub const fn has_first_bonus(score: u8) -> bool {
    score % 5 == 1
}

/// Why a match state is not valid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StateError {
    /// A score that no team can stand on between tricks.
    UnreachableScore(u8),
    /// A team with more than 2 games, or more than one team with 2.
    BadGames([u8; 4]),
    /// The dealer team is not 0..=3.
    BadDealerTeam(u8),
    /// A text form that could not be read.
    Parse(String),
}

impl fmt::Display for StateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StateError::UnreachableScore(s) => write!(
                f,
                "score {s} is not reachable (a team's score is 0, 5, …, 40 or 11, 16, …, 36)"
            ),
            StateError::BadGames(g) => {
                write!(f, "games won {g:?} are not a match in progress or just won")
            }
            StateError::BadDealerTeam(d) => write!(f, "dealer team {d} is not 0..=3"),
            StateError::Parse(s) => write!(f, "cannot read the state: {s}"),
        }
    }
}

impl std::error::Error for StateError {}

/// The match between hands: the four scores of the current game, the games each team has won,
/// and the team of the dealer of the next hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MatchState {
    /// Each team's score in the current game.
    pub scores: [u8; 4],
    /// Games each team has won (0, 1, or 2 for the match winner).
    pub games: [u8; 4],
    /// The dealer's team (dealer seat mod 4).
    pub dealer_team: u8,
}

impl MatchState {
    /// A checked state.
    pub fn new(scores: [u8; 4], games: [u8; 4], dealer_team: u8) -> Result<Self, StateError> {
        let s = MatchState {
            scores,
            games,
            dealer_team,
        };
        s.validate()?;
        Ok(s)
    }

    /// The start of a match: all zero, with this dealer team.
    pub fn start(dealer_team: u8) -> Self {
        MatchState {
            scores: [0; 4],
            games: [0; 4],
            dealer_team: dealer_team % 4,
        }
    }

    /// The state the public view shows (meaningful between hands, or as "the scores now").
    pub fn from_public(p: &PublicState) -> Self {
        MatchState {
            scores: p.scores,
            games: p.games_won,
            dealer_team: p.dealer % 4,
        }
    }

    /// Check scores, games and dealer team.
    pub fn validate(&self) -> Result<(), StateError> {
        if self.dealer_team > 3 {
            return Err(StateError::BadDealerTeam(self.dealer_team));
        }
        let twos = self.games.iter().filter(|&&g| g == 2).count();
        if self.games.iter().any(|&g| g > 2) || twos > 1 {
            return Err(StateError::BadGames(self.games));
        }
        if twos == 0 {
            for &s in &self.scores {
                if score_index(s).is_none() {
                    return Err(StateError::UnreachableScore(s));
                }
            }
        }
        Ok(())
    }

    /// The match winner, if a team has won 2 games.
    pub fn winner(&self) -> Option<u8> {
        (0..4u8).find(|&t| self.games[usize::from(t)] >= 2)
    }

    /// Relabel the teams: team `t` becomes team `(t + by) mod 4` (scores, games and dealer).
    pub fn rotate(&self, by: u8) -> Self {
        let by = usize::from(by % 4);
        let mut scores = [0u8; 4];
        let mut games = [0u8; 4];
        for t in 0..4 {
            scores[(t + by) % 4] = self.scores[t];
            games[(t + by) % 4] = self.games[t];
        }
        MatchState {
            scores,
            games,
            dealer_team: ((usize::from(self.dealer_team) + by) % 4) as u8,
        }
    }

    /// The canonical form (dealer team 0) and the rotation `d` that maps it back:
    /// `self == canonical.rotate(d)`.
    pub fn canonical(&self) -> (MatchState, u8) {
        let d = self.dealer_team % 4;
        (self.rotate(4 - d), d)
    }

    /// The table index of this state's canonical form, or `None` for a finished match.
    ///
    /// # Panics
    /// If the state is not valid (see [`MatchState::validate`]).
    pub fn canonical_index(&self) -> Option<usize> {
        if let Err(e) = self.validate() {
            panic!("invalid match state {self}: {e}");
        }
        if self.winner().is_some() {
            return None;
        }
        let (c, _) = self.canonical();
        Some(canonical_index_of(&c.scores, &c.games))
    }

    /// The canonical state (dealer team 0) stored at a table index.
    ///
    /// # Panics
    /// If `index >= TABLE_STATES`.
    pub fn from_canonical_index(index: usize) -> MatchState {
        assert!(index < TABLE_STATES, "table index out of range");
        let games_bits = index / SCORE_COMBOS;
        let mut rest = index % SCORE_COMBOS;
        let mut scores = [0u8; 4];
        for t in (0..4).rev() {
            scores[t] = SCORE_VALUES[rest % N_SCORES];
            rest /= N_SCORES;
        }
        let games: [u8; 4] = core::array::from_fn(|t| ((games_bits >> t) & 1) as u8);
        MatchState {
            scores,
            games,
            dealer_team: 0,
        }
    }

    /// Total points of the four teams (for the sweep order).
    pub fn total_points(&self) -> u32 {
        self.scores.iter().map(|&s| u32::from(s)).sum()
    }

    /// Total games won (for the sweep order).
    pub fn total_games(&self) -> u32 {
        self.games.iter().map(|&g| u32::from(g)).sum()
    }

    /// Read the text form `scores=36,0,0,0 games=1,0,0,0 dealer_team=0` (fields in any order,
    /// separated by spaces).
    pub fn parse(text: &str) -> Result<Self, StateError> {
        let mut scores = None;
        let mut games = None;
        let mut dealer = None;
        for part in text.split_whitespace() {
            let (key, value) = part
                .split_once('=')
                .ok_or_else(|| StateError::Parse(format!("`{part}` is not key=value")))?;
            match key {
                "scores" => scores = Some(parse_four(value)?),
                "games" => games = Some(parse_four(value)?),
                "dealer_team" => {
                    dealer = Some(
                        value
                            .parse::<u8>()
                            .map_err(|_| StateError::Parse(format!("bad dealer_team `{value}`")))?,
                    )
                }
                _ => return Err(StateError::Parse(format!("unknown field `{key}`"))),
            }
        }
        let scores = scores.ok_or_else(|| StateError::Parse("missing scores=".into()))?;
        let games = games.ok_or_else(|| StateError::Parse("missing games=".into()))?;
        let dealer = dealer.ok_or_else(|| StateError::Parse("missing dealer_team=".into()))?;
        MatchState::new(scores, games, dealer)
    }
}

fn parse_four(value: &str) -> Result<[u8; 4], StateError> {
    let parts: Vec<&str> = value.split(',').collect();
    if parts.len() != 4 {
        return Err(StateError::Parse(format!("`{value}` needs four numbers")));
    }
    let mut out = [0u8; 4];
    for (o, p) in out.iter_mut().zip(&parts) {
        *o = p
            .trim()
            .parse::<u8>()
            .map_err(|_| StateError::Parse(format!("bad number `{p}`")))?;
    }
    Ok(out)
}

impl fmt::Display for MatchState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.scores;
        let g = self.games;
        write!(
            f,
            "scores={},{},{},{} games={},{},{},{} dealer_team={}",
            s[0], s[1], s[2], s[3], g[0], g[1], g[2], g[3], self.dealer_team
        )
    }
}

/// The table index of canonical scores and games (dealer team 0, every game count 0 or 1, every
/// score reachable): `games_bits × 15^4 + idx(s0)·15³ + idx(s1)·15² + idx(s2)·15 + idx(s3)`, with
/// `games_bits = Σ games[t] << t`.
///
/// # Panics
/// If a score is unreachable or a team has 2 games.
pub fn canonical_index_of(scores: &[u8; 4], games: &[u8; 4]) -> usize {
    let mut bits = 0usize;
    for (t, &g) in games.iter().enumerate() {
        assert!(g <= 1, "a finished match has no table entry");
        bits |= usize::from(g) << t;
    }
    let mut idx = 0usize;
    for &s in scores {
        let si = score_index(s).unwrap_or_else(|| panic!("unreachable score {s}"));
        idx = idx * N_SCORES + si;
    }
    bits * SCORE_COMBOS + idx
}

/// The value of a state in its own team order, from a table of canonical values (any closure
/// from canonical index to canonical value): finished matches are certain, everything else is
/// rotated to canonical form, looked up and rotated back.
pub fn value_via<F: Fn(usize) -> MoneyChance>(
    state: &MatchState,
    canonical_value: F,
) -> MoneyChance {
    if let Some(w) = state.winner() {
        return MoneyChance::certain(w);
    }
    let (c, d) = state.canonical();
    canonical_value(canonical_index_of(&c.scores, &c.games)).rotate(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reachable_scores_and_flags() {
        let reachable: Vec<u8> = (0..=60u8).filter(|&s| score_index(s).is_some()).collect();
        assert_eq!(reachable, SCORE_VALUES.to_vec());
        assert!(score_index(6).is_none());
        assert!(score_index(41).is_none());
        for (i, &s) in SCORE_VALUES.iter().enumerate() {
            assert_eq!(score_index(s), Some(i));
            // 5k or 5k+6 (k >= 1): flag exactly when score mod 5 == 1.
            assert!(s % 5 == 0 || (s % 5 == 1 && s >= 11));
            assert_eq!(has_first_bonus(s), s % 5 == 1);
        }
    }

    #[test]
    fn index_round_trip_covers_the_table() {
        let mut seen = vec![false; TABLE_STATES];
        for i in 0..TABLE_STATES {
            let s = MatchState::from_canonical_index(i);
            assert_eq!(s.dealer_team, 0);
            assert!(s.validate().is_ok());
            let j = s.canonical_index().expect("not terminal");
            assert_eq!(i, j);
            assert!(!seen[j]);
            seen[j] = true;
        }
        // The documented layout: C order over [games_bits, s0, s1, s2, s3].
        let s = MatchState::new([40, 0, 11, 5], [1, 0, 1, 1], 0).expect("valid");
        assert_eq!(
            s.canonical_index(),
            Some(0b1101 * 15 * 15 * 15 * 15 + 14 * 15 * 15 * 15 + 3 * 15 + 1)
        );
    }

    #[test]
    fn rotation_and_canonical_form() {
        let s = MatchState::new([5, 11, 0, 40], [1, 0, 0, 1], 2).expect("valid");
        let (c, d) = s.canonical();
        assert_eq!(d, 2);
        assert_eq!(c.dealer_team, 0);
        assert_eq!(c.scores, [0, 40, 5, 11]);
        assert_eq!(c.games, [0, 1, 1, 0]);
        assert_eq!(c.rotate(d), s);
        for by in 0..8 {
            let r = s.rotate(by);
            assert_eq!(
                r.canonical().0,
                c,
                "rotating never changes the canonical form"
            );
            assert_eq!(r.canonical_index(), s.canonical_index());
        }
    }

    #[test]
    fn validation_and_parse() {
        assert!(MatchState::new([6, 0, 0, 0], [0; 4], 0).is_err());
        assert!(MatchState::new([41, 0, 0, 0], [0; 4], 0).is_err());
        assert!(MatchState::new([0; 4], [2, 2, 0, 0], 0).is_err());
        assert!(MatchState::new([0; 4], [3, 0, 0, 0], 0).is_err());
        assert!(MatchState::new([0; 4], [0; 4], 4).is_err());
        let done = MatchState::new([45, 0, 0, 0], [2, 1, 1, 0], 1).expect("finished match");
        assert_eq!(done.winner(), Some(0));
        assert_eq!(done.canonical_index(), None);
        let s = MatchState::parse("scores=36,0,0,0 games=1,0,0,0 dealer_team=3").expect("parse");
        assert_eq!(
            s,
            MatchState::new([36, 0, 0, 0], [1, 0, 0, 0], 3).expect("valid")
        );
        assert_eq!(MatchState::parse(&s.to_string()), Ok(s));
        assert!(MatchState::parse("scores=36,0,0 games=1,0,0,0 dealer_team=3").is_err());
        assert!(MatchState::parse("games=1,0,0,0 dealer_team=3").is_err());
    }

    #[test]
    fn value_via_rotates_back() {
        // A fake canonical table whose value favours canonical team 1 (left of the dealer).
        let fake = |_i: usize| MoneyChance::certain(1);
        let s = MatchState::new([0; 4], [0; 4], 2).expect("valid");
        // Canonical team 1 is real team 3.
        assert_eq!(value_via(&s, fake), MoneyChance::certain(3));
        let done = MatchState::new([0; 4], [0, 0, 2, 1], 2).expect("valid");
        assert_eq!(value_via(&done, fake), MoneyChance::certain(2));
    }
}
