//! The seat-agent interface.
//!
//! A bot controls exactly one seat and only ever receives that seat's [`SeatView`]: it can never
//! see another hand, a burn, or the undealt pack (PLAN §6.2, "one seat per agent"). Every bot must
//! be deterministic given its seed and the views it has received, so that duplicate tournaments
//! replay exactly and leakage tests can compare decisions.

use engine::game::{Action, Rules};
use engine::view::SeatView;

/// A decision-maker for one seat.
pub trait Bot {
    /// A short stable name, used in records and reports (for example `random`, `rules-v1`).
    fn name(&self) -> &str;

    /// Called once at the start of every match the bot takes part in, before any decision.
    /// `seed` is derived by the table from the match seed and the seat, so the same bot at the
    /// same seat of the same match always receives the same seed.
    fn start_match(&mut self, seat: u8, rules: Rules, seed: u64);

    /// Choose one of the `legal` actions (never empty, sorted by card index) for the seat to act,
    /// which is always this bot's seat (`view.seat == view.public.to_act`). Returning an action not
    /// in `legal` is a bot bug; the table refuses it and stops the match with an error.
    fn act(&mut self, view: &SeatView, legal: &[Action]) -> Action;
}

/// A bot factory: builds a fresh bot per seat, so one description can fill several seats.
pub trait BotFactory {
    /// The name the built bots report.
    fn name(&self) -> &str;
    /// Build one bot instance.
    fn build(&self) -> Box<dyn Bot>;
}

impl<F: Fn() -> Box<dyn Bot>> BotFactory for (String, F) {
    fn name(&self) -> &str {
        &self.0
    }
    fn build(&self) -> Box<dyn Bot> {
        (self.1)()
    }
}
