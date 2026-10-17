//! The stopwatch behind the solvers' optional time cap (`time_cap_ms`, used by benchmarks and
//! planned runs only).
//!
//! WebAssembly for the browser (`wasm32-unknown-unknown`) has no clock in Rust's standard
//! library: `std::time::Instant::now` panics there. The engine's WebAssembly API never sets a
//! time cap (its budgets are deals, never time: docs/WASM-API.md), so on that target the
//! stopwatch always reads 0 and a cap can never fire. It never changes a value: a solve stopped
//! by its cap returns no value at all.

/// A started stopwatch.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Stopwatch {
    #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
    start: std::time::Instant,
}

impl Stopwatch {
    /// Start now.
    pub(crate) fn start() -> Stopwatch {
        Stopwatch {
            #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
            start: std::time::Instant::now(),
        }
    }

    /// Whole milliseconds since the start (always 0 without a clock).
    pub(crate) fn elapsed_ms(&self) -> u64 {
        #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
        {
            self.start.elapsed().as_millis() as u64
        }
        #[cfg(all(target_family = "wasm", target_os = "unknown"))]
        {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stopwatch_starts_near_zero() {
        // No sleep: the test only checks the reading is a number of milliseconds that grows.
        let s = Stopwatch::start();
        let a = s.elapsed_ms();
        let b = s.elapsed_ms();
        assert!(b >= a);
        assert!(a < 60_000);
    }
}
