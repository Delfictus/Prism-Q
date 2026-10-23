//! Wall-clock-bounded, canonical-science-codec-event-driven 5-stage phase
//! controller (operator directive 2026-06-15).
//!
//! The locked aspect ratio governs **wall-clock** allocation, not step counts:
//! it partitions the protocol wall budget into per-phase time *envelopes*. Steps
//! stay unbounded inside each envelope → maximum sampling/fidelity, GPU pinned.
//!
//! Phase transitions are **not** fired by the clock. They fire on a **canonical
//! science-codec event** (the `spike_source` taxonomy: LIF / UV / EFP / LADD /
//! COFIRE) detected by the (sensitized) collector — snapped to the real event,
//! never between events, never a bare clock cut. The wall envelope only *bounds*
//! the search:
//!   - **min dwell**: no transition before the phase's window opens (rejects a
//!     premature event firing the phase too early);
//!   - **force ceiling**: if no canonical event has fired by the envelope end,
//!     the transition force-fires at the **next canonical event** (still
//!     science-aligned) so the full 5-stage slate always fits the time frame.
//!
//! This struct holds only the time/decision logic and is integration-agnostic:
//! the chunk loop feeds it elapsed time + the count of clear canonical events
//! observed in the chunk, and it returns whether to advance the phase.

use std::time::Duration;

/// The five CCNS hysteresis stages, in order.
pub const N_PHASES: usize = 5;

/// Phase labels (index order matches the locked-ratio weight order).
pub const PHASE_LABELS: [&str; N_PHASES] =
    ["cold_hold", "ramp", "warm_hold", "ramp_down", "cold_return"];

/// A single phase's wall-clock envelope, measured from that phase's start.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PhaseEnvelope {
    /// No transition may fire before this much time has elapsed in the phase
    /// (rejects premature/noise-adjacent early triggers).
    pub min_dwell: Duration,
    /// Soft ceiling: once elapsed ≥ this, the transition force-fires at the next
    /// canonical event (or immediately if one is present this chunk).
    pub force_after: Duration,
}

/// Reason a transition fired (telemetry / audit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransitionTrigger {
    /// Fired on a clear canonical science-codec event inside the envelope.
    CanonicalEvent,
    /// Past the ceiling and a canonical event arrived — force-fired ON the event.
    ForcedAtCeiling,
    /// Past the ceiling with NO canonical event available — bounded time-fallback
    /// so a flat phase (e.g. cold_hold, which produces no spike-rate burst) cannot
    /// stall the protocol. The wall envelope is the hard upper bound: each phase
    /// lasts at most its wall slice, guaranteeing the full 5-stage panel fits the
    /// time frame. ("Never clock-cut" is honored as: prefer an event within
    /// [min_dwell, ceiling]; only cut when physics genuinely offers none.)
    ForcedAtCeilingNoEvent,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WallBoundPhaseController {
    ratio: [f32; N_PHASES],
    /// Per-phase envelope (min_dwell, force_after), derived from the locked ratio
    /// applied to the total protocol wall budget.
    envelopes: [PhaseEnvelope; N_PHASES],
    /// Fraction of each phase's time slice reserved as min-dwell before a
    /// transition may fire (e.g. 0.5 = must spend half the slice in-phase first).
    min_dwell_fraction: f64,
    current_phase: usize,
    /// Seconds of wall-clock elapsed within the current phase.
    phase_elapsed: Duration,
    /// Total protocol wall budget.
    total_wall: Duration,
}

impl WallBoundPhaseController {
    /// `ratio` = relative weights [cold_hold, ramp, warm_hold, ramp_down,
    /// cold_return]. `total_wall` = the protocol wall budget (the time frame the
    /// full 5-stage slate must fit in). `min_dwell_fraction` ∈ [0,1).
    pub fn new(ratio: [f32; N_PHASES], total_wall: Duration, min_dwell_fraction: f64) -> Self {
        let sum: f32 = ratio.iter().copied().sum::<f32>().max(f32::MIN_POSITIVE);
        let mdf = min_dwell_fraction.clamp(0.0, 0.95);
        let total = total_wall.as_secs_f64();
        let mut envelopes = [PhaseEnvelope {
            min_dwell: Duration::ZERO,
            force_after: Duration::ZERO,
        }; N_PHASES];
        for (i, w) in ratio.iter().enumerate() {
            let slice = total * (*w as f64) / (sum as f64);
            envelopes[i] = PhaseEnvelope {
                min_dwell: Duration::from_secs_f64(slice * mdf),
                force_after: Duration::from_secs_f64(slice),
            };
        }
        Self {
            ratio,
            envelopes,
            min_dwell_fraction: mdf,
            current_phase: 0,
            phase_elapsed: Duration::ZERO,
            total_wall,
        }
    }

    pub fn current_phase(&self) -> usize {
        self.current_phase
    }
    pub fn current_label(&self) -> &'static str {
        PHASE_LABELS[self.current_phase.min(N_PHASES - 1)]
    }
    pub fn envelope(&self, phase: usize) -> PhaseEnvelope {
        self.envelopes[phase.min(N_PHASES - 1)]
    }
    pub fn ratio(&self) -> [f32; N_PHASES] {
        self.ratio
    }
    pub fn min_dwell_fraction(&self) -> f64 {
        self.min_dwell_fraction
    }
    pub fn is_final_phase(&self) -> bool {
        self.current_phase >= N_PHASES - 1
    }

    /// Advance the controller by one chunk.
    ///
    /// `chunk_wall` = wall-clock duration of the chunk just completed.
    /// `canonical_events` = number of CLEAR canonical science-codec events
    /// (LIF/UV/EFP/LADD/COFIRE) the sensitized collector observed in the chunk.
    ///
    /// Returns `Some((new_phase, trigger))` if a transition fired (the caller
    /// should retarget the temperature to `new_phase`), else `None`. The final
    /// phase never transitions onward (the run ends on `total_wall`).
    pub fn on_chunk(
        &mut self,
        chunk_wall: Duration,
        canonical_events: u32,
    ) -> Option<(usize, TransitionTrigger)> {
        self.phase_elapsed = self.phase_elapsed.saturating_add(chunk_wall);
        if self.is_final_phase() {
            return None;
        }
        let env = self.envelopes[self.current_phase];
        if self.phase_elapsed < env.min_dwell {
            // Window not open yet — never transition (reject premature triggers).
            return None;
        }
        let canonical_fired = canonical_events > 0;
        let past_ceiling = self.phase_elapsed >= env.force_after;
        // In [min_dwell, ceiling): fire only on a canonical event (prefer physics).
        // Past the ceiling: fire ON a canonical event if present, else force a
        // bounded time-fallback (ForcedAtCeilingNoEvent) so a flat phase cannot
        // stall — the wall slice is the hard upper bound on phase duration.
        let trigger = if canonical_fired {
            if past_ceiling {
                TransitionTrigger::ForcedAtCeiling
            } else {
                TransitionTrigger::CanonicalEvent
            }
        } else if past_ceiling {
            TransitionTrigger::ForcedAtCeilingNoEvent
        } else {
            return None;
        };
        self.current_phase += 1;
        self.phase_elapsed = Duration::ZERO;
        Some((self.current_phase, trigger))
    }

    /// Whether the total protocol wall budget has been reached (run end).
    pub fn budget_reached(&self, total_run_elapsed: Duration) -> bool {
        total_run_elapsed >= self.total_wall
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: f64) -> Duration {
        Duration::from_secs_f64(s)
    }

    #[test]
    fn envelopes_partition_budget_by_ratio() {
        // 5:10:5:6:4 of 300s = 50:100:50:60:40
        let c = WallBoundPhaseController::new([5.0, 10.0, 5.0, 6.0, 4.0], d(300.0), 0.5);
        let secs: Vec<f64> = (0..N_PHASES)
            .map(|i| c.envelope(i).force_after.as_secs_f64())
            .collect();
        assert!((secs[0] - 50.0).abs() < 1e-6);
        assert!((secs[1] - 100.0).abs() < 1e-6);
        assert!((secs[2] - 50.0).abs() < 1e-6);
        assert!((secs[3] - 60.0).abs() < 1e-6);
        assert!((secs[4] - 40.0).abs() < 1e-6);
        // min dwell = 50% of slice
        assert!((c.envelope(1).min_dwell.as_secs_f64() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn no_transition_before_min_dwell_even_with_events() {
        let mut c = WallBoundPhaseController::new([5.0, 10.0, 5.0, 6.0, 4.0], d(300.0), 0.5);
        // cold_hold slice 50s, min_dwell 25s. A canonical event at 10s must NOT fire.
        assert_eq!(c.on_chunk(d(10.0), 99), None);
        assert_eq!(c.current_phase(), 0);
    }

    #[test]
    fn fires_on_canonical_event_after_min_dwell() {
        let mut c = WallBoundPhaseController::new([5.0, 10.0, 5.0, 6.0, 4.0], d(300.0), 0.5);
        // advance past min dwell (25s) with no events
        assert_eq!(c.on_chunk(d(26.0), 0), None);
        // canonical event now fires the transition
        let t = c.on_chunk(d(1.0), 3);
        assert_eq!(t, Some((1, TransitionTrigger::CanonicalEvent)));
        assert_eq!(c.current_phase(), 1);
    }

    #[test]
    fn past_ceiling_no_event_force_fires_bounded() {
        let mut c = WallBoundPhaseController::new([5.0, 10.0, 5.0, 6.0, 4.0], d(300.0), 0.5);
        // Blow past the 50s cold_hold ceiling with ZERO events: a flat phase
        // (no spike-rate burst) must not stall the protocol — the wall slice is
        // the hard upper bound, so it force-fires the bounded time-fallback.
        let t = c.on_chunk(d(80.0), 0);
        assert_eq!(t, Some((1, TransitionTrigger::ForcedAtCeilingNoEvent)));
        assert_eq!(c.current_phase(), 1);
    }

    #[test]
    fn event_inside_window_beats_ceiling_fallback() {
        let mut c = WallBoundPhaseController::new([5.0, 10.0, 5.0, 6.0, 4.0], d(300.0), 0.5);
        // cold_hold slice 50s, min_dwell 25s, ceiling 50s. A canonical event at
        // 30s (inside the window) fires on the event, before the ceiling fallback.
        assert_eq!(c.on_chunk(d(26.0), 0), None);
        let t = c.on_chunk(d(4.0), 2); // 30s total, < 50s ceiling, event present
        assert_eq!(t, Some((1, TransitionTrigger::CanonicalEvent)));
    }

    #[test]
    fn final_phase_never_transitions() {
        let mut c = WallBoundPhaseController::new([1.0, 1.0, 1.0, 1.0, 1.0], d(50.0), 0.0);
        for _ in 0..4 {
            // drive each phase forward with an event
            c.on_chunk(d(11.0), 5);
        }
        assert_eq!(c.current_phase(), 4);
        assert_eq!(
            c.on_chunk(d(100.0), 99),
            None,
            "final phase has no onward transition"
        );
    }
}
