//! Rocket flight state machine.
//!
//! Determines the current flight phase from ESKF state estimates
//! (velocity, acceleration derived from velocity changes).
//!
//! # States
//!
//! | State       | Description                                           |
//! |-------------|-------------------------------------------------------|
//! | `Pad`       | Pre-ignition: sitting on the launch pad               |
//! | `Ignition`  | Motor ignition detected (large upward acceleration)   |
//! | `Burn`      | Engine burning (sustained upward acceleration)        |
//! | `Coasting`  | Engine off, ascending with decreasing velocity        |
//! | `Apogee`    | Reached maximum altitude (vertical velocity ≈ 0)     |
//! | `Recovery`  | Descending under drag or parachute                   |

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Flight state enum
// ---------------------------------------------------------------------------

/// Discrete rocket flight phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FlightState {
    Pad = 0,
    Ignition = 1,
    Burn = 2,
    Coasting = 3,
    Apogee = 4,
    Recovery = 5,
}

impl FlightState {
    /// Human-readable label for the state.
    pub fn label(self) -> &'static str {
        match self {
            Self::Pad => "pad",
            Self::Ignition => "ignition",
            Self::Burn => "burn",
            Self::Coasting => "coasting",
            Self::Apogee => "apogee",
            Self::Recovery => "recovery",
        }
    }
}

// ---------------------------------------------------------------------------
// Thresholds — tuned for typical hobby-rocket flights
// ---------------------------------------------------------------------------

/// Upward acceleration to declare ignition (m/s²).
/// ESKF velocity is in NED, so "upward" = large negative vel_d change.
const IGNITION_ACCEL_THRESHOLD: f32 = 8.0;

/// Minimum upward acceleration during burn (m/s²).
/// Below this, transition to coasting.
const BURN_ACCEL_THRESHOLD: f32 = 2.0;

/// Downward velocity threshold to declare apogee passed (m/s, NED).
/// vel_d > this after coasting = apogee crossed.
const APOGEE_VEL_D_THRESHOLD: f32 = 0.5;

/// Number of consecutive samples required to confirm a state transition.
/// Prevents noisy single-sample transitions.
const CONFIRM_SAMPLES: u32 = 3;

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

/// Rocket flight state machine operating on ESKF output.
///
/// Call [`StateMachine::update`] once per ESKF sample. The machine outputs
/// the current state and records timestamps when transitions occur.
pub struct StateMachine {
    state: FlightState,
    /// Timestamps (seconds) at which each state was first entered.
    /// Index by FlightState as u8 (0..=5).
    transition_times: [f32; 6],
    prev_vel_d: f32,
    confirm_counter: u32,
    pending_state: Option<FlightState>,
}

impl Default for StateMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl StateMachine {
    pub fn new() -> Self {
        Self {
            state: FlightState::Pad,
            transition_times: [f32::NAN; 6],
            prev_vel_d: 0.0,
            confirm_counter: 0,
            pending_state: None,
        }
    }

    /// Update the state machine with a new ESKF sample.
    ///
    /// # Arguments
    /// * `time_s` — current simulation time (s)
    /// * `vel_n`, `vel_e`, `vel_d` — ESKF velocity in NED (m/s)
    /// * `dt` — time step since last sample (s)
    ///
    /// # Returns
    /// The current flight state *after* this update.
    pub fn update(
        &mut self,
        time_s: f32,
        _vel_n: f32,
        _vel_e: f32,
        vel_d: f32,
        dt: f32,
    ) -> FlightState {
        if dt < 1e-6 {
            self.prev_vel_d = vel_d;
            return self.state;
        }

        // Vertical acceleration estimate (positive upward).
        // In NED, vel_d is positive downward, so upward accel = -(dvel_d / dt).
        let accel_up = -(vel_d - self.prev_vel_d) / dt;
        self.prev_vel_d = vel_d;

        let candidate = match self.state {
            FlightState::Pad => {
                if accel_up > IGNITION_ACCEL_THRESHOLD {
                    Some(FlightState::Ignition)
                } else {
                    None
                }
            }
            FlightState::Ignition => {
                // Immediate transition to Burn once ignition is confirmed
                Some(FlightState::Burn)
            }
            FlightState::Burn => {
                if accel_up < BURN_ACCEL_THRESHOLD {
                    Some(FlightState::Coasting)
                } else {
                    None
                }
            }
            FlightState::Coasting => {
                // vel_d positive = descending in NED
                if vel_d > APOGEE_VEL_D_THRESHOLD {
                    Some(FlightState::Apogee)
                } else {
                    None
                }
            }
            FlightState::Apogee => {
                // Immediate transition to Recovery
                Some(FlightState::Recovery)
            }
            FlightState::Recovery => {
                // Terminal state
                None
            }
        };

        // Confirmation logic: require CONFIRM_SAMPLES consecutive agreeing samples
        if let Some(next) = candidate {
            if self.pending_state == Some(next) {
                self.confirm_counter += 1;
            } else {
                self.pending_state = Some(next);
                self.confirm_counter = 1;
            }

            // Ignition→Burn and Apogee→Recovery are instantaneous (1 sample)
            let needed = match next {
                FlightState::Burn | FlightState::Recovery => 1,
                _ => CONFIRM_SAMPLES,
            };

            if self.confirm_counter >= needed {
                self.state = next;
                let idx = next as usize;
                if self.transition_times[idx].is_nan() {
                    // Record the time when confirmation started, not when it completed
                    self.transition_times[idx] =
                        time_s - (self.confirm_counter.saturating_sub(1) as f32) * dt;
                }
                self.pending_state = None;
                self.confirm_counter = 0;
            }
        } else {
            self.pending_state = None;
            self.confirm_counter = 0;
        }

        self.state
    }

    /// Get the transition time for a given state, or NaN if not yet reached.
    pub fn transition_time(&self, state: FlightState) -> f32 {
        self.transition_times[state as usize]
    }

    /// Get all transition times as an array indexed by state.
    pub fn transition_times(&self) -> &[f32; 6] {
        &self.transition_times
    }

    /// Get the current state.
    pub fn current_state(&self) -> FlightState {
        self.state
    }
}

// ---------------------------------------------------------------------------
// Batch processing function — runs state machine on ESKF output arrays
// ---------------------------------------------------------------------------

/// Run the state machine over ESKF velocity arrays.
///
/// # Arguments
/// * `times_s` — time array (s)
/// * `vel_n`, `vel_e`, `vel_d` — ESKF velocity arrays (NED, m/s)
///
/// # Returns
/// A tuple of:
/// * `states` — per-sample state (u8, 0..=5)
/// * `transition_times` — [pad, ignition, burn, coasting, apogee, recovery]
///   time in seconds, NaN if not reached
#[pyfunction]
pub fn detect_flight_states(
    times_s: Vec<f32>,
    vel_n: Vec<f32>,
    vel_e: Vec<f32>,
    vel_d: Vec<f32>,
) -> (Vec<u8>, [f32; 6]) {
    let n = times_s.len();
    let mut sm = StateMachine::new();
    let mut states = Vec::with_capacity(n);

    for i in 0..n {
        let dt = if i > 0 {
            times_s[i] - times_s[i - 1]
        } else {
            0.0
        };
        let s = sm.update(times_s[i], vel_n[i], vel_e[i], vel_d[i], dt);
        states.push(s as u8);
    }

    (states, *sm.transition_times())
}

/// Run state detection on ground-truth sim arrays (XYZ frame, Y=up).
///
/// Uses acceleration_y (upward) and velocity_y to detect states from
/// the simulation truth data directly.
///
/// # Returns
/// * `states` — per-sample state (u8)
/// * `transition_times` — [pad, ignition, burn, coasting, apogee, recovery]
#[pyfunction]
pub fn detect_truth_states(
    times_s: Vec<f32>,
    _accel_y: Vec<f32>,
    vel_y: Vec<f32>,
    thrust_n: Vec<f32>,
) -> (Vec<u8>, [f32; 6]) {
    let n = times_s.len();
    let mut states = Vec::with_capacity(n);
    let mut transition_times = [f32::NAN; 6];
    let mut current_state = FlightState::Pad;
    transition_times[FlightState::Pad as usize] = 0.0;

    for i in 0..n {
        let next = match current_state {
            FlightState::Pad => {
                // Truth ignition: thrust goes positive
                if thrust_n[i] > 1.0 {
                    Some(FlightState::Ignition)
                } else {
                    None
                }
            }
            FlightState::Ignition => Some(FlightState::Burn),
            FlightState::Burn => {
                // Truth burnout: thrust drops to ~0
                if thrust_n[i] < 1.0 {
                    Some(FlightState::Coasting)
                } else {
                    None
                }
            }
            FlightState::Coasting => {
                // Truth apogee: velocity_y crosses zero downward
                if vel_y[i] <= 0.0 {
                    Some(FlightState::Apogee)
                } else {
                    None
                }
            }
            FlightState::Apogee => Some(FlightState::Recovery),
            FlightState::Recovery => None,
        };

        if let Some(next_state) = next {
            current_state = next_state;
            let idx = next_state as usize;
            if transition_times[idx].is_nan() {
                transition_times[idx] = times_s[i];
            }
        }

        states.push(current_state as u8);
    }

    (states, transition_times)
}

// ---------------------------------------------------------------------------
// State label utility
// ---------------------------------------------------------------------------

/// Convert a state u8 to its string label.
pub fn state_label(s: u8) -> &'static str {
    match s {
        0 => "pad",
        1 => "ignition",
        2 => "burn",
        3 => "coasting",
        4 => "apogee",
        5 => "recovery",
        _ => "unknown",
    }
}
