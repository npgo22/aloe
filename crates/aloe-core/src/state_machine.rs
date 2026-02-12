// ---------------------------------------------------------------------------
// Enums & Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FlightState {
    Pad = 0,
    Ascent = 1,  // Powered Flight
    Coast = 2,   // Unpowered Ascent
    Descent = 3, // Drogue/Main Descent
    Landed = 4,  // Ground hit
}

impl FlightState {
    pub fn label(self) -> &'static str {
        match self {
            Self::Pad => "Pad",
            Self::Ascent => "Ascent",
            Self::Coast => "Coast",
            Self::Descent => "Descent",
            Self::Landed => "Landed",
        }
    }
}

pub const NUM_STAGES: usize = 5;

#[derive(Debug, Clone, Copy)]
pub struct StateMachineConfig {
    pub launch_accel_thresh: f32,
    pub launch_vel_thresh: f32,
    pub burnout_accel_thresh: f32,
    pub min_ascent_time: f32,
    pub apogee_descent_thresh: f32,
    pub min_coast_time: f32,
    pub landing_vel_thresh: f32,
    pub landing_alt_thresh: f32,
    pub landing_confirm_window: f32,
}

impl Default for StateMachineConfig {
    fn default() -> Self {
        Self {
            launch_accel_thresh: 20.0,
            launch_vel_thresh: 10.0,
            burnout_accel_thresh: 2.0,
            min_ascent_time: 0.5,
            apogee_descent_thresh: 1.0,
            min_coast_time: 2.0,
            landing_vel_thresh: 0.5,
            landing_alt_thresh: 100.0,
            landing_confirm_window: 2.0,
        }
    }
}

/// Inputs required by the state machine update.
#[derive(Debug, Clone, Copy)]
pub struct StateInput {
    pub time: f32,
    pub altitude: f32,      // Up is positive (AGL)
    pub velocity_down: f32, // NED frame (positive = down)
    pub accel_down: f32,    // NED frame (positive = down)
}

// ---------------------------------------------------------------------------
// State Machine
// ---------------------------------------------------------------------------

pub struct StateMachine {
    config: StateMachineConfig,
    state: FlightState,

    /// Time when the current state was entered.
    state_start_time: f32,

    /// Historical transition timestamps.
    transition_times: [f32; NUM_STAGES],

    // Internal counters
    landing_detect_timer: f32,
}

impl Default for StateMachine {
    fn default() -> Self {
        Self::new(StateMachineConfig::default())
    }
}

impl StateMachine {
    pub fn new(config: StateMachineConfig) -> Self {
        let mut transition_times = [0.0; NUM_STAGES];
        // We use 0.0 or -1.0 to indicate "not reached" in no_std if NAN is annoying,
        // but f32::NAN is available in core.
        transition_times.fill(f32::NAN);
        transition_times[FlightState::Pad as usize] = 0.0;

        Self {
            config,
            state: FlightState::Pad,
            state_start_time: 0.0,
            transition_times,
            landing_detect_timer: 0.0,
        }
    }

    pub fn update(&mut self, input: StateInput, dt: f32) -> FlightState {
        let time_in_state = input.time - self.state_start_time;

        let next_state = match self.state {
            FlightState::Pad => {
                // Launch: High Upward Accel OR High Upward Velocity
                // Accel Down < -Thresh  OR  Vel Down < -Thresh
                let launch_accel = input.accel_down < -self.config.launch_accel_thresh;
                let launch_vel = input.velocity_down < -self.config.launch_vel_thresh;

                if launch_accel || launch_vel {
                    Some(FlightState::Ascent)
                } else {
                    None
                }
            }
            FlightState::Ascent => {
                // Burnout: Weak upward acceleration
                // Accel Down > -Thresh (means upward push is gone)
                let weak_accel = input.accel_down > -self.config.burnout_accel_thresh;

                if weak_accel && time_in_state > self.config.min_ascent_time {
                    Some(FlightState::Coast)
                } else {
                    None
                }
            }
            FlightState::Coast => {
                // Apogee: Velocity turns positive (Down)
                let descending = input.velocity_down > self.config.apogee_descent_thresh;

                if descending && time_in_state > self.config.min_coast_time {
                    Some(FlightState::Descent)
                } else {
                    None
                }
            }
            FlightState::Descent => {
                // Landing: Velocity near zero AND Low Altitude
                // Using libm::fabs or simple if check for no_std abs()
                let vel_abs = if input.velocity_down < 0.0 {
                    -input.velocity_down
                } else {
                    input.velocity_down
                };

                let low_vel = vel_abs < self.config.landing_vel_thresh;
                let low_alt = input.altitude < self.config.landing_alt_thresh;

                if low_vel && low_alt {
                    self.landing_detect_timer += dt;
                } else {
                    self.landing_detect_timer = 0.0;
                }

                if self.landing_detect_timer > self.config.landing_confirm_window {
                    Some(FlightState::Landed)
                } else {
                    None
                }
            }
            FlightState::Landed => None,
        };

        if let Some(new_state) = next_state {
            self.transition_to(new_state, input.time);
        }

        self.state
    }

    fn transition_to(&mut self, new_state: FlightState, time: f32) {
        self.state = new_state;
        self.state_start_time = time;
        self.transition_times[new_state as usize] = time;
        self.landing_detect_timer = 0.0;
    }

    pub fn current_state(&self) -> FlightState {
        self.state
    }

    pub fn transition_time(&self, state: FlightState) -> f32 {
        self.transition_times[state as usize]
    }
}
