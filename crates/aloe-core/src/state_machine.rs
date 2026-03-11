//! # Rocket Flight State Machine
//!
//! This module implements a finite state machine (FSM) for detecting and tracking the discrete
//! phases of rocket flight. The state machine uses sensor-based heuristics to transition between
//! five primary states, enabling stage-specific filter tuning and event detection.
//!
//! ## Physical Basis: Rocket Flight Phases
//!
//! A typical hobby rocket flight follows a predictable sequence of phases, each characterized
//! by distinct kinematic properties:
//!
//! ### 1. Pad (Pre-launch)
//! - **Duration**: $t < t_{ignition}$
//! - **Kinematics**: $\mathbf{v} \approx \mathbf{0}$, $\mathbf{a} \approx \mathbf{g}$
//! - **Purpose**: Initialize sensor biases, establish launch site coordinates
//!
//! ### 2. Ascent (Powered Flight)
//! - **Duration**: $t_{ignition} < t < t_{burnout}$
//! - **Kinematics**: High upward acceleration from motor thrust
//! - **Equation of Motion**:
//!
//! $$\mathbf{a} = \frac{T}{m(t)}\hat{\mathbf{z}} - \mathbf{g} - \frac{1}{2}\rho v^2 C_D \frac{A}{m}$$
//!
//! where:
//! - $T$: Thrust force (N)
//! - $m(t) = m_{dry} + m_{prop}(1 - t/t_{burn})$: Time-varying mass (kg)
//! - $\rho$: Air density (kg/m³)
//! - $C_D$: Drag coefficient
//! - $A$: Reference area (m²)
//!
//! **Detection criterion**: $|a_z| > 20~\text{m/s}^2$ OR $|v_z| > 10~\text{m/s}$
//!
//! ### 3. Coast (Unpowered Ascent)
//! - **Duration**: $t_{burnout} < t < t_{apogee}$
//! - **Kinematics**: Deceleration due to gravity and drag, still climbing
//! - **Equation of Motion**:
//!
//! $$\mathbf{a} = -\mathbf{g} - \frac{1}{2}\rho v^2 C_D \frac{A}{m_{dry}}$$
//!
//! **Detection criterion**: $|a_z| < 2~\text{m/s}^2$ AND $t - t_{ascent} > 0.5~\text{s}$
//!
//! ### 4. Descent (Post-apogee)
//! - **Duration**: $t_{apogee} < t < t_{landing}$
//! - **Kinematics**: Falling, possibly with parachute deployment
//! - **Terminal Velocity** (with parachute):
//!
//! $$v_{terminal} = \sqrt{\frac{2mg}{\rho C_D A_{chute}}}$$
//!
//! **Detection criterion**: $v_z > 1~\text{m/s}$ (positive = downward in NED)
//! AND $t - t_{coast} > 2~\text{s}$
//!
//! ### 5. Landed (Ground Impact)
//! - **Duration**: $t > t_{landing}$
//! - **Kinematics**: $\mathbf{v} \approx \mathbf{0}$, $h < 100~\text{m}$
//!
//! **Detection criterion**: $|v| < 0.5~\text{m/s}$ AND $h < 100~\text{m}$ for 2 seconds
//!
//! ## High-Velocity Barometer Degradation
//!
//! At high velocities (>Mach 0.5 ≈ 170 m/s), dynamic pressure effects corrupt static pressure
//! measurements from the barometer. The measured static pressure $P_s$ includes a dynamic component:
//!
//! $$P_{measured} = P_s + \Delta P_{dynamic}$$
//!
//! where the dynamic pressure error scales with velocity squared:
//!
//! $$\Delta P \approx k \cdot \frac{1}{2}\rho v^2$$
//!
//! The filter tracks this condition via [`is_high_velocity_baro_degraded()`](StateMachine::is_high_velocity_baro_degraded)
//! and can increase the barometer measurement noise variance $R_{baro}$ accordingly.
//!
//! ## State Transition Diagram
//!
//! ```plantuml
//! @startuml
//! !theme plain
//! skinparam state {
//!   BackgroundColor<<pad>> LightGray
//!   BackgroundColor<<ascent>> LightGreen
//!   BackgroundColor<<coast>> LightBlue
//!   BackgroundColor<<descent>> LightYellow
//!   BackgroundColor<<landed>> LightCoral
//! }
//!
//! [*] --> Pad
//! Pad --> Ascent : |a| > 20 m/s² OR\n|v| > 10 m/s
//! Ascent --> Coast : |a| < 2 m/s² AND\nt > 0.5s
//! Coast --> Descent : vz > 1 m/s AND\nt > 2s\n(apogee detected)
//! Descent --> Landed : |v| < 0.5 m/s AND\nh < 100m\nfor 2s
//! Landed --> [*]
//!
//! note right of Pad
//!   Initialize biases
//!   Establish home position
//! end note
//!
//! note right of Ascent
//!   High-g environment
//!   Baro may degrade at >Mach 0.5
//! end note
//!
//! note right of Coast
//!   Ballistic trajectory
//!   Peak altitude detection
//! end note
//!
//! note right of Descent
//!   Parachute deployment
//!   Terminal velocity
//! end note
//!
//! note right of Landed
//!   Recovery mode
//!   Transmit final position
//! end note
//!
//! @enduml
//! ```
//!
//! ## Hysteresis and Confirmation Windows
//!
//! To prevent spurious transitions due to sensor noise, the state machine employs:
//!
//! 1. **Minimum time-in-state requirements**: Prevents rapid oscillations
//! 2. **Confirmation windows**: Requires sustained condition (e.g., 2s for landing)
//! 3. **Threshold margins**: Acceleration/velocity thresholds with safety margins
//!
//! ## Usage Example
//!
//! ```
//! use aloe_core::state_machine::{StateMachine, StateMachineConfig, StateInput};
//!
//! let mut sm = StateMachine::new(StateMachineConfig::default());
//!
//! // Update at each timestep with sensor data
//! let input = StateInput {
//!     time: 1.5,  // seconds
//!     altitude: 150.0,  // meters AGL
//!     velocity_down: -50.0,  // m/s (negative = upward in NED)
//!     accel_down: -120.0,  // m/s² (negative = upward)
//! };
//!
//! let state = sm.update(input, 0.001);  // 1ms timestep
//! assert_eq!(state.label(), "Ascent");
//!
//! // Check if barometer is reliable
//! if sm.is_high_velocity_baro_degraded() {
//!     // Increase baro measurement noise variance
//! }
//! ```
//!
//! ## Configuration Parameters
//!
//! All thresholds are configurable via [`StateMachineConfig`]:
//! - `launch_accel_thresh`: Pad → Ascent acceleration (20 m/s²)
//! - `launch_vel_thresh`: Pad → Ascent velocity (10 m/s)
//! - `burnout_accel_thresh`: Ascent → Coast acceleration (2 m/s²)
//! - `apogee_descent_thresh`: Coast → Descent velocity (1 m/s)
//! - `landing_vel_thresh`: Descent → Landed velocity (0.5 m/s)
//! - `landing_alt_thresh`: Descent → Landed altitude (100 m)
//! - `high_velocity_baro_thresh`: Baro degradation velocity (~Mach 0.5 = 170 m/s)
//!
//! // ---------------------------------------------------------------------------
//! // Enums & Config
//! // ---------------------------------------------------------------------------

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
    /// Velocity threshold (m/s) above which barometer is unreliable (~Mach 0.5 = 170 m/s)
    pub high_velocity_baro_thresh: f32,
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
            high_velocity_baro_thresh: 170.0, // ~Mach 0.5 at sea level
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

    /// Flag indicating high-velocity flight where barometer may be unreliable
    high_velocity_baro_degraded: bool,
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
            high_velocity_baro_degraded: false,
        }
    }

    pub fn update(&mut self, input: StateInput, dt: f32) -> FlightState {
        let time_in_state = input.time - self.state_start_time;

        // Check for high-velocity conditions where barometer is unreliable
        // Velocity magnitude: sqrt(vx² + vy² + vz²), but we only have velocity_down
        // For a rough check, we can use abs(velocity_down) as a lower bound
        let vel_abs = if input.velocity_down < 0.0 {
            -input.velocity_down
        } else {
            input.velocity_down
        };
        self.high_velocity_baro_degraded = vel_abs > self.config.high_velocity_baro_thresh;

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

    /// Returns true if the rocket is in high-velocity flight where barometer may be unreliable.
    /// This occurs when vertical velocity exceeds ~Mach 0.5 (170 m/s), causing dynamic pressure
    /// effects that can corrupt static pressure measurements.
    pub fn is_high_velocity_baro_degraded(&self) -> bool {
        self.high_velocity_baro_degraded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input(time: f32, altitude: f32, velocity_down: f32, accel_down: f32) -> StateInput {
        StateInput {
            time,
            altitude,
            velocity_down,
            accel_down,
        }
    }

    #[test]
    fn test_initial_state_is_pad() {
        let sm = StateMachine::new(StateMachineConfig::default());
        assert_eq!(sm.current_state(), FlightState::Pad);
        assert_eq!(sm.current_state().label(), "Pad");
    }

    #[test]
    fn test_pad_to_ascent_via_accel() {
        let mut sm = StateMachine::new(StateMachineConfig::default());
        let dt = 0.001;

        // Pad state
        let state = sm.update(make_input(0.0, 0.0, 0.0, -9.8), dt);
        assert_eq!(state, FlightState::Pad);

        // High acceleration triggers launch (>20 m/s² upward = accel_down < -20)
        let state = sm.update(make_input(0.5, 0.1, -5.0, -100.0), dt);
        assert_eq!(state, FlightState::Ascent);
    }

    #[test]
    fn test_pad_to_ascent_via_velocity() {
        let mut sm = StateMachine::new(StateMachineConfig::default());
        let dt = 0.001;

        // High velocity triggers launch (>10 m/s upward = velocity_down < -10)
        let state = sm.update(make_input(0.5, 0.5, -50.0, -9.8), dt);
        assert_eq!(state, FlightState::Ascent);
    }

    #[test]
    fn test_ascent_to_coast() {
        let mut sm = StateMachine::new(StateMachineConfig::default());
        let dt = 0.001;

        // Trigger launch
        sm.update(make_input(0.5, 0.1, -50.0, -100.0), dt);
        assert_eq!(sm.current_state(), FlightState::Ascent);

        // Burnout: low acceleration after minimum ascent time
        // Need time_in_state > 0.5s AND accel_down > -2.0
        let state = sm.update(make_input(1.5, 500.0, -100.0, -1.0), dt);
        assert_eq!(state, FlightState::Coast);
    }

    #[test]
    fn test_coast_to_descent() {
        let mut sm = StateMachine::new(StateMachineConfig::default());
        let dt = 0.001;

        // Fast forward to coast
        sm.update(make_input(0.5, 0.1, -50.0, -100.0), dt); // Launch
        sm.update(make_input(1.5, 500.0, -100.0, -1.0), dt); // Coast

        // Apogee: velocity turns positive (downward) after minimum coast time
        // Need time_in_state > 2.0s AND velocity_down > 1.0
        let state = sm.update(make_input(5.0, 1000.0, 5.0, -9.8), dt);
        assert_eq!(state, FlightState::Descent);
    }

    #[test]
    fn test_high_velocity_baro_degraded() {
        let mut sm = StateMachine::new(StateMachineConfig::default());
        let dt = 0.001;

        // Below threshold
        sm.update(make_input(1.0, 100.0, -100.0, -50.0), dt);
        assert!(!sm.is_high_velocity_baro_degraded());

        // Above threshold (>170 m/s)
        sm.update(make_input(1.5, 200.0, -200.0, -50.0), dt);
        assert!(sm.is_high_velocity_baro_degraded());

        // Back below threshold
        sm.update(make_input(2.0, 300.0, -100.0, -20.0), dt);
        assert!(!sm.is_high_velocity_baro_degraded());
    }

    #[test]
    fn test_transition_times_recorded() {
        let mut sm = StateMachine::new(StateMachineConfig::default());
        let dt = 0.001;

        // Pad transition time is 0.0
        assert_eq!(sm.transition_time(FlightState::Pad), 0.0);

        // Trigger launch at t=0.5
        sm.update(make_input(0.5, 0.1, -50.0, -100.0), dt);
        assert_eq!(sm.transition_time(FlightState::Ascent), 0.5);
    }

    #[test]
    fn test_all_state_labels() {
        assert_eq!(FlightState::Pad.label(), "Pad");
        assert_eq!(FlightState::Ascent.label(), "Ascent");
        assert_eq!(FlightState::Coast.label(), "Coast");
        assert_eq!(FlightState::Descent.label(), "Descent");
        assert_eq!(FlightState::Landed.label(), "Landed");
    }

    #[test]
    fn test_custom_config() {
        let config = StateMachineConfig {
            launch_accel_thresh: 50.0, // Higher threshold
            launch_vel_thresh: 20.0,
            ..Default::default()
        };
        let mut sm = StateMachine::new(config);
        let dt = 0.001;

        // Normal launch acceleration (20 m/s²) should NOT trigger with higher threshold
        sm.update(make_input(0.5, 0.1, -5.0, -25.0), dt);
        assert_eq!(sm.current_state(), FlightState::Pad);

        // But higher acceleration should trigger
        sm.update(make_input(1.0, 0.2, -15.0, -60.0), dt);
        assert_eq!(sm.current_state(), FlightState::Ascent);
    }
}
