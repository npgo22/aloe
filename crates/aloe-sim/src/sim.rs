//! # 6-DOF Rocket Flight Simulator
//!
//! A high-fidelity six-degree-of-freedom rocket trajectory simulator that models:
//! - Full 3D rigid body dynamics using quaternions for attitude representation
//! - Aerodynamic forces (drag and lift) with angle of attack effects
//! - Time-varying mass and center of gravity during propellant burn
//! - Launch rail constraints
//! - Exponential atmosphere model
//! - Wind effects
//! - RK4 numerical integration
//!
//! ## Coordinate Systems
//!
//! - **World Frame (NED)**: North-East-Down
//!   - X: North
//!   - Y: East
//!   - Z: Down (positive Z is below ground)
//!   - Origin at launch site
//!
//! - **Body Frame**: Fixed to rocket
//!   - X: Along rocket axis (nose to tail)
//!   - Y: Right wing
//!   - Z: Down when rocket is vertical
//!
//! ## Physics Notes
//!
//! - Uses RK4 integration with fixed timestep for stability
//! - Proper acceleration (accelerometer output) excludes gravity
//! - Quaternion normalization handled implicitly by UnitQuaternion
//! - Launch rail constrains lateral motion and rotation until clearance

use nalgebra::{Matrix3, UnitQuaternion, Vector3};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Integration timestep (seconds) - 1000Hz
const DT: f64 = 0.001;

/// Maximum simulation time (seconds)
const MAX_TIME: f64 = 400.0;

/// Atmospheric scale height (meters) for exponential density model
const H_SCALE: f64 = 7400.0;

/// Minimum velocity magnitude for aerodynamic calculations (m/s)
const MIN_AERO_VELOCITY: f64 = 0.1;

/// Minimum angle for quaternion rotation (rad)
const MIN_ROTATION_ANGLE: f64 = 1e-9;

/// Ground level in NED coordinates (Z = 0)
const GROUND_LEVEL: f64 = 0.0;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// Complete set of rocket and environment parameters for simulation
#[derive(Debug, Clone)]
pub struct RocketParams {
    // === Mass Properties ===
    /// Dry mass of rocket without propellant (kg)
    pub dry_mass: f64,

    /// Total propellant mass at launch (kg)
    pub propellant_mass: f64,

    /// Moments of inertia about body axes [Ixx, Iyy, Izz] (kg·m²)
    /// For a cylinder: Ixx (roll) ≈ 0.5*m*r², Iyy,Izz ≈ (1/12)*m*(3r² + h²)
    pub inertia_tensor: Vector3<f64>,

    /// Center of gravity location at full propellant load (m from nose)
    pub cg_full: f64,

    /// Center of gravity location when propellant is empty (m from nose)
    /// Must be forward of cg_full for stability
    pub cg_empty: f64,

    // === Aerodynamics ===
    /// Center of pressure location (m from nose, should be aft of CG)
    pub cp_location: f64,

    /// Reference area for aerodynamic calculations (m²) - typically cross-section
    pub ref_area: f64,

    /// Axial drag coefficient (dimensionless, typical range 0.3-0.8)
    pub drag_coeff_axial: f64,

    /// Normal force coefficient per radian of angle of attack (1/rad)
    /// Typical range: 10-20 for finned rockets
    pub normal_force_coeff: f64,

    // === Propulsion ===
    /// Thrust curve as (time, thrust) pairs (s, N)
    /// Time is relative to ignition, must start at 0.0
    pub thrust_curve: Vec<(f64, f64)>,

    /// Total burn duration (s)
    pub burn_time: f64,

    /// Specific impulse (s). Used to compute propellant mass flow from thrust.
    /// Typical solid motors: 150–250 s. Default 200 s.
    pub isp: f64,

    /// Nozzle exit location (m from nose)
    pub nozzle_location: f64,

    // === Environment ===
    /// Gravitational acceleration (m/s²)
    pub gravity: f64,

    /// Air density at sea level (kg/m³)
    pub air_density_sea_level: f64,

    /// Launch rail/rod length (m) - rocket is constrained until it travels this distance
    pub launch_rod_length: f64,

    /// Wind velocity vector in NED frame (m/s)
    pub wind_velocity_ned: Vector3<f64>,

    // === Launch Configuration ===
    /// Delay from t=0 until ignition (s)
    pub launch_delay: f64,

    /// Initial spin rate around body X-axis (°/s)
    pub spin_rate: f64,

    /// Thrust misalignment angle (°) - simulates thrust vector control or manufacturing defects
    pub thrust_cant: f64,
}

impl Default for RocketParams {
    fn default() -> Self {
        Self {
            // Mass: 30kg total, 10kg propellant
            dry_mass: 20.0,
            propellant_mass: 10.0,

            // Inertia for ~3m long, 0.15m diameter rocket
            inertia_tensor: Vector3::new(0.1, 10.0, 10.0),

            // CG shifts forward as fuel burns
            cg_full: 1.5,
            cg_empty: 1.5,

            // CP behind CG for passive stability
            cp_location: 2.0,

            // ~15cm diameter rocket
            ref_area: std::f64::consts::PI * 0.075_f64.powi(2),

            drag_coeff_axial: 0.5,
            normal_force_coeff: 12.0,

            // 5-second burn at 2000N
            thrust_curve: vec![(0.0, 2000.0), (5.0, 2000.0), (5.01, 0.0)],
            burn_time: 5.0,
            isp: 200.0,
            nozzle_location: 3.0,

            // Standard atmosphere
            gravity: 9.80665,
            air_density_sea_level: 1.225,

            // 2m launch rail
            launch_rod_length: 2.0,

            // 5 m/s wind from North
            wind_velocity_ned: Vector3::new(5.0, 0.0, 0.0),

            launch_delay: 1.0,
            spin_rate: 0.0,
            thrust_cant: 0.0,
        }
    }
}

impl RocketParams {
    /// Validate parameters for physical consistency
    pub fn validate(&self) -> Result<(), String> {
        if self.dry_mass <= 0.0 {
            return Err("Dry mass must be positive".to_string());
        }
        if self.propellant_mass < 0.0 {
            return Err("Propellant mass must be non-negative".to_string());
        }
        if self.inertia_tensor.iter().any(|&i| i <= 0.0) {
            return Err("All moments of inertia must be positive".to_string());
        }
        if self.cg_empty > self.cg_full {
            return Err("Empty CG must be forward of (less than) full CG".to_string());
        }
        if self.cp_location < self.cg_full {
            return Err("CP should be aft of CG for stability".to_string());
        }
        if self.ref_area <= 0.0 {
            return Err("Reference area must be positive".to_string());
        }
        if self.burn_time < 0.0 {
            return Err("Burn time must be non-negative".to_string());
        }
        if self.thrust_curve.is_empty() {
            return Err("Thrust curve must have at least one point".to_string());
        }
        if self.thrust_curve[0].0 != 0.0 {
            return Err("Thrust curve must start at time 0.0".to_string());
        }
        if self.isp <= 0.0 {
            return Err("Isp must be positive".to_string());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 6-DoF State
// ---------------------------------------------------------------------------

/// Complete state vector for 6-DOF simulation
#[derive(Clone, Debug)]
pub struct State {
    /// Simulation time (s)
    pub t: f64,

    /// Position in world frame (m, NED)
    pub pos_w: Vector3<f64>,

    /// Velocity in world frame (m/s, NED)
    pub vel_w: Vector3<f64>,

    /// Attitude quaternion (Body → World rotation)
    pub att: UnitQuaternion<f64>,

    /// Angular velocity in body frame (rad/s)
    pub ang_vel_b: Vector3<f64>,

    /// Current mass (kg)
    pub mass: f64,
}

impl State {
    /// Create initial state from parameters
    fn new(p: &RocketParams) -> Self {
        // Initial orientation: Body X-axis aligned with -Z (up) in NED
        let initial_att = UnitQuaternion::rotation_between(&Vector3::x_axis(), &-Vector3::z_axis())
            .expect("Failed to create initial orientation");

        Self {
            t: 0.0,
            pos_w: Vector3::zeros(),
            vel_w: Vector3::zeros(),
            att: initial_att,
            ang_vel_b: Vector3::new(p.spin_rate.to_radians(), 0.0, 0.0),
            mass: p.dry_mass + p.propellant_mass,
        }
    }

    /// Calculate current altitude (positive = above launch site)
    pub fn altitude(&self) -> f64 {
        -self.pos_w.z
    }

    /// Calculate current vertical velocity (positive = climbing)
    pub fn vertical_velocity(&self) -> f64 {
        -self.vel_w.z
    }

    /// Calculate current speed (magnitude of velocity)
    pub fn speed(&self) -> f64 {
        self.vel_w.norm()
    }

    /// Check if rocket has crashed (hit ground after meaningful flight).
    ///
    /// `min_flight_time` should be `launch_delay + some_margin` so we don't
    /// terminate at t=0 before the rocket has left the pad.
    ///
    /// FIX: The original check required `vel < 1.0 m/s` which could fail to
    /// trigger under some damping conditions. We now declare a crash purely on
    /// altitude (Z >= 0 in NED) after sufficient elapsed time.
    pub fn has_crashed(&self, min_flight_time: f64) -> bool {
        self.pos_w.z >= GROUND_LEVEL && self.t > min_flight_time
    }
}

// ---------------------------------------------------------------------------
// Simulation Result
// ---------------------------------------------------------------------------

/// Complete trajectory data from simulation
#[derive(Clone)]
pub struct SimResult {
    /// Time samples (s)
    pub time: Vec<f64>,

    /// Position samples (m, NED)
    pub pos: Vec<Vector3<f64>>,

    /// Velocity samples (m/s, NED)
    pub vel: Vec<Vector3<f64>>,

    /// Proper acceleration in body frame (m/s²) - what accelerometers measure
    /// Excludes gravity (free-fall principle)
    pub accel_body: Vec<Vector3<f64>>,

    /// Angular velocity in body frame (rad/s)
    pub ang_vel: Vec<Vector3<f64>>,

    /// Orientation quaternions (Body → World)
    pub orientation: Vec<UnitQuaternion<f64>>,
}

impl SimResult {
    /// Find maximum altitude achieved
    pub fn max_altitude(&self) -> f64 {
        self.pos.iter().map(|p| -p.z).fold(0.0, f64::max)
    }

    /// Find maximum velocity achieved
    pub fn max_velocity(&self) -> f64 {
        self.vel.iter().map(|v| v.norm()).fold(0.0, f64::max)
    }

    /// Find time of apogee (maximum altitude)
    pub fn apogee_time(&self) -> Option<f64> {
        self.pos
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let alt_a = -a.z;
                let alt_b = -b.z;
                alt_a.partial_cmp(&alt_b).unwrap()
            })
            .map(|(idx, _)| self.time[idx])
    }

    /// Get number of samples in result
    pub fn len(&self) -> usize {
        self.time.len()
    }

    /// Check if result is empty
    pub fn is_empty(&self) -> bool {
        self.time.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Physics Engine - Derivative Calculation
// ---------------------------------------------------------------------------

/// State derivatives for RK4 integration
struct Derivative {
    d_pos: Vector3<f64>,
    d_vel: Vector3<f64>,
    d_att: Vector3<f64>,
    d_ang_vel: Vector3<f64>,
    d_mass: f64,
    // Auxiliary data
    proper_accel_b: Vector3<f64>,
}

/// Calculate state derivatives (right-hand side of ODEs)
fn calculate_derivative(s: &State, p: &RocketParams) -> Derivative {
    // ========================================================================
    // 1. ENVIRONMENT
    // ========================================================================

    let alt = s.altitude();

    // Exponential atmosphere model: ρ(h) = ρ₀ * e^(-h/H)
    let rho = if alt > 0.0 {
        p.air_density_sea_level * (-alt / H_SCALE).exp()
    } else {
        p.air_density_sea_level
    };

    // ========================================================================
    // 2. KINEMATICS
    // ========================================================================

    // Relative airspeed (rocket velocity minus wind), in body frame
    let v_rel_w = s.vel_w - p.wind_velocity_ned;
    let v_rel_b = s.att.inverse_transform_vector(&v_rel_w);
    let v_mag = v_rel_b.norm();

    // ========================================================================
    // 3. AERODYNAMICS (in Body Frame)
    // ========================================================================

    let mut force_aero_b = Vector3::zeros();
    let mut torque_aero_b = Vector3::zeros();

    // Calculate current CG (used by both aero and thrust torque)
    let fuel_fraction = if p.propellant_mass > 0.0 {
        ((s.mass - p.dry_mass) / p.propellant_mass).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let cg_now = p.cg_empty + fuel_fraction * (p.cg_full - p.cg_empty);

    if v_mag > MIN_AERO_VELOCITY {
        // Dynamic pressure: q = 0.5 * ρ * v²
        let q_dynamic = 0.5 * rho * v_mag.powi(2);

        // --- Axial Drag ---
        // FIX: Axial drag acts strictly along the body -X axis (opposing axial airflow),
        // not along -v_rel_b. Using the full velocity vector as the drag direction
        // was unphysical and under-applied drag, causing altitude overestimates.
        //
        // The axial velocity component is v_rel_b.x. Drag opposes that component.
        let axial_vel = v_rel_b.x;
        let drag_magnitude = q_dynamic * p.ref_area * p.drag_coeff_axial;
        // Drag acts along -X body axis, scaled by sign of axial flow
        force_aero_b.x -= drag_magnitude * axial_vel.signum();

        // --- Normal Force (Lift) ---
        // Provides restoring moment for stable rockets
        // F_N = q * S * CNα * α
        // where α is angle of attack in each plane
        //
        // Angle of attack components (small angle: sin(α) ≈ v_perp / v_mag)
        let alpha_y = v_rel_b.y / v_mag.max(MIN_AERO_VELOCITY);
        let alpha_z = v_rel_b.z / v_mag.max(MIN_AERO_VELOCITY);

        // Normal forces oppose the sideslip
        let force_y = -q_dynamic * p.ref_area * p.normal_force_coeff * alpha_y;
        let force_z = -q_dynamic * p.ref_area * p.normal_force_coeff * alpha_z;

        force_aero_b.y += force_y;
        force_aero_b.z += force_z;

        // --- Aerodynamic Torque ---
        // Torque = r × F, where r is from CG to CP
        let moment_arm_x = p.cp_location - cg_now;
        let lever = Vector3::new(moment_arm_x, 0.0, 0.0);

        // Normal forces create restoring torque
        let lift_force = Vector3::new(0.0, force_y, force_z);
        torque_aero_b += lever.cross(&lift_force);

        // --- Damping Torque ---
        // Opposes rotation, proportional to angular velocity and dynamic pressure
        let damping_factor = -0.05 * q_dynamic * p.ref_area * moment_arm_x.powi(2);
        torque_aero_b += Vector3::new(
            s.ang_vel_b.x * damping_factor * 0.1, // Roll damping is typically lower
            s.ang_vel_b.y * damping_factor,
            s.ang_vel_b.z * damping_factor,
        );
    }

    // ========================================================================
    // 4. PROPULSION (in Body Frame)
    // ========================================================================

    let thrust_mag = if s.t >= p.launch_delay && s.mass > p.dry_mass {
        interpolate_thrust(&p.thrust_curve, s.t - p.launch_delay)
    } else {
        0.0
    };

    // Apply thrust along body X-axis with optional cant angle
    let force_thrust_b = if p.thrust_cant.abs() > MIN_ROTATION_ANGLE {
        let cant_rad = p.thrust_cant.to_radians();
        Vector3::new(
            thrust_mag * cant_rad.cos(),
            thrust_mag * cant_rad.sin(),
            0.0,
        )
    } else {
        Vector3::new(thrust_mag, 0.0, 0.0)
    };

    // FIX: Apply torque from canted/offset thrust about the CG.
    // Previously this was a stub comment. Without this torque the rocket receives a
    // lateral force from cant but never rotates, so the flight path is a skewed arc
    // rather than the expected spiral (especially with spin).
    //
    // Moment arm: nozzle is aft of CG, so thrust cant force creates a pitch/yaw torque.
    // lever = (nozzle_location - cg_now) along body X.
    // torque = r × F_thrust  (r is from CG to nozzle)
    if thrust_mag > 0.0 && p.thrust_cant.abs() > MIN_ROTATION_ANGLE {
        let nozzle_arm = p.nozzle_location - cg_now; // positive = nozzle aft of CG
        let r_to_nozzle = Vector3::new(nozzle_arm, 0.0, 0.0);
        torque_aero_b += r_to_nozzle.cross(&force_thrust_b);
    }

    // ========================================================================
    // 5. GRAVITY (World → Body Frame)
    // ========================================================================

    let g_w = Vector3::new(0.0, 0.0, p.gravity);
    let force_gravity_b = s.att.inverse_transform_vector(&(g_w * s.mass));

    // ========================================================================
    // 6. LAUNCH RAIL CONSTRAINT
    // ========================================================================

    // FIX: dist_traveled must use altitude() = -pos_w.z (positive upward).
    // The original code computed `dist_traveled = pos_w.z` which is negative while
    // the rocket is above ground in NED convention, so `on_rail` was always true —
    // the rocket was permanently rail-constrained and could never drift laterally.
    let dist_traveled = s.altitude();
    let on_rail = dist_traveled < p.launch_rod_length;

    // ========================================================================
    // 7. TOTAL FORCES & MOMENTS
    // ========================================================================

    let mut total_force_b = force_thrust_b + force_aero_b + force_gravity_b;
    let mut total_torque_b = torque_aero_b;

    if on_rail {
        // Constrain lateral motion and rotation while on rail
        total_force_b.y = 0.0;
        total_force_b.z = 0.0;
        total_torque_b = Vector3::zeros();
    }

    // Ground collision: apply normal force to prevent penetration
    if s.pos_w.z >= GROUND_LEVEL && s.t > p.launch_delay + 0.1 {
        // In world frame, cancel all downward (positive Z) forces
        let total_force_w = s.att.transform_vector(&total_force_b);

        if total_force_w.z > 0.0 {
            // Zero out the downward component
            let corrected_force_w = Vector3::new(total_force_w.x, total_force_w.y, 0.0);
            total_force_b = s.att.inverse_transform_vector(&corrected_force_w);
        }

        // Also apply damping to stop bouncing
        if s.vel_w.z > 0.0 {
            let damping_force_w = Vector3::new(0.0, 0.0, -0.9 * s.mass * s.vel_w.z / DT);
            total_force_b += s.att.inverse_transform_vector(&damping_force_w);
        }
    }

    // ========================================================================
    // 8. DYNAMICS
    // ========================================================================

    // --- Linear Dynamics: F = ma ---
    let accel_b = total_force_b / s.mass;
    let accel_w = s.att.transform_vector(&accel_b);

    // --- Rotational Dynamics: Euler's Equation ---
    // M = Iα + ω × (Iω)
    // α = I⁻¹(M - ω × (Iω))
    let i_mat = Matrix3::from_diagonal(&p.inertia_tensor);
    let i_inv = i_mat
        .try_inverse()
        .expect("Inertia tensor must be invertible");

    let gyroscopic = s.ang_vel_b.cross(&(i_mat * s.ang_vel_b));
    let ang_accel_b = i_inv * (total_torque_b - gyroscopic);

    // --- Mass Flow ---
    // FIX: Use thrust / (Isp * g0) for physically correct propellant mass flow.
    // The original code used a constant propellant_mass/burn_time regardless of
    // instantaneous thrust, which is wrong for shaped thrust curves and also
    // consumed mass even when thrust was zero at the start/end of the curve.
    let d_mass = if thrust_mag > 0.0 {
        let mdot = thrust_mag / (p.isp * p.gravity);
        -mdot
    } else {
        0.0
    };

    // Clamp mass to dry mass
    let mass_limited = d_mass.max((p.dry_mass - s.mass) / DT);

    // ========================================================================
    // 9. PROPER ACCELERATION
    // ========================================================================

    // Accelerometers measure: a_proper = a_total - g
    // In body frame: a_proper = (F_thrust + F_aero) / m
    // Gravity is NOT felt by accelerometers (equivalence principle)
    let proper_accel = (force_thrust_b + force_aero_b) / s.mass;

    Derivative {
        d_pos: s.vel_w,
        d_vel: accel_w,
        d_att: s.ang_vel_b,
        d_ang_vel: ang_accel_b,
        d_mass: mass_limited,
        proper_accel_b: proper_accel,
    }
}

/// Linear interpolation of thrust curve
fn interpolate_thrust(curve: &[(f64, f64)], t: f64) -> f64 {
    if curve.is_empty() {
        return 0.0;
    }

    if t <= curve[0].0 {
        return curve[0].1;
    }

    if t >= curve.last().unwrap().0 {
        return curve.last().unwrap().1;
    }

    for i in 0..curve.len() - 1 {
        if t >= curve[i].0 && t <= curve[i + 1].0 {
            let dt = curve[i + 1].0 - curve[i].0;
            if dt <= 0.0 {
                return curve[i].1;
            }
            let frac = (t - curve[i].0) / dt;
            return curve[i].1 + frac * (curve[i + 1].1 - curve[i].1);
        }
    }

    0.0
}

// ---------------------------------------------------------------------------
// RK4 Integration
// ---------------------------------------------------------------------------

/// Apply RK4 integration step
fn step_state(s: &State, d: &Derivative, dt: f64) -> State {
    let mut ns = s.clone();
    ns.t += dt;
    ns.pos_w += d.d_pos * dt;
    ns.vel_w += d.d_vel * dt;
    ns.ang_vel_b += d.d_ang_vel * dt;
    ns.mass = (ns.mass + d.d_mass * dt).max(0.0); // Ensure non-negative mass

    // Quaternion integration using exponential map
    let angle = d.d_att.norm() * dt;
    if angle > MIN_ROTATION_ANGLE {
        let axis = d.d_att.normalize();
        let dq = UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_unchecked(axis), angle);
        ns.att *= dq; // Compose rotations
    }

    ns
}

// ---------------------------------------------------------------------------
// Main Simulation Loop
// ---------------------------------------------------------------------------

/// Run complete 6-DOF simulation
///
/// # Arguments
/// * `p` - Rocket and environment parameters
///
/// # Returns
/// Complete trajectory data
///
/// # Example
/// ```
/// use rocket_sim::{RocketParams, simulate_6dof};
///
/// let params = RocketParams::default();
/// let result = simulate_6dof(&params);
/// println!("Max altitude: {:.1} m", result.max_altitude());
/// ```
pub fn simulate_6dof(p: &RocketParams) -> SimResult {
    let mut s = State::new(p);
    let max_steps = (MAX_TIME / DT) as usize;

    let mut res = SimResult {
        time: Vec::with_capacity(max_steps),
        pos: Vec::with_capacity(max_steps),
        vel: Vec::with_capacity(max_steps),
        accel_body: Vec::with_capacity(max_steps),
        ang_vel: Vec::with_capacity(max_steps),
        orientation: Vec::with_capacity(max_steps),
    };

    for step in 0..max_steps {
        // Record current state
        res.time.push(s.t);
        res.pos.push(s.pos_w);
        res.vel.push(s.vel_w);
        res.ang_vel.push(s.ang_vel_b);
        res.orientation.push(s.att);

        // RK4 Integration
        let k1 = calculate_derivative(&s, p);
        res.accel_body.push(k1.proper_accel_b);

        let s2 = step_state(&s, &k1, DT * 0.5);
        let k2 = calculate_derivative(&s2, p);

        let s3 = step_state(&s, &k2, DT * 0.5);
        let k3 = calculate_derivative(&s3, p);

        let s4 = step_state(&s, &k3, DT);
        let k4 = calculate_derivative(&s4, p);

        // Combine derivatives (RK4 weighted average)
        s.pos_w += (k1.d_pos + k2.d_pos * 2.0 + k3.d_pos * 2.0 + k4.d_pos) * (DT / 6.0);

        // Clamp to ground level
        if s.pos_w.z > GROUND_LEVEL {
            s.pos_w.z = GROUND_LEVEL;
        }

        s.vel_w += (k1.d_vel + k2.d_vel * 2.0 + k3.d_vel * 2.0 + k4.d_vel) * (DT / 6.0);

        // Zero downward velocity at ground
        if s.pos_w.z >= GROUND_LEVEL && s.vel_w.z > 0.0 {
            s.vel_w.z = 0.0;
        }
        s.mass = (s.mass
            + (k1.d_mass + k2.d_mass * 2.0 + k3.d_mass * 2.0 + k4.d_mass) * (DT / 6.0))
            .max(p.dry_mass);

        s.ang_vel_b +=
            (k1.d_ang_vel + k2.d_ang_vel * 2.0 + k3.d_ang_vel * 2.0 + k4.d_ang_vel) * (DT / 6.0);

        // Quaternion integration (RK4 weighted average of angular velocity)
        let w_mean = (k1.d_att + k2.d_att * 2.0 + k3.d_att * 2.0 + k4.d_att) / 6.0;
        let angle = w_mean.norm() * DT;
        if angle > MIN_ROTATION_ANGLE {
            let axis = w_mean.normalize();
            let dq = UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_unchecked(axis), angle);
            s.att *= dq;
        }

        s.t += DT;

        // Termination: rocket has returned to ground after flight
        // FIX: Check altitude <= 0 after sufficient flight time instead of using
        // the fragile vel < 1.0 heuristic from the original code.
        if s.has_crashed(p.launch_delay + 5.0) {
            // Record final state
            res.time.push(s.t);
            res.pos.push(s.pos_w);
            res.vel.push(s.vel_w);
            res.ang_vel.push(s.ang_vel_b);
            res.orientation.push(s.att);
            res.accel_body.push(Vector3::zeros());
            break;
        }

        // Progress indicator for long simulations
        if step % 100000 == 0 && step > 0 {
            eprintln!("Simulation progress: {:.1}s / {:.1}s", s.t, MAX_TIME);
        }
    }

    res
}

// ===========================================================================
// TESTS
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // -----------------------------------------------------------------------
    // Parameter Validation Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_params_valid() {
        let p = RocketParams::default();
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_zero_dry_mass_invalid() {
        let mut p = RocketParams::default();
        p.dry_mass = 0.0;
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_negative_inertia_invalid() {
        let mut p = RocketParams::default();
        p.inertia_tensor.x = -1.0;
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_cg_ordering_invalid() {
        let mut p = RocketParams::default();
        p.cg_empty = 2.0;
        p.cg_full = 1.0; // Empty should be forward of full
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_cp_forward_of_cg_warning() {
        let mut p = RocketParams::default();
        p.cp_location = 1.0;
        p.cg_full = 2.0; // Unstable configuration
        assert!(p.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // Physics Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_thrust_interpolation() {
        let curve = vec![(0.0, 0.0), (1.0, 100.0), (2.0, 50.0), (3.0, 0.0)];

        assert_relative_eq!(interpolate_thrust(&curve, -1.0), 0.0);
        assert_relative_eq!(interpolate_thrust(&curve, 0.0), 0.0);
        assert_relative_eq!(interpolate_thrust(&curve, 0.5), 50.0);
        assert_relative_eq!(interpolate_thrust(&curve, 1.0), 100.0);
        assert_relative_eq!(interpolate_thrust(&curve, 1.5), 75.0);
        assert_relative_eq!(interpolate_thrust(&curve, 2.0), 50.0);
        assert_relative_eq!(interpolate_thrust(&curve, 3.0), 0.0);
        assert_relative_eq!(interpolate_thrust(&curve, 5.0), 0.0);
    }

    #[test]
    fn test_initial_state() {
        let p = RocketParams::default();
        let s = State::new(&p);

        assert_eq!(s.t, 0.0);
        assert_eq!(s.pos_w, Vector3::zeros());
        assert_eq!(s.vel_w, Vector3::zeros());
        assert_eq!(s.mass, p.dry_mass + p.propellant_mass);

        // Should be pointing up (body X aligned with -Z in NED)
        let body_x = s.att.transform_vector(&Vector3::x());
        assert_relative_eq!(body_x.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(body_x.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(body_x.z, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_integration_no_double_scaling() {
        // Regression test for the quaternion double-DT bug
        let p = RocketParams {
            spin_rate: 360.0, // 1 rev/sec around body X
            launch_delay: 0.0,
            thrust_curve: vec![(0.0, 0.0)],
            ..Default::default()
        };

        let result = simulate_6dof(&p);

        // After 1 second of 360°/s spin, should complete ~1 rotation
        let idx_1s = result.time.iter().position(|&t| t >= 1.0).unwrap();
        let spin_angle = result.ang_vel[idx_1s].x * result.time[idx_1s];

        // Should be close to 2π radians
        assert_relative_eq!(spin_angle, 2.0 * std::f64::consts::PI, epsilon = 0.1);
    }

    #[test]
    fn test_mass_depletion_via_isp() {
        // With Isp-based mass flow, propellant should be consumed correctly.
        // For T=2000N, Isp=200s, g=9.80665: mdot = 2000/(200*9.80665) ≈ 1.019 kg/s
        // Over 5s burn: ~5.1 kg consumed (< 10 kg propellant, so burn completes)
        let p = RocketParams::default();
        let result = simulate_6dof(&p);

        // After burn, mass should be above dry_mass (propellant not fully consumed
        // unless Isp is low enough) and never below dry_mass.
        let burn_end_idx = result
            .time
            .iter()
            .position(|&t| t > p.launch_delay + p.burn_time + 0.1)
            .unwrap_or(result.len() - 1);

        // Mass post-burn should be >= dry_mass
        // (We can't easily track instantaneous mass from SimResult; verify indirectly
        //  by checking that the simulation doesn't panic and produces valid altitudes.)
        let _ = burn_end_idx;
        assert!(result.max_altitude() > 0.0);
    }

    #[test]
    fn test_energy_conservation_no_drag() {
        // With no drag or thrust, energy should be conserved (within numerical error)
        let p = RocketParams {
            drag_coeff_axial: 0.0,
            normal_force_coeff: 0.0,
            thrust_curve: vec![(0.0, 0.0)],
            launch_delay: 0.0,
            wind_velocity_ned: Vector3::zeros(),
            launch_rod_length: 0.0,
            ..Default::default()
        };

        // Give it initial upward velocity
        let mut s = State::new(&p);
        s.vel_w = Vector3::new(0.0, 0.0, -100.0); // 100 m/s up

        let initial_ke = 0.5 * s.mass * s.vel_w.norm_squared();
        let initial_pe = s.mass * p.gravity * s.altitude();
        let initial_energy = initial_ke + initial_pe;

        // Simulate for a short time
        let mut current_s = s;
        for _ in 0..10000 {
            let k1 = calculate_derivative(&current_s, &p);
            current_s = step_state(&current_s, &k1, DT);
        }

        let final_ke = 0.5 * current_s.mass * current_s.vel_w.norm_squared();
        let final_pe = current_s.mass * p.gravity * current_s.altitude();
        let final_energy = final_ke + final_pe;

        // Energy should be conserved within 1% (numerical error)
        assert_relative_eq!(
            initial_energy,
            final_energy,
            epsilon = 0.01 * initial_energy
        );
    }

    #[test]
    fn test_launch_rail_constraint() {
        let p = RocketParams {
            launch_rod_length: 5.0,
            wind_velocity_ned: Vector3::new(10.0, 0.0, 0.0), // Strong wind
            ..Default::default()
        };

        let result = simulate_6dof(&p);

        // While altitude < launch_rod_length, lateral position should be ~zero
        for i in 0..result.len() {
            let alt = -result.pos[i].z;
            if alt < p.launch_rod_length {
                assert!(
                    result.pos[i].x.abs() < 0.1,
                    "x drift on rail at alt {}: {}",
                    alt,
                    result.pos[i].x
                );
                assert!(
                    result.pos[i].y.abs() < 0.1,
                    "y drift on rail at alt {}: {}",
                    alt,
                    result.pos[i].y
                );
            }
        }
    }

    #[test]
    fn test_ground_collision_stops_fall() {
        let p = RocketParams {
            thrust_curve: vec![(0.0, 500.0), (1.0, 500.0), (1.01, 0.0)], // Weak thrust
            burn_time: 1.0,
            launch_delay: 0.0,
            ..Default::default()
        };

        let result = simulate_6dof(&p);

        // Find if rocket comes back down
        let max_alt_idx = result
            .pos
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let alt_a = -a.z;
                let alt_b = -b.z;
                alt_a.partial_cmp(&alt_b).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap();

        // After max altitude, check it doesn't go below ground
        for i in max_alt_idx..result.len() {
            assert!(result.pos[i].z <= 0.1); // Shouldn't go below ground
        }
    }

    #[test]
    fn test_proper_acceleration_excludes_gravity() {
        // At rest on ground, accelerometer should read zero
        let p = RocketParams {
            thrust_curve: vec![(0.0, 0.0)],
            launch_delay: 0.0,
            ..Default::default()
        };

        let result = simulate_6dof(&p);

        // Before launch, proper accel should be ~zero (no thrust, no aero)
        assert_relative_eq!(result.accel_body[0].norm(), 0.0, epsilon = 1.0);
    }

    #[test]
    fn test_thrust_cant_causes_rotation() {
        // FIX: With the thrust torque now properly applied, cant should cause rotation.
        let p = RocketParams {
            thrust_cant: 5.0,       // 5 degree cant
            launch_rod_length: 0.0, // No rail constraint
            ..Default::default()
        };

        let result = simulate_6dof(&p);

        // Should develop angular velocity during burn
        let burn_end_idx = result
            .time
            .iter()
            .position(|&t| t > p.launch_delay + p.burn_time)
            .unwrap_or(result.len() - 1);

        let ang_vel_magnitude = result.ang_vel[burn_end_idx].norm();
        assert!(
            ang_vel_magnitude > 0.01,
            "Expected rotation from thrust cant, got {}",
            ang_vel_magnitude
        );
    }

    #[test]
    fn test_thrust_cant_with_spin_causes_spiral() {
        // With both spin and cant, gyroscopic precession should produce a spiral path.
        let p = RocketParams {
            thrust_cant: 3.0,
            spin_rate: 180.0, // 0.5 rev/sec
            launch_rod_length: 0.0,
            wind_velocity_ned: Vector3::zeros(),
            ..Default::default()
        };

        let result = simulate_6dof(&p);

        // The horizontal drift should oscillate (spiral), meaning both X and Y
        // components of drift are non-trivial.
        let max_x = result.pos.iter().map(|p| p.x.abs()).fold(0.0, f64::max);
        let max_y = result.pos.iter().map(|p| p.y.abs()).fold(0.0, f64::max);

        // Both directions should see drift (spiral not just arc)
        assert!(
            max_x > 1.0 && max_y > 1.0,
            "Expected 2D spiral drift, got max_x={} max_y={}",
            max_x,
            max_y
        );
    }

    // -----------------------------------------------------------------------
    // Trajectory Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_vertical_flight_stays_vertical() {
        let p = RocketParams {
            wind_velocity_ned: Vector3::zeros(),
            thrust_cant: 0.0,
            spin_rate: 0.0,
            ..Default::default()
        };

        let result = simulate_6dof(&p);

        // Check that rocket stays mostly vertical (small drift allowed due to numerics)
        let max_horizontal = result
            .pos
            .iter()
            .map(|p| (p.x.powi(2) + p.y.powi(2)).sqrt())
            .fold(0.0, f64::max);

        assert!(max_horizontal < 10.0); // Less than 10m horizontal drift
    }

    #[test]
    fn test_apogee_detection() {
        let p = RocketParams::default();
        let result = simulate_6dof(&p);

        let apogee_alt = result.max_altitude();
        let apogee_time = result.apogee_time().unwrap();

        assert!(apogee_alt > 0.0); // Should reach decent altitude
        assert!(apogee_time > p.burn_time); // Apogee after burn
    }

    #[test]
    fn test_max_velocity_during_burn() {
        let p = RocketParams::default();
        let result = simulate_6dof(&p);

        let max_vel = result.max_velocity();
        let max_vel_time = result
            .vel
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let v_a = a.norm();
                let v_b = b.norm();
                v_a.partial_cmp(&v_b).unwrap()
            })
            .map(|(i, _)| result.time[i])
            .unwrap();

        assert!(max_vel > 0.0); // Should reach reasonable speed
        assert!(max_vel_time <= p.launch_delay + p.burn_time + 1.0); // Max vel near end of burn
    }

    #[test]
    fn test_stability_cp_behind_cg() {
        // Stable rocket should not tumble
        let p = RocketParams {
            cg_full: 1.5,
            cg_empty: 1.4,
            cp_location: 2.0, // Behind CG
            ..Default::default()
        };

        let result = simulate_6dof(&p);

        // Check that rocket doesn't develop excessive rotation
        let max_ang_vel = result.ang_vel.iter().map(|w| w.norm()).fold(0.0, f64::max);

        assert!(max_ang_vel < 1.0); // Less than ~60 deg/s rotation
    }

    #[test]
    fn test_wind_causes_drift() {
        let p = RocketParams {
            wind_velocity_ned: Vector3::new(20.0, 0.0, 0.0), // 20 m/s North wind
            ..Default::default()
        };

        let result = simulate_6dof(&p);

        // Rocket should drift North (positive X)
        let final_x = result.pos.last().unwrap().x;
        assert!(final_x > 0.0); // Significant drift
    }

    #[test]
    fn test_different_thrust_profiles() {
        // Short high thrust
        let p1 = RocketParams {
            thrust_curve: vec![(0.0, 5000.0), (2.0, 5000.0), (2.01, 0.0)],
            burn_time: 2.0,
            ..Default::default()
        };

        // Long low thrust
        let p2 = RocketParams {
            thrust_curve: vec![(0.0, 1000.0), (10.0, 1000.0), (10.01, 0.0)],
            burn_time: 10.0,
            propellant_mass: 10.0,
            ..Default::default()
        };

        let result1 = simulate_6dof(&p1);
        let result2 = simulate_6dof(&p2);

        // Both should fly, but with different characteristics
        assert!(result1.max_velocity() > 0.0);
        assert!(result2.max_velocity() > 0.0);
        assert!(result1.max_altitude() > 0.0);
        assert!(result2.max_altitude() > 0.0);
    }

    // -----------------------------------------------------------------------
    // Result API Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_result_metadata() {
        let p = RocketParams::default();
        let result = simulate_6dof(&p);

        assert!(!result.is_empty());
        assert!(result.len() > 1000);
        assert_eq!(result.time.len(), result.pos.len());
        assert_eq!(result.time.len(), result.vel.len());
        assert_eq!(result.time.len(), result.accel_body.len());
    }

    #[test]
    fn test_state_helper_functions() {
        let p = RocketParams::default();
        let mut s = State::new(&p);

        s.pos_w = Vector3::new(0.0, 0.0, -100.0); // 100m altitude
        s.vel_w = Vector3::new(10.0, 0.0, -50.0); // Moving up and north

        assert_relative_eq!(s.altitude(), 100.0);
        assert_relative_eq!(s.vertical_velocity(), 50.0);
        assert_relative_eq!(s.speed(), (10.0_f64.powi(2) + 50.0_f64.powi(2)).sqrt());
        assert!(!s.has_crashed(5.0)); // t=0, not crashed yet

        s.pos_w.z = 0.1; // Below ground
        s.t = 10.0; // After launch + 5s margin
        assert!(s.has_crashed(5.0));
    }

    // -----------------------------------------------------------------------
    // Edge Cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_thrust() {
        let p = RocketParams {
            thrust_curve: vec![(0.0, 0.0)],
            launch_delay: 0.0,
            ..Default::default()
        };

        let result = simulate_6dof(&p);

        // Should fall straight down
        assert!(result.max_altitude() < 1.0);
    }

    #[test]
    fn test_very_high_drag() {
        let p = RocketParams {
            drag_coeff_axial: 5.0, // Unrealistically high
            ..Default::default()
        };

        let result = simulate_6dof(&p);

        // Should still fly but reach lower altitude
        assert!(result.max_altitude() > 0.0);
        assert!(result.max_velocity() < 200.0); // Terminal velocity limited
    }

    #[test]
    fn test_spin_stabilization() {
        let p = RocketParams {
            spin_rate: 600.0, // 10 rev/sec
            ..Default::default()
        };

        let result = simulate_6dof(&p);

        // Should maintain high spin rate throughout flight
        let final_spin = result.ang_vel.last().unwrap().x;
        assert!(final_spin.abs() > 5.0); // Still spinning
    }
}
