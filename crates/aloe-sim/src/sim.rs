use nalgebra::{Matrix3, UnitQuaternion, Vector3};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const DT: f64 = 0.001; // 1kHz physics for stability
const MAX_TIME: f64 = 300.0;
const H_SCALE: f64 = 7400.0;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct RocketParams {
    // Mass Properties
    pub dry_mass: f64,                // kg
    pub propellant_mass: f64,         // kg
    pub inertia_tensor: Vector3<f64>, // [Ix, Iy, Iz] diagonal (kg*m^2)
    pub cg_location: f64,             // Meters from nose (positive X)

    // Aerodynamics
    pub cp_location: f64,        // Meters from nose (positive X)
    pub ref_area: f64,           // m^2 (cross section)
    pub drag_coeff_axial: f64,   // Cd (Axial)
    pub normal_force_coeff: f64, // Cna (Lift slope per radian of AoA)

    // Propulsion
    pub thrust_curve: Vec<(f64, f64)>, // (time, force) points
    pub burn_time: f64,
    pub nozzle_location: f64, // Meters from nose (usually length of rocket)

    // Environment
    pub gravity: f64,                    // m/s²
    pub air_density_sea_level: f64,      // kg/m³ at sea level
    pub launch_rod_length: f64,          // m
    pub wind_velocity_ned: Vector3<f64>, // m/s
    pub launch_delay: f64,               // s
    pub spin_rate: f64,                  // °/s
    pub thrust_cant: f64,                // °
}

impl Default for RocketParams {
    fn default() -> Self {
        Self {
            dry_mass: 20.0,
            propellant_mass: 10.0,
            inertia_tensor: Vector3::new(0.1, 10.0, 10.0), // Long thin rod approx
            cg_location: 1.5,
            cp_location: 2.0, // Stable: CP behind CG
            ref_area: 0.018,
            drag_coeff_axial: 0.5,
            normal_force_coeff: 12.0, // Typical for fins
            thrust_curve: vec![(0.0, 2000.0), (5.0, 2000.0), (5.1, 0.0)],
            burn_time: 5.0,
            nozzle_location: 3.0,
            gravity: 9.80665,
            air_density_sea_level: 1.225,
            launch_rod_length: 2.0,
            wind_velocity_ned: Vector3::new(5.0, 0.0, 0.0), // 5m/s North wind
            launch_delay: 1.0,
            spin_rate: 0.0,
            thrust_cant: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// 6-DoF State
// ---------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub struct State {
    pub t: f64,
    pub pos_w: Vector3<f64>,      // Position (World NED)
    pub vel_w: Vector3<f64>,      // Velocity (World NED)
    pub att: UnitQuaternion<f64>, // Attitude (Body -> World)
    pub ang_vel_b: Vector3<f64>,  // Angular Velocity (Body Frame)
    pub mass: f64,
}

impl State {
    fn new(p: &RocketParams) -> Self {
        // Initial orientation: Pointing Up (-Z in NED)
        // Rotate vector (1,0,0) [Body X] to (0,0,-1) [World -Z]
        let initial_att =
            UnitQuaternion::rotation_between(&Vector3::x_axis(), &-Vector3::z_axis()).unwrap();

        Self {
            t: 0.0,
            pos_w: Vector3::zeros(), // Launch at 0,0,0
            vel_w: Vector3::zeros(),
            att: initial_att,
            ang_vel_b: Vector3::new(p.spin_rate.to_radians(), 0.0, 0.0), // Spin around X axis
            mass: p.dry_mass + p.propellant_mass,
        }
    }
}

// ---------------------------------------------------------------------------
// Simulation Result
// ---------------------------------------------------------------------------
#[derive(Clone)]
pub struct SimResult {
    pub time: Vec<f64>,
    pub pos: Vec<Vector3<f64>>,
    pub vel: Vec<Vector3<f64>>,
    pub accel_body: Vec<Vector3<f64>>, // Proper acceleration (what accelerometer sees)
    pub ang_vel: Vec<Vector3<f64>>,
    pub orientation: Vec<UnitQuaternion<f64>>,
}

// ---------------------------------------------------------------------------
// Physics Engine
// ---------------------------------------------------------------------------
struct Derivative {
    d_pos: Vector3<f64>,
    d_vel: Vector3<f64>,
    d_att: Vector3<f64>, // Angular velocity vector for quaternion integration
    d_ang_vel: Vector3<f64>,
    d_mass: f64,
    // Aux
    proper_accel_b: Vector3<f64>,
}

fn calculate_derivative(s: &State, p: &RocketParams) -> Derivative {
    // 1. Environment
    let alt = -s.pos_w.z;
    let rho = if alt > 0.0 {
        p.air_density_sea_level * (-alt / H_SCALE).exp()
    } else {
        p.air_density_sea_level
    };

    // 2. Kinematics
    // Relative air velocity in Body Frame
    // v_rel = v_rocket - v_wind
    let v_rel_w = s.vel_w - p.wind_velocity_ned;
    let v_rel_b = s.att.inverse_transform_vector(&v_rel_w);
    let v_mag = v_rel_b.norm();

    // 3. Aerodynamics (Body Frame)
    let mut force_aero_b = Vector3::zeros();
    let mut torque_aero_b = Vector3::zeros();

    if v_mag > 0.1 {
        let q_dynamic = 0.5 * rho * v_mag.powi(2);

        // Axial Drag (X-axis)
        // F_drag = q * S * Cd * -direction
        // Technically depends on angle of attack, simplified here to component
        let axial_drag = -q_dynamic * p.ref_area * p.drag_coeff_axial * (v_rel_b.x / v_mag);
        force_aero_b.x += axial_drag;

        // Normal Lift (Y/Z axes) - Restoring Force
        // Angle of Attack (alpha) approximation: v_lateral / v_axial
        // F_normal = q * S * Cna * alpha
        let alpha_y = v_rel_b.y / v_mag; // Sideslip
        let alpha_z = v_rel_b.z / v_mag; // Angle of attack

        let force_y = -q_dynamic * p.ref_area * p.normal_force_coeff * alpha_y;
        let force_z = -q_dynamic * p.ref_area * p.normal_force_coeff * alpha_z;

        force_aero_b.y += force_y;
        force_aero_b.z += force_z;

        // Aerodynamic Torque
        // Acts at CP. Torque = r x F
        // r = vector from CG to CP
        // We assume CG moves but CP is constant.
        let cg_now = if s.mass > p.dry_mass {
            // Simple linear CG shift model
            let fuel_ratio = (s.mass - p.dry_mass) / p.propellant_mass;
            p.cg_location * 0.9 + (p.cg_location + 0.5) * 0.1 * fuel_ratio
        } else {
            p.cg_location
        };

        let arm_x = p.cp_location - cg_now; // positive if CP is behind CG
        let lever = Vector3::new(arm_x, 0.0, 0.0);

        // Only normal forces create torque (axial drag passes through axis)
        let lift_force = Vector3::new(0.0, force_y, force_z);
        torque_aero_b += lever.cross(&lift_force);

        // Damping Torque (Rotational Drag)
        // T_damp = -k * omega * v^2 ?? Simplified: -constant * omega * q
        let damping_factor = -0.05 * q_dynamic * p.ref_area * arm_x.powi(2);
        torque_aero_b += Vector3::new(
            s.ang_vel_b.x * damping_factor * 0.1, // Roll damping is usually lower
            s.ang_vel_b.y * damping_factor,
            s.ang_vel_b.z * damping_factor,
        );
    }

    // 4. Propulsion (Body Frame)
    let thrust_mag = if s.t >= p.launch_delay {
        interpolate_thrust(&p.thrust_curve, s.t - p.launch_delay)
    } else {
        0.0
    };
    let mut force_thrust_b = Vector3::new(thrust_mag, 0.0, 0.0);
    // Apply thrust cant
    if p.thrust_cant.abs() > 1e-6 {
        let cant_angle = p.thrust_cant.to_radians();
        // Cant around Z axis (yaw)
        force_thrust_b.y = thrust_mag * cant_angle.sin();
        force_thrust_b.x = thrust_mag * cant_angle.cos();
    }
    // Assuming thrust through centerline, so 0 torque unless gimballing added

    // 5. Gravity (World Frame -> Body Frame)
    let g_w = Vector3::new(0.0, 0.0, p.gravity); // Down
    let force_gravity_b = s.att.inverse_transform_vector(&(g_w * s.mass));

    // 6. Launch Rod Constraint
    // If on rod, cancel lateral forces and moments
    let dist_traveled = s.pos_w.norm(); // Approximate
    let on_rod = dist_traveled < p.launch_rod_length;

    // 7. Total Forces & Moments
    let mut total_force_b = force_thrust_b + force_aero_b + force_gravity_b;
    let mut total_torque_b = torque_aero_b;

    if on_rod {
        // Constrain lateral motion and rotation
        total_force_b.y = 0.0;
        total_force_b.z = 0.0;
        total_torque_b = Vector3::zeros();
    }

    // Normal force constraint at ground (prevent falling through earth)
    if s.pos_w.z >= 0.0 && total_force_b.x < 0.0 {
        // Simplified ground clamp
    }

    // 8. Dynamics
    // Linear: F = ma
    let accel_b = total_force_b / s.mass;
    let accel_w = s.att.transform_vector(&accel_b);

    // Rotational: Euler's Eq: M = I * alpha + w x (I * w)
    // alpha = I_inv * (M - w x (I * w))
    let i_mat = Matrix3::from_diagonal(&p.inertia_tensor);
    let i_inv = i_mat.try_inverse().unwrap();
    let gyroscopic = s.ang_vel_b.cross(&(i_mat * s.ang_vel_b));
    let ang_accel_b = i_inv * (total_torque_b - gyroscopic);

    // Mass flow
    let d_mass = if thrust_mag > 0.0 {
        -p.propellant_mass / p.burn_time
    } else {
        0.0
    };

    // 9. Proper Acceleration (What accelerometers measure)
    // a_meas = a_body - g_body (Rotation handled implicitly)
    // Actually simpler: a_meas = (F_aero + F_thrust) / m
    // Gravity is NOT felt by accelerometers (freefall principle)
    let proper_accel = (force_thrust_b + force_aero_b) / s.mass;

    Derivative {
        d_pos: s.vel_w,
        d_vel: accel_w,
        d_att: s.ang_vel_b,
        d_ang_vel: ang_accel_b,
        d_mass,
        proper_accel_b: proper_accel,
    }
}

// Helper: Linear interpolation for thrust curve
fn interpolate_thrust(curve: &[(f64, f64)], t: f64) -> f64 {
    if t < 0.0 || t > curve.last().unwrap().0 {
        return 0.0;
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
// Main Loop
// ---------------------------------------------------------------------------
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

    for _ in 0..max_steps {
        // Record
        res.time.push(s.t);
        res.pos.push(s.pos_w);
        res.vel.push(s.vel_w);
        res.ang_vel.push(s.ang_vel_b);
        res.orientation.push(s.att);

        // RK4 Step
        let k1 = calculate_derivative(&s, p);

        // Log proper acceleration from k1 (start of step)
        res.accel_body.push(k1.proper_accel_b);

        let s2 = step_state(&s, &k1, DT * 0.5);
        let k2 = calculate_derivative(&s2, p);

        let s3 = step_state(&s, &k2, DT * 0.5);
        let k3 = calculate_derivative(&s3, p);

        let s4 = step_state(&s, &k3, DT);
        let k4 = calculate_derivative(&s4, p);

        // Combine
        s.pos_w += (k1.d_pos + k2.d_pos * 2.0 + k3.d_pos * 2.0 + k4.d_pos) * (DT / 6.0);
        s.vel_w += (k1.d_vel + k2.d_vel * 2.0 + k3.d_vel * 2.0 + k4.d_vel) * (DT / 6.0);
        s.mass += (k1.d_mass + k2.d_mass * 2.0 + k3.d_mass * 2.0 + k4.d_mass) * (DT / 6.0);

        // Angular Velocity integration
        s.ang_vel_b +=
            (k1.d_ang_vel + k2.d_ang_vel * 2.0 + k3.d_ang_vel * 2.0 + k4.d_ang_vel) * (DT / 6.0);

        // Quaternion integration
        // dq/dt = 0.5 * q * omega
        // We use mean angular velocity for the step to propagate rotation
        let w_mean = (k1.d_att + k2.d_att * 2.0 + k3.d_att * 2.0 + k4.d_att) * (DT / 6.0);
        let angle = w_mean.norm() * DT;
        if angle > 1e-9 {
            let axis = w_mean.normalize();
            let dq = UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_unchecked(axis), angle);
            s.att *= dq; // Local update
        }

        s.t += DT;

        // Termination: Crash
        if s.pos_w.z > 0.0 && s.t > 1.0 {
            break;
        }
    }

    res
}

fn step_state(s: &State, d: &Derivative, dt: f64) -> State {
    let mut ns = s.clone();
    ns.t += dt;
    ns.pos_w += d.d_pos * dt;
    ns.vel_w += d.d_vel * dt;
    ns.ang_vel_b += d.d_ang_vel * dt;
    ns.mass += d.d_mass * dt;

    // Simple Euler step for quaternion in intermediate RK stages
    let angle = d.d_att.norm() * dt;
    if angle > 1e-9 {
        let axis = d.d_att.normalize();
        let dq = UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_unchecked(axis), angle);
        ns.att *= dq;
    }
    ns
}
