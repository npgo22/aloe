//! # Rocket Flight Simulation
//!
//! A high-performance 3-DoF (translational) rocket flight simulator that
//! integrates the equations of motion using a simple forward-Euler scheme at
//! 10 ms time steps.
//!
//! ## Physics model
//!
//! The simulator models the following forces on a rigid rocket:
//!
//! | Force       | Description |
//! |-------------|-------------|
//! | **Thrust**  | Constant magnitude along the body axis. Optionally canted at [`RocketParams::thrust_cant`] degrees and rotated by [`RocketParams::spin_rate`] to produce a helical trajectory. |
//! | **Gravity** | Acts downward (−Y) with magnitude `m · g`. |
//! | **Drag**    | Quadratic drag opposing the velocity vector: `½ ρ v² Cd A`. |
//! | **Wind**    | Constant lateral forces proportional to wind speed along X and Z. |
//!
//! ## Coordinate system
//!
//! * **X** – downrange (north)
//! * **Y** – altitude (up)
//! * **Z** – crosswind (east)
//!
//! ## Example (from Python via PyO3)
//!
//! ```python
//! from aloe_core import simulate_rocket_rs
//!
//! result = simulate_rocket_rs(
//!     dry_mass=50.0,
//!     propellant_mass=150.0,
//!     thrust=15000.0,
//!     burn_time=25.0,
//!     drag_coeff=0.40,
//!     ref_area=0.03,
//!     gravity=9.81,
//!     wind_speed=3.0,
//!     wind_speed_z=0.0,
//!     air_density=1.225,
//!     launch_delay=1.0,
//!     spin_rate=0.0,
//!     thrust_cant=0.0,
//! )
//! # result is a dict of str → list[float]
//! ```

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Integration time step in seconds (100 Hz).
const DT: f64 = 0.01;

/// Maximum simulation duration in seconds.
const MAX_TIME: f64 = 300.0;

/// Input parameters for the rocket flight simulation.
///
/// All fields mirror the Python [`RocketParams`](../../aloe/sim.py) dataclass.
/// See module-level docs for the physics model.
#[derive(Debug, Clone)]
pub struct RocketParams {
    /// Mass of the rocket without propellant (kg).
    pub dry_mass: f64,
    /// Mass of the propellant (kg).
    pub propellant_mass: f64,
    /// Engine thrust force (N).
    pub thrust: f64,
    /// Duration the engine fires (s).
    pub burn_time: f64,
    /// Aerodynamic drag coefficient (dimensionless).
    pub drag_coeff: f64,
    /// Reference cross-sectional area for drag (m²).
    pub ref_area: f64,
    /// Gravitational acceleration (m/s²).
    pub gravity: f64,
    /// Wind speed along X axis (m/s).
    pub wind_speed: f64,
    /// Crosswind speed along Z axis (m/s).
    pub wind_speed_z: f64,
    /// Ambient air density (kg/m³).
    pub air_density: f64,
    /// Pre-launch idle time on the pad (s).
    pub launch_delay: f64,
    /// Rocket roll rate around longitudinal axis (°/s).
    pub spin_rate: f64,
    /// Thrust vector cant angle from longitudinal axis (°).
    pub thrust_cant: f64,
}

/// Result of a single simulation run.
///
/// Each field is a `Vec<f64>` of length equal to the number of time steps
/// completed (simulation terminates early when the rocket returns to ground).
///
/// Column names match the Polars DataFrame columns produced by the Python
/// `simulate_rocket()` function.
#[derive(Debug, Clone)]
pub struct SimResult {
    /// Simulation time (s).
    pub time_s: Vec<f64>,
    /// Position along the X (downrange / north) axis (m).
    pub position_x_m: Vec<f64>,
    /// Altitude / Y position (m). Always ≥ 0 during flight.
    pub altitude_m: Vec<f64>,
    /// Position along the Z (crosswind / east) axis (m).
    pub position_z_m: Vec<f64>,
    /// Velocity along X (m/s).
    pub velocity_x_ms: Vec<f64>,
    /// Velocity along Y (m/s). Positive = upward.
    pub velocity_y_ms: Vec<f64>,
    /// Velocity along Z (m/s).
    pub velocity_z_ms: Vec<f64>,
    /// Acceleration along X (m/s²).
    pub acceleration_x_ms2: Vec<f64>,
    /// Acceleration along Y (m/s²).
    pub acceleration_y_ms2: Vec<f64>,
    /// Acceleration along Z (m/s²).
    pub acceleration_z_ms2: Vec<f64>,
    /// Current vehicle mass (kg). Decreases linearly during the burn.
    pub mass_kg: Vec<f64>,
    /// Aerodynamic drag force magnitude (N).
    pub drag_force_n: Vec<f64>,
    /// Thrust force magnitude (N). Zero outside the burn window.
    pub thrust_n: Vec<f64>,
    /// Spin rate around the body axis (°/s). Constant.
    pub spin_rate_dps: Vec<f64>,
}

/// Run the rocket flight simulation with the given parameters.
///
/// This is the core physics loop. It uses forward-Euler integration at a
/// fixed 10 ms time step ([`DT`]) and terminates when either:
///
/// 1. The rocket returns to ground (`altitude < 0` after launch), or
/// 2. The maximum simulation time ([`MAX_TIME`] = 300 s) is reached.
///
/// # Arguments
///
/// * `p` – Physical and environmental parameters (see [`RocketParams`]).
///
/// # Returns
///
/// A [`SimResult`] containing time-series arrays for all state variables.
///
/// # Physics
///
/// At each time step the simulator:
///
/// 1. Computes the current vehicle mass (constant on pad, linearly decreasing
///    during burn, constant after burnout).
/// 2. Determines thrust: a constant force directed upward (Y), optionally
///    canted and rotated by the spin rate to sweep out a cone.
/// 3. Computes quadratic aerodynamic drag opposing the velocity vector.
/// 4. Applies constant wind forces along X and Z.
/// 5. Sums forces, divides by mass → acceleration.
/// 6. Integrates velocity and position (forward Euler).
pub fn simulate(p: &RocketParams) -> SimResult {
    let max_steps = (MAX_TIME / DT) as usize;

    // Pre-allocate output arrays with estimated capacity.
    // Typical flight ≈ 85 s → ~8 500 steps, but we allocate for max.
    let mut r = SimResult {
        time_s: Vec::with_capacity(max_steps),
        position_x_m: Vec::with_capacity(max_steps),
        altitude_m: Vec::with_capacity(max_steps),
        position_z_m: Vec::with_capacity(max_steps),
        velocity_x_ms: Vec::with_capacity(max_steps),
        velocity_y_ms: Vec::with_capacity(max_steps),
        velocity_z_ms: Vec::with_capacity(max_steps),
        acceleration_x_ms2: Vec::with_capacity(max_steps),
        acceleration_y_ms2: Vec::with_capacity(max_steps),
        acceleration_z_ms2: Vec::with_capacity(max_steps),
        mass_kg: Vec::with_capacity(max_steps),
        drag_force_n: Vec::with_capacity(max_steps),
        thrust_n: Vec::with_capacity(max_steps),
        spin_rate_dps: Vec::with_capacity(max_steps),
    };

    // State variables
    let mut t: f64 = 0.0;
    let mut x: f64 = 0.0;
    let mut y: f64 = 0.0;
    let mut z: f64 = 0.0;
    let mut vx: f64 = 0.0;
    let mut vy: f64 = 0.0;
    let mut vz: f64 = 0.0;

    let cant_rad = p.thrust_cant.to_radians();
    let spin_rad_s = p.spin_rate.to_radians();
    let mass_flow = p.propellant_mass / p.burn_time;
    let wind_fx = p.wind_speed * 0.5;
    let wind_fz = p.wind_speed_z * 0.5;
    let half_rho_cd_a = 0.5 * p.air_density * p.drag_coeff * p.ref_area;

    for _step in 0..max_steps {
        let t_ign = t - p.launch_delay;
        let on_pad = t_ign < 0.0;

        // ── Mass ──────────────────────────────────────────────
        let current_mass = if on_pad {
            p.dry_mass + p.propellant_mass
        } else if t_ign > p.burn_time {
            p.dry_mass
        } else {
            p.dry_mass + p.propellant_mass - mass_flow * t_ign
        };

        // ── Thrust ────────────────────────────────────────────
        let thrust_mag = if t_ign >= 0.0 && t_ign <= p.burn_time {
            p.thrust
        } else {
            0.0
        };

        // Thrust vector: optionally canted + spinning
        let (thrust_x, thrust_y, thrust_z) =
            if thrust_mag > 0.0 && cant_rad > 0.0 && p.spin_rate != 0.0 {
                let phase = spin_rad_s * t_ign;
                (
                    thrust_mag * libm::sin(cant_rad) * libm::cos(phase),
                    thrust_mag * libm::cos(cant_rad),
                    thrust_mag * libm::sin(cant_rad) * libm::sin(phase),
                )
            } else {
                (0.0, thrust_mag, 0.0)
            };

        // ── Drag ──────────────────────────────────────────────
        let v_sq = vx * vx + vy * vy + vz * vz;
        let v_total = libm::sqrt(v_sq);
        let (drag_force, drag_x, drag_y, drag_z) = if v_total > 0.0 {
            let d = half_rho_cd_a * v_sq;
            let inv_v = 1.0 / v_total;
            (d, -d * vx * inv_v, -d * vy * inv_v, -d * vz * inv_v)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        // ── Gravity + wind ────────────────────────────────────
        let gravity_y = -current_mass * p.gravity;

        // ── Net force → acceleration ──────────────────────────
        let (ax, ay, az, drag_out) = if on_pad {
            // On the pad: ground reaction balances all forces
            (0.0, 0.0, 0.0, 0.0)
        } else {
            let fx = thrust_x + drag_x + wind_fx;
            let fy = thrust_y + drag_y + gravity_y;
            let fz = thrust_z + drag_z + wind_fz;
            (fx / current_mass, fy / current_mass, fz / current_mass, drag_force)
        };

        // ── Record state ──────────────────────────────────────
        r.time_s.push(t);
        r.position_x_m.push(x);
        r.altitude_m.push(y);
        r.position_z_m.push(z);
        r.velocity_x_ms.push(vx);
        r.velocity_y_ms.push(vy);
        r.velocity_z_ms.push(vz);
        r.acceleration_x_ms2.push(ax);
        r.acceleration_y_ms2.push(ay);
        r.acceleration_z_ms2.push(az);
        r.mass_kg.push(current_mass);
        r.drag_force_n.push(drag_out);
        r.thrust_n.push(thrust_mag);
        r.spin_rate_dps.push(p.spin_rate);

        // ── Integrate (skip if sitting on pad) ────────────────
        if !on_pad {
            vx += ax * DT;
            vy += ay * DT;
            vz += az * DT;
            x += vx * DT;
            y += vy * DT;
            z += vz * DT;
        }

        // ── Termination: rocket back on ground ────────────────
        if y < 0.0 && t > p.launch_delay {
            break;
        }

        t += DT;
    }

    r
}

/// Compute derived columns: total velocity and total acceleration.
///
/// This mirrors the `df.with_columns(...)` call in the Python version,
/// appending two derived arrays in-place.
///
/// # Returns
///
/// * `(velocity_total_ms, acceleration_total_ms2)` – each a `Vec<f64>` of
///   the same length as the primary time-series in `r`.
pub fn compute_derived(r: &SimResult) -> (Vec<f64>, Vec<f64>) {
    let n = r.time_s.len();
    let mut vel_total = Vec::with_capacity(n);
    let mut acc_total = Vec::with_capacity(n);
    for i in 0..n {
        let vx = r.velocity_x_ms[i];
        let vy = r.velocity_y_ms[i];
        let vz = r.velocity_z_ms[i];
        vel_total.push(libm::sqrt(vx * vx + vy * vy + vz * vz));

        let ax = r.acceleration_x_ms2[i];
        let ay = r.acceleration_y_ms2[i];
        let az = r.acceleration_z_ms2[i];
        acc_total.push(libm::sqrt(ax * ax + ay * ay + az * az));
    }
    (vel_total, acc_total)
}

/// PyO3 wrapper that exposes [`simulate`] + [`compute_derived`] to Python.
///
/// Accepts individual parameter arguments (matching `RocketParams` fields)
/// and returns a Python `dict[str, list[float]]` ready for Polars DataFrame
/// construction.
///
/// See module-level docs for a usage example.
#[pyfunction]
#[pyo3(signature = (
    dry_mass = 50.0,
    propellant_mass = 150.0,
    thrust = 15000.0,
    burn_time = 25.0,
    drag_coeff = 0.40,
    ref_area = 0.03,
    gravity = 9.81,
    wind_speed = 3.0,
    wind_speed_z = 0.0,
    air_density = 1.225,
    launch_delay = 1.0,
    spin_rate = 0.0,
    thrust_cant = 0.0,
))]
#[allow(clippy::too_many_arguments)]
pub fn simulate_rocket_rs(
    py: Python<'_>,
    dry_mass: f64,
    propellant_mass: f64,
    thrust: f64,
    burn_time: f64,
    drag_coeff: f64,
    ref_area: f64,
    gravity: f64,
    wind_speed: f64,
    wind_speed_z: f64,
    air_density: f64,
    launch_delay: f64,
    spin_rate: f64,
    thrust_cant: f64,
) -> PyResult<Py<PyDict>> {
    let params = RocketParams {
        dry_mass,
        propellant_mass,
        thrust,
        burn_time,
        drag_coeff,
        ref_area,
        gravity,
        wind_speed,
        wind_speed_z,
        air_density,
        launch_delay,
        spin_rate,
        thrust_cant,
    };

    let r = simulate(&params);
    let (vel_total, acc_total) = compute_derived(&r);

    // Pack into a Python dict
    let dict = PyDict::new(py);
    dict.set_item("time_s", r.time_s)?;
    dict.set_item("position_x_m", r.position_x_m)?;
    dict.set_item("altitude_m", r.altitude_m)?;
    dict.set_item("position_z_m", r.position_z_m)?;
    dict.set_item("velocity_x_ms", r.velocity_x_ms)?;
    dict.set_item("velocity_y_ms", r.velocity_y_ms)?;
    dict.set_item("velocity_z_ms", r.velocity_z_ms)?;
    dict.set_item("acceleration_x_ms2", r.acceleration_x_ms2)?;
    dict.set_item("acceleration_y_ms2", r.acceleration_y_ms2)?;
    dict.set_item("acceleration_z_ms2", r.acceleration_z_ms2)?;
    dict.set_item("mass_kg", r.mass_kg)?;
    dict.set_item("drag_force_N", r.drag_force_n)?;
    dict.set_item("thrust_N", r.thrust_n)?;
    dict.set_item("spin_rate_dps", r.spin_rate_dps)?;
    dict.set_item("velocity_total_ms", vel_total)?;
    dict.set_item("acceleration_total_ms2", acc_total)?;

    Ok(dict.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: default parameters should produce a flight that reaches
    /// roughly 31 742 m apogee and lasts about 85 s.
    #[test]
    fn test_default_flight() {
        let p = RocketParams {
            dry_mass: 50.0,
            propellant_mass: 150.0,
            thrust: 15000.0,
            burn_time: 25.0,
            drag_coeff: 0.40,
            ref_area: 0.03,
            gravity: 9.81,
            wind_speed: 3.0,
            wind_speed_z: 0.0,
            air_density: 1.225,
            launch_delay: 1.0,
            spin_rate: 0.0,
            thrust_cant: 0.0,
        };

        let r = simulate(&p);

        // Should have a reasonable number of steps
        assert!(r.time_s.len() > 1000, "Flight too short: {} steps", r.time_s.len());

        // Apogee should be close to 31 742 m
        let max_alt = r.altitude_m.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (max_alt - 31742.0).abs() < 50.0,
            "Apogee {:.1} m not near 31742 m", max_alt
        );

        // Flight should end when rocket returns to ground
        // (the last recorded step may still have y >= 0 because we record
        //  state *before* integration; the break fires on the *next* step)
        let last_alt = *r.altitude_m.last().unwrap();
        assert!(last_alt <= 5.0, "Last altitude {last_alt} should be near ground");
    }

    /// Verify derived columns have the correct length.
    #[test]
    fn test_derived_columns() {
        let p = RocketParams {
            dry_mass: 50.0,
            propellant_mass: 150.0,
            thrust: 15000.0,
            burn_time: 25.0,
            drag_coeff: 0.40,
            ref_area: 0.03,
            gravity: 9.81,
            wind_speed: 3.0,
            wind_speed_z: 0.0,
            air_density: 1.225,
            launch_delay: 1.0,
            spin_rate: 0.0,
            thrust_cant: 0.0,
        };
        let r = simulate(&p);
        let (vt, at) = compute_derived(&r);
        assert_eq!(vt.len(), r.time_s.len());
        assert_eq!(at.len(), r.time_s.len());
    }

    /// Test with spin + cant to exercise the helical thrust path.
    #[test]
    fn test_spin_cant() {
        let p = RocketParams {
            dry_mass: 50.0,
            propellant_mass: 150.0,
            thrust: 15000.0,
            burn_time: 25.0,
            drag_coeff: 0.40,
            ref_area: 0.03,
            gravity: 9.81,
            wind_speed: 0.0,
            wind_speed_z: 0.0,
            air_density: 1.225,
            launch_delay: 0.0,
            spin_rate: 720.0,
            thrust_cant: 2.0,
        };
        let r = simulate(&p);

        // With spin+cant, lateral positions should be non-trivial
        let max_x = r.position_x_m.iter().cloned().fold(0.0_f64, f64::max);
        let max_z = r.position_z_m.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            max_x.abs() > 1.0 || max_z.abs() > 1.0,
            "Spin+cant should produce lateral displacement"
        );
    }
}
