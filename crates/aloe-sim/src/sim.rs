//! 6-DoF liquid-rocket simulation interface backed by JSBSim.

use aloe_jsbsim::{simulate as simulate_jsbsim, JsbsimRocketParams};
use nalgebra::{UnitQuaternion, Vector3};

const THRUST_THRESHOLD_N: f64 = 10.0;
const DEFAULT_SIM_DT_S: f64 = 0.001;
const DEFAULT_MAX_TIME_S: f64 = 400.0;
const MAX_SIM_SAMPLES: usize = 1_000_000;

/// Complete set of rocket and environment parameters for simulation.
#[derive(Debug, Clone)]
pub struct RocketParams {
    pub dry_mass: f64,
    pub fuel_mass: f64,
    pub oxidizer_mass: f64,
    pub inertia_tensor: Vector3<f64>,
    pub cg_full: f64,
    pub cg_empty: f64,
    pub cp_location: f64,
    pub ref_area: f64,
    pub drag_coeff_axial: f64,
    pub normal_force_coeff: f64,
    pub thrust_curve: Vec<(f64, f64)>,
    pub burn_time: f64,
    pub isp: f64,
    pub nozzle_location: f64,
    pub gravity: f64,
    pub air_density_sea_level: f64,
    pub launch_rod_length: f64,
    pub wind_velocity_ned: Vector3<f64>,
    pub launch_delay: f64,
    pub spin_rate: f64,
    pub thrust_cant: f64,
    pub nozzle_exit_pressure_psf: f64,
    pub nozzle_area_ft2: f64,
    pub pad_static_friction: f64,
    pub pad_dynamic_friction: f64,
    pub pad_spring_coeff_lbs_ft: f64,
    pub pad_damping_coeff_lbs_ft_s: f64,
    pub sim_dt: f64,
    pub max_time: f64,
}

impl Default for RocketParams {
    fn default() -> Self {
        Self {
            dry_mass: 20.0,
            fuel_mass: 10.0 / 3.0,
            oxidizer_mass: 20.0 / 3.0,
            inertia_tensor: Vector3::new(0.1, 10.0, 10.0),
            cg_full: 1.5,
            cg_empty: 1.5,
            cp_location: 2.0,
            ref_area: std::f64::consts::PI * 0.075_f64.powi(2),
            drag_coeff_axial: 0.5,
            normal_force_coeff: 12.0,
            thrust_curve: vec![(0.0, 2000.0), (5.0, 2000.0), (5.01, 0.0)],
            burn_time: 5.0,
            isp: 200.0,
            nozzle_location: 3.0,
            gravity: 9.80665,
            air_density_sea_level: 1.225,
            launch_rod_length: 2.0,
            wind_velocity_ned: Vector3::new(5.0, 0.0, 0.0),
            launch_delay: 1.0,
            spin_rate: 0.0,
            thrust_cant: 0.0,
            nozzle_exit_pressure_psf: 2116.22,
            nozzle_area_ft2: 0.01,
            pad_static_friction: 0.8,
            pad_dynamic_friction: 0.4,
            pad_spring_coeff_lbs_ft: 10_000.0,
            pad_damping_coeff_lbs_ft_s: 5_000.0,
            sim_dt: DEFAULT_SIM_DT_S,
            max_time: DEFAULT_MAX_TIME_S,
        }
    }
}

impl RocketParams {
    pub fn total_reactant_mass(&self) -> f64 {
        self.fuel_mass + self.oxidizer_mass
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.dry_mass <= 0.0 {
            return Err("Dry mass must be positive".to_string());
        }
        if self.fuel_mass < 0.0 {
            return Err("Fuel mass must be non-negative".to_string());
        }
        if self.oxidizer_mass < 0.0 {
            return Err("Oxidizer mass must be non-negative".to_string());
        }
        if (self.fuel_mass == 0.0) != (self.oxidizer_mass == 0.0) {
            return Err(
                "Liquid rocket configurations require both fuel and oxidizer masses".to_string(),
            );
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
        if !self.gravity.is_finite() || self.gravity <= 0.0 {
            return Err("Gravity must be positive".to_string());
        }
        if !self.air_density_sea_level.is_finite() || self.air_density_sea_level <= 0.0 {
            return Err("Sea-level air density must be positive".to_string());
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
        if self.thrust_curve.iter().any(|(_, thrust)| *thrust > 1e-6)
            && (self.fuel_mass <= 0.0 || self.oxidizer_mass <= 0.0)
        {
            return Err(
                "Liquid rocket configurations with thrust require positive fuel and oxidizer masses"
                    .to_string(),
            );
        }
        if self.isp <= 0.0 {
            return Err("Isp must be positive".to_string());
        }
        if !self.sim_dt.is_finite() || self.sim_dt <= 0.0 {
            return Err("Simulation dt must be positive".to_string());
        }
        if !self.max_time.is_finite() || self.max_time <= 0.0 {
            return Err("Maximum simulation time must be positive".to_string());
        }
        if self.nozzle_area_ft2 <= 0.0 {
            return Err("Nozzle area must be positive".to_string());
        }
        if self.pad_static_friction < 0.0 || self.pad_dynamic_friction < 0.0 {
            return Err("Pad friction must be non-negative".to_string());
        }
        if self.pad_spring_coeff_lbs_ft <= 0.0 || self.pad_damping_coeff_lbs_ft_s <= 0.0 {
            return Err("Pad spring and damping coefficients must be positive".to_string());
        }
        let sample_count = self.max_time / self.sim_dt;
        if !sample_count.is_finite() || sample_count > MAX_SIM_SAMPLES as f64 {
            return Err(format!(
                "Simulation sample count exceeds limit ({MAX_SIM_SAMPLES}); increase sim_dt or reduce max_time"
            ));
        }
        Ok(())
    }
}

/// Complete trajectory data from simulation.
#[derive(Clone)]
pub struct SimResult {
    pub time: Vec<f64>,
    pub pos: Vec<Vector3<f64>>,
    pub vel: Vec<Vector3<f64>>,
    pub accel_body: Vec<Vector3<f64>>,
    pub ang_vel: Vec<Vector3<f64>>,
    pub orientation: Vec<UnitQuaternion<f64>>,
    pub ascent_time: Option<f64>,
    pub coast_time: Option<f64>,
    pub descent_time: Option<f64>,
}

impl SimResult {
    pub fn max_altitude(&self) -> f64 {
        self.pos
            .iter()
            .filter_map(|p| {
                let altitude = -p.z;
                altitude.is_finite().then_some(altitude)
            })
            .fold(0.0, f64::max)
    }

    pub fn max_velocity(&self) -> f64 {
        self.vel.iter().map(|v| v.norm()).fold(0.0, f64::max)
    }

    pub fn apogee_time(&self) -> Option<f64> {
        self.pos
            .iter()
            .enumerate()
            .filter(|(_, pos)| (-pos.z).is_finite())
            .max_by(|(_, a), (_, b)| (-a.z).total_cmp(&(-b.z)))
            .map(|(idx, _)| self.time[idx])
    }

    pub fn len(&self) -> usize {
        self.time.len()
    }

    pub fn is_empty(&self) -> bool {
        self.time.is_empty()
    }
}

fn interpolate_thrust(curve: &[(f64, f64)], t: f64) -> f64 {
    if curve.is_empty() {
        return 0.0;
    }
    if t <= curve[0].0 {
        return curve[0].1;
    }
    let last = curve.last().unwrap();
    if t >= last.0 {
        return last.1;
    }
    for window in curve.windows(2) {
        let [(t0, f0), (t1, f1)] = [window[0], window[1]];
        if t >= t0 && t <= t1 {
            let span = t1 - t0;
            if span <= 0.0 {
                return f0;
            }
            let frac = (t - t0) / span;
            return f0 + frac * (f1 - f0);
        }
    }
    0.0
}

fn detect_ascent_time(time: &[f64], params: &RocketParams) -> Option<f64> {
    time.iter().copied().find(|&t| {
        t >= params.launch_delay
            && interpolate_thrust(&params.thrust_curve, t - params.launch_delay)
                > THRUST_THRESHOLD_N
    })
}

fn detect_coast_time(time: &[f64], params: &RocketParams, ascent_time: Option<f64>) -> Option<f64> {
    let ascent_time = ascent_time?;
    time.iter().copied().find(|&t| {
        t >= ascent_time
            && interpolate_thrust(&params.thrust_curve, t - params.launch_delay)
                <= THRUST_THRESHOLD_N
    })
}

fn detect_descent_time(time: &[f64], pos: &[Vector3<f64>], coast_time: Option<f64>) -> Option<f64> {
    let coast_time = coast_time?;
    time.iter()
        .enumerate()
        .filter(|(_, t)| **t >= coast_time)
        .filter(|(idx, _)| (-pos[*idx].z).is_finite())
        .max_by(|(idx_a, _), (idx_b, _)| (-pos[*idx_a].z).total_cmp(&(-pos[*idx_b].z)))
        .map(|(_, t)| *t)
}

/// Run the complete 6-DoF simulation through JSBSim.
pub fn simulate_6dof(params: &RocketParams) -> SimResult {
    params
        .validate()
        .unwrap_or_else(|err| panic!("invalid rocket parameters: {err}"));

    let jsbsim_result = simulate_jsbsim(&JsbsimRocketParams {
        dry_mass: params.dry_mass,
        fuel_mass: params.fuel_mass,
        oxidizer_mass: params.oxidizer_mass,
        inertia_tensor: params.inertia_tensor,
        cg_full: params.cg_full,
        cg_empty: params.cg_empty,
        cp_location: params.cp_location,
        ref_area: params.ref_area,
        drag_coeff_axial: params.drag_coeff_axial,
        normal_force_coeff: params.normal_force_coeff,
        thrust_curve: params.thrust_curve.clone(),
        burn_time: params.burn_time,
        isp: params.isp,
        nozzle_location: params.nozzle_location,
        gravity: params.gravity,
        air_density_sea_level: params.air_density_sea_level,
        launch_rod_length: params.launch_rod_length,
        wind_velocity_ned: params.wind_velocity_ned,
        launch_delay: params.launch_delay,
        spin_rate: params.spin_rate,
        thrust_cant: params.thrust_cant,
        nozzle_exit_pressure_psf: params.nozzle_exit_pressure_psf,
        nozzle_area_ft2: params.nozzle_area_ft2,
        pad_static_friction: params.pad_static_friction,
        pad_dynamic_friction: params.pad_dynamic_friction,
        pad_spring_coeff_lbs_ft: params.pad_spring_coeff_lbs_ft,
        pad_damping_coeff_lbs_ft_s: params.pad_damping_coeff_lbs_ft_s,
        sim_dt: params.sim_dt,
        max_time: params.max_time,
    })
    .unwrap_or_else(|err| panic!("JSBSim simulation failed: {err:#}"));

    let ascent_time = detect_ascent_time(&jsbsim_result.time, params);
    let coast_time = detect_coast_time(&jsbsim_result.time, params, ascent_time);
    let descent_time = detect_descent_time(&jsbsim_result.time, &jsbsim_result.pos, coast_time);

    SimResult {
        time: jsbsim_result.time,
        pos: jsbsim_result.pos,
        vel: jsbsim_result.vel,
        accel_body: jsbsim_result.accel_body,
        ang_vel: jsbsim_result.ang_vel,
        orientation: jsbsim_result.orientation,
        ascent_time,
        coast_time,
        descent_time,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params_valid() {
        assert!(RocketParams::default().validate().is_ok());
    }

    #[test]
    fn test_zero_dry_mass_invalid() {
        let params = RocketParams {
            dry_mass: 0.0,
            ..Default::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_negative_inertia_invalid() {
        let params = RocketParams {
            inertia_tensor: Vector3::new(-1.0, 1.0, 1.0),
            ..Default::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_non_positive_sim_dt_invalid() {
        let params = RocketParams {
            sim_dt: 0.0,
            ..Default::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_non_positive_gravity_invalid() {
        let params = RocketParams {
            gravity: 0.0,
            ..Default::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_non_positive_air_density_invalid() {
        let params = RocketParams {
            air_density_sea_level: 0.0,
            ..Default::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_thrust_requires_reactants() {
        let params = RocketParams {
            fuel_mass: 0.0,
            oxidizer_mass: 0.0,
            ..Default::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_excessive_sample_count_invalid() {
        let params = RocketParams {
            sim_dt: 1.0e-5,
            max_time: 20.0,
            ..Default::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_result_metadata() {
        let result = simulate_6dof(&RocketParams::default());
        assert!(!result.is_empty());
        assert_eq!(result.time.len(), result.pos.len());
        assert_eq!(result.time.len(), result.vel.len());
        assert_eq!(result.time.len(), result.accel_body.len());
        assert_eq!(result.time.len(), result.ang_vel.len());
        assert_eq!(result.time.len(), result.orientation.len());
    }

    #[test]
    fn test_apogee_detection() {
        let params = RocketParams::default();
        let result = simulate_6dof(&params);
        assert!(result.max_altitude() > 0.0);
        assert!(result.apogee_time().is_some());
    }

    #[test]
    fn test_state_transition_detection() {
        let params = RocketParams::default();
        let result = simulate_6dof(&params);

        assert!(result.ascent_time.is_some());
        assert!(result.coast_time.is_some());
        assert!(result.descent_time.is_some());

        let ascent = result.ascent_time.unwrap();
        let coast = result.coast_time.unwrap();
        let descent = result.descent_time.unwrap();

        assert!(ascent >= params.launch_delay);
        assert!(ascent < coast);
        assert!(coast < descent);
    }

    #[test]
    fn test_zero_thrust_stays_near_pad() {
        let params = RocketParams {
            thrust_curve: vec![(0.0, 0.0)],
            burn_time: 0.0,
            launch_delay: 0.0,
            ..Default::default()
        };
        let result = simulate_6dof(&params);
        assert!(result.max_altitude() < 2.0);
    }

    #[test]
    fn test_wind_causes_horizontal_drift() {
        let params = RocketParams {
            wind_velocity_ned: Vector3::new(20.0, 0.0, 0.0),
            ..Default::default()
        };
        let result = simulate_6dof(&params);
        let final_horizontal = result.pos.last().unwrap().xy().norm();
        assert!(final_horizontal > 0.0);
    }

    #[test]
    fn test_longer_launch_rod_increases_altitude_in_wind() {
        let base = RocketParams {
            dry_mass: 2.0,
            fuel_mass: 0.25,
            oxidizer_mass: 0.5,
            thrust_curve: vec![(0.0, 350.0), (2.2, 350.0), (2.21, 0.0)],
            burn_time: 2.2,
            isp: 185.0,
            ref_area: 0.0045,
            drag_coeff_axial: 0.45,
            normal_force_coeff: 12.0,
            cg_full: 0.8,
            cg_empty: 0.74,
            cp_location: 1.15,
            inertia_tensor: Vector3::new(0.03, 0.35, 0.35),
            wind_velocity_ned: Vector3::new(12.0, 0.0, 0.0),
            thrust_cant: 2.0,
            launch_rod_length: 0.5,
            max_time: 60.0,
            ..Default::default()
        };

        let short_rail = simulate_6dof(&base);
        let long_rail = simulate_6dof(&RocketParams {
            launch_rod_length: 8.0,
            ..base
        });

        assert!(long_rail.max_altitude() > short_rail.max_altitude() + 100.0);
    }

    #[test]
    fn test_explicit_thrust_curve_does_not_require_burn_time() {
        let params = RocketParams {
            thrust_curve: vec![(0.0, 600.0), (1.0, 600.0), (1.01, 0.0)],
            burn_time: 0.0,
            launch_delay: 0.0,
            fuel_mass: 0.4,
            oxidizer_mass: 0.8,
            dry_mass: 2.0,
            ..Default::default()
        };
        let result = simulate_6dof(&params);
        assert!(result.max_altitude() > 5.0);
    }
}
