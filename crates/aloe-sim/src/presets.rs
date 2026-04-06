//! Rocket configuration presets for common scenarios

use crate::sim::RocketParams;
use nalgebra::Vector3;

fn gui_aligned(mut params: RocketParams) -> RocketParams {
    params.wind_velocity_ned = Vector3::new(0.0, 0.0, 0.0);
    params.launch_rod_length = 3.0;
    params
}

fn constant_thrust_curve(thrust_n: f64, burn_time_s: f64) -> Vec<(f64, f64)> {
    vec![
        (0.0, thrust_n),
        (burn_time_s, thrust_n),
        (burn_time_s + 0.01, 0.0),
    ]
}

/// Approximately 3k-ft apogee under current JSBSim liquid/rail modeling.
pub fn altitude_3k_ft() -> RocketParams {
    gui_aligned(RocketParams {
        dry_mass: 0.3,
        fuel_mass: 0.1 / 3.0,
        oxidizer_mass: 0.2 / 3.0,
        thrust_curve: constant_thrust_curve(320.0, 2.0),
        burn_time: 2.0,
        isp: 170.0,
        ref_area: 0.002,
        drag_coeff_axial: 0.45,
        normal_force_coeff: 10.0,
        cg_full: 0.5,
        cg_empty: 0.45,
        cp_location: 0.75,
        inertia_tensor: Vector3::new(0.00006, 0.006, 0.006),
        nozzle_location: 0.8,
        ..Default::default()
    })
}

/// Approximately 5k-ft apogee under current JSBSim liquid/rail modeling.
pub fn altitude_5k_ft() -> RocketParams {
    gui_aligned(RocketParams {
        dry_mass: 0.7,
        fuel_mass: 0.08,
        oxidizer_mass: 0.16,
        thrust_curve: constant_thrust_curve(500.0, 2.5),
        burn_time: 2.5,
        isp: 180.0,
        ref_area: 0.003,
        drag_coeff_axial: 0.42,
        cg_full: 0.65,
        cg_empty: 0.58,
        cp_location: 0.92,
        inertia_tensor: Vector3::new(0.00018, 0.018, 0.018),
        nozzle_location: 1.0,
        ..Default::default()
    })
}

/// Approximately 10k-ft apogee under current JSBSim liquid/rail modeling.
pub fn altitude_10k_ft() -> RocketParams {
    gui_aligned(RocketParams {
        dry_mass: 4.0,
        fuel_mass: 2.0 / 3.0,
        oxidizer_mass: 4.0 / 3.0,
        thrust_curve: constant_thrust_curve(900.0, 3.0),
        burn_time: 3.0,
        isp: 190.0,
        ref_area: 0.008,
        drag_coeff_axial: 0.4,
        normal_force_coeff: 12.0,
        cg_full: 0.8,
        cg_empty: 0.7,
        cp_location: 1.2,
        inertia_tensor: Vector3::new(0.0025, 0.25, 0.25),
        nozzle_location: 1.4,
        ..Default::default()
    })
}

/// Approximately 12k-ft apogee under current JSBSim liquid/rail modeling.
pub fn altitude_12k_ft() -> RocketParams {
    gui_aligned(RocketParams {
        dry_mass: 6.0,
        fuel_mass: 2.5 / 3.0,
        oxidizer_mass: 5.0 / 3.0,
        thrust_curve: constant_thrust_curve(1200.0, 3.5),
        burn_time: 3.5,
        isp: 210.0,
        ref_area: 0.010, // ~11 cm diameter
        drag_coeff_axial: 0.38,
        normal_force_coeff: 12.0,
        cg_full: 1.0,
        cg_empty: 0.9,
        cp_location: 1.4,
        inertia_tensor: Vector3::new(0.004, 0.4, 0.4),
        nozzle_location: 1.7,
        ..Default::default()
    })
}

/// Approximately 15k-ft apogee under current JSBSim liquid/rail modeling.
pub fn altitude_15k_ft() -> RocketParams {
    gui_aligned(RocketParams {
        dry_mass: 12.0,
        fuel_mass: 8.0 / 3.0,
        oxidizer_mass: 16.0 / 3.0,
        thrust_curve: constant_thrust_curve(2000.0, 4.5),
        burn_time: 4.5,
        isp: 210.0,
        ref_area: 0.015,
        drag_coeff_axial: 0.35,
        normal_force_coeff: 12.0,
        cg_full: 1.2,
        cg_empty: 1.1,
        cp_location: 1.6,
        inertia_tensor: Vector3::new(0.006, 1.2, 1.2),
        nozzle_location: 2.0,
        ..Default::default()
    })
}

/// Approximately 30k-ft apogee under current JSBSim liquid/rail modeling.
pub fn altitude_30k_ft() -> RocketParams {
    gui_aligned(RocketParams {
        dry_mass: 25.0,
        fuel_mass: 5.0,
        oxidizer_mass: 10.0,
        thrust_curve: constant_thrust_curve(4000.0, 6.0),
        burn_time: 6.0,
        isp: 230.0,
        ref_area: 0.020, // ~16 cm diameter
        drag_coeff_axial: 0.32,
        normal_force_coeff: 12.0,
        cg_full: 1.8,
        cg_empty: 1.6,
        cp_location: 2.5,
        inertia_tensor: Vector3::new(0.015, 3.0, 3.0),
        nozzle_location: 3.0,
        ..Default::default()
    })
}

/// High-drag test vehicle.
/// Current JSBSim apogee is about 380 ft and is mainly useful as a behavioral test case.
pub fn high_drag_test() -> RocketParams {
    gui_aligned(RocketParams {
        dry_mass: 2.0,
        fuel_mass: 0.5 / 3.0,
        oxidizer_mass: 1.0 / 3.0,
        thrust_curve: constant_thrust_curve(200.0, 2.0),
        burn_time: 2.0,
        isp: 180.0,
        ref_area: 0.02,        // Large frontal area
        drag_coeff_axial: 2.0, // Very high drag
        normal_force_coeff: 12.0,
        cg_full: 0.8,
        cg_empty: 0.75,
        cp_location: 1.2,
        inertia_tensor: Vector3::new(0.001, 0.2, 0.2),
        nozzle_location: 1.3,
        ..Default::default()
    })
}

/// Spin-stabilized rocket (marginally stable).
pub fn spin_stabilized() -> RocketParams {
    gui_aligned(RocketParams {
        dry_mass: 1.0,
        fuel_mass: 0.1,
        oxidizer_mass: 0.2,
        thrust_curve: vec![(0.0, 150.0), (2.5, 150.0), (2.51, 0.0)],
        burn_time: 2.5,
        isp: 170.0,
        ref_area: 0.003,
        drag_coeff_axial: 0.5,
        cg_full: 0.5,
        cg_empty: 0.45,
        cp_location: 0.52, // Minimal stability margin - spin provides additional stability
        inertia_tensor: Vector3::new(0.1, 0.1, 0.001),
        spin_rate: 600.0, // 10 rev/sec for gyroscopic stability
        thrust_cant: 2.0, // Canted nozzle for spin-up
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apogee_ft(params: RocketParams) -> f64 {
        let result = crate::sim::simulate_6dof(&params);
        result.max_altitude() * 3.28084
    }

    #[test]
    fn test_all_presets_valid() {
        assert!(altitude_3k_ft().validate().is_ok());
        assert!(altitude_5k_ft().validate().is_ok());
        assert!(altitude_10k_ft().validate().is_ok());
        assert!(altitude_12k_ft().validate().is_ok());
        assert!(altitude_15k_ft().validate().is_ok());
        assert!(altitude_30k_ft().validate().is_ok());
        assert!(high_drag_test().validate().is_ok());
        assert!(spin_stabilized().validate().is_ok());
    }

    #[test]
    #[ignore = "measurement helper for JSBSim preset retuning"]
    fn print_preset_apogees() {
        let presets = [
            ("3k-ft", altitude_3k_ft()),
            ("5k-ft", altitude_5k_ft()),
            ("10k-ft", altitude_10k_ft()),
            ("12k-ft", altitude_12k_ft()),
            ("15k-ft", altitude_15k_ft()),
            ("30k-ft", altitude_30k_ft()),
            ("high-drag", high_drag_test()),
            ("spin-stabilized", spin_stabilized()),
        ];

        for (name, params) in presets {
            println!("{name}: {:.1} ft", apogee_ft(params));
        }
    }
}
