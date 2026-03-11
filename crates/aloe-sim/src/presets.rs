//! Rocket configuration presets for common scenarios

use crate::sim::RocketParams;
use nalgebra::Vector3;

/// 5k ft (1524m) apogee - Small model rocket
/// Typical high-power model rocket with F/G motor
pub fn altitude_5k_ft() -> RocketParams {
    RocketParams {
        dry_mass: 0.5,
        propellant_mass: 0.15,
        thrust_curve: vec![(0.0, 80.0), (2.5, 80.0), (2.51, 0.0)],
        burn_time: 2.5,
        isp: 180.0,
        ref_area: 0.003, // ~6 cm diameter
        drag_coeff_axial: 0.42,
        cg_full: 0.60,
        cg_empty: 0.55,
        cp_location: 0.85,
        inertia_tensor: Vector3::new(0.01, 0.01, 0.0001),
        ..Default::default()
    }
}

/// 12k ft (3658m) apogee - Mid-power high-altitude
/// High-power rocket with K-class motor
pub fn altitude_12k_ft() -> RocketParams {
    RocketParams {
        dry_mass: 6.0,
        propellant_mass: 2.5,
        thrust_curve: vec![
            (0.0, 1200.0),
            (0.5, 1200.0),
            (2.5, 1000.0),
            (3.5, 600.0),
            (3.51, 0.0),
        ],
        burn_time: 3.5,
        isp: 210.0,
        ref_area: 0.010, // ~11 cm diameter
        drag_coeff_axial: 0.38,
        cg_full: 1.0,
        cg_empty: 0.9,
        cp_location: 1.4,
        inertia_tensor: Vector3::new(0.4, 0.4, 0.004),
        ..Default::default()
    }
}

/// 30k ft (9144m) apogee - High-altitude research rocket
/// High-power rocket with multiple M-class motors
pub fn altitude_30k_ft() -> RocketParams {
    RocketParams {
        dry_mass: 25.0,
        propellant_mass: 15.0,
        thrust_curve: vec![
            (0.0, 4000.0),
            (1.0, 4000.0),
            (4.0, 3500.0),
            (6.0, 2500.0),
            (6.01, 0.0),
        ],
        burn_time: 6.0,
        isp: 230.0,
        ref_area: 0.020, // ~16 cm diameter
        drag_coeff_axial: 0.32,
        cg_full: 1.8,
        cg_empty: 1.6,
        cp_location: 2.5,
        inertia_tensor: Vector3::new(3.0, 3.0, 0.015),
        ..Default::default()
    }
}

/// High-drag test vehicle (low altitude)
/// Useful for testing filter performance in high-drag regime
pub fn high_drag_test() -> RocketParams {
    RocketParams {
        dry_mass: 2.0,
        propellant_mass: 0.5,
        thrust_curve: vec![(0.0, 200.0), (2.0, 200.0), (2.01, 0.0)],
        burn_time: 2.0,
        isp: 180.0,
        ref_area: 0.02,        // Large frontal area
        drag_coeff_axial: 2.0, // Very high drag
        cg_full: 0.8,
        cg_empty: 0.75,
        cp_location: 1.2,
        inertia_tensor: Vector3::new(0.2, 0.2, 0.001),
        ..Default::default()
    }
}

/// Spin-stabilized rocket (marginally stable)
/// Uses gyroscopic stability with minimal aerodynamic stability
pub fn spin_stabilized() -> RocketParams {
    RocketParams {
        dry_mass: 1.0,
        propellant_mass: 0.3,
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_presets_valid() {
        assert!(altitude_5k_ft().validate().is_ok());
        assert!(altitude_12k_ft().validate().is_ok());
        assert!(altitude_30k_ft().validate().is_ok());
        assert!(high_drag_test().validate().is_ok());
        assert!(spin_stabilized().validate().is_ok());
    }
}
