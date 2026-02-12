//! Parameter definitions for rocket simulation.

/// Parameter specification with bounds and step size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamSpec {
    /// Human-readable label.
    pub label: &'static str,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Step size for sliders.
    pub step: f64,
}

impl ParamSpec {
    /// Create a new parameter specification.
    pub const fn new(label: &'static str, min: f64, max: f64, step: f64) -> Self {
        Self {
            label,
            min,
            max,
            step,
        }
    }
}

/// Rocket physical parameters.
pub mod rocket {
    use super::ParamSpec;

    pub const DRY_MASS: ParamSpec = ParamSpec::new("Dry Mass (kg)", 5.0, 100.0, 1.0);
    pub const PROPELLANT_MASS: ParamSpec = ParamSpec::new("Propellant (kg)", 10.0, 300.0, 5.0);
    pub const THRUST: ParamSpec = ParamSpec::new("Thrust (N)", 500.0, 25000.0, 250.0);
    pub const BURN_TIME: ParamSpec = ParamSpec::new("Burn Time (s)", 2.0, 45.0, 0.5);
    pub const DRAG_COEFF: ParamSpec = ParamSpec::new("Drag Coeff.", 0.15, 1.5, 0.05);
    pub const REF_AREA: ParamSpec = ParamSpec::new("Ref. Area (m²)", 0.005, 0.08, 0.002);
    pub const LAUNCH_DELAY: ParamSpec = ParamSpec::new("Launch Delay (s)", 0.0, 30.0, 0.5);
    pub const SPIN_RATE: ParamSpec = ParamSpec::new("Spin Rate (°/s)", 0.0, 3600.0, 30.0);
    pub const THRUST_CANT: ParamSpec = ParamSpec::new("Thrust Cant (°)", 0.0, 10.0, 0.1);
}

/// Environmental parameters.
pub mod environment {
    use super::ParamSpec;

    pub const GRAVITY: ParamSpec = ParamSpec::new("Gravity (m/s²)", 1.0, 15.0, 0.1);
    pub const WIND_SPEED: ParamSpec = ParamSpec::new("Wind X (m/s)", 0.0, 25.0, 1.0);
    pub const WIND_SPEED_Z: ParamSpec = ParamSpec::new("Crosswind Z (m/s)", -25.0, 25.0, 1.0);
    pub const AIR_DENSITY: ParamSpec = ParamSpec::new("Air Density (kg/m³)", 0.3, 1.8, 0.02);
}

/// Sensor rate parameters.
pub mod sensor_rates {
    use super::ParamSpec;

    pub const BMI088_ACCEL_HZ: ParamSpec = ParamSpec::new("BMI088 Accel Hz", 100.0, 1600.0, 100.0);
    pub const BMI088_GYRO_HZ: ParamSpec = ParamSpec::new("BMI088 Gyro Hz", 100.0, 2000.0, 100.0);
    pub const ADXL375_HZ: ParamSpec = ParamSpec::new("ADXL375 Hz", 100.0, 3200.0, 100.0);
    pub const MS5611_HZ: ParamSpec = ParamSpec::new("MS5611 Hz", 10.0, 122.0, 5.0);
    pub const LIS3MDL_HZ: ParamSpec = ParamSpec::new("LIS3MDL Hz", 10.0, 155.0, 5.0);
    pub const LC29H_HZ: ParamSpec = ParamSpec::new("GPS Hz", 1.0, 10.0, 1.0);
}

/// Sensor latency parameters.
pub mod sensor_latency {
    use super::ParamSpec;

    pub const BMI088_ACCEL_MS: ParamSpec = ParamSpec::new("BMI088 Accel (ms)", 0.1, 50.0, 0.5);
    pub const BMI088_GYRO_MS: ParamSpec = ParamSpec::new("BMI088 Gyro (ms)", 1.0, 200.0, 5.0);
    pub const ADXL375_MS: ParamSpec = ParamSpec::new("ADXL375 (ms)", 0.1, 10.0, 0.1);
    pub const MS5611_MS: ParamSpec = ParamSpec::new("MS5611 (ms)", 1.0, 50.0, 1.0);
    pub const LIS3MDL_MS: ParamSpec = ParamSpec::new("LIS3MDL (ms)", 1.0, 100.0, 5.0);
    pub const LC29H_MS: ParamSpec = ParamSpec::new("GPS TTFF (ms)", 100.0, 26000.0, 100.0);
}

/// ESKF tuning parameters.
pub mod eskf {
    use super::ParamSpec;

    pub const ACCEL_NOISE_DENSITY: ParamSpec =
        ParamSpec::new("Accel Noise (m/s²/√Hz)", 0.001, 20.0, 0.1);
    pub const GYRO_NOISE_DENSITY: ParamSpec =
        ParamSpec::new("Gyro Noise (rad/s/√Hz)", 0.00001, 0.5, 0.001);
    pub const ACCEL_BIAS_INSTABILITY: ParamSpec =
        ParamSpec::new("Accel Bias Inst.", 1e-8, 1e-2, 1e-5);
    pub const GYRO_BIAS_INSTABILITY: ParamSpec =
        ParamSpec::new("Gyro Bias Inst.", 1e-8, 0.01, 1e-6);
    pub const POS_PROCESS_NOISE: ParamSpec =
        ParamSpec::new("Pos Proc Noise (m/√s)", 0.0001, 10.0, 0.01);
    pub const R_GPS_POS: ParamSpec = ParamSpec::new("R GPS Pos (m²)", 0.01, 500.0, 1.0);
    pub const R_GPS_VEL: ParamSpec = ParamSpec::new("R GPS Vel ((m/s)²)", 0.001, 100.0, 0.05);
    pub const R_BARO: ParamSpec = ParamSpec::new("R Baro (m²)", 0.001, 200.0, 0.5);
    pub const R_MAG: ParamSpec = ParamSpec::new("R Mag", 0.0001, 10.0, 0.01);
}

/// Flight stages.
pub const FLIGHT_STAGES: [&str; 4] = ["pad", "burn", "coasting", "recovery"];

/// Number of flight stages.
pub const NUM_STAGES: usize = FLIGHT_STAGES.len();

/// Per-stage ESKF tuning defaults.
pub const ESKF_TUNING_DEFAULTS: [(&str, [f32; 4]); 9] = [
    ("accel_noise_density", [0.2236, 0.02430, 0.01, 0.01]),
    ("gyro_noise_density", [0.03728, 0.01389, 0.1, 0.01389]),
    ("accel_bias_instability", [0.01, 0.002683, 1e-6, 1e-6]),
    ("gyro_bias_instability", [3.728e-5, 1e-5, 1e-7, 1e-3]),
    ("pos_process_noise", [1.0, 0.1389, 0.004394, 0.007197]),
    ("r_gps_pos", [61.05, 0.1, 0.1, 0.1]),
    ("r_gps_vel", [0.07197, 0.04394, 0.04394, 0.01]),
    ("r_baro", [0.1, 2.236, 50.0, 50.0]),
    ("r_mag", [1.0, 0.01179, 1.0, 0.002683]),
];

/// Get default tuning value for a parameter and stage.
pub fn eskf_tuning_default(param: &str, stage_idx: usize) -> Option<f32> {
    for (name, values) in ESKF_TUNING_DEFAULTS.iter() {
        if *name == param {
            return values.get(stage_idx).copied();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eskf_tuning_defaults() {
        assert!((eskf_tuning_default("accel_noise_density", 0).unwrap() - 0.2236).abs() < 0.0001);
        assert!((eskf_tuning_default("r_baro", 2).unwrap() - 50.0).abs() < 0.1);
        assert!(eskf_tuning_default("unknown_param", 0).is_none());
        assert!(eskf_tuning_default("accel_noise_density", 10).is_none());
    }

    #[test]
    fn test_param_spec() {
        let spec = ParamSpec::new("Test", 0.0, 100.0, 1.0);
        assert_eq!(spec.label, "Test");
        assert_eq!(spec.min, 0.0);
        assert_eq!(spec.max, 100.0);
        assert_eq!(spec.step, 1.0);
    }
}
