use crate::sim::SimResult;
use nalgebra::Vector3;
use rand_distr::{Distribution, Normal}; // Assuming sim is in the same crate

pub struct SensorConfig {
    pub noise_scale: f64,
    pub accel_noise_std: f64,   // m/s^2
    pub gyro_noise_std: f64,    // rad/s
    pub mag_noise_std: f64,     // Gauss
    pub baro_noise_std: f64,    // Pascals or meters
    pub gps_pos_noise_std: f64, // meters
    pub gps_vel_noise_std: f64, // m/s

    // Biases (Walking bias not implemented for simplicity, just static)
    pub accel_bias: Vector3<f64>,
    pub gyro_bias: Vector3<f64>,

    pub seed: u64,

    // Enable flags
    pub accel_enabled: bool,
    pub gyro_enabled: bool,
    pub mag_enabled: bool,
    pub baro_enabled: bool,
    pub gps_enabled: bool,

    // Saturation limits (realistic sensor ranges)
    pub accel_saturation: f64, // m/s² (e.g., 200.0 for BMI088)
    pub gyro_saturation: f64,  // rad/s (e.g., 34.9 for 2000 deg/s)
}

impl Default for SensorConfig {
    fn default() -> Self {
        Self {
            noise_scale: 1.0,
            accel_noise_std: 0.01,
            gyro_noise_std: 0.001,
            mag_noise_std: 0.001,
            baro_noise_std: 0.1,
            gps_pos_noise_std: 1.0,
            gps_vel_noise_std: 0.1,
            accel_bias: Vector3::zeros(),
            gyro_bias: Vector3::zeros(),
            seed: 42,
            accel_enabled: true,
            gyro_enabled: true,
            mag_enabled: true,
            baro_enabled: true,
            gps_enabled: true,
            accel_saturation: 200.0, // BMI088: ±200 m/s² (~20g)
            gyro_saturation: 34.9,   // BMI088: 2000 deg/s = 34.9 rad/s
        }
    }
}

pub struct SensorData {
    pub time: Vec<f64>,
    pub accel_meas: Vec<Option<Vector3<f64>>>,
    pub gyro_meas: Vec<Option<Vector3<f64>>>,
    pub mag_meas: Vec<Option<Vector3<f64>>>,
    pub baro_pressure: Vec<Option<f64>>,
    pub gps_pos: Vec<Option<Vector3<f64>>>,
    pub gps_vel: Vec<Option<Vector3<f64>>>,
}

pub fn generate_sensor_data(sim: &SimResult, cfg: &SensorConfig) -> SensorData {
    let mut rng = rand::rngs::ThreadRng::default();

    let n = sim.time.len();
    let mut data = SensorData {
        time: sim.time.clone(),
        accel_meas: Vec::with_capacity(n),
        gyro_meas: Vec::with_capacity(n),
        mag_meas: Vec::with_capacity(n),
        baro_pressure: Vec::with_capacity(n),
        gps_pos: Vec::with_capacity(n),
        gps_vel: Vec::with_capacity(n),
    };

    // Distributions
    let d_accel = Normal::new(0.0, cfg.noise_scale * cfg.accel_noise_std).unwrap();
    let d_gyro = Normal::new(0.0, cfg.noise_scale * cfg.gyro_noise_std).unwrap();
    let d_mag = Normal::new(0.0, cfg.noise_scale * cfg.mag_noise_std).unwrap();
    let d_baro = Normal::new(0.0, cfg.noise_scale * cfg.baro_noise_std).unwrap();
    let d_gps_p = Normal::new(0.0, cfg.noise_scale * cfg.gps_pos_noise_std).unwrap();
    let d_gps_v = Normal::new(0.0, cfg.noise_scale * cfg.gps_vel_noise_std).unwrap();

    // Constant Field Definitions (NED)
    // Example: ~0.5 Gauss, dipping down (Northern hemisphere)
    let mag_field_ned = Vector3::new(0.25, 0.0, 0.45);

    for i in 0..n {
        // 1. IMU (Accelerometer)
        if cfg.accel_enabled {
            let proper_accel_true = sim.accel_body[i];

            let ax = proper_accel_true.x + cfg.accel_bias.x + d_accel.sample(&mut rng);
            let ay = proper_accel_true.y + cfg.accel_bias.y + d_accel.sample(&mut rng);
            let az = proper_accel_true.z + cfg.accel_bias.z + d_accel.sample(&mut rng);

            // Apply saturation (clamp to sensor range)
            let ax_sat = ax.clamp(-cfg.accel_saturation, cfg.accel_saturation);
            let ay_sat = ay.clamp(-cfg.accel_saturation, cfg.accel_saturation);
            let az_sat = az.clamp(-cfg.accel_saturation, cfg.accel_saturation);

            data.accel_meas.push(Some(Vector3::new(ax_sat, ay_sat, az_sat)));
        } else {
            data.accel_meas.push(None);
        }

        // 2. Gyroscope
        if cfg.gyro_enabled {
            let gx = sim.ang_vel[i].x + cfg.gyro_bias.x + d_gyro.sample(&mut rng);
            let gy = sim.ang_vel[i].y + cfg.gyro_bias.y + d_gyro.sample(&mut rng);
            let gz = sim.ang_vel[i].z + cfg.gyro_bias.z + d_gyro.sample(&mut rng);

            // Apply saturation (clamp to sensor range)
            let gx_sat = gx.clamp(-cfg.gyro_saturation, cfg.gyro_saturation);
            let gy_sat = gy.clamp(-cfg.gyro_saturation, cfg.gyro_saturation);
            let gz_sat = gz.clamp(-cfg.gyro_saturation, cfg.gyro_saturation);

            data.gyro_meas.push(Some(Vector3::new(gx_sat, gy_sat, gz_sat)));
        } else {
            data.gyro_meas.push(None);
        }

        // 3. Magnetometer
        if cfg.mag_enabled {
            let mag_body = sim.orientation[i].inverse_transform_vector(&mag_field_ned);

            let mx = mag_body.x + d_mag.sample(&mut rng);
            let my = mag_body.y + d_mag.sample(&mut rng);
            let mz = mag_body.z + d_mag.sample(&mut rng);
            data.mag_meas.push(Some(Vector3::new(mx, my, mz)));
        } else {
            data.mag_meas.push(None);
        }

        // 4. Barometer (Pressure)
        if cfg.baro_enabled {
            let true_alt = -sim.pos[i].z;
            let p0 = 101325.0; // Pa
            let h_scale = 8500.0; // m
            let static_pressure = p0 * (-true_alt / h_scale).exp();

            // Add dynamic pressure effect (pitot-static error)
            // At high velocities, airflow into the static port causes errors
            let v_mag = sim.vel[i].norm();
            let rho = p0 / (287.0 * 288.0) * (-true_alt / h_scale).exp(); // Air density
            let q_dynamic = 0.5 * rho * v_mag.powi(2); // Dynamic pressure

            // Model coupling of dynamic pressure into static port
            // Realistic coupling: ~5-15% depending on velocity and rocket geometry
            // At high speeds (>Mach 0.5), flow separation increases coupling
            let mach = v_mag / 343.0; // Speed of sound at sea level
            let coupling_factor = if mach < 0.3 {
                0.05 // Subsonic, good design
            } else if mach < 0.8 {
                0.05 + 0.15 * (mach - 0.3) / 0.5 // Transonic increase
            } else {
                0.20 // High transonic/supersonic regime
            };
            let pressure_error = q_dynamic * coupling_factor;

            // Velocity-dependent noise: vibration and turbulence increase with speed
            let velocity_noise_factor = 1.0 + (v_mag / 100.0).min(5.0);
            let noise_component = d_baro.sample(&mut rng) * velocity_noise_factor;

            let meas_pressure = static_pressure + pressure_error + noise_component;
            data.baro_pressure.push(Some(meas_pressure));
        } else {
            data.baro_pressure.push(None);
        }

        // 5. GPS
        if cfg.gps_enabled {
            let px = sim.pos[i].x + d_gps_p.sample(&mut rng);
            let py = sim.pos[i].y + d_gps_p.sample(&mut rng);
            let pz = sim.pos[i].z + d_gps_p.sample(&mut rng); // GPS Altitude usually noisy
            data.gps_pos.push(Some(Vector3::new(px, py, pz)));

            let vx = sim.vel[i].x + d_gps_v.sample(&mut rng);
            let vy = sim.vel[i].y + d_gps_v.sample(&mut rng);
            let vz = sim.vel[i].z + d_gps_v.sample(&mut rng);
            data.gps_vel.push(Some(Vector3::new(vx, vy, vz)));
        } else {
            data.gps_pos.push(None);
            data.gps_vel.push(None);
        }
    }

    data
}
