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
        }
    }
}

pub struct SensorData {
    pub time: Vec<f64>,
    pub accel_meas: Vec<Vector3<f64>>,
    pub gyro_meas: Vec<Vector3<f64>>,
    pub mag_meas: Vec<Vector3<f64>>,
    pub baro_pressure: Vec<f64>,
    pub gps_pos: Vec<Vector3<f64>>,
    pub gps_vel: Vec<Vector3<f64>>,
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
            data.accel_meas.push(Vector3::new(ax, ay, az));
        } else {
            data.accel_meas.push(Vector3::zeros());
        }

        // 2. Gyroscope
        if cfg.gyro_enabled {
            let gx = sim.ang_vel[i].x + cfg.gyro_bias.x + d_gyro.sample(&mut rng);
            let gy = sim.ang_vel[i].y + cfg.gyro_bias.y + d_gyro.sample(&mut rng);
            let gz = sim.ang_vel[i].z + cfg.gyro_bias.z + d_gyro.sample(&mut rng);
            data.gyro_meas.push(Vector3::new(gx, gy, gz));
        } else {
            data.gyro_meas.push(Vector3::zeros());
        }

        // 3. Magnetometer
        if cfg.mag_enabled {
            let mag_body = sim.orientation[i].inverse_transform_vector(&mag_field_ned);

            let mx = mag_body.x + d_mag.sample(&mut rng);
            let my = mag_body.y + d_mag.sample(&mut rng);
            let mz = mag_body.z + d_mag.sample(&mut rng);
            data.mag_meas.push(Vector3::new(mx, my, mz));
        } else {
            data.mag_meas.push(Vector3::zeros());
        }

        // 4. Barometer (Pressure)
        if cfg.baro_enabled {
            let true_alt = -sim.pos[i].z;
            let p0 = 101325.0; // Pa
            let h_scale = 8500.0; // m
            let true_pressure = p0 * (-true_alt / h_scale).exp();
            let meas_pressure = true_pressure + d_baro.sample(&mut rng);
            data.baro_pressure.push(meas_pressure);
        } else {
            data.baro_pressure.push(101325.0);
        }

        // 5. GPS
        if cfg.gps_enabled {
            let px = sim.pos[i].x + d_gps_p.sample(&mut rng);
            let py = sim.pos[i].y + d_gps_p.sample(&mut rng);
            let pz = sim.pos[i].z + d_gps_p.sample(&mut rng); // GPS Altitude usually noisy
            data.gps_pos.push(Vector3::new(px, py, pz));

            let vx = sim.vel[i].x + d_gps_v.sample(&mut rng);
            let vy = sim.vel[i].y + d_gps_v.sample(&mut rng);
            let vz = sim.vel[i].z + d_gps_v.sample(&mut rng);
            data.gps_vel.push(Vector3::new(vx, vy, vz));
        } else {
            data.gps_pos.push(Vector3::zeros());
            data.gps_vel.push(Vector3::zeros());
        }
    }

    data
}
