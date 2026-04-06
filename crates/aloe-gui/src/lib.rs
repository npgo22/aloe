//! Aloe GUI - Web interface for rocket simulation
//!
//! Features:
//! - Configuration panel with tabs (Rocket, ENV, Sensors, Filter)
//! - Real-time simulation via API
//! - Multiple chart types (2D and 3D)
//! - Sensor data visualization
//! - Filter error statistics

pub mod tracing_init;

use aloe_sim::{
    filter::{run_filter, FilterConfig},
    sensor::{generate_sensor_data, ChaosConfig, SensorConfig},
    sim::{simulate_6dof, RocketParams},
};
use axum::{
    body::Body,
    extract::Path,
    http::{header, HeaderValue, Response, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use nalgebra::Vector3;
use rust_embed::RustEmbed;
use serde::{Deserialize, Serialize};
use std::sync::{LazyLock, Mutex};
use tracing::{debug, error, info, instrument, warn};

#[derive(RustEmbed)]
#[folder = "../aloe-gui/frontend-dist/browser"]
struct FrontendAssets;

#[derive(Serialize, Clone)]
struct SimpleErrorStats {
    min: f64,
    max: f64,
    mean: f64,
    std: f64,
    rmse: f64,
    mae: f64,
    p95: f64,
    n: usize,
}

#[derive(Serialize)]
struct EskfErrorStats {
    pos_n: SimpleErrorStats,
    pos_e: SimpleErrorStats,
    pos_d: SimpleErrorStats,
    vel_n: SimpleErrorStats,
    vel_e: SimpleErrorStats,
    vel_d: SimpleErrorStats,
    pos_3d: SimpleErrorStats,
}

#[derive(Serialize)]
struct QuantizedFlightErrorStats {
    pos_n: SimpleErrorStats,
    pos_e: SimpleErrorStats,
    alt: SimpleErrorStats,
    vel_n: SimpleErrorStats,
    vel_e: SimpleErrorStats,
    vel_d: SimpleErrorStats,
    pos_3d: SimpleErrorStats,
}

#[derive(Serialize)]
struct QuantRoundtripErrorStats {
    pos_n: SimpleErrorStats,
    pos_e: SimpleErrorStats,
    alt: SimpleErrorStats,
    vel_n: SimpleErrorStats,
    vel_e: SimpleErrorStats,
    vel_d: SimpleErrorStats,
}

#[derive(Serialize)]
struct QuantRecoveryErrorStats {
    lat: SimpleErrorStats,
    lon: SimpleErrorStats,
    alt: SimpleErrorStats,
    horiz: SimpleErrorStats,
}

#[derive(Serialize)]
struct StateDetectionErrorStats {
    burn: SimpleErrorStats,
    coast: SimpleErrorStats,
    rec: SimpleErrorStats,
}

#[derive(Serialize)]
struct ErrorStats {
    eskf: Option<EskfErrorStats>,
    quantized_flight: Option<QuantizedFlightErrorStats>,
    quant_roundtrip: Option<QuantRoundtripErrorStats>,
    quant_recovery: Option<QuantRecoveryErrorStats>,
    state_detection: Option<StateDetectionErrorStats>,
}

const SCALAR_TARGET_POINTS: usize = 1200;

#[derive(Clone)]
struct SimCacheEntry {
    rocket: RocketConfig,
    environment: EnvironmentConfig,
    sim_result: aloe_sim::sim::SimResult,
}

static SIM_CACHE: LazyLock<Mutex<Option<SimCacheEntry>>> = LazyLock::new(|| Mutex::new(None));

/// Creates the Axum router with all routes
pub fn create_router() -> Router {
    Router::new()
        .route("/api/simulate", post(handle_simulate))
        .route("/", get(handle_index))
        .route("/{*path}", get(handle_static))
}

#[derive(Debug, Clone, Deserialize)]
struct RocketConfig {
    dry_mass: f64,
    fuel_mass: f64,
    oxidizer_mass: f64,
    cg_full: f64,
    cg_empty: f64,
    cp_location: f64,
    inertia_x: f64,
    inertia_y: f64,
    inertia_z: f64,
    thrust: f64,
    burn_time: f64,
    drag_coeff: f64,
    normal_force_coeff: f64,
    ref_area: f64,
    isp: f64,
    nozzle_location: f64,
    launch_delay: f64,
    launch_rod_length: f64,
    spin_rate: f64,
    thrust_cant: f64,
    nozzle_exit_pressure_psf: f64,
    nozzle_area_ft2: f64,
    pad_static_friction: f64,
    pad_dynamic_friction: f64,
    pad_spring_coeff_lbs_ft: f64,
    pad_damping_coeff_lbs_ft_s: f64,
}

impl PartialEq for RocketConfig {
    fn eq(&self, other: &Self) -> bool {
        self.dry_mass.to_bits() == other.dry_mass.to_bits()
            && self.fuel_mass.to_bits() == other.fuel_mass.to_bits()
            && self.oxidizer_mass.to_bits() == other.oxidizer_mass.to_bits()
            && self.cg_full.to_bits() == other.cg_full.to_bits()
            && self.cg_empty.to_bits() == other.cg_empty.to_bits()
            && self.cp_location.to_bits() == other.cp_location.to_bits()
            && self.inertia_x.to_bits() == other.inertia_x.to_bits()
            && self.inertia_y.to_bits() == other.inertia_y.to_bits()
            && self.inertia_z.to_bits() == other.inertia_z.to_bits()
            && self.thrust.to_bits() == other.thrust.to_bits()
            && self.burn_time.to_bits() == other.burn_time.to_bits()
            && self.drag_coeff.to_bits() == other.drag_coeff.to_bits()
            && self.normal_force_coeff.to_bits() == other.normal_force_coeff.to_bits()
            && self.ref_area.to_bits() == other.ref_area.to_bits()
            && self.isp.to_bits() == other.isp.to_bits()
            && self.nozzle_location.to_bits() == other.nozzle_location.to_bits()
            && self.launch_delay.to_bits() == other.launch_delay.to_bits()
            && self.launch_rod_length.to_bits() == other.launch_rod_length.to_bits()
            && self.spin_rate.to_bits() == other.spin_rate.to_bits()
            && self.thrust_cant.to_bits() == other.thrust_cant.to_bits()
            && self.nozzle_exit_pressure_psf.to_bits() == other.nozzle_exit_pressure_psf.to_bits()
            && self.nozzle_area_ft2.to_bits() == other.nozzle_area_ft2.to_bits()
            && self.pad_static_friction.to_bits() == other.pad_static_friction.to_bits()
            && self.pad_dynamic_friction.to_bits() == other.pad_dynamic_friction.to_bits()
            && self.pad_spring_coeff_lbs_ft.to_bits() == other.pad_spring_coeff_lbs_ft.to_bits()
            && self.pad_damping_coeff_lbs_ft_s.to_bits()
                == other.pad_damping_coeff_lbs_ft_s.to_bits()
    }
}

impl Eq for RocketConfig {}

#[derive(Debug, Clone, Deserialize)]
struct EnvironmentConfig {
    gravity: f64,
    wind_north: f64,
    wind_east: f64,
    wind_down: f64,
    air_density: f64,
    sim_dt: f64,
    max_time: f64,
}

impl PartialEq for EnvironmentConfig {
    fn eq(&self, other: &Self) -> bool {
        self.gravity.to_bits() == other.gravity.to_bits()
            && self.wind_north.to_bits() == other.wind_north.to_bits()
            && self.wind_east.to_bits() == other.wind_east.to_bits()
            && self.wind_down.to_bits() == other.wind_down.to_bits()
            && self.air_density.to_bits() == other.air_density.to_bits()
            && self.sim_dt.to_bits() == other.sim_dt.to_bits()
            && self.max_time.to_bits() == other.max_time.to_bits()
    }
}

impl Eq for EnvironmentConfig {}

#[derive(Debug, Clone, Deserialize)]
struct SensorRequestConfig {
    noise_scale: f64,
    seed: u64,
    bmi088_accel_enabled: bool,
    bmi088_gyro_enabled: bool,
    adxl375_enabled: bool,
    lis3mdl_enabled: bool,
    ms5611_enabled: bool,
    gps_enabled: bool,
    bmi088_accel_rate_hz: f64,
    bmi088_gyro_rate_hz: f64,
    adxl375_rate_hz: f64,
    lis3mdl_rate_hz: f64,
    ms5611_rate_hz: f64,
    gps_rate_hz: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct StageTuningConfig {
    accel_noise_density: f64,
    gyro_noise_density: f64,
    accel_bias_instability: f64,
    gyro_bias_instability: f64,
    pos_process_noise: f64,
    r_gps_pos: f64,
    r_gps_vel: f64,
    r_baro: f64,
    r_mag: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct FilterRequestConfig {
    ground_pressure_mbar: f64,
    mag_declination_deg: f64,
    mag_dip_deg: f64,
    home_lat_deg: f64,
    home_lon_deg: f64,
    home_alt_m: f64,
    launch_accel_thresh: f64,
    launch_vel_thresh: f64,
    burnout_accel_thresh: f64,
    min_ascent_time: f64,
    apogee_descent_thresh: f64,
    min_coast_time: f64,
    landing_vel_thresh: f64,
    landing_alt_thresh: f64,
    landing_confirm_window: f64,
    high_velocity_baro_thresh: f64,
    stage_tuning: Vec<StageTuningConfig>,
}

#[derive(Debug, Clone, Deserialize)]
struct SimulationOptions {
    no_sensors: bool,
    no_filter: bool,
}

/// Simulation configuration from typed JSON API
#[derive(Debug, Clone, Deserialize)]
struct SimConfig {
    rocket: RocketConfig,
    environment: EnvironmentConfig,
    sensors: SensorRequestConfig,
    filter: FilterRequestConfig,
    options: SimulationOptions,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            // 30km Sounding Rocket Preset
            rocket: RocketConfig {
                dry_mass: 80.0,
                fuel_mass: 40.0,
                oxidizer_mass: 80.0,
                cg_full: 1.5,
                cg_empty: 1.4,
                cp_location: 2.0,
                inertia_x: 0.1,
                inertia_y: 3.0,
                inertia_z: 3.0,
                thrust: 18000.0,
                burn_time: 12.0,
                drag_coeff: 0.38,
                normal_force_coeff: 12.0,
                ref_area: 0.045,
                isp: 200.0,
                nozzle_location: 3.0,
                launch_delay: 1.0,
                launch_rod_length: 3.0,
                spin_rate: 0.0,
                thrust_cant: 0.0,
                nozzle_exit_pressure_psf: 2116.22,
                nozzle_area_ft2: 0.01,
                pad_static_friction: 0.8,
                pad_dynamic_friction: 0.4,
                pad_spring_coeff_lbs_ft: 10000.0,
                pad_damping_coeff_lbs_ft_s: 5000.0,
            },
            environment: EnvironmentConfig {
                gravity: 9.81,
                wind_north: 5.0,
                wind_east: 0.0,
                wind_down: 0.0,
                air_density: 1.225,
                sim_dt: 0.001,
                max_time: 400.0,
            },
            sensors: SensorRequestConfig {
                noise_scale: 1.0,
                seed: 42,
                bmi088_accel_enabled: true,
                bmi088_gyro_enabled: true,
                adxl375_enabled: true,
                lis3mdl_enabled: true,
                ms5611_enabled: true,
                gps_enabled: true,
                bmi088_accel_rate_hz: 1000.0,
                bmi088_gyro_rate_hz: 1000.0,
                adxl375_rate_hz: 3200.0,
                lis3mdl_rate_hz: 100.0,
                ms5611_rate_hz: 50.0,
                gps_rate_hz: 10.0,
            },
            filter: FilterRequestConfig {
                ground_pressure_mbar: 1013.25,
                mag_declination_deg: 0.0,
                mag_dip_deg: 60.0,
                home_lat_deg: 35.0,
                home_lon_deg: -106.0,
                home_alt_m: 1500.0,
                launch_accel_thresh: 20.0,
                launch_vel_thresh: 10.0,
                burnout_accel_thresh: 2.0,
                min_ascent_time: 0.5,
                apogee_descent_thresh: 1.0,
                min_coast_time: 2.0,
                landing_vel_thresh: 0.5,
                landing_alt_thresh: 100.0,
                landing_confirm_window: 2.0,
                high_velocity_baro_thresh: 170.0,
                stage_tuning: vec![
                    StageTuningConfig {
                        accel_noise_density: 0.2236,
                        gyro_noise_density: 0.03728,
                        accel_bias_instability: 0.01,
                        gyro_bias_instability: 3.728e-5,
                        pos_process_noise: 1.0,
                        r_gps_pos: 61.05,
                        r_gps_vel: 0.07197,
                        r_baro: 0.1,
                        r_mag: 1.0,
                    },
                    StageTuningConfig {
                        accel_noise_density: 0.02430,
                        gyro_noise_density: 0.01389,
                        accel_bias_instability: 0.002683,
                        gyro_bias_instability: 1e-5,
                        pos_process_noise: 0.1389,
                        r_gps_pos: 0.1,
                        r_gps_vel: 0.04394,
                        r_baro: 2.236,
                        r_mag: 0.01179,
                    },
                    StageTuningConfig {
                        accel_noise_density: 0.01,
                        gyro_noise_density: 0.1,
                        accel_bias_instability: 1e-6,
                        gyro_bias_instability: 1e-7,
                        pos_process_noise: 0.004394,
                        r_gps_pos: 0.1,
                        r_gps_vel: 0.04394,
                        r_baro: 50.0,
                        r_mag: 1.0,
                    },
                    StageTuningConfig {
                        accel_noise_density: 0.01,
                        gyro_noise_density: 0.01389,
                        accel_bias_instability: 1e-6,
                        gyro_bias_instability: 1e-3,
                        pos_process_noise: 0.007197,
                        r_gps_pos: 0.1,
                        r_gps_vel: 0.01,
                        r_baro: 50.0,
                        r_mag: 0.002683,
                    },
                ],
            },
            options: SimulationOptions {
                no_sensors: false,
                no_filter: false,
            },
        }
    }
}

/// Handle simulation request
#[instrument(skip(config))]
async fn handle_simulate(Json(config): Json<SimConfig>) -> Json<FullSimulationResponse> {
    info!("Received /api/simulate request");
    debug!("Parsed Config: {:?}", config);

    let config_for_run = config.clone();
    match tokio::task::spawn_blocking(move || {
        std::panic::catch_unwind(|| run_full_simulation(&config_for_run))
    })
    .await
    {
        Ok(Ok(results)) => {
            info!(
                "Simulation completed successfully. Returning {} points.",
                results.time.len()
            );
            Json(results)
        }
        Ok(Err(payload)) => {
            let message = if let Some(s) = payload.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };

            error!("Simulation PANIC: {}", message);

            Json(FullSimulationResponse {
                time: vec![],
                altitude: vec![],
                velocity: vec![],
                acceleration: vec![],
                force: vec![],
                mass: vec![],
                position_x: vec![],
                position_y: vec![],
                position_z: vec![],
                velocity_x: vec![],
                velocity_y: vec![],
                velocity_z: vec![],
                state_changes_sim: vec![],
                state_changes_eskf: vec![],
                sensor_data: GuiSensorData::empty(),
                filter_data: FilterData::empty(),
                error_stats: None,
                true_accel_x: vec![],
                true_accel_y: vec![],
                true_accel_z: vec![],
                true_gyro_x: vec![],
                true_gyro_y: vec![],
                true_gyro_z: vec![],
                apogee: 0.0,
                max_velocity: 0.0,
                flight_time: 0.0,
                success: false,
                error_message: Some(message),
            })
        }
        Err(join_err) => {
            let message = format!("simulation task failed: {join_err}");
            error!("{}", message);

            Json(FullSimulationResponse {
                time: vec![],
                altitude: vec![],
                velocity: vec![],
                acceleration: vec![],
                force: vec![],
                mass: vec![],
                position_x: vec![],
                position_y: vec![],
                position_z: vec![],
                velocity_x: vec![],
                velocity_y: vec![],
                velocity_z: vec![],
                state_changes_sim: vec![],
                state_changes_eskf: vec![],
                sensor_data: GuiSensorData::empty(),
                filter_data: FilterData::empty(),
                error_stats: None,
                true_accel_x: vec![],
                true_accel_y: vec![],
                true_accel_z: vec![],
                true_gyro_x: vec![],
                true_gyro_y: vec![],
                true_gyro_z: vec![],
                apogee: 0.0,
                max_velocity: 0.0,
                flight_time: 0.0,
                success: false,
                error_message: Some(message),
            })
        }
    }
}

async fn handle_index() -> impl IntoResponse {
    serve_embedded_asset("index.html")
}

async fn handle_static(Path(path): Path<String>) -> impl IntoResponse {
    let normalized = if path.is_empty() {
        "index.html"
    } else {
        path.as_str()
    };
    serve_embedded_asset(normalized)
}

fn serve_embedded_asset(path: &str) -> Response<Body> {
    if let Some(file) = FrontendAssets::get(path) {
        let mime = mime_guess::from_path(path).first_or_octet_stream();
        let mut response = Response::new(Body::from(file.data.into_owned()));
        *response.status_mut() = StatusCode::OK;
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_str(mime.as_ref())
                .unwrap_or_else(|_| HeaderValue::from_static("application/octet-stream")),
        );
        return response;
    }

    Response::builder()
        .status(StatusCode::NOT_FOUND)
        .body(Body::empty())
        .expect("response builder should not fail")
}

#[derive(Serialize)]
struct FullSimulationResponse {
    time: Vec<f64>,
    altitude: Vec<f64>,
    velocity: Vec<f64>,
    acceleration: Vec<f64>,
    force: Vec<f64>,
    mass: Vec<f64>,
    position_x: Vec<f64>,
    position_y: Vec<f64>,
    position_z: Vec<f64>,
    velocity_x: Vec<f64>, // NED north velocity (m/s)
    velocity_y: Vec<f64>, // NED east velocity (m/s)
    velocity_z: Vec<f64>, // NED down velocity (m/s)
    state_changes_sim: Vec<StateChange>,
    state_changes_eskf: Vec<StateChange>,
    sensor_data: GuiSensorData,
    filter_data: FilterData,
    error_stats: Option<ErrorStats>,
    // True sensor values (perfect, no noise)
    true_accel_x: Vec<f64>,
    true_accel_y: Vec<f64>,
    true_accel_z: Vec<f64>,
    true_gyro_x: Vec<f64>,
    true_gyro_y: Vec<f64>,
    true_gyro_z: Vec<f64>,
    // Key metrics
    apogee: f64,       // Maximum altitude (m)
    max_velocity: f64, // Maximum velocity (m/s)
    flight_time: f64,  // Total flight time (s)
    success: bool,
    /// Optional error message (populated when simulation fails)
    error_message: Option<String>,
}

#[derive(Serialize)]
struct StateChange {
    time: f64,
    state: String,
    description: String,
}

#[derive(Serialize)]
struct GuiSensorData {
    accel_x: Vec<f64>,
    accel_y: Vec<f64>,
    accel_z: Vec<f64>,
    gyro_x: Vec<f64>,
    gyro_y: Vec<f64>,
    gyro_z: Vec<f64>,
    baro_pressure: Vec<f64>,
    mag_x: Vec<f64>,
    mag_y: Vec<f64>,
    mag_z: Vec<f64>,
    gps_x: Vec<Option<f64>>,
    gps_y: Vec<Option<f64>>,
    gps_z: Vec<Option<f64>>,
    gps_vel_x: Vec<Option<f64>>,
    gps_vel_y: Vec<Option<f64>>,
    gps_vel_z: Vec<Option<f64>>,
    adxl_x: Vec<f64>,
    adxl_y: Vec<f64>,
    adxl_z: Vec<f64>,
}

impl GuiSensorData {
    fn empty() -> Self {
        Self {
            accel_x: vec![],
            accel_y: vec![],
            accel_z: vec![],
            gyro_x: vec![],
            gyro_y: vec![],
            gyro_z: vec![],
            baro_pressure: vec![],
            mag_x: vec![],
            mag_y: vec![],
            mag_z: vec![],
            gps_x: vec![],
            gps_y: vec![],
            gps_z: vec![],
            gps_vel_x: vec![],
            gps_vel_y: vec![],
            gps_vel_z: vec![],
            adxl_x: vec![],
            adxl_y: vec![],
            adxl_z: vec![],
        }
    }
}

#[derive(Serialize)]
struct FilterData {
    est_pos_x: Vec<f64>,
    est_pos_y: Vec<f64>,
    est_pos_z: Vec<f64>,
    est_vel_x: Vec<f64>,
    est_vel_y: Vec<f64>,
    est_vel_z: Vec<f64>,
    #[serde(skip)]
    est_vel_mag: Vec<f64>,
    quantized_est_pos_x: Vec<f64>,
    quantized_est_pos_y: Vec<f64>,
    quantized_est_pos_z: Vec<f64>,
    quantized_est_vel_x: Vec<f64>,
    quantized_est_vel_y: Vec<f64>,
    quantized_est_vel_z: Vec<f64>,
}

impl FilterData {
    fn empty() -> Self {
        Self {
            est_pos_x: vec![],
            est_pos_y: vec![],
            est_pos_z: vec![],
            est_vel_x: vec![],
            est_vel_y: vec![],
            est_vel_z: vec![],
            est_vel_mag: vec![],
            quantized_est_pos_x: vec![],
            quantized_est_pos_y: vec![],
            quantized_est_pos_z: vec![],
            quantized_est_vel_x: vec![],
            quantized_est_vel_y: vec![],
            quantized_est_vel_z: vec![],
        }
    }
}

/// Interpolates simulation data (high freq) to match filter timestamps (lower freq)
fn align_ground_truth(sim_time: &[f64], sim_data: &[f64], filter_time: &[f64]) -> Vec<f64> {
    if sim_time.is_empty() || sim_data.is_empty() {
        warn!("align_ground_truth received empty sim data");
        return vec![];
    }

    let mut aligned = Vec::with_capacity(filter_time.len());
    let mut sim_idx = 0;

    for &t_target in filter_time {
        // Find the two sim points bounding this time
        while sim_idx < sim_time.len() - 1 && sim_time[sim_idx + 1] < t_target {
            sim_idx += 1;
        }

        if sim_idx >= sim_time.len() - 1 {
            aligned.push(sim_data.last().copied().unwrap_or(0.0));
            continue;
        }

        let t0 = sim_time[sim_idx];
        let t1 = sim_time[sim_idx + 1];
        let v0 = sim_data[sim_idx];
        let v1 = sim_data[sim_idx + 1];

        // Linear interpolation
        let denom = t1 - t0;
        let frac = if denom.abs() > 1e-9 {
            (t_target - t0) / denom
        } else {
            0.0
        };
        aligned.push(v0 + frac * (v1 - v0));
    }
    aligned
}

/// Run full 6-DOF simulation
#[instrument(skip(config), fields(
    dry_mass = config.rocket.dry_mass,
    thrust = config.rocket.thrust,
    burn_time = config.rocket.burn_time
))]
fn run_full_simulation(config: &SimConfig) -> FullSimulationResponse {
    info!("Starting run_full_simulation");
    let rocket_params = RocketParams {
        dry_mass: config.rocket.dry_mass,
        fuel_mass: config.rocket.fuel_mass,
        oxidizer_mass: config.rocket.oxidizer_mass,
        inertia_tensor: Vector3::new(
            config.rocket.inertia_x,
            config.rocket.inertia_y,
            config.rocket.inertia_z,
        ),
        cg_full: config.rocket.cg_full,
        cg_empty: config.rocket.cg_empty,
        cp_location: config.rocket.cp_location,
        ref_area: config.rocket.ref_area,
        drag_coeff_axial: config.rocket.drag_coeff,
        normal_force_coeff: config.rocket.normal_force_coeff,
        thrust_curve: vec![
            (0.0, config.rocket.thrust),
            (config.rocket.burn_time, config.rocket.thrust),
            (config.rocket.burn_time + 0.1, 0.0),
        ],
        burn_time: config.rocket.burn_time,
        isp: config.rocket.isp,
        nozzle_location: config.rocket.nozzle_location,
        gravity: config.environment.gravity,
        air_density_sea_level: config.environment.air_density,
        launch_rod_length: config.rocket.launch_rod_length,
        wind_velocity_ned: Vector3::new(
            config.environment.wind_north,
            config.environment.wind_east,
            config.environment.wind_down,
        ),
        launch_delay: config.rocket.launch_delay,
        spin_rate: config.rocket.spin_rate,
        thrust_cant: config.rocket.thrust_cant,
        nozzle_exit_pressure_psf: config.rocket.nozzle_exit_pressure_psf,
        nozzle_area_ft2: config.rocket.nozzle_area_ft2,
        pad_static_friction: config.rocket.pad_static_friction,
        pad_dynamic_friction: config.rocket.pad_dynamic_friction,
        pad_spring_coeff_lbs_ft: config.rocket.pad_spring_coeff_lbs_ft,
        pad_damping_coeff_lbs_ft_s: config.rocket.pad_damping_coeff_lbs_ft_s,
        sim_dt: config.environment.sim_dt,
        max_time: config.environment.max_time,
    };

    let cached_sim = {
        let cache = SIM_CACHE.lock().expect("simulation cache poisoned");
        cache.as_ref().and_then(|entry| {
            if entry.rocket == config.rocket && entry.environment == config.environment {
                Some(entry.sim_result.clone())
            } else {
                None
            }
        })
    };

    let sim_result = if let Some(sim_result) = cached_sim {
        debug!("Reusing cached physics simulation result");
        sim_result
    } else {
        debug!("Running physics sim_6dof...");
        let sim_result = simulate_6dof(&rocket_params);
        let mut cache = SIM_CACHE.lock().expect("simulation cache poisoned");
        *cache = Some(SimCacheEntry {
            rocket: config.rocket.clone(),
            environment: config.environment.clone(),
            sim_result: sim_result.clone(),
        });
        sim_result
    };
    debug!("Physics complete. Steps: {}", sim_result.time.len());

    if let Some(final_time) = sim_result.time.last() {
        debug!("Sim Final Time: {:.2}s", final_time);
    } else {
        error!("Sim returned NO TIME STEPS");
    }

    // Convert SimResult to the GUI format
    let time: Vec<f64> = sim_result.time.clone();
    let altitude: Vec<f64> = sim_result.pos.iter().map(|p| -p.z).collect(); // Convert from NED to altitude
    let velocity: Vec<f64> = sim_result.vel.iter().map(|v| v.norm()).collect();
    let acceleration: Vec<f64> = sim_result.accel_body.iter().map(|a| a.x).collect(); // Axial acceleration
    let force: Vec<f64> = sim_result
        .accel_body
        .iter()
        .zip(&sim_result.vel)
        .map(|(a, _)| a.x * (rocket_params.dry_mass + rocket_params.total_reactant_mass()))
        .collect(); // Approximate
    let mass: Vec<f64> = time
        .iter()
        .map(|&t| {
            if t < config.rocket.burn_time {
                rocket_params.dry_mass
                    + rocket_params.total_reactant_mass()
                        * (1.0 - t / config.rocket.burn_time).max(0.0)
            } else {
                rocket_params.dry_mass
            }
        })
        .collect();
    let position_x: Vec<f64> = sim_result.pos.iter().map(|p| p.x).collect();
    let position_y: Vec<f64> = sim_result.pos.iter().map(|p| p.y).collect();
    let position_z: Vec<f64> = sim_result.pos.iter().map(|p| -p.z).collect(); // NED to altitude

    // Extract velocity components (NED frame)
    let velocity_x: Vec<f64> = sim_result.vel.iter().map(|v| v.x).collect(); // North
    let velocity_y: Vec<f64> = sim_result.vel.iter().map(|v| v.y).collect(); // East
    let velocity_z: Vec<f64> = sim_result.vel.iter().map(|v| v.z).collect(); // Down

    // Extract true sensor values (perfect, no noise)
    let true_accel_x: Vec<f64> = sim_result.accel_body.iter().map(|a| a.x).collect();
    let true_accel_y: Vec<f64> = sim_result.accel_body.iter().map(|a| a.y).collect();
    let true_accel_z: Vec<f64> = sim_result.accel_body.iter().map(|a| a.z).collect();
    let true_gyro_x: Vec<f64> = sim_result.ang_vel.iter().map(|g| g.x).collect();
    let true_gyro_y: Vec<f64> = sim_result.ang_vel.iter().map(|g| g.y).collect();
    let true_gyro_z: Vec<f64> = sim_result.ang_vel.iter().map(|g| g.z).collect();

    // Generate state changes for sim
    let state_changes_sim = generate_state_changes(
        &time,
        &sim_result.pos,
        &sim_result.vel,
        sim_result.ascent_time,
        sim_result.coast_time,
        sim_result.descent_time,
    );

    if config.options.no_sensors {
        info!("Sensors DISABLED by config.");

        let scalar_step = time.len().div_ceil(SCALAR_TARGET_POINTS).max(1);
        let time = downsample_vec(&time, scalar_step);
        let altitude = downsample_vec(&altitude, scalar_step);
        let velocity = downsample_vec(&velocity, scalar_step);
        let acceleration = downsample_vec(&acceleration, scalar_step);
        let force = downsample_vec(&force, scalar_step);
        let mass = downsample_vec(&mass, scalar_step);
        let position_x = downsample_vec(&position_x, scalar_step);
        let position_y = downsample_vec(&position_y, scalar_step);
        let position_z = downsample_vec(&position_z, scalar_step);
        let velocity_x = downsample_vec(&velocity_x, scalar_step);
        let velocity_y = downsample_vec(&velocity_y, scalar_step);
        let velocity_z = downsample_vec(&velocity_z, scalar_step);

        // Calculate key metrics
        let apogee = altitude.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let max_velocity = velocity.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let flight_time = *time.last().unwrap_or(&0.0);

        debug!(
            "Simulation (no sensors): time steps {}, apogee {:.2}m, max_vel {:.2}m/s, flight_time {:.2}s",
            time.len(),
            apogee,
            max_velocity,
            flight_time
        );

        FullSimulationResponse {
            time,
            altitude,
            velocity,
            acceleration,
            force,
            mass,
            position_x,
            position_y,
            position_z,
            velocity_x: velocity_x.clone(),
            velocity_y: velocity_y.clone(),
            velocity_z: velocity_z.clone(),
            state_changes_sim,
            state_changes_eskf: vec![],
            sensor_data: GuiSensorData::empty(),
            filter_data: FilterData::empty(),
            error_stats: None,
            true_accel_x: vec![],
            true_accel_y: vec![],
            true_accel_z: vec![],
            true_gyro_x: vec![],
            true_gyro_y: vec![],
            true_gyro_z: vec![],
            apogee,
            max_velocity,
            flight_time,
            success: true,
            error_message: None,
        }
    } else {
        debug!("Generating Sensor Data...");
        let sensor_config = SensorConfig {
            noise_scale: config.sensors.noise_scale,
            accel_noise_std: 0.01,
            gyro_noise_std: 0.001,
            mag_noise_std: 0.001,
            baro_noise_std: 0.1,
            gps_pos_noise_std: 1.0,
            gps_vel_noise_std: 0.1,
            accel_bias: Vector3::zeros(),
            gyro_bias: Vector3::zeros(),
            seed: config.sensors.seed,
            accel_enabled: config.sensors.bmi088_accel_enabled || config.sensors.adxl375_enabled,
            gyro_enabled: config.sensors.bmi088_gyro_enabled,
            mag_enabled: config.sensors.lis3mdl_enabled,
            baro_enabled: config.sensors.ms5611_enabled,
            gps_enabled: config.sensors.gps_enabled,
            bmi088_accel_rate_hz: config.sensors.bmi088_accel_rate_hz,
            bmi088_gyro_rate_hz: config.sensors.bmi088_gyro_rate_hz,
            adxl375_rate_hz: config.sensors.adxl375_rate_hz,
            lis3mdl_rate_hz: config.sensors.lis3mdl_rate_hz,
            ms5611_rate_hz: config.sensors.ms5611_rate_hz,
            gps_rate_hz: config.sensors.gps_rate_hz,
            accel_saturation: 200.0,
            gyro_saturation: 34.9,
            chaos: ChaosConfig::default(),
        };
        let sensor_data_sim = generate_sensor_data(&sim_result, &sensor_config);
        debug!("Sensor Data Steps: {}", sensor_data_sim.time.len());

        // Generate sensor data for GUI
        let gui_sensor_data = GuiSensorData {
            accel_x: if config.sensors.bmi088_accel_enabled {
                hold_last_vector_component(&sensor_data_sim.bmi088_accel_meas, |v| v.x)
            } else {
                vec![]
            },
            accel_y: if config.sensors.bmi088_accel_enabled {
                hold_last_vector_component(&sensor_data_sim.bmi088_accel_meas, |v| v.y)
            } else {
                vec![]
            },
            accel_z: if config.sensors.bmi088_accel_enabled {
                hold_last_vector_component(&sensor_data_sim.bmi088_accel_meas, |v| v.z)
            } else {
                vec![]
            },
            gyro_x: if config.sensors.bmi088_gyro_enabled {
                hold_last_vector_component(&sensor_data_sim.gyro_meas, |v| v.x)
            } else {
                vec![]
            },
            gyro_y: if config.sensors.bmi088_gyro_enabled {
                hold_last_vector_component(&sensor_data_sim.gyro_meas, |v| v.y)
            } else {
                vec![]
            },
            gyro_z: if config.sensors.bmi088_gyro_enabled {
                hold_last_vector_component(&sensor_data_sim.gyro_meas, |v| v.z)
            } else {
                vec![]
            },
            baro_pressure: if config.sensors.ms5611_enabled {
                hold_last_scalar_samples(&sensor_data_sim.baro_pressure)
            } else {
                vec![]
            },
            mag_x: if config.sensors.lis3mdl_enabled {
                hold_last_vector_component(&sensor_data_sim.mag_meas, |v| v.x)
            } else {
                vec![]
            },
            mag_y: if config.sensors.lis3mdl_enabled {
                hold_last_vector_component(&sensor_data_sim.mag_meas, |v| v.y)
            } else {
                vec![]
            },
            mag_z: if config.sensors.lis3mdl_enabled {
                hold_last_vector_component(&sensor_data_sim.mag_meas, |v| v.z)
            } else {
                vec![]
            },
            gps_x: if config.sensors.gps_enabled {
                sensor_data_sim
                    .gps_pos
                    .iter()
                    .map(|opt| opt.map(|v| v.x))
                    .collect()
            } else {
                vec![]
            },
            gps_y: if config.sensors.gps_enabled {
                sensor_data_sim
                    .gps_pos
                    .iter()
                    .map(|opt| opt.map(|v| v.y))
                    .collect()
            } else {
                vec![]
            },
            gps_z: if config.sensors.gps_enabled {
                sensor_data_sim
                    .gps_pos
                    .iter()
                    .map(|opt| opt.map(|v| v.z))
                    .collect()
            } else {
                vec![]
            },
            gps_vel_x: if config.sensors.gps_enabled {
                sensor_data_sim
                    .gps_vel
                    .iter()
                    .map(|opt| opt.map(|v| v.x))
                    .collect()
            } else {
                vec![]
            },
            gps_vel_y: if config.sensors.gps_enabled {
                sensor_data_sim
                    .gps_vel
                    .iter()
                    .map(|opt| opt.map(|v| v.y))
                    .collect()
            } else {
                vec![]
            },
            gps_vel_z: if config.sensors.gps_enabled {
                sensor_data_sim
                    .gps_vel
                    .iter()
                    .map(|opt| opt.map(|v| v.z))
                    .collect()
            } else {
                vec![]
            },
            adxl_x: if config.sensors.adxl375_enabled {
                hold_last_vector_component(&sensor_data_sim.adxl375_accel_meas, |v| v.x)
            } else {
                vec![]
            },
            adxl_y: if config.sensors.adxl375_enabled {
                hold_last_vector_component(&sensor_data_sim.adxl375_accel_meas, |v| v.y)
            } else {
                vec![]
            },
            adxl_z: if config.sensors.adxl375_enabled {
                hold_last_vector_component(&sensor_data_sim.adxl375_accel_meas, |v| v.z)
            } else {
                vec![]
            },
        };

        debug!("Running Filter...");
        let filter_config = FilterConfig {
            ground_pressure_mbar: config.filter.ground_pressure_mbar,
            mag_declination_deg: config.filter.mag_declination_deg,
            mag_dip_deg: config.filter.mag_dip_deg,
            home_lat_deg: config.filter.home_lat_deg,
            home_lon_deg: config.filter.home_lon_deg,
            home_alt_m: config.filter.home_alt_m,
            launch_accel_thresh: config.filter.launch_accel_thresh,
            launch_vel_thresh: config.filter.launch_vel_thresh,
            burnout_accel_thresh: config.filter.burnout_accel_thresh,
            min_ascent_time: config.filter.min_ascent_time,
            apogee_descent_thresh: config.filter.apogee_descent_thresh,
            min_coast_time: config.filter.min_coast_time,
            landing_vel_thresh: config.filter.landing_vel_thresh,
            landing_alt_thresh: config.filter.landing_alt_thresh,
            landing_confirm_window: config.filter.landing_confirm_window,
            high_velocity_baro_thresh: config.filter.high_velocity_baro_thresh,
            accel_noise_density: config
                .filter
                .stage_tuning
                .iter()
                .map(|stage| stage.accel_noise_density)
                .collect(),
            gyro_noise_density: config
                .filter
                .stage_tuning
                .iter()
                .map(|stage| stage.gyro_noise_density)
                .collect(),
            accel_bias_instability: config
                .filter
                .stage_tuning
                .iter()
                .map(|stage| stage.accel_bias_instability)
                .collect(),
            gyro_bias_instability: config
                .filter
                .stage_tuning
                .iter()
                .map(|stage| stage.gyro_bias_instability)
                .collect(),
            pos_process_noise: config
                .filter
                .stage_tuning
                .iter()
                .map(|stage| stage.pos_process_noise)
                .collect(),
            r_gps_pos: config
                .filter
                .stage_tuning
                .iter()
                .map(|stage| stage.r_gps_pos)
                .collect(),
            r_gps_vel: config
                .filter
                .stage_tuning
                .iter()
                .map(|stage| stage.r_gps_vel)
                .collect(),
            r_baro: config
                .filter
                .stage_tuning
                .iter()
                .map(|stage| stage.r_baro)
                .collect(),
            r_mag: config
                .filter
                .stage_tuning
                .iter()
                .map(|stage| stage.r_mag)
                .collect(),
        };
        let (filter_data, state_changes_eskf, error_stats) = if config.options.no_filter {
            info!("Filter DISABLED by config.");
            (FilterData::empty(), vec![], None)
        } else {
            let filter_result = run_filter(&sim_result, &sensor_data_sim, &filter_config);
            debug!("Filter Result Steps: {}", filter_result.position.len());

            let filter_data_temp = FilterData {
                est_pos_x: filter_result.position.iter().map(|p| p.x).collect(),
                est_pos_y: filter_result.position.iter().map(|p| p.y).collect(),
                est_pos_z: filter_result.position.iter().map(|p| p.z).collect(),
                est_vel_x: filter_result.velocity.iter().map(|v| v.x).collect(),
                est_vel_y: filter_result.velocity.iter().map(|v| v.y).collect(),
                est_vel_z: filter_result.velocity.iter().map(|v| v.z).collect(),
                est_vel_mag: vec![],
                quantized_est_pos_x: vec![],
                quantized_est_pos_y: vec![],
                quantized_est_pos_z: vec![],
                quantized_est_vel_x: vec![],
                quantized_est_vel_y: vec![],
                quantized_est_vel_z: vec![],
            };

            let est_vel_mag: Vec<f64> = filter_data_temp
                .est_vel_x
                .iter()
                .zip(&filter_data_temp.est_vel_y)
                .zip(&filter_data_temp.est_vel_z)
                .map(|((x, y), z)| (x * x + y * y + z * z).sqrt())
                .collect();

            // Compute quantized positions
            let quantized_est_pos_x: Vec<f64> = filter_data_temp
                .est_pos_x
                .iter()
                .map(|&x| x as i16 as f64)
                .collect();
            let quantized_est_pos_y: Vec<f64> = filter_data_temp
                .est_pos_y
                .iter()
                .map(|&y| y as i16 as f64)
                .collect();
            let quantized_est_pos_z: Vec<f64> = filter_data_temp
                .est_pos_z
                .iter()
                .map(|&z| ((z * 100.0) as i32 as f64) / 100.0)
                .collect();

            // Compute quantized velocities (stored as i16 in dm/s, per FlightData in quantize.rs)
            let quantized_est_vel_x: Vec<f64> = filter_data_temp
                .est_vel_x
                .iter()
                .map(|&vx| ((vx * 10.0) as i16 as f64) / 10.0)
                .collect();
            let quantized_est_vel_y: Vec<f64> = filter_data_temp
                .est_vel_y
                .iter()
                .map(|&vy| ((vy * 10.0) as i16 as f64) / 10.0)
                .collect();
            let quantized_est_vel_z: Vec<f64> = filter_data_temp
                .est_vel_z
                .iter()
                .map(|&vz| ((vz * 10.0) as i16 as f64) / 10.0)
                .collect();

            let filter_data = FilterData {
                est_pos_x: filter_data_temp.est_pos_x,
                est_pos_y: filter_data_temp.est_pos_y,
                est_pos_z: filter_data_temp.est_pos_z,
                est_vel_x: filter_data_temp.est_vel_x,
                est_vel_y: filter_data_temp.est_vel_y,
                est_vel_z: filter_data_temp.est_vel_z,
                est_vel_mag,
                quantized_est_pos_x,
                quantized_est_pos_y,
                quantized_est_pos_z,
                quantized_est_vel_x,
                quantized_est_vel_y,
                quantized_est_vel_z,
            };

            let state_changes_eskf = generate_state_changes(
                &filter_result.time,
                &filter_result.position,
                &filter_result.velocity,
                filter_result.ascent_time,
                filter_result.coast_time,
                filter_result.descent_time,
            );

            debug!("Calculating Error Stats...");
            let true_pos_z: Vec<f64> = sim_result.pos.iter().map(|p| p.z).collect();
            let true_vel_x: Vec<f64> = sim_result.vel.iter().map(|v| v.x).collect();
            let true_vel_y: Vec<f64> = sim_result.vel.iter().map(|v| v.y).collect();
            let true_vel_z: Vec<f64> = sim_result.vel.iter().map(|v| v.z).collect();
            let true_pos_n: Vec<f64> = sim_result.pos.iter().map(|p| p.x).collect();
            let true_pos_e: Vec<f64> = sim_result.pos.iter().map(|p| p.y).collect();

            let filter_time = &filter_result.time;
            let aligned_true_pos_n = align_ground_truth(&sim_result.time, &true_pos_n, filter_time);
            let aligned_true_pos_e = align_ground_truth(&sim_result.time, &true_pos_e, filter_time);
            let aligned_true_pos_z = align_ground_truth(&sim_result.time, &true_pos_z, filter_time);
            let aligned_true_vel_x = align_ground_truth(&sim_result.time, &true_vel_x, filter_time);
            let aligned_true_vel_y = align_ground_truth(&sim_result.time, &true_vel_y, filter_time);
            let aligned_true_vel_z = align_ground_truth(&sim_result.time, &true_vel_z, filter_time);

            let error_stats = calculate_error_stats(
                PositionData {
                    x: &aligned_true_pos_n,
                    y: &aligned_true_pos_e,
                    z: &aligned_true_pos_z,
                },
                PositionData {
                    x: &filter_data.est_pos_x,
                    y: &filter_data.est_pos_y,
                    z: &filter_data.est_pos_z,
                },
                PositionData {
                    x: &filter_data.quantized_est_pos_x,
                    y: &filter_data.quantized_est_pos_y,
                    z: &filter_data.quantized_est_pos_z,
                },
                VelocityData {
                    x: &aligned_true_vel_x,
                    y: &aligned_true_vel_y,
                    z: &aligned_true_vel_z,
                },
                VelocityData {
                    x: &filter_data.est_vel_x,
                    y: &filter_data.est_vel_y,
                    z: &filter_data.est_vel_z,
                },
                VelocityData {
                    x: &filter_data.quantized_est_vel_x,
                    y: &filter_data.quantized_est_vel_y,
                    z: &filter_data.quantized_est_vel_z,
                },
                sim_result.ascent_time,
                sim_result.coast_time,
                sim_result.descent_time,
                filter_result.ascent_time,
                filter_result.coast_time,
                filter_result.descent_time,
                filter_config.home_lat_deg,
                filter_config.home_lon_deg,
                filter_config.home_alt_m,
            );

            (filter_data, state_changes_eskf, Some(error_stats))
        };

        debug!("=== BEFORE DOWNSAMPLING ===");
        debug!("position_x length: {}", position_x.len());
        debug!("position_y length: {}", position_y.len());
        debug!("position_z length: {}", position_z.len());

        let max_alt = altitude.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        debug!("Max altitude: {}", max_alt);

        // Downsample data for frontend performance.
        // Keep one shared cadence across time-series and trajectory data so every plotted
        // channel can be indexed against the same time axis without frontend bookkeeping.
        let target_points = SCALAR_TARGET_POINTS;
        let (
            time,
            altitude,
            velocity,
            acceleration,
            force,
            mass,
            position_x,
            position_y,
            position_z,
            velocity_x,
            velocity_y,
            velocity_z,
            true_accel_x,
            true_accel_y,
            true_accel_z,
            true_gyro_x,
            true_gyro_y,
            true_gyro_z,
            sensor_data,
            filter_data,
        ) = if time.len() > target_points {
            let step = time.len().div_ceil(target_points);
            (
                downsample_vec(&time, step),
                downsample_vec(&altitude, step),
                downsample_vec(&velocity, step),
                downsample_vec(&acceleration, step),
                downsample_vec(&force, step),
                downsample_vec(&mass, step),
                downsample_vec(&position_x, step),
                downsample_vec(&position_y, step),
                downsample_vec(&position_z, step),
                downsample_vec(&velocity_x, step),
                downsample_vec(&velocity_y, step),
                downsample_vec(&velocity_z, step),
                downsample_vec(&true_accel_x, step),
                downsample_vec(&true_accel_y, step),
                downsample_vec(&true_accel_z, step),
                downsample_vec(&true_gyro_x, step),
                downsample_vec(&true_gyro_y, step),
                downsample_vec(&true_gyro_z, step),
                GuiSensorData {
                    accel_x: downsample_vec(&gui_sensor_data.accel_x, step),
                    accel_y: downsample_vec(&gui_sensor_data.accel_y, step),
                    accel_z: downsample_vec(&gui_sensor_data.accel_z, step),
                    gyro_x: downsample_vec(&gui_sensor_data.gyro_x, step),
                    gyro_y: downsample_vec(&gui_sensor_data.gyro_y, step),
                    gyro_z: downsample_vec(&gui_sensor_data.gyro_z, step),
                    baro_pressure: downsample_vec(&gui_sensor_data.baro_pressure, step),
                    mag_x: downsample_vec(&gui_sensor_data.mag_x, step),
                    mag_y: downsample_vec(&gui_sensor_data.mag_y, step),
                    mag_z: downsample_vec(&gui_sensor_data.mag_z, step),
                    gps_x: downsample_option_vec(&gui_sensor_data.gps_x, step),
                    gps_y: downsample_option_vec(&gui_sensor_data.gps_y, step),
                    gps_z: downsample_option_vec(&gui_sensor_data.gps_z, step),
                    gps_vel_x: downsample_option_vec(&gui_sensor_data.gps_vel_x, step),
                    gps_vel_y: downsample_option_vec(&gui_sensor_data.gps_vel_y, step),
                    gps_vel_z: downsample_option_vec(&gui_sensor_data.gps_vel_z, step),
                    adxl_x: downsample_vec(&gui_sensor_data.adxl_x, step),
                    adxl_y: downsample_vec(&gui_sensor_data.adxl_y, step),
                    adxl_z: downsample_vec(&gui_sensor_data.adxl_z, step),
                },
                FilterData {
                    est_pos_x: downsample_vec(&filter_data.est_pos_x, step),
                    est_pos_y: downsample_vec(&filter_data.est_pos_y, step),
                    est_pos_z: downsample_vec(&filter_data.est_pos_z, step),
                    est_vel_x: downsample_vec(&filter_data.est_vel_x, step),
                    est_vel_y: downsample_vec(&filter_data.est_vel_y, step),
                    est_vel_z: downsample_vec(&filter_data.est_vel_z, step),
                    est_vel_mag: downsample_vec(&filter_data.est_vel_mag, step),
                    quantized_est_pos_x: downsample_vec(&filter_data.quantized_est_pos_x, step),
                    quantized_est_pos_y: downsample_vec(&filter_data.quantized_est_pos_y, step),
                    quantized_est_pos_z: downsample_vec(&filter_data.quantized_est_pos_z, step),
                    quantized_est_vel_x: downsample_vec(&filter_data.quantized_est_vel_x, step),
                    quantized_est_vel_y: downsample_vec(&filter_data.quantized_est_vel_y, step),
                    quantized_est_vel_z: downsample_vec(&filter_data.quantized_est_vel_z, step),
                },
            )
        } else {
            (
                time,
                altitude,
                velocity,
                acceleration,
                force,
                mass,
                position_x,
                position_y,
                position_z,
                velocity_x,
                velocity_y,
                velocity_z,
                true_accel_x,
                true_accel_y,
                true_accel_z,
                true_gyro_x,
                true_gyro_y,
                true_gyro_z,
                gui_sensor_data,
                filter_data,
            )
        };

        // Calculate key metrics
        let apogee = altitude.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let max_velocity = velocity.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let flight_time = *time.last().unwrap_or(&0.0);

        info!(
            "Simulation (with sensors): apogee {:.2}m, max_vel {:.2}m/s, flight_time {:.2}s",
            apogee, max_velocity, flight_time
        );

        FullSimulationResponse {
            time,
            altitude,
            velocity,
            acceleration,
            force,
            mass,
            position_x,
            position_y,
            position_z,
            velocity_x,
            velocity_y,
            velocity_z,
            state_changes_sim,
            state_changes_eskf,
            sensor_data,
            filter_data,
            error_stats,
            true_accel_x,
            true_accel_y,
            true_accel_z,
            true_gyro_x,
            true_gyro_y,
            true_gyro_z,
            apogee,
            max_velocity,
            flight_time,
            success: true,
            error_message: None,
        }
    }
}

#[derive(Debug)]
struct PositionData<'a> {
    x: &'a [f64],
    y: &'a [f64],
    z: &'a [f64],
}

#[derive(Debug)]
struct VelocityData<'a> {
    x: &'a [f64],
    y: &'a [f64],
    z: &'a [f64],
}

#[allow(clippy::too_many_arguments)]
fn calculate_error_stats(
    true_pos: PositionData,
    est_pos: PositionData,
    quantized_pos: PositionData,
    true_vel: VelocityData,
    est_vel: VelocityData,
    quantized_vel: VelocityData,
    sim_ascent_time: Option<f64>,
    sim_coast_time: Option<f64>,
    sim_descent_time: Option<f64>,
    filter_ascent_time: Option<f64>,
    filter_coast_time: Option<f64>,
    filter_descent_time: Option<f64>,
    home_lat_deg: f64,
    home_lon_deg: f64,
    home_alt_m: f64,
) -> ErrorStats {
    // Generate position errors for N, E, D components
    // PositionData: .x = North, .y = East, .z = Down
    let n = true_pos.x.len();
    let pos_n_errors: Vec<f64> = (0..n).map(|i| est_pos.x[i] - true_pos.x[i]).collect(); // north error
    let pos_e_errors: Vec<f64> = (0..n).map(|i| est_pos.y[i] - true_pos.y[i]).collect(); // east error
    let pos_d_errors: Vec<f64> = (0..n).map(|i| est_pos.z[i] - true_pos.z[i]).collect(); // down error

    // 3D position errors
    let pos_3d_errors: Vec<f64> = (0..n)
        .map(|i| {
            (pos_n_errors[i].powi(2) + pos_e_errors[i].powi(2) + pos_d_errors[i].powi(2)).sqrt()
        })
        .collect();

    // Velocity errors components
    let vel_n_errors: Vec<f64> = (0..n).map(|i| est_vel.x[i] - true_vel.x[i]).collect();
    let vel_e_errors: Vec<f64> = (0..n).map(|i| est_vel.y[i] - true_vel.y[i]).collect();
    let vel_d_errors: Vec<f64> = (0..n).map(|i| est_vel.z[i] - true_vel.z[i]).collect();

    // Helper functions for statistics
    let calc_stats = |data: &[f64]| -> (f64, f64, f64, f64, f64, f64, f64) {
        let finite: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
        if finite.is_empty() {
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
        let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        let variance = finite.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / finite.len() as f64;
        let std = variance.sqrt();
        let rmse = (finite.iter().map(|x| x.powi(2)).sum::<f64>() / finite.len() as f64).sqrt();
        let mae = finite.iter().map(|x| x.abs()).sum::<f64>() / finite.len() as f64;
        let mut sorted = finite;
        let p95_idx = (0.95 * sorted.len() as f64) as usize;
        if p95_idx < sorted.len() {
            sorted.select_nth_unstable_by(p95_idx, |a, b| a.total_cmp(b));
        }
        let p95 = sorted.get(p95_idx).copied().unwrap_or(0.0);
        (min, max, mean, std, rmse, mae, p95)
    };

    let (pos_n_min, pos_n_max, pos_n_mean, pos_n_std, pos_n_rmse, pos_n_mae, pos_n_p95) =
        calc_stats(&pos_n_errors);
    let (pos_e_min, pos_e_max, pos_e_mean, pos_e_std, pos_e_rmse, pos_e_mae, pos_e_p95) =
        calc_stats(&pos_e_errors);
    let (pos_d_min, pos_d_max, pos_d_mean, pos_d_std, pos_d_rmse, pos_d_mae, pos_d_p95) =
        calc_stats(&pos_d_errors);
    let (vel_n_min, vel_n_max, vel_n_mean, vel_n_std, vel_n_rmse, vel_n_mae, vel_n_p95) =
        calc_stats(&vel_n_errors);
    let (vel_e_min, vel_e_max, vel_e_mean, vel_e_std, vel_e_rmse, vel_e_mae, vel_e_p95) =
        calc_stats(&vel_e_errors);
    let (vel_d_min, vel_d_max, vel_d_mean, vel_d_std, vel_d_rmse, vel_d_mae, vel_d_p95) =
        calc_stats(&vel_d_errors);
    let (pos_3d_min, pos_3d_max, pos_3d_mean, pos_3d_std, pos_3d_rmse, pos_3d_mae, pos_3d_p95) =
        calc_stats(&pos_3d_errors);

    // Quantized flight errors (ESKF vs quantized ESKF)
    let quant_pos_n_errors: Vec<f64> = (0..n).map(|i| est_pos.x[i] - quantized_pos.x[i]).collect();
    let quant_pos_e_errors: Vec<f64> = (0..n).map(|i| est_pos.y[i] - quantized_pos.y[i]).collect();
    let quant_alt_errors: Vec<f64> = (0..n)
        .map(|i| (-est_pos.z[i]) - (-quantized_pos.z[i]))
        .collect();
    let quant_pos_3d_errors: Vec<f64> = (0..n)
        .map(|i| {
            let dn = quant_pos_n_errors[i];
            let de = quant_pos_e_errors[i];
            let dalt = quant_alt_errors[i];
            (dn * dn + de * de + dalt * dalt).sqrt()
        })
        .collect();
    let quant_vel_n_errors: Vec<f64> = (0..n).map(|i| est_vel.x[i] - quantized_vel.x[i]).collect();
    let quant_vel_e_errors: Vec<f64> = (0..n).map(|i| est_vel.y[i] - quantized_vel.y[i]).collect();
    let quant_vel_d_errors: Vec<f64> = (0..n).map(|i| est_vel.z[i] - quantized_vel.z[i]).collect();

    let (
        quant_pos_n_min,
        quant_pos_n_max,
        quant_pos_n_mean,
        quant_pos_n_std,
        quant_pos_n_rmse,
        quant_pos_n_mae,
        quant_pos_n_p95,
    ) = calc_stats(&quant_pos_n_errors);
    let (
        quant_pos_e_min,
        quant_pos_e_max,
        quant_pos_e_mean,
        quant_pos_e_std,
        quant_pos_e_rmse,
        quant_pos_e_mae,
        quant_pos_e_p95,
    ) = calc_stats(&quant_pos_e_errors);
    let (
        quant_alt_min,
        quant_alt_max,
        quant_alt_mean,
        quant_alt_std,
        quant_alt_rmse,
        quant_alt_mae,
        quant_alt_p95,
    ) = calc_stats(&quant_alt_errors);
    let (
        quant_vel_n_min,
        quant_vel_n_max,
        quant_vel_n_mean,
        quant_vel_n_std,
        quant_vel_n_rmse,
        quant_vel_n_mae,
        quant_vel_n_p95,
    ) = calc_stats(&quant_vel_n_errors);
    let (
        quant_vel_e_min,
        quant_vel_e_max,
        quant_vel_e_mean,
        quant_vel_e_std,
        quant_vel_e_rmse,
        quant_vel_e_mae,
        quant_vel_e_p95,
    ) = calc_stats(&quant_vel_e_errors);
    let (
        quant_vel_d_min,
        quant_vel_d_max,
        quant_vel_d_mean,
        quant_vel_d_std,
        quant_vel_d_rmse,
        quant_vel_d_mae,
        quant_vel_d_p95,
    ) = calc_stats(&quant_vel_d_errors);
    let (
        quant_pos_3d_min,
        quant_pos_3d_max,
        quant_pos_3d_mean,
        quant_pos_3d_std,
        quant_pos_3d_rmse,
        quant_pos_3d_mae,
        quant_pos_3d_p95,
    ) = calc_stats(&quant_pos_3d_errors);

    // Quant roundtrip errors (true vs quantized)
    let quant_roundtrip_pos_n_errors: Vec<f64> =
        (0..n).map(|i| quantized_pos.x[i] - true_pos.x[i]).collect();
    let quant_roundtrip_pos_e_errors: Vec<f64> =
        (0..n).map(|i| quantized_pos.y[i] - true_pos.y[i]).collect();
    let quant_roundtrip_alt_errors: Vec<f64> = (0..n)
        .map(|i| (-quantized_pos.z[i]) - (-true_pos.z[i]))
        .collect(); // altitude error
    let _quant_roundtrip_pos_3d_errors: Vec<f64> = (0..n)
        .map(|i| {
            let dn = quant_roundtrip_pos_n_errors[i];
            let de = quant_roundtrip_pos_e_errors[i];
            let dalt = quant_roundtrip_alt_errors[i];
            (dn * dn + de * de + dalt * dalt).sqrt()
        })
        .collect();
    let quant_roundtrip_vel_n_errors: Vec<f64> =
        (0..n).map(|i| quantized_vel.x[i] - true_vel.x[i]).collect();
    let quant_roundtrip_vel_e_errors: Vec<f64> =
        (0..n).map(|i| quantized_vel.y[i] - true_vel.y[i]).collect();
    let quant_roundtrip_vel_d_errors: Vec<f64> =
        (0..n).map(|i| quantized_vel.z[i] - true_vel.z[i]).collect();

    let (
        quant_roundtrip_pos_n_min,
        quant_roundtrip_pos_n_max,
        quant_roundtrip_pos_n_mean,
        quant_roundtrip_pos_n_std,
        quant_roundtrip_pos_n_rmse,
        quant_roundtrip_pos_n_mae,
        quant_roundtrip_pos_n_p95,
    ) = calc_stats(&quant_roundtrip_pos_n_errors);
    let (
        quant_roundtrip_pos_e_min,
        quant_roundtrip_pos_e_max,
        quant_roundtrip_pos_e_mean,
        quant_roundtrip_pos_e_std,
        quant_roundtrip_pos_e_rmse,
        quant_roundtrip_pos_e_mae,
        quant_roundtrip_pos_e_p95,
    ) = calc_stats(&quant_roundtrip_pos_e_errors);
    let (
        quant_roundtrip_alt_min,
        quant_roundtrip_alt_max,
        quant_roundtrip_alt_mean,
        quant_roundtrip_alt_std,
        quant_roundtrip_alt_rmse,
        quant_roundtrip_alt_mae,
        quant_roundtrip_alt_p95,
    ) = calc_stats(&quant_roundtrip_alt_errors);
    let (
        quant_roundtrip_vel_n_min,
        quant_roundtrip_vel_n_max,
        quant_roundtrip_vel_n_mean,
        quant_roundtrip_vel_n_std,
        quant_roundtrip_vel_n_rmse,
        quant_roundtrip_vel_n_mae,
        quant_roundtrip_vel_n_p95,
    ) = calc_stats(&quant_roundtrip_vel_n_errors);
    let (
        quant_roundtrip_vel_e_min,
        quant_roundtrip_vel_e_max,
        quant_roundtrip_vel_e_mean,
        quant_roundtrip_vel_e_std,
        quant_roundtrip_vel_e_rmse,
        quant_roundtrip_vel_e_mae,
        quant_roundtrip_vel_e_p95,
    ) = calc_stats(&quant_roundtrip_vel_e_errors);
    let (
        quant_roundtrip_vel_d_min,
        quant_roundtrip_vel_d_max,
        quant_roundtrip_vel_d_mean,
        quant_roundtrip_vel_d_std,
        quant_roundtrip_vel_d_rmse,
        quant_roundtrip_vel_d_mae,
        quant_roundtrip_vel_d_p95,
    ) = calc_stats(&quant_roundtrip_vel_d_errors);

    ErrorStats {
        eskf: Some(EskfErrorStats {
            pos_n: SimpleErrorStats {
                min: pos_n_min,
                max: pos_n_max,
                mean: pos_n_mean,
                std: pos_n_std,
                rmse: pos_n_rmse,
                mae: pos_n_mae,
                p95: pos_n_p95,
                n,
            },
            pos_e: SimpleErrorStats {
                min: pos_e_min,
                max: pos_e_max,
                mean: pos_e_mean,
                std: pos_e_std,
                rmse: pos_e_rmse,
                mae: pos_e_mae,
                p95: pos_e_p95,
                n,
            },
            pos_d: SimpleErrorStats {
                min: pos_d_min,
                max: pos_d_max,
                mean: pos_d_mean,
                std: pos_d_std,
                rmse: pos_d_rmse,
                mae: pos_d_mae,
                p95: pos_d_p95,
                n,
            },
            vel_n: SimpleErrorStats {
                min: vel_n_min,
                max: vel_n_max,
                mean: vel_n_mean,
                std: vel_n_std,
                rmse: vel_n_rmse,
                mae: vel_n_mae,
                p95: vel_n_p95,
                n,
            },
            vel_e: SimpleErrorStats {
                min: vel_e_min,
                max: vel_e_max,
                mean: vel_e_mean,
                std: vel_e_std,
                rmse: vel_e_rmse,
                mae: vel_e_mae,
                p95: vel_e_p95,
                n,
            },
            vel_d: SimpleErrorStats {
                min: vel_d_min,
                max: vel_d_max,
                mean: vel_d_mean,
                std: vel_d_std,
                rmse: vel_d_rmse,
                mae: vel_d_mae,
                p95: vel_d_p95,
                n,
            },
            pos_3d: SimpleErrorStats {
                min: pos_3d_min,
                max: pos_3d_max,
                mean: pos_3d_mean,
                std: pos_3d_std,
                rmse: pos_3d_rmse,
                mae: pos_3d_mae,
                p95: pos_3d_p95,
                n,
            },
        }),
        quantized_flight: Some(QuantizedFlightErrorStats {
            pos_n: SimpleErrorStats {
                min: quant_pos_n_min,
                max: quant_pos_n_max,
                mean: quant_pos_n_mean,
                std: quant_pos_n_std,
                rmse: quant_pos_n_rmse,
                mae: quant_pos_n_mae,
                p95: quant_pos_n_p95,
                n,
            },
            pos_e: SimpleErrorStats {
                min: quant_pos_e_min,
                max: quant_pos_e_max,
                mean: quant_pos_e_mean,
                std: quant_pos_e_std,
                rmse: quant_pos_e_rmse,
                mae: quant_pos_e_mae,
                p95: quant_pos_e_p95,
                n,
            },
            alt: SimpleErrorStats {
                min: quant_alt_min,
                max: quant_alt_max,
                mean: quant_alt_mean,
                std: quant_alt_std,
                rmse: quant_alt_rmse,
                mae: quant_alt_mae,
                p95: quant_alt_p95,
                n,
            },
            vel_n: SimpleErrorStats {
                min: quant_vel_n_min,
                max: quant_vel_n_max,
                mean: quant_vel_n_mean,
                std: quant_vel_n_std,
                rmse: quant_vel_n_rmse,
                mae: quant_vel_n_mae,
                p95: quant_vel_n_p95,
                n,
            },
            vel_e: SimpleErrorStats {
                min: quant_vel_e_min,
                max: quant_vel_e_max,
                mean: quant_vel_e_mean,
                std: quant_vel_e_std,
                rmse: quant_vel_e_rmse,
                mae: quant_vel_e_mae,
                p95: quant_vel_e_p95,
                n,
            },
            vel_d: SimpleErrorStats {
                min: quant_vel_d_min,
                max: quant_vel_d_max,
                mean: quant_vel_d_mean,
                std: quant_vel_d_std,
                rmse: quant_vel_d_rmse,
                mae: quant_vel_d_mae,
                p95: quant_vel_d_p95,
                n,
            },
            pos_3d: SimpleErrorStats {
                min: quant_pos_3d_min,
                max: quant_pos_3d_max,
                mean: quant_pos_3d_mean,
                std: quant_pos_3d_std,
                rmse: quant_pos_3d_rmse,
                mae: quant_pos_3d_mae,
                p95: quant_pos_3d_p95,
                n,
            },
        }),
        quant_roundtrip: Some(QuantRoundtripErrorStats {
            pos_n: SimpleErrorStats {
                min: quant_roundtrip_pos_n_min,
                max: quant_roundtrip_pos_n_max,
                mean: quant_roundtrip_pos_n_mean,
                std: quant_roundtrip_pos_n_std,
                rmse: quant_roundtrip_pos_n_rmse,
                mae: quant_roundtrip_pos_n_mae,
                p95: quant_roundtrip_pos_n_p95,
                n,
            },
            pos_e: SimpleErrorStats {
                min: quant_roundtrip_pos_e_min,
                max: quant_roundtrip_pos_e_max,
                mean: quant_roundtrip_pos_e_mean,
                std: quant_roundtrip_pos_e_std,
                rmse: quant_roundtrip_pos_e_rmse,
                mae: quant_roundtrip_pos_e_mae,
                p95: quant_roundtrip_pos_e_p95,
                n,
            },
            alt: SimpleErrorStats {
                min: quant_roundtrip_alt_min,
                max: quant_roundtrip_alt_max,
                mean: quant_roundtrip_alt_mean,
                std: quant_roundtrip_alt_std,
                rmse: quant_roundtrip_alt_rmse,
                mae: quant_roundtrip_alt_mae,
                p95: quant_roundtrip_alt_p95,
                n,
            },
            vel_n: SimpleErrorStats {
                min: quant_roundtrip_vel_n_min,
                max: quant_roundtrip_vel_n_max,
                mean: quant_roundtrip_vel_n_mean,
                std: quant_roundtrip_vel_n_std,
                rmse: quant_roundtrip_vel_n_rmse,
                mae: quant_roundtrip_vel_n_mae,
                p95: quant_roundtrip_vel_n_p95,
                n,
            },
            vel_e: SimpleErrorStats {
                min: quant_roundtrip_vel_e_min,
                max: quant_roundtrip_vel_e_max,
                mean: quant_roundtrip_vel_e_mean,
                std: quant_roundtrip_vel_e_std,
                rmse: quant_roundtrip_vel_e_rmse,
                mae: quant_roundtrip_vel_e_mae,
                p95: quant_roundtrip_vel_e_p95,
                n,
            },
            vel_d: SimpleErrorStats {
                min: quant_roundtrip_vel_d_min,
                max: quant_roundtrip_vel_d_max,
                mean: quant_roundtrip_vel_d_mean,
                std: quant_roundtrip_vel_d_std,
                rmse: quant_roundtrip_vel_d_rmse,
                mae: quant_roundtrip_vel_d_mae,
                p95: quant_roundtrip_vel_d_p95,
                n,
            },
        }),
        quant_recovery: {
            // Calculate recovery position errors (final landing position)
            // Use last position in arrays (landing position)
            let n = true_pos.x.len();
            if n == 0 {
                None
            } else {
                let final_idx = n - 1;

                // Final true position in NED
                let true_north_m = true_pos.x[final_idx];
                let true_east_m = true_pos.y[final_idx];
                let true_down_m = true_pos.z[final_idx];

                // Final quantized position in NED
                let quant_north_m = quantized_pos.x[final_idx];
                let quant_east_m = quantized_pos.y[final_idx];
                let quant_down_m = quantized_pos.z[final_idx];

                // Convert NED to geographic coordinates
                // Approximation: 1 deg latitude ≈ 111,111 m
                // 1 deg longitude ≈ 111,111 m * cos(lat)
                const M_PER_DEG_LAT: f64 = 111_111.0;
                let m_per_deg_lon = M_PER_DEG_LAT * home_lat_deg.to_radians().cos();

                // True final position in geographic
                let true_lat = home_lat_deg + true_north_m / M_PER_DEG_LAT;
                let true_lon = home_lon_deg + true_east_m / m_per_deg_lon;
                let true_alt_msl = home_alt_m + (-true_down_m); // NED down is negative up

                // Quantized final position in geographic
                let quant_lat = home_lat_deg + quant_north_m / M_PER_DEG_LAT;
                let quant_lon = home_lon_deg + quant_east_m / m_per_deg_lon;
                let quant_alt_msl = home_alt_m + (-quant_down_m);

                // Calculate errors (quantized - true)
                let lat_error_deg = quant_lat - true_lat;
                let lon_error_deg = quant_lon - true_lon;
                let alt_error_m = quant_alt_msl - true_alt_msl;
                let horiz_error_m = ((quant_north_m - true_north_m).powi(2)
                    + (quant_east_m - true_east_m).powi(2))
                .sqrt();

                // Create stats for single measurement
                let make_single_stat = |val: f64| SimpleErrorStats {
                    min: val,
                    max: val,
                    mean: val,
                    std: 0.0,
                    rmse: val.abs(),
                    mae: val.abs(),
                    p95: val,
                    n: 1,
                };

                Some(QuantRecoveryErrorStats {
                    lat: make_single_stat(lat_error_deg),
                    lon: make_single_stat(lon_error_deg),
                    alt: make_single_stat(alt_error_m),
                    horiz: make_single_stat(horiz_error_m),
                })
            }
        },
        state_detection: {
            // Calculate state detection time delays (ESKF - Simulation)
            // Positive delay means ESKF detected later than simulation
            // Negative delay means ESKF detected earlier than simulation

            let burn_delay = match (filter_ascent_time, sim_ascent_time) {
                (Some(f), Some(s)) => Some(f - s),
                _ => None,
            };

            let coast_delay = match (filter_coast_time, sim_coast_time) {
                (Some(f), Some(s)) => Some(f - s),
                _ => None,
            };

            let rec_delay = match (filter_descent_time, sim_descent_time) {
                (Some(f), Some(s)) => Some(f - s),
                _ => None,
            };

            // Helper to create SimpleErrorStats for a single delay value
            let stats_from_delay = |delay: Option<f64>| -> SimpleErrorStats {
                match delay {
                    Some(d) => SimpleErrorStats {
                        min: d,
                        max: d,
                        mean: d,
                        std: 0.0,
                        rmse: d.abs(),
                        mae: d.abs(),
                        p95: d,
                        n: 1,
                    },
                    None => SimpleErrorStats {
                        min: 0.0,
                        max: 0.0,
                        mean: 0.0,
                        std: 0.0,
                        rmse: 0.0,
                        mae: 0.0,
                        p95: 0.0,
                        n: 0,
                    },
                }
            };

            Some(StateDetectionErrorStats {
                burn: stats_from_delay(burn_delay),
                coast: stats_from_delay(coast_delay),
                rec: stats_from_delay(rec_delay),
            })
        },
    }
}

/// Downsample with anti-aliasing using a moving average filter
/// This prevents resonance/aliasing artifacts in the downsampled data
fn downsample_vec<T>(vec: &[T], step: usize) -> Vec<T>
where
    T: Clone + std::ops::Add<Output = T> + std::ops::Div<f64, Output = T> + Default,
{
    if step <= 1 {
        return vec.to_vec();
    }

    let target_points = vec.len().div_ceil(step);
    let mut result = Vec::with_capacity(target_points);

    // Use moving average as anti-aliasing filter
    // Window size is the decimation factor
    for i in 0..target_points {
        let center_idx = i * step;
        let start = center_idx.saturating_sub(step / 2);
        let end = (center_idx + step / 2).min(vec.len());

        // Compute average over window
        let mut sum = T::default();
        for item in vec.iter().take(end).skip(start) {
            sum = sum + item.clone();
        }
        let count = (end - start) as f64;
        result.push(sum / count);
    }

    result
}

fn downsample_option_vec<T: Clone>(vec: &[Option<T>], step: usize) -> Vec<Option<T>> {
    if step <= 1 {
        return vec.to_vec();
    }

    let target_points = vec.len().div_ceil(step);
    let mut result = Vec::with_capacity(target_points);

    for i in 0..target_points {
        let center_idx = i * step;
        let start = center_idx.saturating_sub(step / 2);
        let end = (center_idx + step / 2).min(vec.len());

        let sample = vec
            .iter()
            .take(end)
            .skip(start)
            .find_map(|item| item.clone())
            .or_else(|| vec.get(center_idx).cloned().flatten());

        result.push(sample);
    }

    result
}

fn hold_last_scalar_samples(samples: &[Option<f64>]) -> Vec<f64> {
    let mut last_sample = None;

    samples
        .iter()
        .map(|sample| match sample {
            Some(value) => {
                last_sample = Some(*value);
                *value
            }
            None => last_sample.unwrap_or(0.0),
        })
        .collect()
}

fn hold_last_vector_component<F>(samples: &[Option<Vector3<f64>>], component: F) -> Vec<f64>
where
    F: Fn(&Vector3<f64>) -> f64,
{
    let mut last_sample = None;

    samples
        .iter()
        .map(|sample| match sample.as_ref() {
            Some(value) => {
                let component_value = component(value);
                last_sample = Some(component_value);
                component_value
            }
            None => last_sample.unwrap_or(0.0),
        })
        .collect()
}

fn generate_state_changes(
    time: &[f64],
    pos: &[Vector3<f64>],
    vel: &[Vector3<f64>],
    ascent_time: Option<f64>,
    coast_time: Option<f64>,
    descent_time: Option<f64>,
) -> Vec<StateChange> {
    let mut state_changes = vec![StateChange {
        time: 0.0,
        state: "Pad".to_string(),
        description: "On Pad".to_string(),
    }];

    if let Some(t) = ascent_time.filter(|t| *t > 0.0) {
        state_changes.push(StateChange {
            time: t,
            state: "Ascent".to_string(),
            description: "Powered ascent".to_string(),
        });
    }

    if let Some(t) = coast_time.filter(|t| *t > 0.0) {
        state_changes.push(StateChange {
            time: t,
            state: "Coast".to_string(),
            description: "Ballistic coast".to_string(),
        });
    }

    if !time.is_empty() && !pos.is_empty() {
        let apogee_index = pos
            .iter()
            .enumerate()
            .filter(|(_, sample)| (-sample.z).is_finite())
            .max_by(|(_, a), (_, b)| (-a.z).total_cmp(&(-b.z)))
            .map(|(idx, _)| idx);

        if let Some(index) = apogee_index {
            state_changes.push(StateChange {
                time: time[index],
                state: "Apogee".to_string(),
                description: format!("Apogee at {:.1}m", -pos[index].z),
            });
        }
    }

    if let Some(t) = descent_time.filter(|t| *t > 0.0) {
        state_changes.push(StateChange {
            time: t,
            state: "Descent".to_string(),
            description: "Descent".to_string(),
        });
    }

    if let (Some(&last_time), Some(last_pos), Some(last_vel)) =
        (time.last(), pos.last(), vel.last())
    {
        let altitude = -last_pos.z;
        let speed = last_vel.norm();
        if altitude <= 10.0 || speed <= 5.0 {
            state_changes.push(StateChange {
                time: last_time,
                state: "Landed".to_string(),
                description: "Ground impact".to_string(),
            });
        }
    }

    state_changes.sort_by(|a, b| {
        a.time
            .total_cmp(&b.time)
            .then_with(|| a.state.cmp(&b.state))
    });
    state_changes.dedup_by(|a, b| a.state == b.state);

    state_changes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Full simulation takes too long
    fn test_simulation_runs() {
        let config = SimConfig::default();
        let results = run_full_simulation(&config);

        assert!(!results.time.is_empty());
        assert!(!results.altitude.is_empty());
        assert_eq!(results.time.len(), results.altitude.len());
        assert!(results.success);
    }
}
