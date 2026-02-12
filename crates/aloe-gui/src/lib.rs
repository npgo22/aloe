//! Aloe GUI - Web interface for rocket simulation
//!
//! Features:
//! - Configuration panel with tabs (Rocket, ENV, Sensors, Filter)
//! - Real-time simulation via API
//! - Multiple chart types (2D and 3D)
//! - Sensor data visualization
//! - Filter error statistics

use aloe_sim::{
    filter::{run_filter, FilterConfig},
    sensor::{generate_sensor_data, SensorConfig},
    sim::{simulate_6dof, RocketParams},
};
use axum::{extract::Query, routing::get, Json, Router};
use log::{debug, error, info, warn};
use nalgebra::Vector3;
use serde::Serialize;
use std::collections::HashMap;
use tower_http::services::ServeDir;

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

/// Creates the Axum router with all routes
pub fn create_router() -> Router {
    Router::new()
        .route("/api/simulate", get(handle_simulate))
        .route("/api/chart/{chart_type}", get(handle_chart_data))
        .nest_service("/static", ServeDir::new("crates/aloe-gui/static"))
        .fallback_service(ServeDir::new("crates/aloe-gui/templates"))
}

/// Simulation configuration from query params
#[derive(Debug, Clone)]
struct SimConfig {
    // Rocket params
    dry_mass: f64,
    propellant_mass: f64,
    cg_full: f64,
    cg_empty: f64,
    thrust: f64,
    burn_time: f64,
    drag_coeff: f64,
    ref_area: f64,
    launch_delay: f64,
    spin_rate: f64,
    thrust_cant: f64,
    // ENV params
    gravity: f64,
    wind_north: f64,
    wind_east: f64,
    air_density: f64,
    // Sensor params
    noise_scale: f64,
    seed: u64,
    no_sensors: bool,
    no_filter: bool,
    bmi088_accel_enabled: bool,
    bmi088_gyro_enabled: bool,
    adxl375_enabled: bool,
    lis3mdl_enabled: bool,
    ms5611_enabled: bool,
    gps_enabled: bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            // 30km Sounding Rocket Preset
            dry_mass: 80.0,
            propellant_mass: 120.0,
            cg_full: 1.5,
            cg_empty: 1.4,
            thrust: 18000.0,
            burn_time: 12.0,
            drag_coeff: 0.38,
            ref_area: 0.045,
            launch_delay: 1.0,
            spin_rate: 0.0,
            thrust_cant: 0.0,
            gravity: 9.81,
            wind_north: 5.0,
            wind_east: 0.0,
            air_density: 1.225,
            noise_scale: 1.0,
            seed: 42,
            no_sensors: false,
            no_filter: false,
            bmi088_accel_enabled: true,
            bmi088_gyro_enabled: true,
            adxl375_enabled: true,
            lis3mdl_enabled: true,
            ms5611_enabled: true,
            gps_enabled: true,
        }
    }
}

/// Parse config from query parameters
fn parse_config(params: &HashMap<String, String>) -> SimConfig {
    let mut config = SimConfig::default();

    macro_rules! parse_param {
        ($field:ident, $name:expr, $type:ty) => {
            if let Some(val) = params.get($name).and_then(|v| v.parse::<$type>().ok()) {
                config.$field = val;
            }
        };
    }

    parse_param!(dry_mass, "dry_mass", f64);
    parse_param!(propellant_mass, "propellant_mass", f64);
    parse_param!(cg_full, "cg_full", f64);
    parse_param!(cg_empty, "cg_empty", f64);
    parse_param!(thrust, "thrust", f64);
    parse_param!(burn_time, "burn_time", f64);
    parse_param!(drag_coeff, "drag_coeff", f64);
    parse_param!(ref_area, "ref_area", f64);
    parse_param!(launch_delay, "launch_delay", f64);
    parse_param!(spin_rate, "spin_rate", f64);
    parse_param!(thrust_cant, "thrust_cant", f64);
    parse_param!(gravity, "gravity", f64);
    parse_param!(wind_north, "wind_north", f64);
    parse_param!(wind_east, "wind_east", f64);
    parse_param!(air_density, "air_density", f64);
    parse_param!(noise_scale, "noise_scale", f64);

    if let Some(seed) = params.get("seed").and_then(|v| v.parse::<u64>().ok()) {
        config.seed = seed;
    }

    parse_param!(no_sensors, "no_sensors", bool);
    parse_param!(no_filter, "no_filter", bool);
    parse_param!(bmi088_accel_enabled, "bmi088_accel_enabled", bool);
    parse_param!(bmi088_gyro_enabled, "bmi088_gyro_enabled", bool);
    parse_param!(adxl375_enabled, "adxl375_enabled", bool);
    parse_param!(lis3mdl_enabled, "lis3mdl_enabled", bool);
    parse_param!(ms5611_enabled, "ms5611_enabled", bool);
    parse_param!(gps_enabled, "gps_enabled", bool);

    config
}

/// Handle simulation request
async fn handle_simulate(
    Query(params): Query<HashMap<String, String>>,
) -> Json<FullSimulationResponse> {
    info!("Received /api/simulate request");
    let config = parse_config(&params);
    debug!("Parsed Config: {:?}", config);

    // Run simulation under a panic catcher so we can return structured errors to the UI
    match std::panic::catch_unwind(|| run_full_simulation(&config)) {
        Ok(results) => {
            info!(
                "Simulation completed successfully. Returning {} points.",
                results.time.len()
            );
            Json(results)
        }
        Err(payload) => {
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
                state_changes_sim: vec![],
                state_changes_eskf: vec![],
                sensor_data: GuiSensorData::empty(),
                filter_data: FilterData::empty(),
                error_stats: None,
                success: false,
                error_message: Some(message),
            })
        }
    }
}

/// Handle specific chart data requests
async fn handle_chart_data(
    axum::extract::Path(chart_type): axum::extract::Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Json<ChartData> {
    let config = parse_config(&params);
    let chart_data = generate_chart_data(&chart_type, &config);
    Json(chart_data)
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
    state_changes_sim: Vec<StateChange>,
    state_changes_eskf: Vec<StateChange>,
    sensor_data: GuiSensorData,
    filter_data: FilterData,
    error_stats: Option<ErrorStats>,
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
    gps_x: Vec<f64>,
    gps_y: Vec<f64>,
    gps_z: Vec<f64>,
    gps_vel_x: Vec<f64>,
    gps_vel_y: Vec<f64>,
    gps_vel_z: Vec<f64>,
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
        }
    }
}

#[derive(Serialize)]
struct ChartData {
    time: Vec<f64>,
    data: Vec<f64>,
    data_3d: Option<(Vec<f64>, Vec<f64>, Vec<f64>)>,
    title: String,
    y_label: String,
    chart_type: String,
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
fn run_full_simulation(config: &SimConfig) -> FullSimulationResponse {
    info!("Starting run_full_simulation");
    let rocket_params = RocketParams {
        dry_mass: config.dry_mass,
        propellant_mass: config.propellant_mass,
        inertia_tensor: Vector3::new(0.1, 10.0, 10.0), // Approximate
        cg_full: config.cg_full,
        cg_empty: config.cg_empty,
        cp_location: 2.0,
        ref_area: config.ref_area,
        drag_coeff_axial: config.drag_coeff,
        normal_force_coeff: 12.0,
        thrust_curve: vec![
            (0.0, config.thrust),
            (config.burn_time, config.thrust),
            (config.burn_time + 0.1, 0.0),
        ],
        burn_time: config.burn_time,
        isp: 200.0,
        nozzle_location: 3.0,
        gravity: config.gravity,
        air_density_sea_level: config.air_density,
        launch_rod_length: 2.0,
        wind_velocity_ned: Vector3::new(config.wind_north, config.wind_east, 0.0),
        launch_delay: config.launch_delay,
        spin_rate: config.spin_rate,
        thrust_cant: config.thrust_cant,
    };

    debug!("Running physics sim_6dof...");
    let sim_result = simulate_6dof(&rocket_params);
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
        .map(|(a, _)| a.x * (rocket_params.dry_mass + rocket_params.propellant_mass))
        .collect(); // Approximate
    let mass: Vec<f64> = time
        .iter()
        .map(|&t| {
            if t < config.burn_time {
                rocket_params.dry_mass
                    + rocket_params.propellant_mass * (1.0 - t / config.burn_time).max(0.0)
            } else {
                rocket_params.dry_mass
            }
        })
        .collect();
    let position_x: Vec<f64> = sim_result.pos.iter().map(|p| p.x).collect();
    let position_y: Vec<f64> = sim_result.pos.iter().map(|p| p.y).collect();
    let position_z: Vec<f64> = sim_result.pos.iter().map(|p| -p.z).collect(); // NED to altitude

    // Generate state changes for sim
    let state_changes_sim = generate_state_changes(&time, &sim_result.pos, &sim_result.vel);

    if config.no_sensors {
        info!("Sensors DISABLED by config.");

        // Log basic stats for debugging
        let max_alt = altitude.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        debug!(
            "Simulation (no sensors): time steps {}, max alt {:.2}, final pos_z {:.2}",
            time.len(),
            max_alt,
            position_z.last().unwrap_or(&0.0)
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
            state_changes_sim,
            state_changes_eskf: vec![],
            sensor_data: GuiSensorData::empty(),
            filter_data: FilterData::empty(),
            error_stats: None,
            success: true,
            error_message: None,
        }
    } else {
        debug!("Generating Sensor Data...");
        let sensor_config = SensorConfig {
            noise_scale: config.noise_scale,
            accel_noise_std: 0.01,
            gyro_noise_std: 0.001,
            mag_noise_std: 0.001,
            baro_noise_std: 0.1,
            gps_pos_noise_std: 1.0,
            gps_vel_noise_std: 0.1,
            accel_bias: Vector3::zeros(),
            gyro_bias: Vector3::zeros(),
            seed: config.seed,
            accel_enabled: config.bmi088_accel_enabled || config.adxl375_enabled,
            gyro_enabled: config.bmi088_gyro_enabled,
            mag_enabled: config.lis3mdl_enabled,
            baro_enabled: config.ms5611_enabled,
            gps_enabled: config.gps_enabled,
        };
        let sensor_data_sim = generate_sensor_data(&sim_result, &sensor_config);
        debug!("Sensor Data Steps: {}", sensor_data_sim.time.len());

        // Generate sensor data for GUI
        let gui_sensor_data = GuiSensorData {
            accel_x: sensor_data_sim.accel_meas.iter().map(|v| v.x).collect(),
            accel_y: sensor_data_sim.accel_meas.iter().map(|v| v.y).collect(),
            accel_z: sensor_data_sim.accel_meas.iter().map(|v| v.z).collect(),
            gyro_x: sensor_data_sim.gyro_meas.iter().map(|v| v.x).collect(),
            gyro_y: sensor_data_sim.gyro_meas.iter().map(|v| v.y).collect(),
            gyro_z: sensor_data_sim.gyro_meas.iter().map(|v| v.z).collect(),
            baro_pressure: sensor_data_sim.baro_pressure.clone(),
            mag_x: sensor_data_sim.mag_meas.iter().map(|v| v.x).collect(),
            mag_y: sensor_data_sim.mag_meas.iter().map(|v| v.y).collect(),
            mag_z: sensor_data_sim.mag_meas.iter().map(|v| v.z).collect(),
            gps_x: sensor_data_sim.gps_pos.iter().map(|v| v.x).collect(),
            gps_y: sensor_data_sim.gps_pos.iter().map(|v| v.y).collect(),
            gps_z: sensor_data_sim.gps_pos.iter().map(|v| v.z).collect(),
            gps_vel_x: sensor_data_sim.gps_vel.iter().map(|v| v.x).collect(),
            gps_vel_y: sensor_data_sim.gps_vel.iter().map(|v| v.y).collect(),
            gps_vel_z: sensor_data_sim.gps_vel.iter().map(|v| v.z).collect(),
            adxl_x: sensor_data_sim.accel_meas.iter().map(|v| v.x).collect(),
            adxl_y: sensor_data_sim.accel_meas.iter().map(|v| v.y).collect(),
            adxl_z: sensor_data_sim.accel_meas.iter().map(|v| v.z).collect(),
        };

        debug!("Running Filter...");
        let filter_config = FilterConfig::default();
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
            .map(|&z| ((-z * 100.0) as i32 as f64) / 100.0)
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
        };

        // Generate state changes for ESKF
        let state_changes_eskf =
            generate_state_changes(&time, &filter_result.position, &filter_result.velocity);

        debug!("Calculating Error Stats...");
        // Prepare true data for error stats
        let true_pos_z: Vec<f64> = sim_result.pos.iter().map(|p| p.z).collect(); // down
        let true_vel_x: Vec<f64> = sim_result.vel.iter().map(|v| v.x).collect();
        let true_vel_y: Vec<f64> = sim_result.vel.iter().map(|v| v.y).collect();
        let true_vel_z: Vec<f64> = sim_result.vel.iter().map(|v| v.z).collect();

        // Align ground truth to filter timestamps for accurate error stats
        let filter_time = &sensor_data_sim.time;
        let true_pos_x: Vec<f64> = sim_result.pos.iter().map(|p| p.y).collect(); // east
        let true_pos_y: Vec<f64> = sim_result.pos.iter().map(|p| p.x).collect(); // north

        let aligned_true_pos_x = align_ground_truth(&sim_result.time, &true_pos_x, filter_time);
        let aligned_true_pos_y = align_ground_truth(&sim_result.time, &true_pos_y, filter_time);
        let aligned_true_pos_z = align_ground_truth(&sim_result.time, &true_pos_z, filter_time);
        let aligned_true_vel_x = align_ground_truth(&sim_result.time, &true_vel_x, filter_time);
        let aligned_true_vel_y = align_ground_truth(&sim_result.time, &true_vel_y, filter_time);
        let aligned_true_vel_z = align_ground_truth(&sim_result.time, &true_vel_z, filter_time);

        // Calculate error statistics before downsampling
        let error_stats = calculate_error_stats(
            PositionData {
                x: &aligned_true_pos_x,
                y: &aligned_true_pos_y,
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
        );

        debug!("=== BEFORE DOWNSAMPLING ===");
        debug!("position_x length: {}", position_x.len());
        debug!("position_y length: {}", position_y.len());
        debug!("position_z length: {}", position_z.len());

        let max_alt = altitude.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        debug!("Max altitude: {}", max_alt);

        // Downsample data for frontend performance
        let target_points = 6000;
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
                    gps_x: downsample_vec(&gui_sensor_data.gps_x, step),
                    gps_y: downsample_vec(&gui_sensor_data.gps_y, step),
                    gps_z: downsample_vec(&gui_sensor_data.gps_z, step),
                    gps_vel_x: downsample_vec(&gui_sensor_data.gps_vel_x, step),
                    gps_vel_y: downsample_vec(&gui_sensor_data.gps_vel_y, step),
                    gps_vel_z: downsample_vec(&gui_sensor_data.gps_vel_z, step),
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
                gui_sensor_data,
                filter_data,
            )
        };

        let max_alt = altitude.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        info!(
            "Simulation (with sensors): time steps {}, max alt {:.2}, final pos_z {:.2}",
            time.len(),
            max_alt,
            position_z.last().unwrap_or(&0.0)
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
            state_changes_sim,
            state_changes_eskf,
            sensor_data,
            filter_data,
            error_stats: Some(ErrorStats {
                eskf: error_stats.eskf,
                quantized_flight: error_stats.quantized_flight,
                quant_roundtrip: error_stats.quant_roundtrip,
                quant_recovery: error_stats.quant_recovery,
                state_detection: error_stats.state_detection,
            }),
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

fn calculate_error_stats(
    true_pos: PositionData,
    est_pos: PositionData,
    quantized_pos: PositionData,
    true_vel: VelocityData,
    est_vel: VelocityData,
) -> ErrorStats {
    // Generate position errors for N, E, D components
    let n = true_pos.x.len();
    let pos_n_errors: Vec<f64> = (0..n).map(|i| est_pos.y[i] - true_pos.y[i]).collect(); // north error
    let pos_e_errors: Vec<f64> = (0..n).map(|i| est_pos.x[i] - true_pos.x[i]).collect(); // east error
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
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std = variance.sqrt();
        let rmse = (data.iter().map(|x| x.powi(2)).sum::<f64>() / data.len() as f64).sqrt();
        let mae = data.iter().map(|x| x.abs()).sum::<f64>() / data.len() as f64;
        let mut sorted = data.to_vec();
        let p95_idx = (0.95 * data.len() as f64) as usize;
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
    let quant_pos_n_errors: Vec<f64> = (0..n).map(|i| est_pos.y[i] - quantized_pos.y[i]).collect();
    let quant_pos_e_errors: Vec<f64> = (0..n).map(|i| est_pos.x[i] - quantized_pos.x[i]).collect();
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
    let quant_vel_n_errors: Vec<f64> = vec![0.0; n]; // Not quantized
    let quant_vel_e_errors: Vec<f64> = vec![0.0; n];
    let quant_vel_d_errors: Vec<f64> = vec![0.0; n];

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
        (0..n).map(|i| quantized_pos.y[i] - true_pos.y[i]).collect();
    let quant_roundtrip_pos_e_errors: Vec<f64> =
        (0..n).map(|i| quantized_pos.x[i] - true_pos.x[i]).collect();
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
    let quant_roundtrip_vel_n_errors: Vec<f64> = vec![0.0; n]; // Not quantized
    let quant_roundtrip_vel_e_errors: Vec<f64> = vec![0.0; n];
    let quant_roundtrip_vel_d_errors: Vec<f64> = vec![0.0; n];

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
        quant_recovery: None,
        state_detection: None,
    }
}

fn downsample_vec<T: Clone>(vec: &[T], step: usize) -> Vec<T> {
    vec.iter().step_by(step).cloned().collect()
}

fn generate_state_changes(
    time: &[f64],
    pos: &[Vector3<f64>],
    vel: &[Vector3<f64>],
) -> Vec<StateChange> {
    let mut state_changes = vec![StateChange {
        time: 0.0,
        state: "Pad".to_string(),
        description: "On Pad".to_string(),
    }];
    let mut last_state = "Pad";
    for i in 0..time.len() {
        let t = time[i];
        let vel_mag = vel[i].norm();
        let alt = -pos[i].z;
        let on_ground = alt < 0.1 && vel_mag < 1.0;

        let state = if t < 1.0 {
            "Pad"
        } else if t < 6.0 {
            "Ascent"
        } else if vel_mag > 5.0 && !on_ground {
            "Coast"
        } else if alt > 10.0 {
            "Descent"
        } else {
            "Landed"
        };

        if state != last_state {
            state_changes.push(StateChange {
                time: t,
                state: state.to_string(),
                description: match state {
                    "Pad" => "Pre-launch on pad",
                    "Ascent" => "Powered ascent",
                    "Coast" => "Ballistic coast",
                    "Descent" => "Free fall descent",
                    "Landed" => "Ground impact",
                    _ => state,
                }
                .to_string(),
            });
            last_state = state;
        }
    }

    state_changes
}

fn generate_chart_data(chart_type: &str, config: &SimConfig) -> ChartData {
    let results = run_full_simulation(config);

    match chart_type {
        "altitude" => ChartData {
            time: results.time.clone(),
            data: results.altitude,
            data_3d: None,
            title: "Altitude vs Time".to_string(),
            y_label: "Altitude (m)".to_string(),
            chart_type: "2d".to_string(),
        },
        "velocity" => ChartData {
            time: results.time.clone(),
            data: results.velocity,
            data_3d: None,
            title: "Velocity vs Time".to_string(),
            y_label: "Velocity (m/s)".to_string(),
            chart_type: "2d".to_string(),
        },
        "acceleration" => ChartData {
            time: results.time.clone(),
            data: results.acceleration,
            data_3d: None,
            title: "Acceleration vs Time".to_string(),
            y_label: "Acceleration (m/s²)".to_string(),
            chart_type: "2d".to_string(),
        },
        "force" => ChartData {
            time: results.time.clone(),
            data: results.force,
            data_3d: None,
            title: "Net Force vs Time".to_string(),
            y_label: "Force (N)".to_string(),
            chart_type: "2d".to_string(),
        },
        "mass" => ChartData {
            time: results.time.clone(),
            data: results.mass,
            data_3d: None,
            title: "Mass vs Time".to_string(),
            y_label: "Mass (kg)".to_string(),
            chart_type: "2d".to_string(),
        },
        "trajectory" => ChartData {
            time: results.time.clone(),
            data: vec![],
            data_3d: Some((results.position_y, results.position_x, results.position_z)), // ENU: East, North, Up
            title: "3D Flight Path".to_string(),
            y_label: "Position (m)".to_string(),
            chart_type: "3d".to_string(),
        },
        _ => ChartData {
            time: results.time,
            data: results.altitude,
            data_3d: None,
            title: "Altitude vs Time".to_string(),
            y_label: "Altitude (m)".to_string(),
            chart_type: "2d".to_string(),
        },
    }
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

    #[test]
    #[ignore] // Full simulation takes too long
    fn test_chart_generation() {
        let config = SimConfig::default();
        let chart = generate_chart_data("altitude", &config);

        assert!(!chart.time.is_empty());
        assert!(!chart.data.is_empty());
        assert_eq!(chart.title, "Altitude vs Time");
    }

    #[test]
    #[ignore] // Full simulation takes too long
    fn test_3d_trajectory() {
        let config = SimConfig::default();
        let chart = generate_chart_data("trajectory", &config);

        assert!(chart.data_3d.is_some());
        let (x, y, z) = chart.data_3d.unwrap();
        assert!(!x.is_empty());
        assert!(!y.is_empty());
        assert!(!z.is_empty());
    }
}
