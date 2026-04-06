use aloe_sim::{
    generate_sensor_data, run_filter, simulate_6dof, ChaosConfig, FilterConfig, FilterResult,
    RocketParams, SensorConfig, SensorData, SimResult,
};
use anyhow::Result;
use clap::{Parser, ValueEnum};
use nalgebra::Vector3;
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Clone)]
struct ThrustCurvePoint {
    time_s: f64,
    thrust_n: f64,
}

impl FromStr for ThrustCurvePoint {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut parts = value.split(':');
        let time_s = parts
            .next()
            .ok_or_else(|| "missing thrust curve time".to_string())?
            .parse::<f64>()
            .map_err(|_| format!("invalid thrust curve time: {value}"))?;
        let thrust_n = parts
            .next()
            .ok_or_else(|| "missing thrust curve thrust".to_string())?
            .parse::<f64>()
            .map_err(|_| format!("invalid thrust curve thrust: {value}"))?;
        if parts.next().is_some() {
            return Err(format!("invalid thrust curve point format: {value}"));
        }
        Ok(Self { time_s, thrust_n })
    }
}

/// Public function that can be called from the main binary
pub fn run_cli_main(args: &[&str]) -> Result<()> {
    let args = Args::parse_from(args);
    main_inner(args)
}

#[derive(Parser, Debug)]
#[command(name = "aloe-cli")]
#[command(about = "Hobby-rocket 6-DoF flight simulator")]
#[command(version)]
pub struct Args {
    /// Output directory for simulation results (CSV/JSON files)
    #[arg(short, long, default_value = "output")]
    output_dir: PathBuf,

    /// Output file format (csv or json)
    #[arg(short, long, value_enum, default_value = "csv")]
    format: OutputFormat,

    /// Run a single simulation instead of parameter sweep
    #[arg(long)]
    single: bool,

    // ── Simulation parameters ─────────────────────────────────
    /// Rocket dry mass without propellant (kg). Typical model rockets: 0.1-5 kg, high-power: 5-50 kg
    #[arg(long, default_value_t = 20.0)]
    dry_mass: f64,

    /// Fuel mass at launch (kg) for the liquid engine
    #[arg(long, default_value_t = 10.0 / 3.0)]
    fuel_mass: f64,

    /// Oxidizer mass at launch (kg) for the liquid engine
    #[arg(long, default_value_t = 20.0 / 3.0)]
    oxidizer_mass: f64,

    /// Motor thrust force (N). Example: Estes E9 = 25N, Cesaroni Pro98 = 1500N
    #[arg(long, default_value_t = 2000.0)]
    thrust: f64,

    /// Motor burn time (s). Longer burn = lower acceleration, higher efficiency
    #[arg(long, default_value_t = 5.0)]
    burn_time: f64,

    /// Coefficient of drag (dimensionless). Typical values: 0.3-0.5 for streamlined, 0.5-0.75 for stable
    #[arg(long, default_value_t = 0.5)]
    drag_coeff: f64,

    /// Normal force coefficient per radian. Higher values weathercock more strongly
    #[arg(long, default_value_t = 12.0)]
    normal_force_coeff: f64,

    /// Reference area for drag calculation (m²). Usually cross-sectional area: π*(diameter/2)²
    #[arg(long, default_value_t = 0.018)]
    ref_area: f64,

    /// Center of gravity with full propellant (m from nose)
    #[arg(long, default_value_t = 1.5)]
    cg_full: f64,

    /// Center of gravity at burnout (m from nose)
    #[arg(long, default_value_t = 1.4)]
    cg_empty: f64,

    /// Center of pressure location (m from nose)
    #[arg(long, default_value_t = 2.0)]
    cp_location: f64,

    /// Body-axis roll inertia Ixx (kg·m²)
    #[arg(long, default_value_t = 0.1)]
    inertia_x: f64,

    /// Body-axis pitch inertia Iyy (kg·m²)
    #[arg(long, default_value_t = 15.0)]
    inertia_y: f64,

    /// Body-axis yaw inertia Izz (kg·m²)
    #[arg(long, default_value_t = 15.0)]
    inertia_z: f64,

    /// Effective liquid-engine specific impulse (s)
    #[arg(long, default_value_t = 200.0)]
    isp: f64,

    /// Nozzle exit location from nose (m)
    #[arg(long, default_value_t = 3.0)]
    nozzle_location: f64,

    /// Launch rail / rod length (m)
    #[arg(long, default_value_t = 2.0)]
    launch_rod_length: f64,

    /// Gravitational acceleration (m/s²). Earth = 9.81, varies slightly with latitude/altitude
    #[arg(long, default_value_t = 9.81)]
    gravity: f64,

    /// Wind speed in north direction (m/s). Positive = northward, affects drift
    #[arg(long, alias = "wind-speed", default_value_t = 5.0)]
    wind_north: f64,

    /// Wind speed in east direction (m/s). Positive = eastward, affects drift
    #[arg(long, alias = "wind-speed-z", default_value_t = 0.0)]
    wind_east: f64,

    /// Wind speed in down direction (m/s). Positive = downward, negative = updraft
    #[arg(long, default_value_t = 0.0)]
    wind_down: f64,

    /// Air density at launch site (kg/m³). Sea level = 1.225, decreases with altitude
    #[arg(long, default_value_t = 1.225)]
    air_density: f64,

    /// JSBSim integration step size (s). Smaller = higher fidelity, slower runtime
    #[arg(long, default_value_t = 0.001)]
    sim_dt: f64,

    /// Maximum JSBSim simulation time (s). Longer values capture long coast/descent flights
    #[arg(long, default_value_t = 400.0)]
    max_time: f64,

    /// Delay before engine ignition (s). Used to simulate hold on pad
    #[arg(long, default_value_t = 1.0)]
    launch_delay: f64,

    /// Initial spin rate around body axis (deg/s). Used for spin-stabilized rockets
    #[arg(long, default_value_t = 0.0)]
    spin_rate: f64,

    /// Thrust misalignment angle (deg). Simulates canted nozzle or manufacturing defects
    #[arg(long, default_value_t = 0.0)]
    thrust_cant: f64,

    /// Nozzle exit pressure used in the generated JSBSim nozzle model (psf)
    #[arg(long, default_value_t = 2116.22)]
    nozzle_exit_pressure_psf: f64,

    /// Nozzle exit area used in the generated JSBSim nozzle model (ft²)
    #[arg(long, default_value_t = 0.01)]
    nozzle_area_ft2: f64,

    /// Pad static friction coefficient in JSBSim ground reactions
    #[arg(long, default_value_t = 0.8)]
    pad_static_friction: f64,

    /// Pad dynamic friction coefficient in JSBSim ground reactions
    #[arg(long, default_value_t = 0.4)]
    pad_dynamic_friction: f64,

    /// Pad spring coefficient in JSBSim ground reactions (lbs/ft)
    #[arg(long, default_value_t = 10000.0)]
    pad_spring_coeff_lbs_ft: f64,

    /// Pad damping coefficient in JSBSim ground reactions (lbs/ft/s)
    #[arg(long, default_value_t = 5000.0)]
    pad_damping_coeff_lbs_ft_s: f64,

    /// Explicit thrust-curve points as `time:thrust`, space separated. Example: --thrust-curve "0:0 0.1:800 2.5:600 2.7:0"
    #[arg(long, value_delimiter = ' ')]
    thrust_curve: Vec<ThrustCurvePoint>,

    // ── Sensor options ────────────────────────────────────────
    /// Disable all sensor noise (perfect measurements). Use for debugging or ideal trajectory analysis
    #[arg(long)]
    no_sensors: bool,

    /// Random number generator seed for sensor noise. Same seed = reproducible results
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Sensor noise scaling factor (multiplier). 0.5 = half noise, 2.0 = double noise
    #[arg(long, default_value_t = 1.0)]
    noise_scale: f64,

    // ── Filter options ────────────────────────────────────────
    /// Disable ESKF state estimation. Simulation only, no filtering
    #[arg(long)]
    no_filter: bool,

    /// [DEPRECATED] Use --accel-noise-density-tune instead
    #[arg(long)]
    accel_noise_density: Option<f32>,

    // ── Sweep options ──────────────────────────────────────────
    /// Space-separated list of parameters to sweep. Example: --sweep-params "thrust drag_coeff"
    #[arg(long, value_delimiter = ' ')]
    sweep_params: Option<Vec<String>>,

    /// Number of steps for each swept parameter. Total runs = steps^(num_params)
    #[arg(long, default_value_t = 5)]
    sweep_steps: usize,

    /// Disable specific sensors for failure mode testing. Example: --disable-sensor "gps baro"
    #[arg(long, value_delimiter = ' ')]
    disable_sensor: Vec<String>,

    /// Generate detailed filter performance report with error statistics
    #[arg(long)]
    filter_report: bool,

    // ── ESKF tuning parameters ──────────────────────────────────
    /// Accelerometer noise density (m/s²/√Hz). Controls Q matrix diagonal for accel. Higher = trust accel less
    #[arg(long)]
    accel_noise_density_tune: Option<f32>,

    /// Gyroscope noise density (rad/s/√Hz). Controls Q matrix diagonal for gyro. Higher = trust gyro less
    #[arg(long)]
    gyro_noise_density: Option<f32>,

    /// Accelerometer bias random walk (m/s²). Models slow drift in accel bias over time
    #[arg(long)]
    accel_bias_instability: Option<f32>,

    /// Gyroscope bias random walk (rad/s). Models slow drift in gyro bias over time
    #[arg(long)]
    gyro_bias_instability: Option<f32>,

    /// Position process noise variance (m²). Models uncertainty in position dynamics
    #[arg(long)]
    pos_process_noise: Option<f32>,

    /// GPS position measurement noise variance (m²). R matrix for GPS position updates
    #[arg(long)]
    r_gps_pos: Option<f32>,

    /// GPS velocity measurement noise variance ((m/s)²). R matrix for GPS velocity updates
    #[arg(long)]
    r_gps_vel: Option<f32>,

    /// Barometer measurement noise variance (m²). R matrix for baro altitude updates
    #[arg(long)]
    r_baro: Option<f32>,

    /// Magnetometer measurement noise variance (rad²). R matrix for mag heading updates
    #[arg(long)]
    r_mag: Option<f32>,

    // ── Tune-sweep mode ───────────────────────────────────
    /// Enable filter tuning mode with greedy coordinate descent optimization
    #[arg(long)]
    tune_sweep: bool,

    /// Tuning algorithm: "greedy" (coordinate descent) or "grid" (exhaustive search)
    #[arg(long, default_value = "greedy")]
    tune_mode: String,

    /// Number of steps per tuning parameter. More steps = finer resolution, longer runtime
    #[arg(long, default_value_t = 15)]
    tune_steps: usize,

    /// Space-separated ESKF parameters to tune. Example: --tune-params "r_gps_pos r_baro"
    #[arg(long, value_delimiter = ' ')]
    tune_params: Option<Vec<String>>,

    /// Space-separated flight stages to tune. Example: --tune-stages "Ascent Coast"
    #[arg(long, value_delimiter = ' ')]
    tune_stages: Option<Vec<String>>,

    /// Test filter robustness with systematic sensor failures (GPS dropout, baro noise spike)
    #[arg(long)]
    sensor_failure_test: bool,

    /// Space-separated sensor noise scale factors for Monte Carlo. Example: --tune-noise-scales "0.5 1.0 2.0"
    #[arg(long, value_delimiter = ' ', default_values = ["1.0"])]
    tune_noise_scales: Vec<f64>,

    /// Space-separated RNG seeds for Monte Carlo analysis. Example: --tune-seeds "42 43 44"
    #[arg(long, value_delimiter = ' ', default_values = ["42"])]
    tune_seeds: Vec<u64>,

    /// Magnetic declination at launch site (degrees). Angle between true north and magnetic north
    #[arg(long, default_value_t = 0.0)]
    mag_declination: f32,

    /// Launch site latitude (degrees). Used for geographic coordinate conversion
    #[arg(long, default_value_t = 35.0)]
    home_lat: f32,

    /// Launch site longitude (degrees). Used for geographic coordinate conversion
    #[arg(long, default_value_t = -106.0)]
    home_lon: f32,

    /// Launch site altitude MSL (m). Used as reference for NED coordinate conversions
    #[arg(long, default_value_t = 1500.0)]
    home_alt: f32,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    Csv,
    Json,
}

#[allow(dead_code)]
fn main() -> Result<()> {
    let args = Args::parse();
    main_inner(args)
}

fn main_inner(args: Args) -> Result<()> {
    println!("Aloe Rocket Simulator (6-DoF)");
    println!("=============================\n");

    if args.tune_sweep {
        run_tune_sweep(&args)?;
    } else if args.single {
        run_single(&args)?;
    } else {
        run_sweep(&args)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Single Run
// ---------------------------------------------------------------------------
fn run_single(args: &Args) -> Result<()> {
    println!("Running single simulation...");

    // 1. Setup & Simulate
    let params = build_rocket_params(args);
    let result = simulate_6dof(&params);

    print_sim_stats(&result);

    // 2. Generate Sensors
    let (sensor_data, filter_result) = if !args.no_sensors {
        let config = build_sensor_config(args);
        let s_data = generate_sensor_data(&result, &config);

        // 3. Run Filter (if requested)
        let f_res = if !args.no_filter {
            println!("Running Navigation Filter...");
            let filter_config = build_filter_config(args);
            Some(run_filter(&result, &s_data, &filter_config))
        } else {
            None
        };
        (Some(s_data), f_res)
    } else {
        (None, None)
    };

    // 4. Export
    write_output(args, &result, sensor_data.as_ref(), filter_result.as_ref())?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Sweep Run
// ---------------------------------------------------------------------------
fn run_sweep(args: &Args) -> Result<()> {
    // Simplified sweep implementation
    // We only sweep mass/thrust to demonstrate the new pipeline
    let default_sweep = vec!["thrust".to_string()];
    let sweep_names = args.sweep_params.as_ref().unwrap_or(&default_sweep);

    // Hardcoded simple sweep for demonstration
    // In a full CLI, you'd parse ranges dynamically
    println!("Sweeping parameters: {:?}", sweep_names);

    let steps = args.sweep_steps;
    let mut summary_rows = Vec::new();

    // Just sweeping Thrust for example
    let start_thrust = args.thrust * 0.5;
    let end_thrust = args.thrust * 1.5;

    for i in 0..steps {
        let val = start_thrust + (end_thrust - start_thrust) * (i as f64 / (steps - 1) as f64);

        let mut params = build_rocket_params(args);
        params.thrust_curve = vec![
            (0.0, val),
            (args.burn_time, val),
            (args.burn_time + 0.01, 0.0),
        ];

        let res = simulate_6dof(&params);
        let apogee = res
            .pos
            .iter()
            .map(|p| -p.z)
            .fold(f64::NEG_INFINITY, f64::max);

        println!(
            "Run {}/{} | Thrust: {:.1} N -> Apogee: {:.1} m",
            i + 1,
            steps,
            val,
            apogee
        );

        let mut row = HashMap::new();
        row.insert("thrust".to_string(), val);
        row.insert("apogee".to_string(), apogee);
        summary_rows.push(row);
    }

    // Write summary
    let path = args.output_dir.join("sweep_summary.csv");
    std::fs::create_dir_all(&args.output_dir)?;
    let mut wtr = csv::Writer::from_path(&path)?;
    wtr.write_record(["thrust", "apogee"])?;
    for row in summary_rows {
        wtr.write_record(&[
            format!("{:.2}", row["thrust"]),
            format!("{:.2}", row["apogee"]),
        ])?;
    }

    println!("\nSweep complete. Summary at {:?}", path);
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_rocket_params(args: &Args) -> RocketParams {
    let thrust_curve = if args.thrust_curve.is_empty() {
        vec![
            (0.0, args.thrust),
            (args.burn_time, args.thrust),
            (args.burn_time + 0.01, 0.0),
        ]
    } else {
        args.thrust_curve
            .iter()
            .map(|point| (point.time_s, point.thrust_n))
            .collect()
    };

    RocketParams {
        dry_mass: args.dry_mass,
        fuel_mass: args.fuel_mass,
        oxidizer_mass: args.oxidizer_mass,
        inertia_tensor: Vector3::new(args.inertia_x, args.inertia_y, args.inertia_z),
        cg_full: args.cg_full,
        cg_empty: args.cg_empty,
        cp_location: args.cp_location,
        ref_area: args.ref_area,
        drag_coeff_axial: args.drag_coeff,
        normal_force_coeff: args.normal_force_coeff,
        thrust_curve,
        burn_time: args.burn_time,
        isp: args.isp,
        nozzle_location: args.nozzle_location,
        gravity: args.gravity,
        air_density_sea_level: args.air_density,
        launch_rod_length: args.launch_rod_length,
        wind_velocity_ned: Vector3::new(args.wind_north, args.wind_east, args.wind_down),
        launch_delay: args.launch_delay,
        spin_rate: args.spin_rate,
        thrust_cant: args.thrust_cant,
        nozzle_exit_pressure_psf: args.nozzle_exit_pressure_psf,
        nozzle_area_ft2: args.nozzle_area_ft2,
        pad_static_friction: args.pad_static_friction,
        pad_dynamic_friction: args.pad_dynamic_friction,
        pad_spring_coeff_lbs_ft: args.pad_spring_coeff_lbs_ft,
        pad_damping_coeff_lbs_ft_s: args.pad_damping_coeff_lbs_ft_s,
        sim_dt: args.sim_dt,
        max_time: args.max_time,
    }
}

fn build_sensor_config(args: &Args) -> SensorConfig {
    // Map scalar "noise_scale" to individual sensor sigmas
    let mut cfg = SensorConfig {
        noise_scale: args.noise_scale,
        accel_noise_std: 0.1,   // m/s^2
        gyro_noise_std: 0.002,  // rad/s
        mag_noise_std: 0.001,   // Gauss
        baro_noise_std: 0.5,    // meters
        gps_pos_noise_std: 2.0, // meters
        gps_vel_noise_std: 0.1, // m/s
        accel_bias: Vector3::zeros(),
        gyro_bias: Vector3::zeros(),
        seed: args.seed,
        accel_enabled: true,
        gyro_enabled: true,
        mag_enabled: true,
        baro_enabled: true,
        gps_enabled: true,
        bmi088_accel_rate_hz: 1000.0,
        bmi088_gyro_rate_hz: 1000.0,
        adxl375_rate_hz: 3200.0,
        lis3mdl_rate_hz: 100.0,
        ms5611_rate_hz: 50.0,
        gps_rate_hz: 10.0,
        accel_saturation: 200.0, // BMI088: ±200 m/s²
        gyro_saturation: 34.9,   // BMI088: 2000 deg/s
        chaos: ChaosConfig::default(),
    };

    // Apply sensor disable flags
    for sensor in &args.disable_sensor {
        match sensor.as_str() {
            "accel" => cfg.accel_enabled = false,
            "gyro" => cfg.gyro_enabled = false,
            "mag" => cfg.mag_enabled = false,
            "baro" => cfg.baro_enabled = false,
            "gps" => cfg.gps_enabled = false,
            _ => eprintln!("Warning: unknown sensor '{}' in --disable-sensor", sensor),
        }
    }

    cfg
}

fn build_filter_config(args: &Args) -> FilterConfig {
    let mut cfg = FilterConfig {
        mag_declination_deg: args.mag_declination as f64,
        home_lat_deg: args.home_lat as f64,
        home_lon_deg: args.home_lon as f64,
        home_alt_m: args.home_alt as f64,
        ..FilterConfig::default()
    };

    apply_filter_override(
        &mut cfg.accel_noise_density,
        args.accel_noise_density_tune.or(args.accel_noise_density),
    );
    apply_filter_override(&mut cfg.gyro_noise_density, args.gyro_noise_density);
    apply_filter_override(&mut cfg.accel_bias_instability, args.accel_bias_instability);
    apply_filter_override(&mut cfg.gyro_bias_instability, args.gyro_bias_instability);
    apply_filter_override(&mut cfg.pos_process_noise, args.pos_process_noise);
    apply_filter_override(&mut cfg.r_gps_pos, args.r_gps_pos);
    apply_filter_override(&mut cfg.r_gps_vel, args.r_gps_vel);
    apply_filter_override(&mut cfg.r_baro, args.r_baro);
    apply_filter_override(&mut cfg.r_mag, args.r_mag);

    cfg
}

fn apply_filter_override(values: &mut [f64], override_value: Option<f32>) {
    if let Some(value) = override_value {
        values.fill(value as f64);
    }
}

fn print_sim_stats(result: &SimResult) {
    let max_alt = result
        .pos
        .iter()
        .map(|p| -p.z)
        .fold(f64::NEG_INFINITY, f64::max);
    let flight_time = result.time.last().copied().unwrap_or(0.0);

    println!("\nSimulation Stats:");
    println!("  Steps:       {}", result.time.len());
    println!("  Flight Time: {:.2} s", flight_time);
    println!("  Apogee:      {:.2} m (AGL)", max_alt);
    println!("-----------------------------");
}

fn write_output(
    args: &Args,
    sim: &SimResult,
    _sensors: Option<&SensorData>,
    filter: Option<&FilterResult>,
) -> Result<()> {
    std::fs::create_dir_all(&args.output_dir)?;
    let path = args.output_dir.join("simulation.csv");
    let mut wtr = csv::Writer::from_path(&path)?;

    wtr.write_record([
        "time",
        // Sim Truth
        "true_pos_n",
        "true_pos_e",
        "true_pos_d",
        "true_vel_n",
        "true_vel_e",
        "true_vel_d",
        "true_accel_x",
        "true_accel_y",
        "true_accel_z", // Body frame proper accel
        // Filter Estimates (if available)
        "est_pos_n",
        "est_pos_e",
        "est_pos_d",
        "est_vel_n",
        "est_vel_e",
        "est_vel_d",
        "est_roll",
        "est_pitch",
        "est_yaw",
    ])?;

    let n = sim.time.len();
    for i in 0..n {
        let t = sim.time[i];
        let p = sim.pos[i];
        let v = sim.vel[i];
        let a = sim.accel_body[i];

        // Default "empty" values if filter didn't run
        let (ep, ev, erpy) = if let Some(f) = filter {
            if i < f.time.len() {
                (f.position[i], f.velocity[i], f.orientation_euler[i])
            } else {
                (Vector3::zeros(), Vector3::zeros(), Vector3::zeros())
            }
        } else {
            (Vector3::zeros(), Vector3::zeros(), Vector3::zeros())
        };

        wtr.write_record(&[
            format!("{:.4}", t),
            // Truth
            format!("{:.4}", p.x),
            format!("{:.4}", p.y),
            format!("{:.4}", p.z),
            format!("{:.4}", v.x),
            format!("{:.4}", v.y),
            format!("{:.4}", v.z),
            format!("{:.4}", a.x),
            format!("{:.4}", a.y),
            format!("{:.4}", a.z),
            // Est
            format!("{:.4}", ep.x),
            format!("{:.4}", ep.y),
            format!("{:.4}", ep.z),
            format!("{:.4}", ev.x),
            format!("{:.4}", ev.y),
            format!("{:.4}", ev.z),
            format!("{:.4}", erpy.x),
            format!("{:.4}", erpy.y),
            format!("{:.4}", erpy.z),
        ])?;
    }

    wtr.flush()?;
    println!("Data written to {:?}", path);
    Ok(())
}

// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
/// Metrics computed during filter tuning
struct TuneMetrics {
    /// 3D position RMSE (meters)
    pos3d_rmse_m: f64,
    /// Altitude RMSE (meters)
    alt_rmse_m: f64,
    /// 3D velocity RMSE (m/s)
    vel3d_rmse_m: f64,
    /// Maximum position error (meters)
    pos_max_m: f64,
    /// Maximum altitude error (meters)
    alt_max_m: f64,
    /// Apogee detection error (seconds, positive = late detection)
    apogee_time_error_s: f64,
}

impl TuneMetrics {
    fn zero() -> Self {
        Self {
            pos3d_rmse_m: 0.0,
            alt_rmse_m: 0.0,
            vel3d_rmse_m: 0.0,
            pos_max_m: 0.0,
            alt_max_m: 0.0,
            apogee_time_error_s: 0.0,
        }
    }

    fn average(samples: &[Self]) -> Self {
        if samples.is_empty() {
            return Self::zero();
        }

        let mut sum = Self::zero();
        let mut apogee_sum = 0.0;
        let mut apogee_count = 0usize;

        for sample in samples {
            sum.pos3d_rmse_m += sample.pos3d_rmse_m;
            sum.alt_rmse_m += sample.alt_rmse_m;
            sum.vel3d_rmse_m += sample.vel3d_rmse_m;
            sum.pos_max_m += sample.pos_max_m;
            sum.alt_max_m += sample.alt_max_m;
            if sample.apogee_time_error_s.is_finite() {
                apogee_sum += sample.apogee_time_error_s;
                apogee_count += 1;
            }
        }

        let count = samples.len() as f64;
        Self {
            pos3d_rmse_m: sum.pos3d_rmse_m / count,
            alt_rmse_m: sum.alt_rmse_m / count,
            vel3d_rmse_m: sum.vel3d_rmse_m / count,
            pos_max_m: sum.pos_max_m / count,
            alt_max_m: sum.alt_max_m / count,
            apogee_time_error_s: if apogee_count > 0 {
                apogee_sum / apogee_count as f64
            } else {
                f64::NAN
            },
        }
    }
}

struct SensorSweepCase {
    seed: u64,
    noise_scale: f64,
    sensor_data: SensorData,
}

#[derive(Debug, Clone)]
/// Row data for tune sweep CSV output
struct SummaryRow {
    iteration: usize,
    param_name: String,
    stage: usize,
    value: f64,
    seed: String,
    noise_scale: String,
    pos3d_rmse_m: f64,
    alt_rmse_m: f64,
    vel3d_rmse_m: f64,
    pos_max_m: f64,
    alt_max_m: f64,
    apogee_time_error_s: f64,
}

// ---------------------------------------------------------------------------
fn get_param_spec(name: &str) -> (f64, f64) {
    match name {
        "accel_noise_density" => (0.001, 20.0),
        "gyro_noise_density" => (0.00001, 0.5),
        "accel_bias_instability" => (0.001, 1.0),
        "gyro_bias_instability" => (1e-6, 1e-2),
        "pos_process_noise" => (0.1, 10.0),
        "r_gps_pos" => (1.0, 1000.0),
        "r_gps_vel" => (0.01, 10.0),
        "r_baro" => (0.01, 10.0),
        "r_mag" => (0.1, 10.0),
        _ => (0.0, 1.0),
    }
}

fn compute_tune_metrics(sim: &SimResult, filter: &FilterResult) -> TuneMetrics {
    let n = sim.time.len();
    let mut pos_err_sq = 0.0_f64;
    let mut alt_err_sq = 0.0_f64;
    let mut vel_err_sq = 0.0_f64;
    let mut pos_max: f64 = 0.0;
    let mut alt_max: f64 = 0.0;
    let mut count = 0;

    for i in 0..n {
        if i >= filter.time.len() {
            break;
        }
        // Truth position in NED: N = sim.pos.x, E = sim.pos.y, D = sim.pos.z
        let truth_pos = Vector3::new(sim.pos[i].x, sim.pos[i].y, sim.pos[i].z);
        let est_pos = &filter.position[i];
        let pos_err = truth_pos - Vector3::new(est_pos.x, est_pos.y, est_pos.z);
        let pos_err_norm = pos_err.norm();
        pos_err_sq += pos_err_norm.powi(2);
        pos_max = pos_max.max(pos_err_norm);

        // Altitude error (truth altitude = -z, est altitude = -z)
        let truth_alt = -sim.pos[i].z;
        let est_alt = -est_pos.z;
        let alt_err = (truth_alt - est_alt).abs();
        alt_err_sq += alt_err.powi(2);
        alt_max = alt_max.max(alt_err);

        // Velocity error
        let truth_vel = &sim.vel[i];
        let est_vel = &filter.velocity[i];
        let vel_err = truth_vel - Vector3::new(est_vel.x, est_vel.y, est_vel.z);
        vel_err_sq += vel_err.norm_squared();

        count += 1;
    }

    // Calculate apogee time error
    let apogee_time_error_s = match (sim.descent_time, filter.descent_time) {
        (Some(sim_t), Some(filter_t)) => filter_t - sim_t,
        _ => f64::NAN, // Not detected
    };

    TuneMetrics {
        pos3d_rmse_m: (pos_err_sq / count as f64).sqrt(),
        alt_rmse_m: (alt_err_sq / count as f64).sqrt(),
        vel3d_rmse_m: (vel_err_sq / count as f64).sqrt(),
        pos_max_m: pos_max,
        alt_max_m: alt_max,
        apogee_time_error_s,
    }
}

fn run_tune_sweep(args: &Args) -> Result<()> {
    println!("Running tune-sweep...");

    // Parse tune mode
    let tune_mode = args.tune_mode.as_str();
    let tune_steps = args.tune_steps;
    let tune_stages: Vec<usize> = if let Some(stages_vec) = &args.tune_stages {
        stages_vec
            .iter()
            .map(|s| match s.as_str() {
                "pad" => 0,
                "burn" => 1,
                "coasting" | "coast" => 2,
                "recovery" => 3,
                _ => panic!("Unknown stage: {}", s),
            })
            .collect()
    } else {
        vec![0, 1, 2, 3] // All stages
    };

    let tune_noise_scales = if args.tune_noise_scales.is_empty() {
        vec![args.noise_scale]
    } else {
        args.tune_noise_scales.clone()
    };
    let tune_seeds = if args.tune_seeds.is_empty() {
        vec![args.seed]
    } else {
        args.tune_seeds.clone()
    };

    // Base simulation parameters
    let base_params = build_rocket_params(args);
    let base_sim = simulate_6dof(&base_params);

    // Reuse one truth simulation across all sensor parameter cases.
    let mut sensor_sweep_cases = Vec::new();
    for &noise_scale in &tune_noise_scales {
        for &seed in &tune_seeds {
            let mut cfg = build_sensor_config(args);
            cfg.noise_scale = noise_scale;
            cfg.seed = seed;
            sensor_sweep_cases.push(SensorSweepCase {
                seed,
                noise_scale,
                sensor_data: generate_sensor_data(&base_sim, &cfg),
            });
        }
    }

    // Base filter config
    let mut base_filter_cfg = build_filter_config(args);

    // Compute baseline metrics across all requested sensor cases.
    let baseline_metrics = evaluate_filter_config(&base_sim, &sensor_sweep_cases, &base_filter_cfg);
    let baseline_rmse = baseline_metrics.pos3d_rmse_m;
    println!("Baseline pos3d_rmse = {:.4} m", baseline_rmse);

    let mut summary_rows: Vec<SummaryRow> = Vec::new();
    let case_seed_summary = summarize_seeds(&sensor_sweep_cases);
    let case_noise_summary = summarize_noise_scales(&sensor_sweep_cases);

    if tune_mode == "greedy" {
        // Implement greedy coordinate descent
        let params_to_tune = vec![
            "accel_noise_density",
            "gyro_noise_density",
            "accel_bias_instability",
            "gyro_bias_instability",
            "pos_process_noise",
            "r_gps_pos",
            "r_gps_vel",
            "r_baro",
            "r_mag",
        ];

        let max_greedy_iter = 3; // Limited iterations for simplicity

        for iter in 0..max_greedy_iter {
            println!("Greedy iteration {}", iter + 1);
            let mut improved = false;

            for param_name in &params_to_tune {
                for &stage_idx in &tune_stages {
                    println!("Tuning {} for stage {}", param_name, stage_idx);

                    let (min_val, max_val) = get_param_spec(param_name);
                    let values: Vec<f64> = (0..tune_steps)
                        .map(|i| {
                            let t = i as f64 / (tune_steps - 1) as f64;
                            min_val + t * (max_val - min_val)
                        })
                        .collect();

                    let mut best_val = base_filter_cfg.get_stage_param(stage_idx, param_name);
                    let mut best_rmse = f64::INFINITY;

                    for &val in &values {
                        let mut cfg = base_filter_cfg.clone();
                        cfg.set_stage_param(stage_idx, param_name, val);

                        let metrics = evaluate_filter_config(&base_sim, &sensor_sweep_cases, &cfg);
                        let rmse = metrics.pos3d_rmse_m;

                        if rmse < best_rmse {
                            best_rmse = rmse;
                            best_val = val;
                        }

                        summary_rows.push(SummaryRow {
                            iteration: iter + 1,
                            param_name: param_name.to_string(),
                            stage: stage_idx,
                            value: val,
                            seed: case_seed_summary.clone(),
                            noise_scale: case_noise_summary.clone(),
                            pos3d_rmse_m: metrics.pos3d_rmse_m,
                            alt_rmse_m: metrics.alt_rmse_m,
                            vel3d_rmse_m: metrics.vel3d_rmse_m,
                            pos_max_m: metrics.pos_max_m,
                            alt_max_m: metrics.alt_max_m,
                            apogee_time_error_s: metrics.apogee_time_error_s,
                        });
                    }

                    // Update base config if improved
                    if best_rmse < baseline_rmse {
                        base_filter_cfg.set_stage_param(stage_idx, param_name, best_val);
                        improved = true;
                        println!(
                            "Improved {} for stage {} to {:.6}, rmse {:.4}",
                            param_name, stage_idx, best_val, best_rmse
                        );
                    }
                }
            }

            if !improved {
                println!("No improvement in iteration {}, stopping", iter + 1);
                break;
            }
        }

        // Output optimised tuning
        let optimised = serde_json::json!({
            "baseline_rmse": baseline_rmse,
            "optimised_rmse": evaluate_filter_config(&base_sim, &sensor_sweep_cases, &base_filter_cfg).pos3d_rmse_m,
            "sensor_cases": {
                "seeds": tune_seeds,
                "noise_scales": tune_noise_scales,
            },
            "tuning": base_filter_cfg.to_json()
        });

        let optimised_path = args.output_dir.join("optimised_tuning.json");
        // Ensure output directory exists before writing the optimised tuning file
        std::fs::create_dir_all(&args.output_dir)?;
        std::fs::write(&optimised_path, serde_json::to_string_pretty(&optimised)?)?;
        println!("Optimised tuning written to {:?}", optimised_path);
    } else {
        // Simple sweep on accel_noise_density for all stages
        let tname = "accel_noise_density";
        let (min, max) = get_param_spec(tname);
        let values: Vec<f64> = (0..tune_steps)
            .map(|i| {
                let t = i as f64 / (tune_steps - 1) as f64;
                min + t * (max - min)
            })
            .collect();

        for &val in &values {
            let mut cfg = base_filter_cfg.clone();
            for &stage in &tune_stages {
                cfg.set_stage_param(stage, tname, val);
            }

            let metrics = evaluate_filter_config(&base_sim, &sensor_sweep_cases, &cfg);

            summary_rows.push(SummaryRow {
                iteration: 0, // Grid sweep has no iterations
                param_name: tname.to_string(),
                stage: 0, // Applied to all stages
                value: val,
                seed: case_seed_summary.clone(),
                noise_scale: case_noise_summary.clone(),
                pos3d_rmse_m: metrics.pos3d_rmse_m,
                alt_rmse_m: metrics.alt_rmse_m,
                vel3d_rmse_m: metrics.vel3d_rmse_m,
                pos_max_m: metrics.pos_max_m,
                alt_max_m: metrics.alt_max_m,
                apogee_time_error_s: metrics.apogee_time_error_s,
            });

            println!("{} = {:.6}  rmse = {:.4}", tname, val, metrics.pos3d_rmse_m);
        }
    }

    // Write CSV with full metrics
    let path = args.output_dir.join("tune_sweep_summary.csv");
    std::fs::create_dir_all(&args.output_dir)?;
    let mut wtr = csv::Writer::from_path(&path)?;
    wtr.write_record([
        "iteration",
        "param_name",
        "stage",
        "value",
        "seed",
        "noise_scale",
        "pos3d_rmse_m",
        "alt_rmse_m",
        "vel3d_rmse_m",
        "pos_max_m",
        "alt_max_m",
        "apogee_time_error_s",
    ])?;
    for row in summary_rows {
        wtr.write_record([
            &format!("{}", row.iteration),
            &row.param_name,
            &format!("{}", row.stage),
            &format!("{:.6}", row.value),
            &row.seed,
            &row.noise_scale,
            &format!("{:.6}", row.pos3d_rmse_m),
            &format!("{:.6}", row.alt_rmse_m),
            &format!("{:.6}", row.vel3d_rmse_m),
            &format!("{:.6}", row.pos_max_m),
            &format!("{:.6}", row.alt_max_m),
            &if row.apogee_time_error_s.is_nan() {
                "NA".to_string()
            } else {
                format!("{:.6}", row.apogee_time_error_s)
            },
        ])?;
    }
    wtr.flush()?;
    println!("Tune-sweep summary written to {:?}", path);

    Ok(())
}

fn evaluate_filter_config(
    base_sim: &SimResult,
    sensor_sweep_cases: &[SensorSweepCase],
    filter_cfg: &FilterConfig,
) -> TuneMetrics {
    let metrics: Vec<TuneMetrics> = sensor_sweep_cases
        .iter()
        .map(|sensor_case| {
            let filter_result = run_filter(base_sim, &sensor_case.sensor_data, filter_cfg);
            compute_tune_metrics(base_sim, &filter_result)
        })
        .collect();

    TuneMetrics::average(&metrics)
}

fn summarize_seeds(sensor_sweep_cases: &[SensorSweepCase]) -> String {
    let mut seeds: Vec<u64> = sensor_sweep_cases.iter().map(|case| case.seed).collect();
    seeds.sort_unstable();
    seeds.dedup();
    seeds
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

fn summarize_noise_scales(sensor_sweep_cases: &[SensorSweepCase]) -> String {
    let mut scales: Vec<f64> = sensor_sweep_cases
        .iter()
        .map(|case| case.noise_scale)
        .collect();
    scales.sort_by(|a, b| a.total_cmp(b));
    scales.dedup_by(|a, b| a.total_cmp(b).is_eq());
    scales
        .iter()
        .map(|scale| format!("{scale:.4}"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_filter_config_preserves_stage_defaults_without_overrides() {
        let args = Args::parse_from(["aloe-cli"]);
        let cfg = build_filter_config(&args);
        let defaults = FilterConfig::default();

        assert_eq!(cfg.accel_noise_density, defaults.accel_noise_density);
        assert_eq!(cfg.gyro_noise_density, defaults.gyro_noise_density);
        assert_eq!(cfg.accel_bias_instability, defaults.accel_bias_instability);
        assert_eq!(cfg.gyro_bias_instability, defaults.gyro_bias_instability);
        assert_eq!(cfg.pos_process_noise, defaults.pos_process_noise);
        assert_eq!(cfg.r_gps_pos, defaults.r_gps_pos);
        assert_eq!(cfg.r_gps_vel, defaults.r_gps_vel);
        assert_eq!(cfg.r_baro, defaults.r_baro);
        assert_eq!(cfg.r_mag, defaults.r_mag);
    }

    #[test]
    fn build_filter_config_applies_explicit_overrides_to_all_stages() {
        let args = Args::parse_from([
            "aloe-cli",
            "--gyro-noise-density",
            "0.25",
            "--r-baro",
            "5.0",
            "--accel-noise-density",
            "0.5",
        ]);
        let cfg = build_filter_config(&args);

        assert!(cfg
            .gyro_noise_density
            .iter()
            .all(|value| (*value - 0.25).abs() < f64::EPSILON));
        assert!(cfg
            .r_baro
            .iter()
            .all(|value| (*value - 5.0).abs() < f64::EPSILON));
        assert!(cfg
            .accel_noise_density
            .iter()
            .all(|value| (*value - 0.5).abs() < f64::EPSILON));
    }

    #[test]
    fn build_rocket_params_includes_backend_runtime_knobs() {
        let args = Args::parse_from([
            "aloe-cli",
            "--wind-down",
            "3.5",
            "--sim-dt",
            "0.002",
            "--max-time",
            "120",
        ]);
        let params = build_rocket_params(&args);

        assert!((params.wind_velocity_ned.z - 3.5).abs() < f64::EPSILON);
        assert!((params.sim_dt - 0.002).abs() < f64::EPSILON);
        assert!((params.max_time - 120.0).abs() < f64::EPSILON);
    }
}
