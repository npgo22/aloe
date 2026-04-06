export type Units = 'metric' | 'imperial';

export interface RocketPreset {
  key: string;
  label: string;
  note?: string;
  values: Partial<SimulationRequest['rocket']> & Partial<SimulationRequest['environment']>;
}

export interface StageTuning {
  accel_noise_density: number;
  gyro_noise_density: number;
  accel_bias_instability: number;
  gyro_bias_instability: number;
  pos_process_noise: number;
  r_gps_pos: number;
  r_gps_vel: number;
  r_baro: number;
  r_mag: number;
}

export type FilterAlgorithm = 'eskf' | 'kalman' | 'information';

export interface SimulationRequest {
  rocket: {
    dry_mass: number;
    fuel_mass: number;
    oxidizer_mass: number;
    cg_full: number;
    cg_empty: number;
    cp_location: number;
    inertia_x: number;
    inertia_y: number;
    inertia_z: number;
    thrust: number;
    burn_time: number;
    drag_coeff: number;
    normal_force_coeff: number;
    ref_area: number;
    isp: number;
    nozzle_location: number;
    launch_delay: number;
    launch_rod_length: number;
    spin_rate: number;
    thrust_cant: number;
    nozzle_exit_pressure_psf: number;
    nozzle_area_ft2: number;
    pad_static_friction: number;
    pad_dynamic_friction: number;
    pad_spring_coeff_lbs_ft: number;
    pad_damping_coeff_lbs_ft_s: number;
  };
  environment: {
    gravity: number;
    wind_north: number;
    wind_east: number;
    wind_down: number;
    air_density: number;
    sim_dt: number;
    max_time: number;
  };
  sensors: {
    noise_scale: number;
    seed: number;
    bmi088_accel_enabled: boolean;
    bmi088_gyro_enabled: boolean;
    adxl375_enabled: boolean;
    lis3mdl_enabled: boolean;
    ms5611_enabled: boolean;
    gps_enabled: boolean;
    bmi088_accel_rate_hz: number;
    bmi088_gyro_rate_hz: number;
    adxl375_rate_hz: number;
    lis3mdl_rate_hz: number;
    ms5611_rate_hz: number;
    gps_rate_hz: number;
  };
  filter: {
    ground_pressure_mbar: number;
    mag_declination_deg: number;
    mag_dip_deg: number;
    home_lat_deg: number;
    home_lon_deg: number;
    home_alt_m: number;
    launch_accel_thresh: number;
    launch_vel_thresh: number;
    burnout_accel_thresh: number;
    min_ascent_time: number;
    apogee_descent_thresh: number;
    min_coast_time: number;
    landing_vel_thresh: number;
    landing_alt_thresh: number;
    landing_confirm_window: number;
    high_velocity_baro_thresh: number;
    stage_tuning: StageTuning[];
    selected_algorithms: FilterAlgorithm[];
    active_algorithm: FilterAlgorithm;
  };
  options: {
    no_sensors: boolean;
    no_filter: boolean;
  };
}

export interface SimpleErrorStats {
  min: number;
  max: number;
  mean: number;
  std: number;
  rmse: number;
  mae: number;
  p95: number;
  n: number;
}

export interface ErrorStatsGroup {
  [key: string]: SimpleErrorStats;
}

export interface SimulationResponse {
  time: number[];
  altitude: number[];
  velocity: number[];
  acceleration: number[];
  force: number[];
  mass: number[];
  position_x: number[];
  position_y: number[];
  position_z: number[];
  velocity_x: number[];
  velocity_y: number[];
  velocity_z: number[];
  state_changes_sim: Array<{ time: number; state: string; description: string }>;
  state_changes_eskf: Array<{ time: number; state: string; description: string }>;
  sensor_data: {
    accel_x: number[];
    accel_y: number[];
    accel_z: number[];
    gyro_x: number[];
    gyro_y: number[];
    gyro_z: number[];
    baro_pressure: number[];
    mag_x: number[];
    mag_y: number[];
    mag_z: number[];
    gps_x: Array<number | null>;
    gps_y: Array<number | null>;
    gps_z: Array<number | null>;
    gps_vel_x: Array<number | null>;
    gps_vel_y: Array<number | null>;
    gps_vel_z: Array<number | null>;
    adxl_x: number[];
    adxl_y: number[];
    adxl_z: number[];
  };
  filter_data: {
    est_pos_x: number[];
    est_pos_y: number[];
    est_pos_z: number[];
    est_vel_x: number[];
    est_vel_y: number[];
    est_vel_z: number[];
    quantized_est_pos_x: number[];
    quantized_est_pos_y: number[];
    quantized_est_pos_z: number[];
    quantized_est_vel_x: number[];
    quantized_est_vel_y: number[];
    quantized_est_vel_z: number[];
  };
  error_stats: {
    eskf: ErrorStatsGroup | null;
    quantized_flight: ErrorStatsGroup | null;
    quant_roundtrip: ErrorStatsGroup | null;
    quant_recovery: ErrorStatsGroup | null;
    state_detection: ErrorStatsGroup | null;
  } | null;
  active_filter_algorithm: FilterAlgorithm;
  available_filter_algorithms: FilterAlgorithm[];
  algorithm_outputs: Record<
    string,
    {
      filter_data: SimulationResponse['filter_data'];
      error_stats: SimulationResponse['error_stats'];
    }
  >;
  true_accel_x: number[];
  true_accel_y: number[];
  true_accel_z: number[];
  true_gyro_x: number[];
  true_gyro_y: number[];
  true_gyro_z: number[];
  apogee: number;
  max_velocity: number;
  flight_time: number;
  success: boolean;
  error_message: string | null;
}

export interface StateChange {
  time: number;
  state: string;
  description: string;
}

export interface StatRow {
  category: string;
  algorithm: string;
  label: string;
  stats: SimpleErrorStats;
  unit?: 'distance' | 'velocity' | 'time' | 'angle';
}

export const STAGE_LABELS = ['Pad', 'Ascent', 'Coast', 'Descent'];

export const TUNING_FIELDS: Array<{ key: keyof StageTuning; label: string; step: string }> = [
  { key: 'accel_noise_density', label: 'Accel Noise Density', step: '0.0001' },
  { key: 'gyro_noise_density', label: 'Gyro Noise Density', step: '0.0001' },
  { key: 'accel_bias_instability', label: 'Accel Bias Instability', step: '0.000001' },
  { key: 'gyro_bias_instability', label: 'Gyro Bias Instability', step: '0.000001' },
  { key: 'pos_process_noise', label: 'Position Process Noise', step: '0.0001' },
  { key: 'r_gps_pos', label: 'GPS Position Variance', step: '0.01' },
  { key: 'r_gps_vel', label: 'GPS Velocity Variance', step: '0.001' },
  { key: 'r_baro', label: 'Barometer Variance', step: '0.01' },
  { key: 'r_mag', label: 'Magnetometer Variance', step: '0.0001' }
];

export const DEFAULT_REQUEST: SimulationRequest = {
  rocket: {
    dry_mass: 12,
    fuel_mass: 8 / 3,
    oxidizer_mass: 16 / 3,
    cg_full: 1.2,
    cg_empty: 1.1,
    cp_location: 1.6,
    inertia_x: 0.006,
    inertia_y: 1.2,
    inertia_z: 1.2,
    thrust: 2000,
    burn_time: 4.5,
    drag_coeff: 0.35,
    normal_force_coeff: 12,
    ref_area: 0.015,
    isp: 210,
    nozzle_location: 2,
    launch_delay: 1,
    launch_rod_length: 3,
    spin_rate: 0,
    thrust_cant: 0,
    nozzle_exit_pressure_psf: 2116.22,
    nozzle_area_ft2: 0.01,
    pad_static_friction: 0.8,
    pad_dynamic_friction: 0.4,
    pad_spring_coeff_lbs_ft: 10000,
    pad_damping_coeff_lbs_ft_s: 5000
  },
  environment: {
    gravity: 9.81,
    wind_north: 0,
    wind_east: 0,
    wind_down: 0,
    air_density: 1.225,
    sim_dt: 0.001,
    max_time: 400
  },
  sensors: {
    noise_scale: 1,
    seed: 42,
    bmi088_accel_enabled: true,
    bmi088_gyro_enabled: true,
    adxl375_enabled: false,
    lis3mdl_enabled: true,
    ms5611_enabled: true,
    gps_enabled: true,
    bmi088_accel_rate_hz: 200,
    bmi088_gyro_rate_hz: 200,
    adxl375_rate_hz: 400,
    lis3mdl_rate_hz: 25,
    ms5611_rate_hz: 20,
    gps_rate_hz: 5
  },
  filter: {
    ground_pressure_mbar: 1013.25,
    mag_declination_deg: 0,
    mag_dip_deg: 60,
    home_lat_deg: 35,
    home_lon_deg: -106,
    home_alt_m: 1500,
    launch_accel_thresh: 20,
    launch_vel_thresh: 10,
    burnout_accel_thresh: 2,
    min_ascent_time: 0.5,
    apogee_descent_thresh: 1,
    min_coast_time: 2,
    landing_vel_thresh: 0.5,
    landing_alt_thresh: 100,
    landing_confirm_window: 2,
    high_velocity_baro_thresh: 170,
    selected_algorithms: ['eskf', 'kalman', 'information'],
    active_algorithm: 'eskf',
    stage_tuning: [
      {
        accel_noise_density: 0.2236,
        gyro_noise_density: 0.03728,
        accel_bias_instability: 0.01,
        gyro_bias_instability: 0.00003728,
        pos_process_noise: 1,
        r_gps_pos: 61.05,
        r_gps_vel: 0.07197,
        r_baro: 0.1,
        r_mag: 1
      },
      {
        accel_noise_density: 0.0243,
        gyro_noise_density: 0.01389,
        accel_bias_instability: 0.002683,
        gyro_bias_instability: 0.00001,
        pos_process_noise: 0.1389,
        r_gps_pos: 0.1,
        r_gps_vel: 0.04394,
        r_baro: 2.236,
        r_mag: 0.01179
      },
      {
        accel_noise_density: 0.01,
        gyro_noise_density: 0.1,
        accel_bias_instability: 0.000001,
        gyro_bias_instability: 0.0000001,
        pos_process_noise: 0.004394,
        r_gps_pos: 0.1,
        r_gps_vel: 0.04394,
        r_baro: 50,
        r_mag: 1
      },
      {
        accel_noise_density: 0.01,
        gyro_noise_density: 0.01389,
        accel_bias_instability: 0.000001,
        gyro_bias_instability: 0.001,
        pos_process_noise: 0.007197,
        r_gps_pos: 0.1,
        r_gps_vel: 0.01,
        r_baro: 50,
        r_mag: 0.002683
      }
    ]
  },
  options: {
    no_sensors: false,
    no_filter: false
  }
};

export const ROCKET_PRESETS: RocketPreset[] = [
  {
    key: '30k-ft',
    label: '30k ft',
    note: 'Measured JSBSim apogee: 30.4k ft',
    values: {
      dry_mass: 25,
      fuel_mass: 5,
      oxidizer_mass: 10,
      thrust: 4000,
      burn_time: 6,
      drag_coeff: 0.32,
      normal_force_coeff: 12,
      ref_area: 0.02,
      isp: 230,
      nozzle_location: 3,
      cg_full: 1.8,
      cg_empty: 1.6,
      cp_location: 2.5,
      inertia_x: 0.015,
      inertia_y: 3,
      inertia_z: 3,
      wind_north: 0,
      wind_east: 0
    }
  },
  {
    key: '15k-ft',
    label: '15k ft',
    note: 'Measured JSBSim apogee: 15.9k ft',
    values: {
      dry_mass: 12,
      fuel_mass: 8 / 3,
      oxidizer_mass: 16 / 3,
      thrust: 2000,
      burn_time: 4.5,
      drag_coeff: 0.35,
      normal_force_coeff: 12,
      ref_area: 0.015,
      isp: 210,
      nozzle_location: 2,
      cg_full: 1.2,
      cg_empty: 1.1,
      cp_location: 1.6,
      inertia_x: 0.006,
      inertia_y: 1.2,
      inertia_z: 1.2,
      wind_north: 0,
      wind_east: 0
    }
  },
  {
    key: '12k-ft',
    label: '12k ft',
    note: 'Measured JSBSim apogee: 12.6k ft',
    values: {
      dry_mass: 6,
      fuel_mass: 2.5 / 3,
      oxidizer_mass: 5 / 3,
      thrust: 1200,
      burn_time: 3.5,
      drag_coeff: 0.38,
      normal_force_coeff: 12,
      ref_area: 0.01,
      isp: 210,
      nozzle_location: 1.7,
      cg_full: 1,
      cg_empty: 0.9,
      cp_location: 1.4,
      inertia_x: 0.004,
      inertia_y: 0.4,
      inertia_z: 0.4,
      wind_north: 0,
      wind_east: 0
    }
  },
  {
    key: '10k-ft',
    label: '10k ft',
    note: 'Measured JSBSim apogee: 10.1k ft',
    values: {
      dry_mass: 4,
      fuel_mass: 2 / 3,
      oxidizer_mass: 4 / 3,
      thrust: 900,
      burn_time: 3,
      drag_coeff: 0.4,
      normal_force_coeff: 12,
      ref_area: 0.008,
      isp: 190,
      nozzle_location: 1.4,
      cg_full: 0.8,
      cg_empty: 0.7,
      cp_location: 1.2,
      inertia_x: 0.0025,
      inertia_y: 0.25,
      inertia_z: 0.25,
      wind_north: 0,
      wind_east: 0
    }
  },
  {
    key: '5k-ft',
    label: '5k ft',
    note: 'Measured JSBSim apogee: 4.9k ft',
    values: {
      dry_mass: 0.7,
      fuel_mass: 0.08,
      oxidizer_mass: 0.16,
      thrust: 500,
      burn_time: 2.5,
      drag_coeff: 0.42,
      normal_force_coeff: 10,
      ref_area: 0.003,
      isp: 180,
      nozzle_location: 1.0,
      cg_full: 0.65,
      cg_empty: 0.58,
      cp_location: 0.92,
      inertia_x: 0.00018,
      inertia_y: 0.018,
      inertia_z: 0.018,
      wind_north: 0,
      wind_east: 0
    }
  },
  {
    key: '3k-ft',
    label: '3k ft',
    note: 'Measured JSBSim apogee: 2.9k ft',
    values: {
      dry_mass: 0.3,
      fuel_mass: 0.1 / 3,
      oxidizer_mass: 0.2 / 3,
      thrust: 320,
      burn_time: 2,
      drag_coeff: 0.45,
      normal_force_coeff: 10,
      ref_area: 0.002,
      isp: 170,
      nozzle_location: 0.8,
      cg_full: 0.5,
      cg_empty: 0.45,
      cp_location: 0.75,
      inertia_x: 0.00006,
      inertia_y: 0.006,
      inertia_z: 0.006,
      wind_north: 0,
      wind_east: 0
    }
  },
  {
    key: 'high-drag',
    label: 'High Drag',
    note: 'Measured JSBSim apogee: 380 ft',
    values: {
      dry_mass: 2,
      fuel_mass: 0.5 / 3,
      oxidizer_mass: 1.0 / 3,
      thrust: 200,
      burn_time: 2,
      drag_coeff: 2,
      normal_force_coeff: 12,
      ref_area: 0.02,
      isp: 180,
      nozzle_location: 1.3,
      cg_full: 0.8,
      cg_empty: 0.75,
      cp_location: 1.2,
      inertia_x: 0.001,
      inertia_y: 0.2,
      inertia_z: 0.2,
      wind_north: 0,
      wind_east: 0
    }
  }
];
