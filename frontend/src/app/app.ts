import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { afterNextRender, Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormBuilder, FormGroup, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { debounceTime } from 'rxjs/operators';
import type { Data, Layout } from 'plotly.js';
import Plotly from 'plotly.js-dist-min';
import { CheckboxModule } from 'primeng/checkbox';
import { InputNumberModule } from 'primeng/inputnumber';
import { MessageModule } from 'primeng/message';
import { TableModule } from 'primeng/table';

import {
  DEFAULT_REQUEST,
  type FilterAlgorithm,
  ROCKET_PRESETS,
  STAGE_LABELS,
  TUNING_FIELDS,
  type ErrorStatsGroup,
  type SimulationRequest,
  type SimulationResponse,
  type StageTuning,
  type StateChange,
  type StatRow,
  type Units
} from './app.models';

type UnitKind =
  | 'distance'
  | 'distance_km'
  | 'velocity'
  | 'acceleration'
  | 'force'
  | 'mass'
  | 'pressure'
  | 'angular_rate'
  | 'magnetic_field';
type FormUnitKey =
  | 'force'
  | 'distance'
  | 'area'
  | 'mass'
  | 'inertia'
  | 'acceleration'
  | 'velocity'
  | 'density'
  | 'pressure_mbar'
  | 'pressure_psf'
  | 'area_ft2_native'
  | 'stiffness'
  | 'damping'
  | 'time'
  | 'angle'
  | 'angular_rate_deg'
  | 'frequency'
  | 'per_rad'
  | 'distance_variance'
  | 'velocity_variance'
  | 'distance_process_noise'
  | 'acceleration_noise_density'
  | 'angular_rate_noise_density'
  | 'acceleration_bias'
  | 'angular_rate_bias'
  | 'angle_variance'
  | 'dimensionless';
type FieldMeta = { label: string; unit?: FormUnitKey };
type StatUnitKind = 'distance' | 'velocity' | 'time' | 'angle';
type StatMeta = { label: string; unit?: StatUnitKind };
type FormUnitSpec = {
  metricSuffix?: string;
  imperialSuffix?: string;
  toDisplay: (nativeValue: number, units: Units) => number;
  fromDisplay: (displayValue: number, units: Units) => number;
};
type FieldSpec = {
  key: string;
  label: string;
  step?: number;
  min?: number;
  minFractionDigits?: number;
  maxFractionDigits?: number;
};
type FieldGroup = { title: string; fields: FieldSpec[] };
type ProcessingState = 'idle' | 'processing' | 'updated';

const FEET_PER_METER = 3.28084;
const FT2_PER_M2 = 10.763910416709722;
const M2_PER_FT2 = 1 / FT2_PER_M2;
const LBF_PER_N = 0.224809;
const LB_PER_KG = 2.20462;
const SLUG_FT2_PER_KG_M2 = 23.73036040423184;
const SLUG_PER_FT3_PER_KG_M3 = 0.0019403203319807647;
const PSI_PER_MBAR = 0.0145037738;
const KPA_PER_PSF = 0.04788025898;
const N_PER_M_PER_LBF_PER_FT = 14.593902937206364;
const DEG_PER_RAD = 57.29577951308232;

const nativeMetricLinearUnit = (metricSuffix: string, imperialSuffix: string, metricToImperial: number): FormUnitSpec => ({
  metricSuffix,
  imperialSuffix,
  toDisplay: (nativeValue, units) => (units === 'imperial' ? nativeValue * metricToImperial : nativeValue),
  fromDisplay: (displayValue, units) => (units === 'imperial' ? displayValue / metricToImperial : displayValue)
});

const nativeImperialLinearUnit = (metricSuffix: string, imperialSuffix: string, imperialToMetric: number): FormUnitSpec => ({
  metricSuffix,
  imperialSuffix,
  toDisplay: (nativeValue, units) => (units === 'imperial' ? nativeValue : nativeValue * imperialToMetric),
  fromDisplay: (displayValue, units) => (units === 'imperial' ? displayValue : displayValue / imperialToMetric)
});

const identityUnit = (metricSuffix?: string, imperialSuffix?: string): FormUnitSpec => ({
  metricSuffix,
  imperialSuffix: imperialSuffix ?? metricSuffix,
  toDisplay: (nativeValue) => nativeValue,
  fromDisplay: (displayValue) => displayValue
});

const FORM_UNIT_SPECS: Record<FormUnitKey, FormUnitSpec> = {
  force: nativeMetricLinearUnit('N', 'lbf', LBF_PER_N),
  distance: nativeMetricLinearUnit('m', 'ft', FEET_PER_METER),
  area: nativeMetricLinearUnit('m²', 'ft²', FT2_PER_M2),
  mass: nativeMetricLinearUnit('kg', 'lb', LB_PER_KG),
  inertia: nativeMetricLinearUnit('kg·m²', 'slug·ft²', SLUG_FT2_PER_KG_M2),
  acceleration: nativeMetricLinearUnit('m/s²', 'ft/s²', FEET_PER_METER),
  velocity: nativeMetricLinearUnit('m/s', 'ft/s', FEET_PER_METER),
  density: nativeMetricLinearUnit('kg/m³', 'slug/ft³', SLUG_PER_FT3_PER_KG_M3),
  pressure_mbar: nativeMetricLinearUnit('mbar', 'psi', PSI_PER_MBAR),
  pressure_psf: nativeImperialLinearUnit('kPa', 'psf', KPA_PER_PSF),
  area_ft2_native: nativeImperialLinearUnit('m²', 'ft²', M2_PER_FT2),
  stiffness: nativeImperialLinearUnit('N/m', 'lbf/ft', N_PER_M_PER_LBF_PER_FT),
  damping: nativeImperialLinearUnit('N·s/m', 'lbf·s/ft', N_PER_M_PER_LBF_PER_FT),
  time: identityUnit('s'),
  angle: identityUnit('deg'),
  angular_rate_deg: identityUnit('deg/s'),
  frequency: identityUnit('Hz'),
  per_rad: identityUnit('/rad'),
  distance_variance: nativeMetricLinearUnit('m²', 'ft²', FT2_PER_M2),
  velocity_variance: nativeMetricLinearUnit('(m/s)²', '(ft/s)²', FT2_PER_M2),
  distance_process_noise: nativeMetricLinearUnit('m/√Hz', 'ft/√Hz', FEET_PER_METER),
  acceleration_noise_density: nativeMetricLinearUnit('m/s²/√Hz', 'ft/s²/√Hz', FEET_PER_METER),
  angular_rate_noise_density: nativeMetricLinearUnit('rad/s/√Hz', 'deg/s/√Hz', DEG_PER_RAD),
  acceleration_bias: nativeMetricLinearUnit('m/s²/√Hz', 'ft/s²/√Hz', FEET_PER_METER),
  angular_rate_bias: nativeMetricLinearUnit('rad/s/√Hz', 'deg/s/√Hz', DEG_PER_RAD),
  angle_variance: identityUnit('rad²'),
  dimensionless: identityUnit()
};

const FIELD_META: Record<string, FieldMeta> = {
  thrust: { label: 'Engine Thrust', unit: 'force' },
  burn_time: { label: 'Engine Burn Time', unit: 'time' },
  drag_coeff: { label: 'Drag Coefficient' },
  normal_force_coeff: { label: 'Normal Force Coeff', unit: 'per_rad' },
  isp: { label: 'Specific Impulse', unit: 'time' },
  nozzle_location: { label: 'Nozzle Location', unit: 'distance' },
  ref_area: { label: 'Reference Area', unit: 'area' },
  dry_mass: { label: 'Dry Mass', unit: 'mass' },
  fuel_mass: { label: 'Fuel Mass', unit: 'mass' },
  oxidizer_mass: { label: 'Oxidizer Mass', unit: 'mass' },
  cg_full: { label: 'CG Full', unit: 'distance' },
  cg_empty: { label: 'CG Empty', unit: 'distance' },
  cp_location: { label: 'CP Location', unit: 'distance' },
  launch_delay: { label: 'Ignition Delay', unit: 'time' },
  launch_rod_length: { label: 'Launch Rod Length', unit: 'distance' },
  spin_rate: { label: 'Spin Rate', unit: 'angular_rate_deg' },
  thrust_cant: { label: 'Thrust Cant', unit: 'angle' },
  inertia_x: { label: 'Roll Inertia Ixx', unit: 'inertia' },
  inertia_y: { label: 'Pitch Inertia Iyy', unit: 'inertia' },
  inertia_z: { label: 'Yaw Inertia Izz', unit: 'inertia' },
  nozzle_exit_pressure_psf: { label: 'Nozzle Exit Pressure', unit: 'pressure_psf' },
  nozzle_area_ft2: { label: 'Nozzle Area', unit: 'area_ft2_native' },
  pad_static_friction: { label: 'Pad Static Friction' },
  pad_dynamic_friction: { label: 'Pad Dynamic Friction' },
  pad_spring_coeff_lbs_ft: { label: 'Pad Spring', unit: 'stiffness' },
  pad_damping_coeff_lbs_ft_s: { label: 'Pad Damping', unit: 'damping' },
  gravity: { label: 'Gravity', unit: 'acceleration' },
  wind_north: { label: 'Wind North', unit: 'velocity' },
  wind_east: { label: 'Wind East', unit: 'velocity' },
  wind_down: { label: 'Wind Down', unit: 'velocity' },
  air_density: { label: 'Air Density', unit: 'density' },
  sim_dt: { label: 'Simulation Step', unit: 'time' },
  max_time: { label: 'Max Simulation Time', unit: 'time' },
  noise_scale: { label: 'Noise Scale' },
  seed: { label: 'RNG Seed' },
  bmi088_accel_rate_hz: { label: 'BMI088 Accel', unit: 'frequency' },
  bmi088_gyro_rate_hz: { label: 'BMI088 Gyro', unit: 'frequency' },
  adxl375_rate_hz: { label: 'ADXL375', unit: 'frequency' },
  lis3mdl_rate_hz: { label: 'LIS3MDL', unit: 'frequency' },
  ms5611_rate_hz: { label: 'MS5611', unit: 'frequency' },
  gps_rate_hz: { label: 'GPS', unit: 'frequency' },
  ground_pressure_mbar: { label: 'Ground Pressure', unit: 'pressure_mbar' },
  mag_declination_deg: { label: 'Mag Declination', unit: 'angle' },
  mag_dip_deg: { label: 'Mag Dip', unit: 'angle' },
  home_lat_deg: { label: 'Home Latitude', unit: 'angle' },
  home_lon_deg: { label: 'Home Longitude', unit: 'angle' },
  home_alt_m: { label: 'Home Altitude', unit: 'distance' },
  launch_accel_thresh: { label: 'Launch Accel Threshold', unit: 'acceleration' },
  launch_vel_thresh: { label: 'Launch Velocity Threshold', unit: 'velocity' },
  burnout_accel_thresh: { label: 'Burnout Accel Threshold', unit: 'acceleration' },
  min_ascent_time: { label: 'Minimum Ascent Time', unit: 'time' },
  apogee_descent_thresh: { label: 'Apogee Descent Threshold', unit: 'velocity' },
  min_coast_time: { label: 'Minimum Coast Time', unit: 'time' },
  landing_vel_thresh: { label: 'Landing Velocity Threshold', unit: 'velocity' },
  landing_alt_thresh: { label: 'Landing Altitude Threshold', unit: 'distance' },
  landing_confirm_window: { label: 'Landing Confirm Window', unit: 'time' },
  high_velocity_baro_thresh: { label: 'High-Velocity Baro Threshold', unit: 'velocity' }
};

const TUNING_FIELD_META: Record<keyof StageTuning, FieldMeta> = {
  accel_noise_density: { label: 'Accel Noise Density', unit: 'acceleration_noise_density' },
  gyro_noise_density: { label: 'Gyro Noise Density', unit: 'angular_rate_noise_density' },
  accel_bias_instability: { label: 'Accel Bias Instability', unit: 'acceleration_bias' },
  gyro_bias_instability: { label: 'Gyro Bias Instability', unit: 'angular_rate_bias' },
  pos_process_noise: { label: 'Position Process Noise', unit: 'distance_process_noise' },
  r_gps_pos: { label: 'GPS Position Variance', unit: 'distance_variance' },
  r_gps_vel: { label: 'GPS Velocity Variance', unit: 'velocity_variance' },
  r_baro: { label: 'Barometer Variance', unit: 'distance_variance' },
  r_mag: { label: 'Magnetometer Direction Variance', unit: 'angle_variance' }
};

const STAT_META: Record<string, StatMeta> = {
  pos_n: { label: 'North Position Error', unit: 'distance' },
  pos_e: { label: 'East Position Error', unit: 'distance' },
  pos_d: { label: 'Down Position Error', unit: 'distance' },
  pos_3d: { label: '3D Position Error', unit: 'distance' },
  vel_n: { label: 'North Velocity Error', unit: 'velocity' },
  vel_e: { label: 'East Velocity Error', unit: 'velocity' },
  vel_d: { label: 'Down Velocity Error', unit: 'velocity' },
  alt: { label: 'Altitude Error', unit: 'distance' },
  horiz: { label: 'Horizontal Recovery Error', unit: 'distance' },
  lat: { label: 'Latitude Error', unit: 'angle' },
  lon: { label: 'Longitude Error', unit: 'angle' },
  burn: { label: 'Burn Detection Delay', unit: 'time' },
  coast: { label: 'Coast Detection Delay', unit: 'time' },
  rec: { label: 'Descent Detection Delay', unit: 'time' }
};

const unitSystems = {
  metric: {
    distance: { factor: 1, suffix: 'm' },
    distance_km: { factor: 0.001, suffix: 'km' },
    velocity: { factor: 1, suffix: 'm/s' },
    acceleration: { factor: 1, suffix: 'm/s²' },
    force: { factor: 1, suffix: 'N' },
    mass: { factor: 1, suffix: 'kg' },
    pressure: { factor: 1, suffix: 'Pa' },
    angular_rate: { factor: 1, suffix: 'rad/s' },
    magnetic_field: { factor: 1, suffix: 'Gauss' }
  },
  imperial: {
    distance: { factor: 3.28084, suffix: 'ft' },
    distance_km: { factor: 3280.84, suffix: 'ft' },
    velocity: { factor: 3.28084, suffix: 'ft/s' },
    acceleration: { factor: 3.28084, suffix: 'ft/s²' },
    force: { factor: 0.224809, suffix: 'lbf' },
    mass: { factor: 2.20462, suffix: 'lb' },
    pressure: { factor: 0.000145038, suffix: 'psi' },
    angular_rate: { factor: 57.2958, suffix: 'deg/s' },
    magnetic_field: { factor: 1, suffix: 'Gauss' }
  }
} as const;

const stateColors: Record<string, string> = {
  Pad: '#3ab27b',
  Ascent: '#f59e0b',
  Burn: '#f97316',
  Coast: '#38bdf8',
  Descent: '#c084fc',
  Recovery: '#c084fc',
  Landed: '#a78bfa'
};

const FILTER_ALGORITHM_LABELS: Record<FilterAlgorithm, string> = {
  eskf: 'ESKF',
  kalman: 'Kalman',
  information: 'Information'
};
const KNOWN_FILTER_ALGORITHMS: readonly FilterAlgorithm[] = ['eskf', 'kalman', 'information'];

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    CheckboxModule,
    InputNumberModule,
    MessageModule,
    TableModule
  ],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  readonly fb = inject(FormBuilder);
  readonly http = inject(HttpClient);
  readonly destroyRef = inject(DestroyRef);

  readonly presets = ROCKET_PRESETS;
  readonly units = signal<Units>((localStorage.getItem('aloe-units') as Units) || 'imperial');
  readonly response = signal<SimulationResponse | null>(null);
  readonly errorMessage = signal<string | null>(null);
  readonly isRunning = signal(false);
  readonly processingState = signal<ProcessingState>('idle');
  readonly statRows = computed(() => this.buildStatRows());
  readonly showProcessingIndicator = computed(() => this.processingState() !== 'idle');
  readonly hasAdxlCharts = computed(() => {
    const data = this.response();
    if (!data) {
      return false;
    }

    return (
      this.hasCompatibleScalarSeries(data.time, data.sensor_data.adxl_x) &&
      this.hasCompatibleScalarSeries(data.time, data.sensor_data.adxl_y) &&
      this.hasCompatibleScalarSeries(data.time, data.sensor_data.adxl_z)
    );
  });
  readonly processingLabel = computed(() => {
    const state = this.processingState();
    if (state === 'processing') {
      return 'Processing';
    }
    if (state === 'updated') {
      return 'Updated';
    }
    return 'Idle';
  });

  private processingDelayHandle: ReturnType<typeof setTimeout> | null = null;
  private processingDoneHandle: ReturnType<typeof setTimeout> | null = null;
  private readonly chartOperations = new Map<string, Promise<void>>();
  private readonly initializedCharts = new Set<string>();
  private pendingSimulationRequest = false;

  readonly unitOptions: Array<{ label: string; value: Units }> = [
    { label: 'Metric', value: 'metric' },
    { label: 'Imperial', value: 'imperial' }
  ];
  readonly numberInputStyle: Record<string, string> = {
    color: '#0f172a',
    '-webkit-text-fill-color': '#0f172a',
    background: '#ffffff',
    'font-weight': '600'
  };

  readonly rocketGroups: FieldGroup[] = [
    {
      title: 'Propulsion and Aero',
      fields: [
        { key: 'thrust', label: this.labelForField('thrust') },
        { key: 'burn_time', label: this.labelForField('burn_time') },
        { key: 'drag_coeff', label: this.labelForField('drag_coeff') },
        { key: 'normal_force_coeff', label: this.labelForField('normal_force_coeff') },
        { key: 'isp', label: this.labelForField('isp') },
        { key: 'nozzle_location', label: this.labelForField('nozzle_location') },
        { key: 'ref_area', label: this.labelForField('ref_area') }
      ]
    },
    {
      title: 'Mass and Stability',
      fields: [
        { key: 'dry_mass', label: this.labelForField('dry_mass') },
        { key: 'fuel_mass', label: this.labelForField('fuel_mass') },
        { key: 'oxidizer_mass', label: this.labelForField('oxidizer_mass') },
        { key: 'cg_full', label: this.labelForField('cg_full') },
        { key: 'cg_empty', label: this.labelForField('cg_empty') },
        { key: 'cp_location', label: this.labelForField('cp_location') }
      ]
    },
    {
      title: 'Launch and Inertia',
      fields: [
        { key: 'launch_delay', label: this.labelForField('launch_delay') },
        { key: 'launch_rod_length', label: this.labelForField('launch_rod_length') },
        { key: 'spin_rate', label: this.labelForField('spin_rate') },
        { key: 'thrust_cant', label: this.labelForField('thrust_cant') },
        { key: 'inertia_x', label: this.labelForField('inertia_x') },
        { key: 'inertia_y', label: this.labelForField('inertia_y') },
        { key: 'inertia_z', label: this.labelForField('inertia_z') }
      ]
    },
    {
      title: 'Low-Level JSBSim',
      fields: [
        { key: 'nozzle_exit_pressure_psf', label: this.labelForField('nozzle_exit_pressure_psf') },
        { key: 'nozzle_area_ft2', label: this.labelForField('nozzle_area_ft2') },
        { key: 'pad_static_friction', label: this.labelForField('pad_static_friction') },
        { key: 'pad_dynamic_friction', label: this.labelForField('pad_dynamic_friction') },
        { key: 'pad_spring_coeff_lbs_ft', label: this.labelForField('pad_spring_coeff_lbs_ft') },
        { key: 'pad_damping_coeff_lbs_ft_s', label: this.labelForField('pad_damping_coeff_lbs_ft_s') }
      ]
    }
  ];

  readonly environmentFields: FieldSpec[] = [
    { key: 'gravity', label: this.labelForField('gravity') },
    { key: 'wind_north', label: this.labelForField('wind_north') },
    { key: 'wind_east', label: this.labelForField('wind_east') },
    { key: 'wind_down', label: this.labelForField('wind_down') },
    { key: 'air_density', label: this.labelForField('air_density') },
    { key: 'sim_dt', label: this.labelForField('sim_dt'), step: 0.0001, min: 0.0001, minFractionDigits: 3, maxFractionDigits: 6 },
    { key: 'max_time', label: this.labelForField('max_time') }
  ];

  readonly sensorValueGroups: FieldGroup[] = [
    {
      title: 'Noise Model',
      fields: [
        { key: 'noise_scale', label: this.labelForField('noise_scale') },
        { key: 'seed', label: this.labelForField('seed') }
      ]
    },
    {
      title: 'Sample Rates',
      fields: [
        { key: 'bmi088_accel_rate_hz', label: this.labelForField('bmi088_accel_rate_hz') },
        { key: 'bmi088_gyro_rate_hz', label: this.labelForField('bmi088_gyro_rate_hz') },
        { key: 'adxl375_rate_hz', label: this.labelForField('adxl375_rate_hz') },
        { key: 'lis3mdl_rate_hz', label: this.labelForField('lis3mdl_rate_hz') },
        { key: 'ms5611_rate_hz', label: this.labelForField('ms5611_rate_hz') },
        { key: 'gps_rate_hz', label: this.labelForField('gps_rate_hz') }
      ]
    }
  ];

  readonly sensorToggleFields: FieldSpec[] = [
    { key: 'bmi088_accel_enabled', label: 'BMI088 Accelerometer' },
    { key: 'bmi088_gyro_enabled', label: 'BMI088 Gyroscope' },
    { key: 'adxl375_enabled', label: 'ADXL375 Accelerometer' },
    { key: 'lis3mdl_enabled', label: 'LIS3MDL Magnetometer' },
    { key: 'ms5611_enabled', label: 'MS5611 Barometer' },
    { key: 'gps_enabled', label: 'GPS' }
  ];

  readonly optionToggleFields: FieldSpec[] = [
    { key: 'no_sensors', label: 'Disable sensor generation' },
    { key: 'no_filter', label: 'Disable filter' }
  ];
  readonly algorithmToggleFields: Array<{ key: FilterAlgorithm; label: string }> = [
    { key: 'eskf', label: FILTER_ALGORITHM_LABELS.eskf },
    { key: 'kalman', label: FILTER_ALGORITHM_LABELS.kalman },
    { key: 'information', label: FILTER_ALGORITHM_LABELS.information }
  ];
  readonly algorithmOptions: Array<{ value: FilterAlgorithm; label: string }> = [
    { value: 'eskf', label: FILTER_ALGORITHM_LABELS.eskf },
    { value: 'kalman', label: FILTER_ALGORITHM_LABELS.kalman },
    { value: 'information', label: FILTER_ALGORITHM_LABELS.information }
  ];

  readonly filterFields: FieldSpec[] = [
    { key: 'ground_pressure_mbar', label: this.labelForField('ground_pressure_mbar') },
    { key: 'mag_declination_deg', label: this.labelForField('mag_declination_deg') },
    { key: 'mag_dip_deg', label: this.labelForField('mag_dip_deg') },
    { key: 'home_lat_deg', label: this.labelForField('home_lat_deg') },
    { key: 'home_lon_deg', label: this.labelForField('home_lon_deg') },
    { key: 'home_alt_m', label: this.labelForField('home_alt_m') }
  ];

  readonly stateMachineFields: FieldSpec[] = [
    { key: 'launch_accel_thresh', label: this.labelForField('launch_accel_thresh') },
    { key: 'launch_vel_thresh', label: this.labelForField('launch_vel_thresh') },
    { key: 'burnout_accel_thresh', label: this.labelForField('burnout_accel_thresh') },
    { key: 'min_ascent_time', label: this.labelForField('min_ascent_time') },
    { key: 'apogee_descent_thresh', label: this.labelForField('apogee_descent_thresh') },
    { key: 'min_coast_time', label: this.labelForField('min_coast_time') },
    { key: 'landing_vel_thresh', label: this.labelForField('landing_vel_thresh') },
    { key: 'landing_alt_thresh', label: this.labelForField('landing_alt_thresh') },
    { key: 'landing_confirm_window', label: this.labelForField('landing_confirm_window') },
    { key: 'high_velocity_baro_thresh', label: this.labelForField('high_velocity_baro_thresh') }
  ];

  readonly form = this.fb.nonNullable.group({
    rocket: this.fb.nonNullable.group({
      dry_mass: DEFAULT_REQUEST.rocket.dry_mass,
      fuel_mass: DEFAULT_REQUEST.rocket.fuel_mass,
      oxidizer_mass: DEFAULT_REQUEST.rocket.oxidizer_mass,
      cg_full: DEFAULT_REQUEST.rocket.cg_full,
      cg_empty: DEFAULT_REQUEST.rocket.cg_empty,
      cp_location: DEFAULT_REQUEST.rocket.cp_location,
      inertia_x: DEFAULT_REQUEST.rocket.inertia_x,
      inertia_y: DEFAULT_REQUEST.rocket.inertia_y,
      inertia_z: DEFAULT_REQUEST.rocket.inertia_z,
      thrust: DEFAULT_REQUEST.rocket.thrust,
      burn_time: DEFAULT_REQUEST.rocket.burn_time,
      drag_coeff: DEFAULT_REQUEST.rocket.drag_coeff,
      normal_force_coeff: DEFAULT_REQUEST.rocket.normal_force_coeff,
      ref_area: DEFAULT_REQUEST.rocket.ref_area,
      isp: DEFAULT_REQUEST.rocket.isp,
      nozzle_location: DEFAULT_REQUEST.rocket.nozzle_location,
      launch_delay: DEFAULT_REQUEST.rocket.launch_delay,
      launch_rod_length: DEFAULT_REQUEST.rocket.launch_rod_length,
      spin_rate: DEFAULT_REQUEST.rocket.spin_rate,
      thrust_cant: DEFAULT_REQUEST.rocket.thrust_cant,
      nozzle_exit_pressure_psf: DEFAULT_REQUEST.rocket.nozzle_exit_pressure_psf,
      nozzle_area_ft2: DEFAULT_REQUEST.rocket.nozzle_area_ft2,
      pad_static_friction: DEFAULT_REQUEST.rocket.pad_static_friction,
      pad_dynamic_friction: DEFAULT_REQUEST.rocket.pad_dynamic_friction,
      pad_spring_coeff_lbs_ft: DEFAULT_REQUEST.rocket.pad_spring_coeff_lbs_ft,
      pad_damping_coeff_lbs_ft_s: DEFAULT_REQUEST.rocket.pad_damping_coeff_lbs_ft_s
    }),
    environment: this.fb.nonNullable.group({
      gravity: DEFAULT_REQUEST.environment.gravity,
      wind_north: DEFAULT_REQUEST.environment.wind_north,
      wind_east: DEFAULT_REQUEST.environment.wind_east,
      wind_down: DEFAULT_REQUEST.environment.wind_down,
      air_density: DEFAULT_REQUEST.environment.air_density,
      sim_dt: DEFAULT_REQUEST.environment.sim_dt,
      max_time: DEFAULT_REQUEST.environment.max_time
    }),
    sensors: this.fb.nonNullable.group({
      noise_scale: DEFAULT_REQUEST.sensors.noise_scale,
      seed: DEFAULT_REQUEST.sensors.seed,
      bmi088_accel_enabled: DEFAULT_REQUEST.sensors.bmi088_accel_enabled,
      bmi088_gyro_enabled: DEFAULT_REQUEST.sensors.bmi088_gyro_enabled,
      adxl375_enabled: DEFAULT_REQUEST.sensors.adxl375_enabled,
      lis3mdl_enabled: DEFAULT_REQUEST.sensors.lis3mdl_enabled,
      ms5611_enabled: DEFAULT_REQUEST.sensors.ms5611_enabled,
      gps_enabled: DEFAULT_REQUEST.sensors.gps_enabled,
      bmi088_accel_rate_hz: DEFAULT_REQUEST.sensors.bmi088_accel_rate_hz,
      bmi088_gyro_rate_hz: DEFAULT_REQUEST.sensors.bmi088_gyro_rate_hz,
      adxl375_rate_hz: DEFAULT_REQUEST.sensors.adxl375_rate_hz,
      lis3mdl_rate_hz: DEFAULT_REQUEST.sensors.lis3mdl_rate_hz,
      ms5611_rate_hz: DEFAULT_REQUEST.sensors.ms5611_rate_hz,
      gps_rate_hz: DEFAULT_REQUEST.sensors.gps_rate_hz
    }),
    filter: this.fb.nonNullable.group({
      ground_pressure_mbar: DEFAULT_REQUEST.filter.ground_pressure_mbar,
      mag_declination_deg: DEFAULT_REQUEST.filter.mag_declination_deg,
      mag_dip_deg: DEFAULT_REQUEST.filter.mag_dip_deg,
      home_lat_deg: DEFAULT_REQUEST.filter.home_lat_deg,
      home_lon_deg: DEFAULT_REQUEST.filter.home_lon_deg,
      home_alt_m: DEFAULT_REQUEST.filter.home_alt_m,
      launch_accel_thresh: DEFAULT_REQUEST.filter.launch_accel_thresh,
      launch_vel_thresh: DEFAULT_REQUEST.filter.launch_vel_thresh,
      burnout_accel_thresh: DEFAULT_REQUEST.filter.burnout_accel_thresh,
      min_ascent_time: DEFAULT_REQUEST.filter.min_ascent_time,
      apogee_descent_thresh: DEFAULT_REQUEST.filter.apogee_descent_thresh,
      min_coast_time: DEFAULT_REQUEST.filter.min_coast_time,
      landing_vel_thresh: DEFAULT_REQUEST.filter.landing_vel_thresh,
      landing_alt_thresh: DEFAULT_REQUEST.filter.landing_alt_thresh,
      landing_confirm_window: DEFAULT_REQUEST.filter.landing_confirm_window,
      high_velocity_baro_thresh: DEFAULT_REQUEST.filter.high_velocity_baro_thresh,
      selected_algorithms: this.fb.nonNullable.control<FilterAlgorithm[]>([
        ...DEFAULT_REQUEST.filter.selected_algorithms
      ]),
      active_algorithm: this.fb.nonNullable.control<FilterAlgorithm>(DEFAULT_REQUEST.filter.active_algorithm),
      stage_tuning: this.fb.array(DEFAULT_REQUEST.filter.stage_tuning.map((stage) => this.createStageTuningGroup(stage)))
    }),
    options: this.fb.nonNullable.group({
      no_sensors: DEFAULT_REQUEST.options.no_sensors,
      no_filter: DEFAULT_REQUEST.options.no_filter
    })
  });

  constructor() {
    if (this.units() === 'imperial') {
      this.rebaseFormUnits('metric', 'imperial');
    }

    this.form.controls.rocket.valueChanges.pipe(debounceTime(120), takeUntilDestroyed(this.destroyRef)).subscribe(() => {
      this.requestRun('full');
    });

    this.form.controls.environment.valueChanges
      .pipe(debounceTime(120), takeUntilDestroyed(this.destroyRef))
      .subscribe(() => {
        this.requestRun('full');
      });

    this.form.controls.sensors.valueChanges.pipe(debounceTime(120), takeUntilDestroyed(this.destroyRef)).subscribe(() => {
      this.requestRun('sensor-filter');
    });

    this.form.controls.filter.valueChanges.pipe(debounceTime(120), takeUntilDestroyed(this.destroyRef)).subscribe(() => {
      this.requestRun('sensor-filter');
    });

    this.form.controls.options.valueChanges.pipe(debounceTime(120), takeUntilDestroyed(this.destroyRef)).subscribe(() => {
      this.requestRun('sensor-filter');
    });

    this.destroyRef.onDestroy(() => {
      this.initializedCharts.forEach((chartId) => {
        const element = document.getElementById(chartId);
        if (element) {
          Plotly.purge(element);
        }
      });
      this.initializedCharts.clear();
      this.chartOperations.clear();
    });

    afterNextRender(() => {
      void this.runSimulation();
    });
  }

  setUnits(units: Units): void {
    const previousUnits = this.units();
    if (previousUnits === units) {
      return;
    }

    this.rebaseFormUnits(previousUnits, units);
    this.units.set(units);
    localStorage.setItem('aloe-units', units);
    queueMicrotask(() => this.renderCharts());
  }

  setUnitsFromChange(value: string | number | undefined): void {
    if (value === 'metric' || value === 'imperial') {
      this.setUnits(value);
    }
  }

  applyPreset(key: string): void {
    const preset = this.presets.find((entry) => entry.key === key);
    if (!preset) {
      return;
    }

    const units = this.units();
    const displayValue = (value: number | undefined, fieldKey: string, fallback: number): number => {
      if (value === undefined) {
        return fallback;
      }
      return FORM_UNIT_SPECS[FIELD_META[fieldKey]?.unit ?? 'dimensionless'].toDisplay(value, units);
    };

    this.form.patchValue({
      rocket: {
        dry_mass: displayValue(preset.values.dry_mass, 'dry_mass', this.form.controls.rocket.controls.dry_mass.value),
        fuel_mass: displayValue(preset.values.fuel_mass, 'fuel_mass', this.form.controls.rocket.controls.fuel_mass.value),
        oxidizer_mass: displayValue(preset.values.oxidizer_mass, 'oxidizer_mass', this.form.controls.rocket.controls.oxidizer_mass.value),
        cg_full: displayValue(preset.values.cg_full, 'cg_full', this.form.controls.rocket.controls.cg_full.value),
        cg_empty: displayValue(preset.values.cg_empty, 'cg_empty', this.form.controls.rocket.controls.cg_empty.value),
        cp_location: displayValue(preset.values.cp_location, 'cp_location', this.form.controls.rocket.controls.cp_location.value),
        inertia_x: displayValue(preset.values.inertia_x, 'inertia_x', this.form.controls.rocket.controls.inertia_x.value),
        inertia_y: displayValue(preset.values.inertia_y, 'inertia_y', this.form.controls.rocket.controls.inertia_y.value),
        inertia_z: displayValue(preset.values.inertia_z, 'inertia_z', this.form.controls.rocket.controls.inertia_z.value),
        thrust: displayValue(preset.values.thrust, 'thrust', this.form.controls.rocket.controls.thrust.value),
        burn_time: displayValue(preset.values.burn_time, 'burn_time', this.form.controls.rocket.controls.burn_time.value),
        drag_coeff: displayValue(preset.values.drag_coeff, 'drag_coeff', this.form.controls.rocket.controls.drag_coeff.value),
        normal_force_coeff: displayValue(preset.values.normal_force_coeff, 'normal_force_coeff', this.form.controls.rocket.controls.normal_force_coeff.value),
        ref_area: displayValue(preset.values.ref_area, 'ref_area', this.form.controls.rocket.controls.ref_area.value),
        isp: displayValue(preset.values.isp, 'isp', this.form.controls.rocket.controls.isp.value),
        nozzle_location: displayValue(preset.values.nozzle_location, 'nozzle_location', this.form.controls.rocket.controls.nozzle_location.value),
        nozzle_exit_pressure_psf: displayValue(preset.values.nozzle_exit_pressure_psf, 'nozzle_exit_pressure_psf', this.form.controls.rocket.controls.nozzle_exit_pressure_psf.value),
        nozzle_area_ft2: displayValue(preset.values.nozzle_area_ft2, 'nozzle_area_ft2', this.form.controls.rocket.controls.nozzle_area_ft2.value),
        pad_static_friction: displayValue(preset.values.pad_static_friction, 'pad_static_friction', this.form.controls.rocket.controls.pad_static_friction.value),
        pad_dynamic_friction: displayValue(preset.values.pad_dynamic_friction, 'pad_dynamic_friction', this.form.controls.rocket.controls.pad_dynamic_friction.value),
        pad_spring_coeff_lbs_ft: displayValue(preset.values.pad_spring_coeff_lbs_ft, 'pad_spring_coeff_lbs_ft', this.form.controls.rocket.controls.pad_spring_coeff_lbs_ft.value),
        pad_damping_coeff_lbs_ft_s: displayValue(preset.values.pad_damping_coeff_lbs_ft_s, 'pad_damping_coeff_lbs_ft_s', this.form.controls.rocket.controls.pad_damping_coeff_lbs_ft_s.value)
      },
      environment: {
        wind_north: displayValue(preset.values.wind_north, 'wind_north', this.form.controls.environment.controls.wind_north.value),
        wind_east: displayValue(preset.values.wind_east, 'wind_east', this.form.controls.environment.controls.wind_east.value),
        wind_down: displayValue(preset.values.wind_down, 'wind_down', this.form.controls.environment.controls.wind_down.value),
        gravity: this.form.controls.environment.controls.gravity.value,
        air_density: this.form.controls.environment.controls.air_density.value,
        sim_dt: displayValue(preset.values.sim_dt, 'sim_dt', this.form.controls.environment.controls.sim_dt.value),
        max_time: displayValue(preset.values.max_time, 'max_time', this.form.controls.environment.controls.max_time.value)
      }
    }, { emitEvent: false });

    this.requestRun('full');
  }

  async runSimulation(): Promise<void> {
    if (this.isRunning()) {
      this.pendingSimulationRequest = true;
      return;
    }

    this.isRunning.set(true);
    this.processingState.set('processing');
    this.errorMessage.set(null);

    try {
      const payload = this.buildSimulationRequest();
      const result = await this.http.post<SimulationResponse>('/api/simulate', payload).toPromise();
      if (!result) {
        throw new Error('No simulation response received');
      }

      this.response.set(result);
      this.errorMessage.set(result.error_message);
      queueMicrotask(() => this.renderCharts());
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Simulation failed';
      this.errorMessage.set(message);
    } finally {
      this.isRunning.set(false);
      this.clearProcessingIndicator();

      if (this.pendingSimulationRequest) {
        this.pendingSimulationRequest = false;
        queueMicrotask(() => {
          this.queueProcessingIndicator();
          void this.runSimulation();
        });
      }
    }
  }

  formatStat(value: number | null | undefined): string {
    if (value === null || value === undefined || Number.isNaN(value) || !Number.isFinite(value)) {
      return '-';
    }
    if (Math.abs(value) < 0.001 && value !== 0) {
      return value.toExponential(3);
    }
    return value.toFixed(4);
  }

  formatStatValue(row: StatRow, value: number | null | undefined): string {
    if (value === null || value === undefined || Number.isNaN(value) || !Number.isFinite(value)) {
      return '-';
    }

    const converted = this.convertStatValue(value, row.unit);
    if (Math.abs(converted) < 0.001 && converted !== 0) {
      return converted.toExponential(3);
    }

    return converted.toFixed(4);
  }

  setStageTuningValue(stageIndex: number, field: keyof StageTuning, value: number | null | undefined): void {
    if (value === null || value === undefined) {
      return;
    }

    this.form.controls.filter.controls.stage_tuning.at(stageIndex).get(field)?.setValue(value);
  }

  private createStageTuningGroup(stage: StageTuning): FormGroup {
    return this.fb.nonNullable.group({
      accel_noise_density: stage.accel_noise_density,
      gyro_noise_density: stage.gyro_noise_density,
      accel_bias_instability: stage.accel_bias_instability,
      gyro_bias_instability: stage.gyro_bias_instability,
      pos_process_noise: stage.pos_process_noise,
      r_gps_pos: stage.r_gps_pos,
      r_gps_vel: stage.r_gps_vel,
      r_baro: stage.r_baro,
      r_mag: stage.r_mag
    });
  }

  private convert(values: number[], kind: UnitKind): number[] {
    const factor = unitSystems[this.units()][kind].factor;
    return values.map((value) => value * factor);
  }

  protected labelForField(key: string): string {
    const meta = FIELD_META[key];
    if (!meta) {
      return key;
    }

    return this.withUnitSuffix(meta.label, meta.unit);
  }

  protected labelForTuningField(key: keyof StageTuning): string {
    const meta = TUNING_FIELD_META[key];
    return this.withUnitSuffix(meta.label, meta.unit);
  }

  private withUnitSuffix(label: string, unitKey?: FormUnitKey): string {
    if (!unitKey) {
      return label;
    }

    const suffix = this.formUnitSuffix(unitKey);
    return suffix ? `${label} (${suffix})` : label;
  }

  private formUnitSuffix(unitKey: FormUnitKey): string {
    const spec = FORM_UNIT_SPECS[unitKey];
    return this.units() === 'imperial' ? spec.imperialSuffix ?? '' : spec.metricSuffix ?? '';
  }

  private convertFormValue(value: number, unitKey: FormUnitKey | undefined, fromUnits: Units, toUnits: Units): number {
    if (!unitKey || fromUnits === toUnits || !Number.isFinite(value)) {
      return value;
    }

    const spec = FORM_UNIT_SPECS[unitKey];
    return spec.toDisplay(spec.fromDisplay(value, fromUnits), toUnits);
  }

  private nativeValueFromDisplay(value: number, unitKey?: FormUnitKey): number {
    if (!unitKey || !Number.isFinite(value)) {
      return value;
    }

    return FORM_UNIT_SPECS[unitKey].fromDisplay(value, this.units());
  }

  private convertStatValue(value: number, unit?: StatUnitKind): number {
    if (!unit || !Number.isFinite(value)) {
      return value;
    }

    if (unit === 'distance') {
      return this.convert([value], 'distance')[0] ?? value;
    }
    if (unit === 'velocity') {
      return this.convert([value], 'velocity')[0] ?? value;
    }
    return value;
  }

  private statUnitSuffix(unit?: StatUnitKind): string {
    if (!unit) {
      return '';
    }
    if (unit === 'distance') {
      return unitSystems[this.units()].distance.suffix;
    }
    if (unit === 'velocity') {
      return unitSystems[this.units()].velocity.suffix;
    }
    if (unit === 'time') {
      return 's';
    }
    if (unit === 'angle') {
      return 'deg';
    }
    return '';
  }

  private rebaseFormUnits(fromUnits: Units, toUnits: Units): void {
    const raw = this.form.getRawValue();

    const convertRecord = (record: Record<string, number>, keys: string[]): Record<string, number> => {
      const next = { ...record };
      keys.forEach((key) => {
        const value = record[key];
        if (typeof value === 'number') {
          next[key] = this.convertFormValue(value, FIELD_META[key]?.unit, fromUnits, toUnits);
        }
      });
      return next;
    };

    const stageTuning = raw.filter.stage_tuning.map((stage) => {
      const nextStage = { ...stage };
      (Object.keys(nextStage) as Array<keyof StageTuning>).forEach((key) => {
        const value = nextStage[key];
        if (typeof value === 'number') {
          nextStage[key] = this.convertFormValue(value, TUNING_FIELD_META[key].unit, fromUnits, toUnits);
        }
      });
      return nextStage;
    });

    this.form.patchValue(
      {
        rocket: convertRecord(raw.rocket as unknown as Record<string, number>, Object.keys(raw.rocket)),
        environment: convertRecord(raw.environment as unknown as Record<string, number>, Object.keys(raw.environment)),
        sensors: convertRecord(raw.sensors as unknown as Record<string, number>, Object.keys(raw.sensors)),
        filter: {
          ...convertRecord(raw.filter as unknown as Record<string, number>, Object.keys(raw.filter).filter((key) => key !== 'stage_tuning')),
          stage_tuning: stageTuning
        }
      },
      { emitEvent: false }
    );
  }

  private buildSimulationRequest(): SimulationRequest {
    const raw = this.form.getRawValue() as unknown as SimulationRequest;

    const convertRecord = <T extends Record<string, unknown>>(record: T): T => {
      const next = { ...record };
      for (const [key, value] of Object.entries(record)) {
        if (typeof value === 'number') {
          (next as Record<string, unknown>)[key] = this.nativeValueFromDisplay(value, FIELD_META[key]?.unit);
        }
      }
      return next;
    };

    const filter = convertRecord(raw.filter);
    const selectedAlgorithms = (raw.filter.selected_algorithms as FilterAlgorithm[]).filter(Boolean);
    filter.selected_algorithms = selectedAlgorithms.length ? selectedAlgorithms : ['eskf'];
    if (!filter.selected_algorithms.includes(raw.filter.active_algorithm as FilterAlgorithm)) {
      filter.active_algorithm = filter.selected_algorithms[0];
    }
    filter.stage_tuning = raw.filter.stage_tuning.map((stage) => {
      const nextStage = { ...stage };
      (Object.keys(nextStage) as Array<keyof StageTuning>).forEach((key) => {
        nextStage[key] = this.nativeValueFromDisplay(nextStage[key], TUNING_FIELD_META[key].unit);
      });
      return nextStage;
    });

    return {
      rocket: convertRecord(raw.rocket),
      environment: convertRecord(raw.environment),
      sensors: raw.sensors,
      filter,
      options: raw.options
    };
  }

  protected toggleAlgorithm(algorithm: FilterAlgorithm, enabled: boolean): void {
    const current = new Set<FilterAlgorithm>(this.form.controls.filter.controls.selected_algorithms.value);
    if (enabled) {
      current.add(algorithm);
    } else {
      current.delete(algorithm);
    }
    if (current.size === 0) {
      current.add('eskf');
    }
    const next = this.algorithmOptions.map((x) => x.value).filter((value) => current.has(value));
    this.form.controls.filter.controls.selected_algorithms.setValue(next);
    const active = this.form.controls.filter.controls.active_algorithm.value;
    if (!current.has(active)) {
      this.form.controls.filter.controls.active_algorithm.setValue(next[0] ?? 'eskf');
    }
  }

  protected isAlgorithmSelected(algorithm: FilterAlgorithm): boolean {
    const current = this.form.controls.filter.controls.selected_algorithms.value;
    return current.includes(algorithm);
  }

  protected setActiveAlgorithm(algorithm: FilterAlgorithm): void {
    if (!this.isAlgorithmSelected(algorithm)) {
      return;
    }
    this.form.controls.filter.controls.active_algorithm.setValue(algorithm);
  }

  private altitudeFromNedDown(values: number[]): number[] {
    return values.map((value) => -value);
  }

  private altitudeFromResponse(data: SimulationResponse): number[] {
    return data.position_z;
  }

  private renderCharts(): void {
    const data = this.response();
    if (!data) {
      return;
    }

    this.renderTrajectory(data);
    this.renderAltitude(data);
    this.renderVelocity(data);
    this.renderAcceleration(data);
    this.renderForce(data);
    this.renderMass(data);
    this.renderAccel(data);
    this.renderGyro(data);
    this.renderMag(data);
    this.renderAdxl(data);
    this.renderBaro(data);
    this.renderGpsPosition(data);
    this.renderGpsVelocity(data);
    this.renderErrorPosition(data);
    this.renderErrorVelocity(data);
    this.renderErrorAltitude(data);
  }

  private activeFilterData(data: SimulationResponse): SimulationResponse['filter_data'] {
    const active = data.active_filter_algorithm;
    const fromMap = data.algorithm_outputs?.[active]?.filter_data;
    return fromMap ?? data.filter_data;
  }

  private getAlgorithmLabel(algorithm: string): string {
    if ((KNOWN_FILTER_ALGORITHMS as readonly string[]).includes(algorithm)) {
      return FILTER_ALGORITHM_LABELS[algorithm as FilterAlgorithm];
    }
    return algorithm.toUpperCase();
  }

  private getActiveAlgorithmLabel(data: SimulationResponse): string {
    return this.getAlgorithmLabel(data.active_filter_algorithm);
  }

  private renderTrajectory(data: SimulationResponse): void {
    const activeFilter = this.activeFilterData(data);
    const hasFilterTrajectory = this.hasTrajectoryData(
      activeFilter.est_pos_x,
      activeFilter.est_pos_y,
      activeFilter.est_pos_z
    );
    const hasQuantizedTrajectory = this.hasTrajectoryData(
      data.filter_data.quantized_est_pos_x,
      data.filter_data.quantized_est_pos_y,
      data.filter_data.quantized_est_pos_z
    );
    const truthStates = this.makeTrajectoryStateTraces(data.state_changes_sim, data, false);
    const eskfStates = hasFilterTrajectory ? this.makeTrajectoryStateTraces(data.state_changes_eskf, data, true) : [];
    const apogeeIndex = this.findMaxIndex(data.position_z);

    const convertedTrueEast = this.convert(data.position_y, 'distance');
    const convertedTrueNorth = this.convert(data.position_x, 'distance');
    const convertedTrueAltitude = this.convert(data.position_z, 'distance');
    const convertedEskfEast = hasFilterTrajectory ? this.convert(activeFilter.est_pos_y, 'distance') : [];
    const convertedEskfNorth = hasFilterTrajectory ? this.convert(activeFilter.est_pos_x, 'distance') : [];
    const convertedEskfAltitude = hasFilterTrajectory ? this.convert(this.altitudeFromNedDown(activeFilter.est_pos_z), 'distance') : [];
    const convertedQuantEast = hasQuantizedTrajectory ? this.convert(data.filter_data.quantized_est_pos_y, 'distance') : [];
    const convertedQuantNorth = hasQuantizedTrajectory ? this.convert(data.filter_data.quantized_est_pos_x, 'distance') : [];
    const convertedQuantAltitude = hasQuantizedTrajectory ? this.convert(this.altitudeFromNedDown(data.filter_data.quantized_est_pos_z), 'distance') : [];

    const traces: Data[] = [
      {
        x: convertedTrueEast,
        y: convertedTrueNorth,
        z: convertedTrueAltitude,
        mode: 'lines',
        type: 'scatter3d',
        name: 'True Path',
        line: { color: '#3ab27b', width: 4 },
        opacity: 0.92
      }
    ];

    if (hasFilterTrajectory) {
      traces.push({
        x: convertedEskfEast,
        y: convertedEskfNorth,
        z: convertedEskfAltitude,
        mode: 'lines',
        type: 'scatter3d',
        name: this.getActiveAlgorithmLabel(data),
        line: { color: '#ff8a5b', width: 3, dash: 'dash' },
        opacity: 0.92
      });
    }

    if (hasQuantizedTrajectory) {
      traces.push({
        x: convertedQuantEast,
        y: convertedQuantNorth,
        z: convertedQuantAltitude,
        mode: 'lines',
        type: 'scatter3d',
        name: 'Quantized',
        line: { color: '#58a6ff', width: 2, dash: 'dot' },
        opacity: 0.85
      });
    }

    traces.push(
      {
        x: [convertedTrueEast[apogeeIndex] || 0],
        y: [convertedTrueNorth[apogeeIndex] || 0],
        z: [convertedTrueAltitude[apogeeIndex] || 0],
        mode: 'text+markers',
        type: 'scatter3d',
        name: 'Apogee',
        text: [`Apogee: ${(this.convert([data.position_z[apogeeIndex] || 0], 'distance')[0] || 0).toFixed(1)} ${unitSystems[this.units()].distance.suffix}`],
        textposition: 'top center',
        textfont: { size: 11, color: '#f97316' },
        marker: {
          size: 12,
          color: '#f97316',
          symbol: 'diamond',
          line: { color: '#ffffff', width: 1.5 }
        }
      },
      ...truthStates,
      ...eskfStates
    );

    const xRange = this.paddedRange(convertedTrueEast, convertedEskfEast, convertedQuantEast);
    const yRange = this.paddedRange(convertedTrueNorth, convertedEskfNorth, convertedQuantNorth);
    const zRange = this.paddedRange(convertedTrueAltitude, convertedEskfAltitude, convertedQuantAltitude);
    const distanceSuffix = unitSystems[this.units()].distance.suffix;

    const layout: Partial<Layout> = {
      title: { text: '3D Flight Path', font: { size: 14 } },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      font: { family: 'Inter, ui-sans-serif, system-ui, sans-serif', color: '#334155', size: 11 },
      scene: {
        bgcolor: '#ffffff',
        xaxis: { title: { text: `East (${distanceSuffix})` }, gridcolor: '#e7e5e4', zerolinecolor: '#d6d3d1', showbackground: false, range: xRange },
        yaxis: { title: { text: `North (${distanceSuffix})` }, gridcolor: '#e7e5e4', zerolinecolor: '#d6d3d1', showbackground: false, range: yRange },
        zaxis: { title: { text: `Altitude (${distanceSuffix})` }, gridcolor: '#e7e5e4', zerolinecolor: '#d6d3d1', showbackground: false, range: zRange },
        camera: { eye: { x: 1.4, y: 1.45, z: 0.9 } },
        aspectmode: 'data'
      },
      legend: { orientation: 'h', x: 1, y: 1, xanchor: 'right', bgcolor: 'rgba(255,255,255,0.92)', bordercolor: '#e7e5e4', borderwidth: 1 },
      margin: { l: 0, r: 0, t: 50, b: 10 }
    };

    this.drawChart('chart-trajectory', traces, layout);
  }

  private renderAltitude(data: SimulationResponse): void {
    const activeFilter = this.activeFilterData(data);
    const trueAltitude = this.altitudeFromResponse(data);
    const estimatedAltitude = this.altitudeFromNedDown(activeFilter.est_pos_z);
    this.render2DChart(
      'chart-altitude',
      data.time,
      this.convert(trueAltitude, 'distance'),
      this.hasCompatibleScalarSeries(data.time, estimatedAltitude) ? this.convert(estimatedAltitude, 'distance') : null,
      'Altitude vs Time',
      'Time (s)',
      `Altitude (${unitSystems[this.units()].distance.suffix})`,
      data
    );
  }

  private renderVelocity(data: SimulationResponse): void {
    const activeFilter = this.activeFilterData(data);
    const estimatedVelocity = this.vectorMagnitude(
      activeFilter.est_vel_x,
      activeFilter.est_vel_y,
      activeFilter.est_vel_z
    );
    this.render2DChart(
      'chart-velocity',
      data.time,
      this.convert(data.velocity, 'velocity'),
      this.hasCompatibleScalarSeries(data.time, estimatedVelocity) ? this.convert(estimatedVelocity, 'velocity') : null,
      'Velocity vs Time',
      'Time (s)',
      `Velocity (${unitSystems[this.units()].velocity.suffix})`,
      data
    );
  }

  private renderAcceleration(data: SimulationResponse): void {
    this.render2DChart(
      'chart-acceleration',
      data.time,
      this.convert(data.acceleration, 'acceleration'),
      null,
      'Acceleration vs Time',
      'Time (s)',
      `Acceleration (${unitSystems[this.units()].acceleration.suffix})`,
      data
    );
  }

  private renderForce(data: SimulationResponse): void {
    this.render2DChart(
      'chart-force',
      data.time,
      this.convert(data.force, 'force'),
      null,
      'Net Force vs Time',
      'Time (s)',
      `Force (${unitSystems[this.units()].force.suffix})`,
      data
    );
  }

  private renderMass(data: SimulationResponse): void {
    this.render2DChart(
      'chart-mass',
      data.time,
      this.convert(data.mass, 'mass'),
      null,
      'Mass vs Time',
      'Time (s)',
      `Mass (${unitSystems[this.units()].mass.suffix})`,
      data
    );
  }

  private renderAccel(data: SimulationResponse): void {
    if (!this.hasSensorSeries(data.time, data.true_accel_x, data.sensor_data.accel_x)) {
      this.clearChart('chart-accel');
      return;
    }

    this.renderMultiSeriesChart(
      'chart-accel',
      [
        { x: data.time, y: this.convert(data.true_accel_x, 'acceleration'), name: 'True North', color: '#ef4444', width: 2, dash: 'solid', opacity: 0.6 },
        { x: data.time, y: this.convert(data.sensor_data.accel_x, 'acceleration'), name: 'Sensor North', color: '#ef4444', width: 1, dash: 'dot' },
        { x: data.time, y: this.convert(data.true_accel_y, 'acceleration'), name: 'True East', color: '#3b82f6', width: 2, dash: 'solid', opacity: 0.6 },
        { x: data.time, y: this.convert(data.sensor_data.accel_y, 'acceleration'), name: 'Sensor East', color: '#3b82f6', width: 1, dash: 'dot' },
        { x: data.time, y: this.convert(data.true_accel_z, 'acceleration'), name: 'True Down', color: '#10b981', width: 2, dash: 'solid', opacity: 0.6 },
        { x: data.time, y: this.convert(data.sensor_data.accel_z, 'acceleration'), name: 'Sensor Down', color: '#10b981', width: 1, dash: 'dot' }
      ],
      'BMI088 Accelerometer',
      `Acceleration (${unitSystems[this.units()].acceleration.suffix})`
    );
  }

  private renderGyro(data: SimulationResponse): void {
    if (!this.hasSensorSeries(data.time, data.true_gyro_x, data.sensor_data.gyro_x)) {
      this.clearChart('chart-gyro');
      return;
    }

    this.renderMultiSeriesChart(
      'chart-gyro',
      [
        { x: data.time, y: this.convert(data.true_gyro_x, 'angular_rate'), name: 'True North', color: '#f59e0b', width: 2, dash: 'solid', opacity: 0.6 },
        { x: data.time, y: this.convert(data.sensor_data.gyro_x, 'angular_rate'), name: 'Sensor North', color: '#f59e0b', width: 1, dash: 'dot' },
        { x: data.time, y: this.convert(data.true_gyro_y, 'angular_rate'), name: 'True East', color: '#a855f7', width: 2, dash: 'solid', opacity: 0.6 },
        { x: data.time, y: this.convert(data.sensor_data.gyro_y, 'angular_rate'), name: 'Sensor East', color: '#a855f7', width: 1, dash: 'dot' },
        { x: data.time, y: this.convert(data.true_gyro_z, 'angular_rate'), name: 'True Down', color: '#06b6d4', width: 2, dash: 'solid', opacity: 0.6 },
        { x: data.time, y: this.convert(data.sensor_data.gyro_z, 'angular_rate'), name: 'Sensor Down', color: '#06b6d4', width: 1, dash: 'dot' }
      ],
      'BMI088 Gyroscope',
      `Angular Rate (${unitSystems[this.units()].angular_rate.suffix})`
    );
  }

  private renderMag(data: SimulationResponse): void {
    if (
      !this.hasCompatibleScalarSeries(data.time, data.sensor_data.mag_x) ||
      !this.hasCompatibleScalarSeries(data.time, data.sensor_data.mag_y) ||
      !this.hasCompatibleScalarSeries(data.time, data.sensor_data.mag_z)
    ) {
      this.clearChart('chart-mag');
      return;
    }

    this.renderMultiSeriesChart(
      'chart-mag',
      [
        { x: data.time, y: this.convert(data.sensor_data.mag_x, 'magnetic_field'), name: 'Mag North', color: '#db2777', width: 1.6 },
        { x: data.time, y: this.convert(data.sensor_data.mag_y, 'magnetic_field'), name: 'Mag East', color: '#8b5cf6', width: 1.6 },
        { x: data.time, y: this.convert(data.sensor_data.mag_z, 'magnetic_field'), name: 'Mag Down', color: '#4f46e5', width: 1.6 }
      ],
      'LIS3MDL Magnetometer',
      `Magnetic Field (${unitSystems[this.units()].magnetic_field.suffix})`
    );
  }

  private renderAdxl(data: SimulationResponse): void {
    if (
      !this.hasCompatibleScalarSeries(data.time, data.sensor_data.adxl_x) ||
      !this.hasCompatibleScalarSeries(data.time, data.sensor_data.adxl_y) ||
      !this.hasCompatibleScalarSeries(data.time, data.sensor_data.adxl_z)
    ) {
      this.clearChart('chart-adxl-yz');
      this.clearChart('chart-adxl-x');
      return;
    }

    this.renderMultiSeriesChart(
      'chart-adxl-yz',
      [
        { x: data.time, y: this.convert(data.sensor_data.adxl_y, 'acceleration'), name: 'ADXL375 East', color: '#2563eb', width: 1.6 },
        { x: data.time, y: this.convert(data.sensor_data.adxl_z, 'acceleration'), name: 'ADXL375 Down', color: '#16a34a', width: 1.6 }
      ],
      'ADXL375 Accelerometer (East & Down)',
      `Acceleration (${unitSystems[this.units()].acceleration.suffix})`
    );

    this.renderMultiSeriesChart(
      'chart-adxl-x',
      [{ x: data.time, y: this.convert(data.sensor_data.adxl_x, 'acceleration'), name: 'ADXL375 North', color: '#f97316', width: 1.6 }],
      'ADXL375 Accelerometer (North)',
      `Acceleration (${unitSystems[this.units()].acceleration.suffix})`
    );
  }

  private renderBaro(data: SimulationResponse): void {
    if (!this.hasCompatibleScalarSeries(data.time, data.sensor_data.baro_pressure)) {
      this.clearChart('chart-baro-pressure');
      this.clearChart('chart-baro-altitude');
      return;
    }

    const baroAltitude = data.sensor_data.baro_pressure.map((pressure) => 8500 * Math.log(101325 / Math.max(pressure, 1)));
    this.renderMultiSeriesChart(
      'chart-baro-pressure',
      [{ x: data.time, y: this.convert(data.sensor_data.baro_pressure, 'pressure'), name: 'MS5611 Pressure', color: '#9333ea', width: 1.6 }],
      'MS5611 Pressure',
      `Pressure (${unitSystems[this.units()].pressure.suffix})`
    );

    this.renderMultiSeriesChart(
      'chart-baro-altitude',
      [{ x: data.time, y: this.convert(baroAltitude, 'distance'), name: 'MS5611 Altitude', color: '#f59e0b', width: 1.6 }],
      'MS5611 Calculated Altitude',
      `Altitude (${unitSystems[this.units()].distance.suffix})`
    );
  }

  private renderGpsPosition(data: SimulationResponse): void {
    if (
      !this.hasCompatibleNullableSeries(data.time, data.sensor_data.gps_x) ||
      !this.hasCompatibleNullableSeries(data.time, data.sensor_data.gps_y) ||
      !this.hasCompatibleNullableSeries(data.time, data.sensor_data.gps_z)
    ) {
      this.clearChart('chart-gps-position');
      return;
    }

    const gpsPosition = this.sparsifyGpsSeries(data.sensor_data.gps_x, data.sensor_data.gps_y, data.sensor_data.gps_z);

    this.renderMultiSeriesChart(
      'chart-gps-position',
      [
        { x: data.time, y: this.convertNullable(gpsPosition.x, 'distance'), name: 'GPS North', color: '#0f766e', width: 1.6, mode: 'lines+markers' },
        { x: data.time, y: this.convertNullable(gpsPosition.y, 'distance'), name: 'GPS East', color: '#ea580c', width: 1.6, mode: 'lines+markers' },
        { x: data.time, y: this.convertNullable(gpsPosition.z, 'distance'), name: 'GPS Down', color: '#78716c', width: 1.6, mode: 'lines+markers' }
      ],
      'GPS Position',
      `Position (${unitSystems[this.units()].distance.suffix})`
    );
  }

  private renderGpsVelocity(data: SimulationResponse): void {
    if (
      !this.hasCompatibleNullableSeries(data.time, data.sensor_data.gps_vel_x) ||
      !this.hasCompatibleNullableSeries(data.time, data.sensor_data.gps_vel_y) ||
      !this.hasCompatibleNullableSeries(data.time, data.sensor_data.gps_vel_z)
    ) {
      this.clearChart('chart-gps-velocity');
      return;
    }

    const gpsVelocity = this.sparsifyGpsSeries(
      data.sensor_data.gps_vel_x,
      data.sensor_data.gps_vel_y,
      data.sensor_data.gps_vel_z
    );

    this.renderMultiSeriesChart(
      'chart-gps-velocity',
      [
        { x: data.time, y: this.convertNullable(gpsVelocity.x, 'velocity'), name: 'GPS Vel North', color: '#475569', width: 1.6, mode: 'lines+markers' },
        { x: data.time, y: this.convertNullable(gpsVelocity.y, 'velocity'), name: 'GPS Vel East', color: '#65a30d', width: 1.6, mode: 'lines+markers' },
        { x: data.time, y: this.convertNullable(gpsVelocity.z, 'velocity'), name: 'GPS Vel Down', color: '#eab308', width: 1.6, mode: 'lines+markers' }
      ],
      'GPS Velocity',
      `Velocity (${unitSystems[this.units()].velocity.suffix})`
    );
  }

  private renderErrorPosition(data: SimulationResponse): void {
    const activeFilter = this.activeFilterData(data);
    if (
      !this.hasCompatibleScalarSeries(data.time, activeFilter.est_pos_x) ||
      !this.hasCompatibleScalarSeries(data.time, activeFilter.est_pos_y) ||
      !this.hasCompatibleScalarSeries(data.time, activeFilter.est_pos_z)
    ) {
      this.clearChart('chart-error-position');
      return;
    }

    const trueDown = this.altitudeFromResponse(data).map((value) => -value);
    const errorNorth = activeFilter.est_pos_x.map((estimate, index) => estimate - (data.position_x[index] || 0));
    const errorEast = activeFilter.est_pos_y.map((estimate, index) => estimate - (data.position_y[index] || 0));
    const errorDown = activeFilter.est_pos_z.map((estimate, index) => estimate - (trueDown[index] || 0));
    const suffix = unitSystems[this.units()].distance.suffix;

    this.renderMultiSeriesChart(
      'chart-error-position',
      [
        { x: data.time, y: this.convert(errorNorth, 'distance'), name: `North Error (${suffix})`, color: '#db2777', width: 1.6 },
        { x: data.time, y: this.convert(errorEast, 'distance'), name: `East Error (${suffix})`, color: '#9333ea', width: 1.6 },
        { x: data.time, y: this.convert(errorDown, 'distance'), name: `Down Error (${suffix})`, color: '#4f46e5', width: 1.6 }
      ],
      'ESKF Position Error vs Time',
      `Error (${suffix})`,
      true
    );
  }

  private renderErrorVelocity(data: SimulationResponse): void {
    const activeFilter = this.activeFilterData(data);
    if (
      !this.hasCompatibleScalarSeries(data.time, activeFilter.est_vel_x) ||
      !this.hasCompatibleScalarSeries(data.time, activeFilter.est_vel_y) ||
      !this.hasCompatibleScalarSeries(data.time, activeFilter.est_vel_z)
    ) {
      this.clearChart('chart-error-velocity');
      return;
    }

    const errorNorth = activeFilter.est_vel_x.map((estimate, index) => estimate - (data.velocity_x[index] || 0));
    const errorEast = activeFilter.est_vel_y.map((estimate, index) => estimate - (data.velocity_y[index] || 0));
    const errorDown = activeFilter.est_vel_z.map((estimate, index) => estimate - (data.velocity_z[index] || 0));
    const suffix = unitSystems[this.units()].velocity.suffix;

    this.renderMultiSeriesChart(
      'chart-error-velocity',
      [
        { x: data.time, y: this.convert(errorNorth, 'velocity'), name: `North Velocity Error (${suffix})`, color: '#db2777', width: 1.6 },
        { x: data.time, y: this.convert(errorEast, 'velocity'), name: `East Velocity Error (${suffix})`, color: '#9333ea', width: 1.6 },
        { x: data.time, y: this.convert(errorDown, 'velocity'), name: `Down Velocity Error (${suffix})`, color: '#4f46e5', width: 1.6 }
      ],
      'ESKF Velocity Error vs Time',
      `Error (${suffix})`,
      true
    );
  }

  private renderErrorAltitude(data: SimulationResponse): void {
    const activeFilter = this.activeFilterData(data);
    if (!this.hasCompatibleScalarSeries(data.time, activeFilter.est_pos_z)) {
      this.clearChart('chart-error-altitude');
      return;
    }

    const trueAltitude = this.altitudeFromResponse(data);
    const estimatedAltitude = this.altitudeFromNedDown(activeFilter.est_pos_z);
    const errorAltitude = estimatedAltitude.map((estimate, index) => estimate - (trueAltitude[index] || 0));
    const suffix = unitSystems[this.units()].distance.suffix;

    this.renderMultiSeriesChart(
      'chart-error-altitude',
      [{ x: data.time, y: this.convert(errorAltitude, 'distance'), name: `Altitude Error (${suffix})`, color: '#f97316', width: 2 }],
      'ESKF Altitude Error vs Time',
      `Altitude Error (${suffix})`,
      true
    );
  }

  private render2DChart(
    elementId: string,
    time: number[],
    simulated: number[],
    estimated: number[] | null,
    title: string,
    xLabel: string,
    yLabel: string,
    data: SimulationResponse
  ): void {
    if (!simulated.length) {
      return;
    }

    const guideStates = this.uniqueStateChanges(data.state_changes_sim);
    const { min: minValue, max: maxValue } = this.finiteRange(simulated, estimated ?? []);
    const valueSpan = Math.max(maxValue - minValue, Math.max(Math.abs(maxValue), 1) * 0.08, 1);
    const padding = valueSpan * 0.1;
    const yRange: [number, number] = [minValue - padding, maxValue + padding];
    const isAltitudeChart = title.includes('Altitude');
    const apogeeIndex = isAltitudeChart ? this.findMaxIndex(simulated) : -1;

    const traces: Data[] = [
      {
        x: time,
        y: simulated,
        mode: 'lines',
        type: 'scatter',
        name: 'Simulated',
        line: { color: '#3ab27b', width: 2.1 }
      }
    ];

    if (estimated?.length) {
      traces.push({
        x: time,
        y: estimated,
        mode: 'lines',
        type: 'scatter',
        name: 'ESKF',
        line: { color: '#ff8a5b', width: 2, dash: 'dash' }
      });
    }

    const shapes: NonNullable<Partial<Layout>['shapes']> = guideStates.map((change) => ({
      type: 'line' as const,
      x0: change.time,
      x1: change.time,
      yref: 'paper' as const,
      y0: 0,
      y1: 1,
      line: { color: '#7c3aed', width: 2, dash: 'dash' as const }
    }));

    if (apogeeIndex >= 0) {
      shapes.push({
        type: 'line',
        x0: time[apogeeIndex] || 0,
        x1: time[apogeeIndex] || 0,
        yref: 'paper',
        y0: 0,
        y1: 1,
        line: { color: '#f97316', width: 2.5, dash: 'dot' }
      });
    }

    const annotations: NonNullable<Partial<Layout>['annotations']> = guideStates.map((change, index) => ({
      x: change.time,
      yref: 'paper' as const,
      y: Math.max(0.14, 0.9 - index * 0.15),
      text: change.state || change.description,
      showarrow: true,
      arrowhead: 2,
      ax: 40,
      ay: 0,
      bgcolor: 'rgba(255,255,255,0.92)',
      bordercolor: '#7c3aed',
      borderwidth: 1,
      font: { size: 11, color: '#0f172a' }
    }));

    if (apogeeIndex >= 0) {
      const apogeeAltitude = simulated[apogeeIndex] || 0;
      const altitudeSpan = Math.max(yRange[1] - yRange[0], 1);
      const annotationY = Math.min(apogeeAltitude + altitudeSpan * 0.07, yRange[1] - altitudeSpan * 0.03);

      annotations.push({
        x: time[apogeeIndex] || 0,
        yref: 'y',
        y: annotationY,
        text: `Apogee: ${apogeeAltitude.toFixed(1)} ${unitSystems[this.units()].distance.suffix} @ ${(time[apogeeIndex] || 0).toFixed(1)} s`,
        showarrow: true,
        arrowhead: 3,
        ax: -52,
        ay: -40,
        bgcolor: 'rgba(249,115,22,0.92)',
        bordercolor: '#f97316',
        borderwidth: 1.5,
        font: { size: 11, color: '#ffffff' }
      });
    }

    const layout: Partial<Layout> = {
      title: { text: title, font: { size: 14 } },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      font: { family: 'Inter, ui-sans-serif, system-ui, sans-serif', color: '#334155', size: 11 },
      xaxis: { title: { text: xLabel }, gridcolor: '#e7e5e4', zeroline: false },
      yaxis: { title: { text: yLabel }, gridcolor: '#e7e5e4', zeroline: false, range: yRange },
      shapes,
      annotations,
      legend: { x: 1, y: 1, xanchor: 'right', bgcolor: 'rgba(255,255,255,0.92)' },
      margin: { l: 56, r: 18, t: 56, b: 42 }
    };

    this.drawChart(elementId, traces, layout);
  }

  private renderMultiSeriesChart(
    elementId: string,
    series: Array<{
      x: number[];
      y: Array<number | null>;
      name: string;
      color: string;
      width: number;
      dash?: 'solid' | 'dot' | 'dash';
      opacity?: number;
      mode?: 'lines' | 'lines+markers';
    }>,
    title: string,
    yLabel: string,
    zeroLine = false
  ): void {
    const normalizedSeries = series
      .map((entry) => ({ ...entry, ...this.trimSeries(entry.x, entry.y) }))
      .filter((entry) => entry.x.length > 0 && entry.y.length > 0);

    if (!normalizedSeries.length) {
      this.clearChart(elementId);
      return;
    }

    const traces: Data[] = normalizedSeries.map((entry) => {
      const trace: Data = {
        x: entry.x,
        y: entry.y,
        mode: entry.mode ?? 'lines',
        type: 'scatter',
        name: entry.name,
        connectgaps: false,
        line: {
          color: entry.color,
          width: entry.width,
          ...(entry.dash ? { dash: entry.dash } : {})
        },
        ...(entry.opacity !== undefined ? { opacity: entry.opacity } : {})
      };

      if (trace.mode === 'lines+markers') {
        trace.marker = { size: 4, color: entry.color };
      }

      return trace;
    });

    const layout: Partial<Layout> = {
      title: { text: title, font: { size: 14 } },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      font: { family: 'Inter, ui-sans-serif, system-ui, sans-serif', color: '#334155', size: 11 },
      xaxis: { title: { text: 'Time (s)' }, gridcolor: '#e7e5e4', zeroline: false },
      yaxis: {
        title: { text: yLabel },
        gridcolor: '#e7e5e4',
        zeroline: zeroLine,
        zerolinecolor: '#94a3b8',
        zerolinewidth: zeroLine ? 1.5 : undefined
      },
      legend: { x: 1, y: 1, xanchor: 'right', bgcolor: 'rgba(255,255,255,0.92)' },
      margin: { l: 56, r: 18, t: 56, b: 42 }
    };

    this.drawChart(elementId, traces, layout);
  }

  private drawChart(elementId: string, traces: Data[], layout: Partial<Layout>, attempt = 0): void {
    this.enqueueChartOperation(elementId, async (element) => {
      const height = this.measureChartHeight(element);
      if (height <= 0) {
        if (attempt >= 2) {
          return;
        }
        requestAnimationFrame(() => this.drawChart(elementId, traces, layout, attempt + 1));
        return;
      }

      const chartLayout: Partial<Layout> = {
        ...layout,
        height
      };

      await this.renderPlot(elementId, element, traces, chartLayout);
    });
  }

  private clearChart(elementId: string): void {
    this.enqueueChartOperation(elementId, async (element) => {
      if (!this.initializedCharts.has(elementId)) {
        element.replaceChildren();
        return;
      }

      const height = this.measureChartHeight(element);
      if (height <= 0) {
        element.replaceChildren();
        this.initializedCharts.delete(elementId);
        return;
      }

      await this.renderPlot(elementId, element, [], this.emptyChartLayout(height));
    });
  }

  private measureChartHeight(element: HTMLElement): number {
    const rectHeight = element.getBoundingClientRect().height;
    if (rectHeight > 0) {
      return Math.round(rectHeight);
    }

    const computed = window.getComputedStyle(element);
    const parseSize = (value: string): number => {
      const parsed = Number.parseFloat(value);
      return Number.isFinite(parsed) ? parsed : 0;
    };

    return Math.round(parseSize(computed.height) || parseSize(computed.minHeight));
  }

  private enqueueChartOperation(elementId: string, operation: (element: HTMLElement) => Promise<void>): void {
    const previous = this.chartOperations.get(elementId) ?? Promise.resolve();
    const next = previous
      .catch(() => undefined)
      .then(async () => {
        const element = document.getElementById(elementId);
        if (!element) {
          this.initializedCharts.delete(elementId);
          return;
        }

        await operation(element);
      })
      .catch((error) => {
        console.error(`Failed to render chart ${elementId}`, error);
      });

    this.chartOperations.set(elementId, next);
  }

  private async renderPlot(
    elementId: string,
    element: HTMLElement,
    traces: Data[],
    layout: Partial<Layout>
  ): Promise<void> {
    const config = { responsive: true, displaylogo: false };

    try {
      if (this.initializedCharts.has(elementId)) {
        await Plotly.react(element, traces, layout, config);
      } else {
        await Plotly.newPlot(element, traces, layout, config);
        this.initializedCharts.add(elementId);
      }
    } catch (error) {
      Plotly.purge(element);
      this.initializedCharts.delete(elementId);
      await Plotly.newPlot(element, traces, layout, config);
      this.initializedCharts.add(elementId);

      if (error instanceof Error) {
        console.warn(`Recovered chart ${elementId} after Plotly error: ${error.message}`);
      } else {
        console.warn(`Recovered chart ${elementId} after Plotly error.`);
      }
    }
  }

  private emptyChartLayout(height: number): Partial<Layout> {
    return {
      height,
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      font: { family: 'Inter, ui-sans-serif, system-ui, sans-serif', color: '#64748b', size: 11 },
      xaxis: { visible: false, showgrid: false, zeroline: false },
      yaxis: { visible: false, showgrid: false, zeroline: false },
      showlegend: false,
      annotations: [
        {
          x: 0.5,
          y: 0.5,
          xref: 'paper',
          yref: 'paper',
          text: 'No data for current run',
          showarrow: false,
          font: { size: 12, color: '#94a3b8' }
        }
      ],
      margin: { l: 24, r: 24, t: 24, b: 24 }
    };
  }

  private convertNullable(values: Array<number | null>, kind: UnitKind): Array<number | null> {
    const factor = unitSystems[this.units()][kind].factor;
    return values.map((value) => (value === null ? null : value * factor));
  }

  private sparsifyGpsSeries(
    x: Array<number | null>,
    y: Array<number | null>,
    z: Array<number | null>
  ): { x: Array<number | null>; y: Array<number | null>; z: Array<number | null> } {
    const sparseX: Array<number | null> = [];
    const sparseY: Array<number | null> = [];
    const sparseZ: Array<number | null> = [];

    for (let index = 0; index < x.length; index += 1) {
      const isLikelyMissingSample = index > 0 && x[index] === null && y[index] === null && z[index] === null;
      sparseX.push(isLikelyMissingSample ? null : x[index]);
      sparseY.push(isLikelyMissingSample ? null : y[index]);
      sparseZ.push(isLikelyMissingSample ? null : z[index]);
    }

    return { x: sparseX, y: sparseY, z: sparseZ };
  }

  private uniqueStateChanges(states: StateChange[]): StateChange[] {
    const seen = new Set<string>();

    return states.filter((change) => {
      const state = change.state || change.description;
      if (!state || state === 'Recovery' || seen.has(state)) {
        return false;
      }

      seen.add(state);
      return true;
    });
  }

  private makeTrajectoryStateTraces(states: StateChange[], data: SimulationResponse, estimated: boolean): Data[] {
    if (estimated && !this.hasTrajectoryData(data.filter_data.est_pos_x, data.filter_data.est_pos_y, data.filter_data.est_pos_z)) {
      return [];
    }

    const seen = new Set<string>();
    return states
      .filter((change) => {
        const state = change.state || change.description;
        if (state === 'Recovery' || seen.has(state)) {
          return false;
        }
        seen.add(state);
        return true;
      })
      .map((change) => {
        const state = change.state || change.description;
        const index = data.time.findIndex((value) => value >= change.time);
        const safeIndex = index >= 0 ? index : data.time.length - 1;
        const x = estimated ? data.filter_data.est_pos_y[safeIndex] || 0 : data.position_y[safeIndex] || 0;
        const y = estimated ? data.filter_data.est_pos_x[safeIndex] || 0 : data.position_x[safeIndex] || 0;
        const z = estimated ? -(data.filter_data.est_pos_z[safeIndex] || 0) : data.position_z[safeIndex] || 0;

        return {
          x: [x],
          y: [y],
          z: [z],
          mode: 'text+markers',
          type: 'scatter3d',
          name: `${estimated ? 'ESKF' : 'Truth'}: ${state}`,
          text: [state],
          textposition: estimated ? 'bottom center' : 'top center',
          textfont: { size: 10, color: stateColors[state] || '#475569' },
          marker: {
            size: estimated ? 8 : 10,
            color: stateColors[state] || '#475569',
            symbol: estimated ? 'diamond' : 'circle',
            line: { color: '#ffffff', width: 1.5 }
          }
        } as unknown as Data;
      });
  }

  private vectorMagnitude(x: number[], y: number[], z: number[]): number[] {
    return x.map((value, index) => Math.sqrt(value * value + y[index] * y[index] + z[index] * z[index]));
  }

  private trimSeries(x: number[], y: Array<number | null>): { x: number[]; y: Array<number | null> } {
    const length = Math.min(x.length, y.length);
    return {
      x: x.slice(0, length),
      y: y.slice(0, length)
    };
  }

  private finiteRange(...series: number[][]): { min: number; max: number } {
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;

    for (const values of series) {
      for (const value of values) {
        if (!Number.isFinite(value)) {
          continue;
        }

        if (value < min) {
          min = value;
        }
        if (value > max) {
          max = value;
        }
      }
    }

    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      return { min: 0, max: 1 };
    }

    return { min, max };
  }

  private paddedRange(...series: number[][]): [number, number] {
    const { min, max } = this.finiteRange(...series);
    const span = Math.max(max - min, 1);
    const padding = span * 0.08;
    return [min - padding, max + padding];
  }

  private hasCompatibleScalarSeries(x: number[], y: number[]): boolean {
    return x.length > 0 && y.length > 0 && x.length === y.length;
  }

  private hasCompatibleNullableSeries(x: number[], y: Array<number | null>): boolean {
    return x.length > 0 && y.length > 0 && x.length === y.length;
  }

  private hasSensorSeries(time: number[], truth: number[], measured: number[]): boolean {
    return this.hasCompatibleScalarSeries(time, truth) && this.hasCompatibleScalarSeries(time, measured);
  }

  private hasTrajectoryData(x: number[], y: number[], z: number[]): boolean {
    return x.length > 0 && x.length === y.length && y.length === z.length;
  }

  private findMaxIndex(values: number[]): number {
    let maxIndex = 0;
    let maxValue = Number.NEGATIVE_INFINITY;

    values.forEach((value, index) => {
      if (value > maxValue) {
        maxValue = value;
        maxIndex = index;
      }
    });

    return maxIndex;
  }

  private queueProcessingIndicator(): void {
    if (this.processingDelayHandle !== null) {
      clearTimeout(this.processingDelayHandle);
    }

    if (this.processingDoneHandle !== null) {
      clearTimeout(this.processingDoneHandle);
      this.processingDoneHandle = null;
    }

    this.processingDelayHandle = setTimeout(() => {
      if (this.isRunning()) {
        this.processingState.set('processing');
      }
      this.processingDelayHandle = null;
    }, 120);
  }

  private clearProcessingIndicator(): void {
    if (this.processingDelayHandle !== null) {
      clearTimeout(this.processingDelayHandle);
      this.processingDelayHandle = null;
    }

    this.processingState.set('updated');

    if (this.processingDoneHandle !== null) {
      clearTimeout(this.processingDoneHandle);
    }

    this.processingDoneHandle = setTimeout(() => {
      this.processingState.set('idle');
      this.processingDoneHandle = null;
    }, 900);
  }

  private buildStatRows(): StatRow[] {
    const groups = this.response()?.error_stats;
    const response = this.response();
    if (!groups || !response) {
      return [];
    }
    const outputs = response.algorithm_outputs ?? {};
    const algorithmKeys = Object.keys(outputs).sort() as FilterAlgorithm[];
    return algorithmKeys.flatMap((algorithmKey) => {
      const algoStats = outputs[algorithmKey]?.error_stats;
      if (!algoStats) {
        return [];
      }
      const algorithmLabel = this.getAlgorithmLabel(algorithmKey);
      return [
        ...this.flattenStats(algorithmLabel, 'Estimate vs True', algoStats.eskf),
        ...this.flattenStats(algorithmLabel, 'Estimate vs Quantized', algoStats.quantized_flight),
        ...this.flattenStats(algorithmLabel, 'True vs Quantized', algoStats.quant_roundtrip),
        ...this.flattenStats(algorithmLabel, 'Landing Recovery', algoStats.quant_recovery),
        ...this.flattenStats(algorithmLabel, 'State Detection', algoStats.state_detection)
      ];
    });
  }

  private flattenStats(algorithm: string, category: string, group: ErrorStatsGroup | null): StatRow[] {
    if (!group) {
      return [];
    }

    return Object.entries(group).map(([label, stats]) => ({
      algorithm,
      category,
      label: this.decorateStatLabel(label),
      stats,
      unit: STAT_META[label]?.unit
    }));
  }

  private decorateStatLabel(label: string): string {
    const meta = STAT_META[label];
    if (!meta) {
      return label;
    }

    const suffix = this.statUnitSuffix(meta.unit);
    return suffix ? `${meta.label} (${suffix})` : meta.label;
  }

  protected readonly stageLabels = STAGE_LABELS;
  protected readonly tuningFields = TUNING_FIELDS;

  protected stageTuningValues(): StageTuning[] {
    return this.form.controls.filter.controls.stage_tuning.getRawValue() as StageTuning[];
  }

  private requestRun(_mode: 'full' | 'sensor-filter'): void {
    this.queueProcessingIndicator();
    void this.runSimulation();
  }
}
