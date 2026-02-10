use crate::lut_data::{cos_lut, pressure_ratio_to_altitude_lut, sin_lut};
use crate::state_machine::StateMachine;
use crate::state_machine::NUM_STAGES;
use nalgebra::{Matrix3, SMatrix, SVector, UnitQuaternion, Vector3};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const GRAVITY: f32 = 9.80665;
const EARTH_RADIUS: f32 = 6_371_000.0;

// ---------------------------------------------------------------------------
// Tunable filter parameters
// ---------------------------------------------------------------------------
/// All ESKF tuning knobs — passed from Python so they can be swept.
#[derive(Clone, Copy, Debug)]
pub struct EskfTuning {
    pub accel_noise_density: f32, // m/s²/√Hz
    pub gyro_noise_density: f32,  // rad/s/√Hz
    pub accel_bias_instability: f32,
    pub gyro_bias_instability: f32,
    pub pos_process_noise: f32, // m/√s
    pub r_gps_pos: f32,         // GPS position variance (m²)
    pub r_gps_vel: f32,         // GPS velocity variance ((m/s)²)
    pub r_baro: f32,            // Baro altitude variance (m²)
    pub r_mag: f32,             // Magnetometer variance
}

impl Default for EskfTuning {
    fn default() -> Self {
        Self {
            accel_noise_density: 0.5,
            gyro_noise_density: 0.005,
            accel_bias_instability: 1e-4,
            gyro_bias_instability: 1e-5,
            pos_process_noise: 0.1,
            r_gps_pos: 9.0,
            r_gps_vel: 0.25,
            r_baro: 4.0,
            r_mag: 0.05,
        }
    }
}

// Accelerometer Blending Thresholds (m/s²)
// BMI088 saturates at ±24g = ±235.44 m/s²
// Only switch to ADXL375 when BMI088 is near saturation
const LOW_G_SATURATION_THRESHOLD: f32 = 156.0;

// History ring-buffer for GPS latency compensation
const BUFFER_SIZE: usize = 256;

// ---------------------------------------------------------------------------
// Type Aliases
// ---------------------------------------------------------------------------
type ErrorCovariance = SMatrix<f32, 15, 15>;

// ---------------------------------------------------------------------------
// Nominal State
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct NominalState {
    pub position: Vector3<f32>,           // NED (m)
    pub velocity: Vector3<f32>,           // NED (m/s)
    pub orientation: UnitQuaternion<f32>, // Body → NED
    pub accel_bias: Vector3<f32>,         // Body frame
    pub gyro_bias: Vector3<f32>,          // Body frame
}

impl NominalState {
    pub fn new() -> Self {
        Self {
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            orientation: UnitQuaternion::identity(),
            accel_bias: Vector3::zeros(),
            gyro_bias: Vector3::zeros(),
        }
    }
}

// ---------------------------------------------------------------------------
// GPS Reference Point
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct GeoReference {
    lat0: f32, // rad
    lon0: f32, // rad
    alt0: f32, // m
}

// ---------------------------------------------------------------------------
// State Snapshot (ring-buffer element)
// ---------------------------------------------------------------------------
#[derive(Clone, Copy)]
struct Snapshot {
    timestamp_us: u64,
    position: Vector3<f32>,
    velocity: Vector3<f32>,
}

impl Default for Snapshot {
    fn default() -> Self {
        Self {
            timestamp_us: 0,
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
        }
    }
}

// ---------------------------------------------------------------------------
// Main Filter
// ---------------------------------------------------------------------------
pub struct RocketEsKf {
    pub state: NominalState,
    pub p_cov: ErrorCovariance,
    pub tuning: EskfTuning,

    ground_pressure: f32,
    mag_reference: Vector3<f32>,
    geo_ref: Option<GeoReference>,

    last_time_us: Option<u64>,
    history: [Snapshot; BUFFER_SIZE],
    history_head: usize,
    history_count: usize,
}

impl RocketEsKf {
    pub fn new(ground_pressure: f32, mag_declination_deg: f32, tuning: EskfTuning) -> Self {
        let mut p = ErrorCovariance::identity();
        p.fixed_view_mut::<3, 3>(0, 0).fill_diagonal(2.0);
        p.fixed_view_mut::<3, 3>(3, 3).fill_diagonal(0.5);
        p.fixed_view_mut::<3, 3>(6, 6).fill_diagonal(0.1);
        p.fixed_view_mut::<3, 3>(9, 9).fill_diagonal(0.2);
        p.fixed_view_mut::<3, 3>(12, 12).fill_diagonal(0.01);

        let dip_rad = 60.0f32.to_radians();
        let dec_rad = mag_declination_deg.to_radians();
        let mn = cos_lut(dip_rad) * cos_lut(dec_rad);
        let me = cos_lut(dip_rad) * sin_lut(dec_rad);
        let md = sin_lut(dip_rad);
        let mag_ref = Vector3::new(mn, me, md).normalize();

        Self {
            state: NominalState::new(),
            p_cov: p,
            tuning,
            ground_pressure,
            mag_reference: mag_ref,
            geo_ref: None,
            last_time_us: None,
            history: [Snapshot::default(); BUFFER_SIZE],
            history_head: 0,
            history_count: 0,
        }
    }

    pub fn set_home_location(&mut self, lat_deg: f32, lon_deg: f32, alt_m: f32) {
        self.geo_ref = Some(GeoReference {
            lat0: lat_deg.to_radians(),
            lon0: lon_deg.to_radians(),
            alt0: alt_m,
        });
        // Initialize position to home location in NED frame
        self.state.position.z = -alt_m;
    }

    // =====================================================================
    // PREDICTION
    // =====================================================================
    pub fn predict(
        &mut self,
        gyro: Option<Vector3<f32>>,
        accel_low: Option<Vector3<f32>>,
        accel_high: Option<Vector3<f32>>,
        time_us: u64,
    ) {
        let dt = match self.last_time_us {
            Some(last) => (time_us.saturating_sub(last)) as f32 * 1e-6,
            None => 0.001,
        };
        self.last_time_us = Some(time_us);
        if dt < 1e-6 {
            return;
        }

        // Get accelerometer measurement if available
        let accel_meas = match (accel_low, accel_high) {
            (Some(low), Some(high)) => Some(self.blend_accels(low, high)),
            (Some(low), None) => Some(low),
            (None, Some(high)) => Some(high),
            (None, None) => None,
        };

        if let Some(accel) = accel_meas {
            // Normal IMU-based prediction
            let a_unbiased = accel - self.state.accel_bias;
            let q_rot = self.state.orientation.to_rotation_matrix();
            let a_ned = q_rot * a_unbiased + Vector3::new(0.0, 0.0, GRAVITY);

            self.state.position += self.state.velocity * dt + 0.5 * a_ned * dt * dt;
            self.state.velocity += a_ned * dt;

            // Only update orientation if we have gyro data
            if let Some(gyro_vec) = gyro {
                let w_unbiased = gyro_vec - self.state.gyro_bias;
                let angle = w_unbiased.norm() * dt;
                if angle > 1e-8 {
                    let axis = w_unbiased.normalize();
                    let dq = UnitQuaternion::from_axis_angle(
                        &nalgebra::Unit::new_normalize(axis),
                        angle,
                    );
                    self.state.orientation *= dq;
                }
            }

            // --- Covariance Propagation ---
            let mut f_x = ErrorCovariance::identity();
            f_x.fixed_view_mut::<3, 3>(0, 3).fill_diagonal(dt);

            let a_skew = skew_symmetric(a_unbiased);
            let vel_att = -(q_rot.matrix() * a_skew) * dt;
            f_x.fixed_view_mut::<3, 3>(3, 6).copy_from(&vel_att);

            let vel_ab = -q_rot.matrix() * dt;
            f_x.fixed_view_mut::<3, 3>(3, 9).copy_from(&vel_ab);

            // Only include attitude dynamics in F matrix if we have gyro
            if let Some(gyro_vec) = gyro {
                let w_unbiased = gyro_vec - self.state.gyro_bias;
                let w_skew = skew_symmetric(w_unbiased);
                let att_att = Matrix3::identity() - w_skew * dt;
                f_x.fixed_view_mut::<3, 3>(6, 6).copy_from(&att_att);
                f_x.fixed_view_mut::<3, 3>(6, 12).fill_diagonal(-dt);
            }

            let mut q = ErrorCovariance::zeros();
            let t = &self.tuning;
            let and2 = t.accel_noise_density * t.accel_noise_density;
            let gnd2 = t.gyro_noise_density * t.gyro_noise_density;
            let pnq = t.pos_process_noise * t.pos_process_noise;
            q.fixed_view_mut::<3, 3>(0, 0).fill_diagonal(pnq * dt);
            q.fixed_view_mut::<3, 3>(3, 3).fill_diagonal(and2 * dt);
            q.fixed_view_mut::<3, 3>(6, 6).fill_diagonal(gnd2 * dt);
            q.fixed_view_mut::<3, 3>(9, 9)
                .fill_diagonal(t.accel_bias_instability * dt);
            q.fixed_view_mut::<3, 3>(12, 12)
                .fill_diagonal(t.gyro_bias_instability * dt);

            // If no gyro, increase attitude uncertainty
            if gyro.is_none() {
                let coast_factor = 10.0;
                q.fixed_view_mut::<3, 3>(6, 6)
                    .fill_diagonal(gnd2 * dt * coast_factor);
            }

            self.p_cov = f_x * self.p_cov * f_x.transpose() + q;
        } else {
            // No accelerometer - coast on position/velocity with high uncertainty
            // Use constant velocity model
            self.state.position += self.state.velocity * dt;
            // Velocity and orientation remain unchanged (we're coasting)

            // High uncertainty growth when no IMU data
            let mut q = ErrorCovariance::zeros();
            let t = &self.tuning;
            let pnq = t.pos_process_noise * t.pos_process_noise;
            // Position uncertainty grows with velocity uncertainty
            q.fixed_view_mut::<3, 3>(0, 0)
                .fill_diagonal(pnq * dt * 100.0);
            // Velocity uncertainty grows significantly without accel
            q.fixed_view_mut::<3, 3>(3, 3).fill_diagonal(100.0 * dt); // 10 m/s/s uncertainty
                                                                      // Attitude uncertainty grows without gyro
            let gnd2 = t.gyro_noise_density * t.gyro_noise_density;
            q.fixed_view_mut::<3, 3>(6, 6)
                .fill_diagonal(gnd2 * dt * 100.0);
            q.fixed_view_mut::<3, 3>(9, 9)
                .fill_diagonal(t.accel_bias_instability * dt);
            q.fixed_view_mut::<3, 3>(12, 12)
                .fill_diagonal(t.gyro_bias_instability * dt);

            // Simple F matrix for constant velocity
            let mut f_x = ErrorCovariance::identity();
            f_x.fixed_view_mut::<3, 3>(0, 3).fill_diagonal(dt);

            self.p_cov = f_x * self.p_cov * f_x.transpose() + q;
        }

        self.p_cov = (self.p_cov + self.p_cov.transpose()) * 0.5;
        self.push_history(time_us);
    }

    // =====================================================================
    // UPDATES
    // =====================================================================

    pub fn update_baro(&mut self, pressure: f32) {
        let mut h = SMatrix::<f32, 1, 15>::zeros();
        h[(0, 2)] = 1.0;

        // Use full ISA model LUT for altitude computation
        // This is accurate up to 50km (unlike the simple hypsometric formula
        // which is only valid up to ~11km)
        let pressure_ratio = pressure / self.ground_pressure;
        let alt_above_ground = pressure_ratio_to_altitude_lut(pressure_ratio);
        let z_meas = -alt_above_ground; // NED: D = -altitude
        let innovation = z_meas - self.state.position.z;
        let r = SMatrix::<f32, 1, 1>::new(self.tuning.r_baro);
        self.apply_correction(&h, &SVector::<f32, 1>::new(innovation), &r);
    }

    pub fn update_mag(&mut self, mag_body: Vector3<f32>) {
        let m_pred_body = self.state.orientation.inverse() * self.mag_reference;
        let innovation = mag_body - m_pred_body;

        let mut h = SMatrix::<f32, 3, 15>::zeros();
        let m_skew = skew_symmetric(m_pred_body);
        h.fixed_view_mut::<3, 3>(0, 6).copy_from(&m_skew);

        let r = Matrix3::identity() * self.tuning.r_mag;
        self.apply_correction(&h, &innovation, &r);
    }

    pub fn update_gps(
        &mut self,
        lat_deg: f32,
        lon_deg: f32,
        alt_m: f32,
        vel_ned: Vector3<f32>,
        gps_time_us: u64,
    ) {
        let home = match self.geo_ref {
            Some(h) => h,
            None => return,
        };

        let past_state = match self.get_historical_state(gps_time_us) {
            Some(s) => s,
            None => return,
        };

        let pos_ned_meas = Self::geodetic_to_ned(lat_deg, lon_deg, alt_m, home);
        let pos_inn = pos_ned_meas - past_state.position;
        let vel_inn = vel_ned - past_state.velocity;

        let mut innovation = SVector::<f32, 6>::zeros();
        innovation.fixed_rows_mut::<3>(0).copy_from(&pos_inn);
        innovation.fixed_rows_mut::<3>(3).copy_from(&vel_inn);

        let mut h = SMatrix::<f32, 6, 15>::zeros();
        h.fixed_view_mut::<3, 3>(0, 0).fill_diagonal(1.0);
        h.fixed_view_mut::<3, 3>(3, 3).fill_diagonal(1.0);

        let mut r = SMatrix::<f32, 6, 6>::zeros();
        r.fixed_view_mut::<3, 3>(0, 0)
            .fill_diagonal(self.tuning.r_gps_pos);
        r.fixed_view_mut::<3, 3>(3, 3)
            .fill_diagonal(self.tuning.r_gps_vel);

        self.apply_correction(&h, &innovation, &r);
    }

    // =====================================================================
    // CORE CORRECTION
    // =====================================================================
    fn apply_correction<const D: usize>(
        &mut self,
        h: &SMatrix<f32, D, 15>,
        innovation: &SVector<f32, D>,
        r: &SMatrix<f32, D, D>,
    ) {
        let s = h * self.p_cov * h.transpose() + r;

        // Innovation-based fault detection gating
        // Reject measurements with normalized innovation > threshold
        const INNOVATION_GATE_THRESHOLD: f32 = 5.0;
        let s_diag_norm = s.diagonal().map(|v| v.sqrt()).norm();
        if s_diag_norm > 1e-10 {
            let normalized_innovation = innovation.norm() / s_diag_norm;
            if normalized_innovation > INNOVATION_GATE_THRESHOLD {
                return; // reject this measurement
            }
        }

        if let Some(s_inv) = s.try_inverse() {
            let k = self.p_cov * h.transpose() * s_inv;
            let dx = k * innovation;

            self.state.position += dx.fixed_rows::<3>(0).into_owned();
            self.state.velocity += dx.fixed_rows::<3>(3).into_owned();

            let theta = dx.fixed_rows::<3>(6).into_owned();
            if theta.norm() > 1e-9 {
                let axis = theta.normalize();
                let dq = UnitQuaternion::from_axis_angle(
                    &nalgebra::Unit::new_normalize(Vector3::new(axis.x, axis.y, axis.z)),
                    theta.norm(),
                );
                self.state.orientation *= dq;
            }

            self.state.accel_bias += dx.fixed_rows::<3>(9).into_owned();
            self.state.gyro_bias += dx.fixed_rows::<3>(12).into_owned();

            let i = ErrorCovariance::identity();
            self.p_cov = (i - k * h) * self.p_cov;
            self.p_cov = (self.p_cov + self.p_cov.transpose()) * 0.5;
        }
    }

    fn blend_accels(&self, low: Vector3<f32>, high: Vector3<f32>) -> Vector3<f32> {
        let mag = low.norm();
        // Only use ADXL375 high-g accel when BMI088 is near saturation
        if mag > LOW_G_SATURATION_THRESHOLD {
            high
        } else {
            low
        }
    }

    // =====================================================================
    // HISTORY RING BUFFER
    // =====================================================================
    fn push_history(&mut self, time_us: u64) {
        self.history[self.history_head] = Snapshot {
            timestamp_us: time_us,
            position: self.state.position,
            velocity: self.state.velocity,
        };
        self.history_head = (self.history_head + 1) % BUFFER_SIZE;
        if self.history_count < BUFFER_SIZE {
            self.history_count += 1;
        }
    }

    fn get_historical_state(&self, target_us: u64) -> Option<Snapshot> {
        if self.history_count == 0 {
            return None;
        }
        for i in 0..self.history_count {
            let idx = (self.history_head + BUFFER_SIZE - 1 - i) % BUFFER_SIZE;
            let snap = &self.history[idx];
            let diff = snap.timestamp_us.abs_diff(target_us);
            if diff < 2000 {
                return Some(*snap);
            }
        }
        None
    }

    fn geodetic_to_ned(lat: f32, lon: f32, alt: f32, ref_geo: GeoReference) -> Vector3<f32> {
        let lat_rad = lat.to_radians();
        let lon_rad = lon.to_radians();
        let d_lat = lat_rad - ref_geo.lat0;
        let d_lon = lon_rad - ref_geo.lon0;
        let n = d_lat * EARTH_RADIUS;
        let e = d_lon * EARTH_RADIUS * cos_lut(ref_geo.lat0);
        let d = ref_geo.alt0 - alt;
        Vector3::new(n, e, d)
    }
}

fn skew_symmetric(v: Vector3<f32>) -> Matrix3<f32> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

// =========================================================================
// PyO3 Wrapper
// =========================================================================

/// Python-facing wrapper around the ES-EKF.
#[pyclass]
pub struct PyRocketEsKf {
    inner: RocketEsKf,
}

#[pymethods]
impl PyRocketEsKf {
    #[new]
    fn new(ground_pressure: f32, mag_declination_deg: f32) -> Self {
        Self {
            inner: RocketEsKf::new(ground_pressure, mag_declination_deg, EskfTuning::default()),
        }
    }

    fn set_home_location(&mut self, lat_deg: f32, lon_deg: f32, alt_m: f32) {
        self.inner.set_home_location(lat_deg, lon_deg, alt_m);
    }

    /// Predict step. Arrays are [x, y, z]. Pass None for missing sensors.
    fn predict(
        &mut self,
        gyro: Option<[f32; 3]>,
        accel_low: Option<[f32; 3]>,
        accel_high: Option<[f32; 3]>,
        time_us: u64,
    ) {
        self.inner.predict(
            gyro.map(Vector3::from),
            accel_low.map(Vector3::from),
            accel_high.map(Vector3::from),
            time_us,
        );
    }

    fn update_baro(&mut self, pressure: f32) {
        self.inner.update_baro(pressure);
    }

    fn update_mag(&mut self, mag_body: [f32; 3]) {
        self.inner.update_mag(Vector3::from(mag_body));
    }

    fn update_gps(
        &mut self,
        lat_deg: f32,
        lon_deg: f32,
        alt_m: f32,
        vel_ned: [f32; 3],
        gps_time_us: u64,
    ) {
        self.inner
            .update_gps(lat_deg, lon_deg, alt_m, Vector3::from(vel_ned), gps_time_us);
    }

    /// Return (position_ned, velocity_ned, orientation_quat_wxyz)
    fn get_state(&self) -> ([f32; 3], [f32; 3], [f32; 4]) {
        let s = &self.inner.state;
        let q = s.orientation.quaternion();
        (
            [s.position.x, s.position.y, s.position.z],
            [s.velocity.x, s.velocity.y, s.velocity.z],
            [q.w, q.i, q.j, q.k],
        )
    }
}

// =========================================================================
// Batch processing: run filter on numpy arrays passed from Python
// =========================================================================

/// Run the ES-EKF over columnar data arrays from the sim.
///
/// Returns a list of (time_us, pos_n, pos_e, pos_d, vel_n, vel_e, vel_d,
///                     qw, qx, qy, qz) tuples for every prediction step,
/// plus the corrections applied at sensor sample instants.
///
/// Arguments are flat f32 slices for each column.  The function figures out
/// which sensor samples are available by checking for NaN (the Python layer
/// replaces null with NaN before calling).
#[pyfunction]
#[pyo3(signature = (
    times_s,
    gyro_x, gyro_y, gyro_z,
    accel_low_x, accel_low_y, accel_low_z,
    accel_high_x, accel_high_y, accel_high_z,
    baro_pressure,
    mag_x, mag_y, mag_z,
    gps_pos_x, gps_pos_y, gps_pos_z,
    gps_vel_x, gps_vel_y, gps_vel_z,
    ground_pressure,
    mag_declination_deg,
    *,
    accel_noise_density = None,
    gyro_noise_density = None,
    accel_bias_instability = None,
    gyro_bias_instability = None,
    pos_process_noise = None,
    r_gps_pos = None,
    r_gps_vel = None,
    r_baro = None,
    r_mag = None,
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn run_eskf_on_arrays(
    times_s: Vec<f32>,
    // Gyro (rad/s)
    gyro_x: Vec<f32>,
    gyro_y: Vec<f32>,
    gyro_z: Vec<f32>,
    // BMI088 low-g accel (m/s²)
    accel_low_x: Vec<f32>,
    accel_low_y: Vec<f32>,
    accel_low_z: Vec<f32>,
    // ADXL375 high-g accel (m/s²)
    accel_high_x: Vec<f32>,
    accel_high_y: Vec<f32>,
    accel_high_z: Vec<f32>,
    // Baro (mbar → Pa inside)
    baro_pressure: Vec<f32>,
    // Mag (normalised body frame)
    mag_x: Vec<f32>,
    mag_y: Vec<f32>,
    mag_z: Vec<f32>,
    // GPS position NED (m) — NaN when no sample
    gps_pos_x: Vec<f32>,
    gps_pos_y: Vec<f32>,
    gps_pos_z: Vec<f32>,
    // GPS velocity NED (m/s)
    gps_vel_x: Vec<f32>,
    gps_vel_y: Vec<f32>,
    gps_vel_z: Vec<f32>,
    // Config
    ground_pressure: f32,
    mag_declination_deg: f32,
    // Per-stage tuning (keyword-only, each Vec has NUM_STAGES elements: pad/burn/coast/recovery)
    // When None, uses uniform default for all stages.
    accel_noise_density: Option<Vec<f32>>,
    gyro_noise_density: Option<Vec<f32>>,
    accel_bias_instability: Option<Vec<f32>>,
    gyro_bias_instability: Option<Vec<f32>>,
    pos_process_noise: Option<Vec<f32>>,
    r_gps_pos: Option<Vec<f32>>,
    r_gps_vel: Option<Vec<f32>>,
    r_baro: Option<Vec<f32>>,
    r_mag: Option<Vec<f32>>,
) -> (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    (Vec<u8>, [f32; NUM_STAGES]),
) {
    let n = times_s.len();
    let defaults = EskfTuning::default();

    // Unpack per-stage tuning vectors (NUM_STAGES elements each) or uniform default
    let and_v =
        accel_noise_density.unwrap_or_else(|| vec![defaults.accel_noise_density; NUM_STAGES]);
    let gnd_v = gyro_noise_density.unwrap_or_else(|| vec![defaults.gyro_noise_density; NUM_STAGES]);
    let abi_v =
        accel_bias_instability.unwrap_or_else(|| vec![defaults.accel_bias_instability; NUM_STAGES]);
    let gbi_v =
        gyro_bias_instability.unwrap_or_else(|| vec![defaults.gyro_bias_instability; NUM_STAGES]);
    let ppn_v = pos_process_noise.unwrap_or_else(|| vec![defaults.pos_process_noise; NUM_STAGES]);
    let rgp_v = r_gps_pos.unwrap_or_else(|| vec![defaults.r_gps_pos; NUM_STAGES]);
    let rgv_v = r_gps_vel.unwrap_or_else(|| vec![defaults.r_gps_vel; NUM_STAGES]);
    let rb_v = r_baro.unwrap_or_else(|| vec![defaults.r_baro; NUM_STAGES]);
    let rm_v = r_mag.unwrap_or_else(|| vec![defaults.r_mag; NUM_STAGES]);

    // Build per-stage tuning array [pad, burn, coasting, recovery]
    let stage_tunings: [EskfTuning; NUM_STAGES] = std::array::from_fn(|i| EskfTuning {
        accel_noise_density: and_v
            .get(i)
            .copied()
            .unwrap_or(defaults.accel_noise_density),
        gyro_noise_density: gnd_v.get(i).copied().unwrap_or(defaults.gyro_noise_density),
        accel_bias_instability: abi_v
            .get(i)
            .copied()
            .unwrap_or(defaults.accel_bias_instability),
        gyro_bias_instability: gbi_v
            .get(i)
            .copied()
            .unwrap_or(defaults.gyro_bias_instability),
        pos_process_noise: ppn_v.get(i).copied().unwrap_or(defaults.pos_process_noise),
        r_gps_pos: rgp_v.get(i).copied().unwrap_or(defaults.r_gps_pos),
        r_gps_vel: rgv_v.get(i).copied().unwrap_or(defaults.r_gps_vel),
        r_baro: rb_v.get(i).copied().unwrap_or(defaults.r_baro),
        r_mag: rm_v.get(i).copied().unwrap_or(defaults.r_mag),
    });

    // Start with pad tuning
    let mut kf = RocketEsKf::new(ground_pressure, mag_declination_deg, stage_tunings[0]);
    let mut sm = StateMachine::new();

    // ── Static pad calibration ──────────────────────────────────────
    // While the rocket sits on the pad the only force is gravity.
    // Expected specific-force in NED = [0, 0, −g]; expected gyro = [0,0,0].
    // Averaging the first quiet samples gives us initial bias estimates.
    {
        let expected_sf = Vector3::new(0.0, 0.0, -GRAVITY);
        let mut gyro_sum = Vector3::<f32>::zeros();
        let mut accel_sum = Vector3::<f32>::zeros();
        let mut cal_n: u32 = 0;

        for i in 0..n {
            let has_gyro = !gyro_x[i].is_nan();
            let has_al = !accel_low_x[i].is_nan();
            let has_ah = !accel_high_x[i].is_nan();
            if !(has_gyro && (has_al || has_ah)) {
                continue;
            }
            let a = if has_al {
                Vector3::new(accel_low_x[i], accel_low_y[i], accel_low_z[i])
            } else {
                Vector3::new(accel_high_x[i], accel_high_y[i], accel_high_z[i])
            };
            // Still on the pad? deviation from expected gravity should be small.
            if (a - expected_sf).norm() > 3.0 {
                break; // launch detected
            }
            gyro_sum += Vector3::new(gyro_x[i], gyro_y[i], gyro_z[i]);
            accel_sum += a;
            cal_n += 1;
        }

        if cal_n >= 20 {
            let inv = 1.0 / cal_n as f32;
            kf.state.gyro_bias = gyro_sum * inv;
            kf.state.accel_bias = accel_sum * inv - expected_sf;
            // Tighten covariance for bias states after calibration
            kf.p_cov.fixed_view_mut::<3, 3>(9, 9).fill_diagonal(0.01);
            kf.p_cov.fixed_view_mut::<3, 3>(12, 12).fill_diagonal(0.001);
        }
    }

    let mut out_time = Vec::with_capacity(n);
    let mut out_pn = Vec::with_capacity(n);
    let mut out_pe = Vec::with_capacity(n);
    let mut out_pd = Vec::with_capacity(n);
    let mut out_vn = Vec::with_capacity(n);
    let mut out_ve = Vec::with_capacity(n);
    let mut out_vd = Vec::with_capacity(n);
    let mut out_qw = Vec::with_capacity(n);
    let mut out_qx = Vec::with_capacity(n);
    let mut out_qy = Vec::with_capacity(n);
    let mut out_qk = Vec::with_capacity(n);
    let mut out_states = Vec::with_capacity(n);

    for i in 0..n {
        let t_us = (times_s[i] * 1_000_000.0) as u64;

        // Run prediction when we have at least one accel source (gyro is optional)
        let gyro = if gyro_x[i].is_nan() {
            None
        } else {
            Some(Vector3::new(gyro_x[i], gyro_y[i], gyro_z[i]))
        };
        let accel_low = if accel_low_x[i].is_nan() {
            None
        } else {
            Some(Vector3::new(accel_low_x[i], accel_low_y[i], accel_low_z[i]))
        };
        let accel_high = if accel_high_x[i].is_nan() {
            None
        } else {
            Some(Vector3::new(
                accel_high_x[i],
                accel_high_y[i],
                accel_high_z[i],
            ))
        };

        // Predict with whatever we have (needs at least one accel)
        kf.predict(gyro, accel_low, accel_high, t_us);

        // Corrections
        if !baro_pressure[i].is_nan() {
            kf.update_baro(baro_pressure[i]);
        }
        if !mag_x[i].is_nan() {
            let m = Vector3::new(mag_x[i], mag_y[i], mag_z[i]).normalize();
            kf.update_mag(m);
        }
        if !gps_pos_x[i].is_nan() {
            // GPS is already in NED metres from Python sim; we do a direct
            // position/velocity update (bypassing geodetic conversion).
            // Build a pseudo-GPS observation matrix:
            let pos_inn =
                Vector3::new(gps_pos_x[i], gps_pos_y[i], gps_pos_z[i]) - kf.state.position;
            let vel_inn =
                Vector3::new(gps_vel_x[i], gps_vel_y[i], gps_vel_z[i]) - kf.state.velocity;

            let mut innovation = SVector::<f32, 6>::zeros();
            innovation.fixed_rows_mut::<3>(0).copy_from(&pos_inn);
            innovation.fixed_rows_mut::<3>(3).copy_from(&vel_inn);

            let mut h = SMatrix::<f32, 6, 15>::zeros();
            h.fixed_view_mut::<3, 3>(0, 0).fill_diagonal(1.0);
            h.fixed_view_mut::<3, 3>(3, 3).fill_diagonal(1.0);

            let mut r = SMatrix::<f32, 6, 6>::zeros();
            r.fixed_view_mut::<3, 3>(0, 0)
                .fill_diagonal(kf.tuning.r_gps_pos);
            r.fixed_view_mut::<3, 3>(3, 3)
                .fill_diagonal(kf.tuning.r_gps_vel);

            kf.apply_correction(&h, &innovation, &r);
        }

        // Step the state machine on ESKF velocity and switch tuning
        let dt = if i > 0 {
            times_s[i] - times_s[i - 1]
        } else {
            0.0
        };
        let s = &kf.state;
        let flight_state = sm.update(times_s[i], s.velocity.x, s.velocity.y, s.velocity.z, dt);
        kf.tuning = stage_tunings[flight_state as usize];

        let q = s.orientation.quaternion();
        out_time.push(times_s[i]);
        out_pn.push(s.position.x);
        out_pe.push(s.position.y);
        out_pd.push(s.position.z);
        out_vn.push(s.velocity.x);
        out_ve.push(s.velocity.y);
        out_vd.push(s.velocity.z);
        out_qw.push(q.w);
        out_qx.push(q.i);
        out_qy.push(q.j);
        out_qk.push(q.k);
        out_states.push(flight_state as u8);
    }

    (
        out_time,
        out_pn,
        out_pe,
        out_pd,
        out_vn,
        out_ve,
        out_vd,
        out_qw,
        out_qx,
        out_qy,
        out_qk,
        (out_states, *sm.transition_times()),
    )
}
