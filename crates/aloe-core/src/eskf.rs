//! # Error-State Kalman Filter for Rocket Navigation
//!
//! This module implements a 15-state Error-State Kalman Filter (ESKF) for 6-DOF rocket state
//! estimation using IMU, GPS, barometer, and magnetometer measurements. The ESKF is preferred
//! over a standard EKF because it maintains the quaternion constraint naturally and has superior
//! numerical properties for highly dynamic vehicles.
//!
//! ## Theory: Error-State Formulation
//!
//! The filter maintains two representations of the state:
//! - **Nominal state** $\mathbf{x}$: Best estimate (position, velocity, quaternion, biases)
//! - **Error state** $\delta\mathbf{x}$: Small deviation from nominal (15-element vector)
//!
//! The true state is the composition:
//!
//! $$\mathbf{x}_{\text{true}} = \mathbf{x} \oplus \delta\mathbf{x}$$
//!
//! where $\oplus$ represents the appropriate composition operator (addition for vectors,
//! quaternion multiplication for attitude).
//!
//! ## State Vector Layout
//!
//! ### Nominal State (21 elements)
//! - **Position** $\mathbf{p} \in \mathbb{R}^3$: NED coordinates (m)
//! - **Velocity** $\mathbf{v} \in \mathbb{R}^3$: NED velocity (m/s)
//! - **Quaternion** $\mathbf{q} \in \mathbb{H}$: Body → NED rotation
//! - **Accel bias** $\mathbf{b}_a \in \mathbb{R}^3$: IMU accelerometer bias (m/s²)
//! - **Gyro bias** $\mathbf{b}_g \in \mathbb{R}^3$: IMU gyroscope bias (rad/s)
//!
//! ### Error State (15 elements)
//!
//! $$\delta\mathbf{x} = \begin{bmatrix} \delta\mathbf{p} \\ \delta\mathbf{v} \\ \delta\boldsymbol{\theta} \\ \delta\mathbf{b}_a \\ \delta\mathbf{b}_g \end{bmatrix}$$
//!
//! where:
//! - $\delta\mathbf{p}$ [0..2]: Position error (m)
//! - $\delta\mathbf{v}$ [3..5]: Velocity error (m/s)
//! - $\delta\boldsymbol{\theta}$ [6..8]: Attitude error as rotation vector (rad)
//! - $\delta\mathbf{b}_a$ [9..11]: Accelerometer bias error (m/s²)
//! - $\delta\mathbf{b}_g$ [12..14]: Gyroscope bias error (rad/s)
//!
//! ## Coordinate System: NED (North-East-Down)
//!
//! - **X-axis**: Points north (positive = north)
//! - **Y-axis**: Points east (positive = east)
//! - **Z-axis**: Points down (positive = toward Earth's center)
//! - **Gravity**: $\mathbf{g} = [0, 0, +9.80665]^T$ m/s²
//! - **Altitude**: Stored as negative Z: $h_{AGL} = -p_z$
//!
//! ## Prediction Step
//!
//! ### IMU Measurement Model
//!
//! Raw IMU measurements include sensor noise and bias:
//!
//! $$\tilde{\mathbf{a}} = \mathbf{a}_m + \mathbf{b}_a + \mathbf{n}_a$$
//! $$\tilde{\boldsymbol{\omega}} = \boldsymbol{\omega}_m + \mathbf{b}_g + \mathbf{n}_g$$
//!
//! where $\mathbf{n}_a$, $\mathbf{n}_g$ are white noise processes.
//!
//! ### Nominal State Propagation
//!
//! $$\dot{\mathbf{p}} = \mathbf{v}$$
//! $$\dot{\mathbf{v}} = \mathbf{R}(\mathbf{q})(\tilde{\mathbf{a}} - \mathbf{b}_a) + \mathbf{g}$$
//! $$\dot{\mathbf{q}} = \frac{1}{2}\mathbf{q} \otimes \boldsymbol{\omega}$$
//!
//! where $\mathbf{R}(\mathbf{q})$ is the rotation matrix (body → NED) and
//! $\boldsymbol{\omega} = \tilde{\boldsymbol{\omega}} - \mathbf{b}_g$.
//!
//! ### Error-State Propagation
//!
//! The error covariance propagates according to:
//!
//! $$\dot{\mathbf{P}} = \mathbf{F}\mathbf{P} + \mathbf{P}\mathbf{F}^T + \mathbf{Q}$$
//!
//! where the state transition matrix $\mathbf{F}$ is:
//!
//! $$\mathbf{F} = \begin{bmatrix} \mathbf{0} & \mathbf{I} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\ \mathbf{0} & \mathbf{0} & -\mathbf{R}[\mathbf{a}]_\times & -\mathbf{R} & \mathbf{0} \\ \mathbf{0} & \mathbf{0} & -[\boldsymbol{\omega}]_\times & \mathbf{0} & -\mathbf{I} \\ \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\ \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} \end{bmatrix}$$
//!
//! and $[\mathbf{v}]_\times$ denotes the skew-symmetric cross-product matrix.
//!
//! ## Update Step (Measurement Fusion)
//!
//! ### GPS Position Update
//!
//! Measurement model:
//! <span class="katex-display">$$
//! \mathbf{z}_{GPS,p} = \mathbf{p} + \mathbf{v}_{GPS,p}
//! $$</span>
//!
//! where <span class="katex-inline">$\mathbf{v}_{GPS,p} \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_{GPS,p})$</span>.
//!
//! Observation matrix: <span class="katex-inline">$\mathbf{H} = [\mathbf{I}_{3\times3} \, \mathbf{0}_{3\times12}]$</span>
//!
//! ### GPS Velocity Update
//!
//! <span class="katex-display">$$
//! \mathbf{z}_{GPS,v} = \mathbf{v} + \mathbf{v}_{GPS,v}
//! $$</span>
//!
//! Observation matrix: <span class="katex-inline">$\mathbf{H} = [\mathbf{0}_{3\times3} \, \mathbf{I}_{3\times3} \, \mathbf{0}_{3\times9}]$</span>
//!
//! ### Barometer Update
//!
//! The barometer measures altitude via static pressure using the barometric formula:
//! <span class="katex-display">$$
//! h = h_0 + H_s \ln\left(\frac{P_0}{P}\right)
//! $$</span>
//!
//! where <span class="katex-inline">$H_s = 7400$</span> m is the scale height.
//!
//! Measurement model:
//! <span class="katex-display">$$
//! z_{baro} = -p_z + v_{baro}
//! $$</span>
//!
//! Observation matrix: <span class="katex-inline">$\mathbf{H} = [\mathbf{e}_3^T \, \mathbf{0}_{1\times12}]$</span> where <span class="katex-inline">$\mathbf{e}_3 = [0, 0, 1]^T$</span>
//!
//! ### Magnetometer Update
//!
//! The magnetometer measures the Earth's magnetic field in body frame. Assuming known field
//! direction in NED (declination angle <span class="katex-inline">$\delta$</span>):
//! <span class="katex-display">$$
//! \mathbf{z}_{mag} = \mathbf{R}^T(\mathbf{q}) \mathbf{m}_{NED} + \mathbf{v}_{mag}
//! $$</span>
//!
//! where <span class="katex-inline">$\mathbf{m}_{NED} = [\cos\delta, 0, \sin\delta]^T$</span> (unit vector).
//!
//! ### Kalman Gain and State Correction
//!
//! <span class="katex-display">$$
//! \begin{aligned}
//! \mathbf{S} &= \mathbf{H}\mathbf{P}\mathbf{H}^T + \mathbf{R} \\
//! \mathbf{K} &= \mathbf{P}\mathbf{H}^T\mathbf{S}^{-1} \\
//! \delta\mathbf{x} &= \mathbf{K}(\mathbf{z} - \mathbf{h}(\mathbf{x})) \\
//! \mathbf{P} &= (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}
//! \end{aligned}
//! $$</span>
//!
//! After each update, the error state is injected into the nominal state and reset:
//! <span class="katex-display">$$
//! \begin{aligned}
//! \mathbf{p} &\leftarrow \mathbf{p} + \delta\mathbf{p} \\
//! \mathbf{v} &\leftarrow \mathbf{v} + \delta\mathbf{v} \\
//! \mathbf{q} &\leftarrow \mathbf{q} \otimes \exp(\delta\boldsymbol{\theta}/2) \\
//! \mathbf{b}_a &\leftarrow \mathbf{b}_a + \delta\mathbf{b}_a \\
//! \mathbf{b}_g &\leftarrow \mathbf{b}_g + \delta\mathbf{b}_g \\
//! \delta\mathbf{x} &\leftarrow \mathbf{0}
//! \end{aligned}
//! $$</span>
//!
//! ## GPS Latency Compensation
//!
//! GPS measurements arrive with latency (typically 50-150ms). The filter maintains a ring buffer
//! of 256 historical states with microsecond timestamps. When a GPS measurement arrives:
//!
//! 1. Find the closest historical state by timestamp
//! 2. Perform the update at that historical time
//! 3. Propagate the correction forward to the current time
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │         Ring Buffer (256 entries)                   │
//! │  [state₀, state₁, ..., stateₙ]                     │
//! │     ↓        ↓           ↓                          │
//! │   t=0ms   t=10ms     t=current                     │
//! │                ↑                                     │
//! │             GPS arrives with                        │
//! │             timestamp t=10ms                        │
//! │             (current time t=160ms)                  │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! ## Filter Architecture
//!
//! ```plantuml
//! @startuml
//! !theme plain
//! skinparam backgroundColor #FEFEFE
//! skinparam component {
//!   BackgroundColor<<imu>> LightBlue
//!   BackgroundColor<<gps>> LightGreen
//!   BackgroundColor<<baro>> LightYellow
//!   BackgroundColor<<mag>> LightCoral
//!   BackgroundColor<<filter>> LightGray
//! }
//!
//! [IMU\nAccel + Gyro] <<imu>> as imu
//! [GPS\nPosition + Velocity] <<gps>> as gps
//! [Barometer\nAltitude] <<baro>> as baro
//! [Magnetometer\nHeading] <<mag>> as mag
//!
//! package "ESKF Core" <<filter>> {
//!   [Prediction\n(IMU Integration)] as predict
//!   [GPS\nUpdate] as gps_update
//!   [Baro\nUpdate] as baro_update
//!   [Mag\nUpdate] as mag_update
//!   [State\nInjection] as inject
//!   database "History\nBuffer\n(256)" as history
//! }
//!
//! [Estimated State\nPosition/Velocity/Attitude] as output
//!
//! imu --> predict : 1000 Hz
//! predict --> history : Store
//! predict --> output
//!
//! gps --> gps_update : 10 Hz\n(delayed)
//! history --> gps_update : Fetch\nhistorical
//! gps_update --> inject
//!
//! baro --> baro_update : 100 Hz
//! baro_update --> inject
//!
//! mag --> mag_update : 100 Hz
//! mag_update --> inject
//!
//! inject --> predict : Corrected\nstate
//! inject --> output
//!
//! @enduml
//! ```
//!
//! ## Tuning Parameters
//!
//! The filter's performance depends critically on the noise covariance matrices:
//!
//! ### Process Noise <span class="katex-inline">$\mathbf{Q}$</span>
//! - `accel_noise_density`: IMU accelerometer noise (m/s²/√Hz)
//! - `gyro_noise_density`: IMU gyroscope noise (rad/s/√Hz)
//! - `accel_bias_instability`: Accel bias random walk (m/s²)
//! - `gyro_bias_instability`: Gyro bias random walk (rad/s)
//! - `pos_process_noise`: Position process noise (m²)
//!
//! ### Measurement Noise <span class="katex-inline">$\mathbf{R}$</span>
//! - `r_gps_pos`: GPS position variance (m²)
//! - `r_gps_vel`: GPS velocity variance ((m/s)²)
//! - `r_baro`: Barometer altitude variance (m²)
//! - `r_mag`: Magnetometer heading variance (rad²)
//!
//! **Pro tip**: These parameters can be tuned per-flight-stage (Pad/Ascent/Coast/Descent/Landed)
//! using greedy coordinate descent to minimize position RMSE. See `aloe-cli --tune-sweep`.
//!
//! ## References
//! - Sola, J. (2017). "Quaternion kinematics for the error-state Kalman filter"
//! - Farrell, J. A. (2008). "Aided Navigation: GPS with High Rate Sensors"
//! - PX4 ECL EKF2 implementation
//! - Crassidis & Junkins (2011). "Optimal Estimation of Dynamic Systems"

use crate::lut_data::{cos_lut, pressure_ratio_to_altitude_lut, sin_lut};
use nalgebra::{
    Matrix1, Matrix2, Matrix3, SMatrix, SVector, Unit, UnitQuaternion, Vector1, Vector2, Vector3,
};

// ---------------------------------------------------------------------------
// Scalar & dimension aliases
// ---------------------------------------------------------------------------
type Scalar = f64;
type V3 = Vector3<Scalar>;
type M3 = Matrix3<Scalar>;
type Quat = UnitQuaternion<Scalar>;
type Cov15 = SMatrix<Scalar, 15, 15>;

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------
const G: Scalar = 9.806_65;
const R_EARTH: Scalar = 6_371_000.0;

// ---------------------------------------------------------------------------
// Filter constants
// ---------------------------------------------------------------------------

/// Ring-buffer depth for GPS latency compensation.
const HIST_LEN: usize = 256;

/// Maximum age of a history snapshot accepted for GPS matching (µs).
const GPS_WINDOW: u64 = 150_000; // 150 ms — covers typical GNSS latency

/// After a successful GPS update, deweight barometer vertical for this long.
/// Baro drifts with weather; GPS altitude references a different datum.
/// While GPS is fresh, letting baro compete causes the vertical channel to
/// oscillate between conflicting references.
const GPS_BARO_INHIBIT_US: u64 = 2_000_000; // 2 s

/// Innovation gate (σ).  Measurements with normalised innovation > this are
/// rejected to protect against outliers.
const GATE_SIGMA: Scalar = 5.0;
const VERTICAL_GATE_SIGMA: Scalar = 20.0;

// ---------------------------------------------------------------------------
// Flight-stage tuning table
// ---------------------------------------------------------------------------

/// Column indices: 0 = pad / pre-launch, 1 = boost, 2 = coast-high, 3 = descent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FlightStage {
    Pad = 0,
    Boost = 1,
    CoastHigh = 2,
    Descent = 3,
}

// ---------------------------------------------------------------------------
// EskfTuning
// ---------------------------------------------------------------------------
/// All noise and covariance parameters for one flight stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EskfTuning {
    /// Accelerometer white-noise density  (m/s² / √Hz)
    pub accel_noise_density: Scalar,
    /// Gyroscope white-noise density      (rad/s / √Hz)
    pub gyro_noise_density: Scalar,
    /// Accelerometer bias random-walk std (m/s² / √Hz)
    pub accel_bias_instability: Scalar,
    /// Gyroscope bias random-walk std     (rad/s / √Hz)
    pub gyro_bias_instability: Scalar,
    /// Extra position process noise std   (m / √Hz)
    pub pos_process_noise: Scalar,
    /// GPS position measurement variance  (m²)
    pub r_gps_pos: Scalar,
    /// GPS velocity measurement variance  ((m/s)²)
    pub r_gps_vel: Scalar,
    /// Barometer altitude variance        (m²)
    pub r_baro: Scalar,
    /// Magnetometer unit-vector variance  (—)
    pub r_mag: Scalar,
}

impl Default for EskfTuning {
    fn default() -> Self {
        Self {
            accel_noise_density: 0.2236,
            gyro_noise_density: 0.03728,
            accel_bias_instability: 0.01,
            gyro_bias_instability: 3.728e-5,
            pos_process_noise: 1.0,
            r_gps_pos: 61.05,
            r_gps_vel: 0.07197,
            r_baro: 0.1,
            r_mag: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Public return type
// ---------------------------------------------------------------------------
/// Result of a predict or update call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterStatus {
    /// State and covariance updated normally.
    Updated,
    /// Predict ran but no IMU data; only position dead-reckoned.
    Coasting,
    /// dt was < 1 µs — skipped to avoid numerical blow-up.
    SkippedSmallDt,
    /// Measurement timestamp too old / no GPS home set.
    SkippedOutdated,
    /// Innovation exceeded the gate; measurement discarded.
    ///
    /// The `f64` is the normalised innovation magnitude.
    RejectedInnovation(Scalar),
    /// Innovation covariance `S` was singular — update skipped.
    SingularMatrix,
}

// ---------------------------------------------------------------------------
// State types
// ---------------------------------------------------------------------------
/// The nominal (non-linear) state estimate.
#[derive(Clone, Copy, Debug)]
pub struct NominalState {
    /// NED position relative to the home origin (m).
    /// `pos.z > 0` means the vehicle is *below* the launch site.
    pub position: V3,
    /// NED velocity (m/s).
    pub velocity: V3,
    /// Body-to-NED rotation quaternion.
    pub orientation: Quat,
    /// Accelerometer bias in body frame (m/s²).
    pub accel_bias: V3,
    /// Gyroscope bias in body frame (rad/s).
    pub gyro_bias: V3,
}

impl NominalState {
    pub fn new() -> Self {
        Self {
            position: V3::zeros(),
            velocity: V3::zeros(),
            orientation: Quat::identity(),
            accel_bias: V3::zeros(),
            gyro_bias: V3::zeros(),
        }
    }
}

impl Default for NominalState {
    fn default() -> Self {
        Self::new()
    }
}

/// Geodetic reference for the home / launch point.
#[derive(Clone, Copy, Debug)]
struct GeoRef {
    lat0_rad: Scalar,
    lon0_rad: Scalar,
    alt0_m: Scalar,
}

/// One slot in the prediction-history ring buffer.
#[derive(Clone, Copy)]
struct Snapshot {
    timestamp_us: u64,
    position: V3,
    velocity: V3,
}

impl Default for Snapshot {
    fn default() -> Self {
        Self {
            timestamp_us: 0,
            position: V3::zeros(),
            velocity: V3::zeros(),
        }
    }
}

// ---------------------------------------------------------------------------
// RocketEsKf
// ---------------------------------------------------------------------------
pub struct RocketEsKf {
    /// Current nominal state estimate.
    pub state: NominalState,
    /// 15×15 error-state covariance matrix.
    pub p_cov: Cov15,
    /// Active tuning parameters (can be swapped between stages).
    pub tuning: EskfTuning,

    ground_pressure: Scalar, // Pa, set at initialisation
    mag_ref_ned: V3,         // unit-vector magnetic field in NED

    geo_ref: Option<GeoRef>,
    last_time_us: Option<u64>,
    /// Timestamp of the most recent accepted GPS update (µs).
    /// Used to deweight barometric altitude while GPS is fresh.
    last_gps_time_us: Option<u64>,

    history: [Snapshot; HIST_LEN],
    history_head: usize,
    history_count: usize,
}

// ─── Construction ────────────────────────────────────────────────────────────
impl RocketEsKf {
    /// Create a new filter.
    ///
    /// # Parameters
    /// - `ground_pressure`      – static pressure at the launch site (Pa)
    /// - `mag_declination_deg`  – magnetic declination, East-positive (°)
    /// - `mag_dip_deg`          – magnetic dip / inclination, down-positive (°)
    /// - `tuning`               – initial noise parameters; swap via
    ///   `filter.tuning = ...` on flight-phase transitions
    pub fn new(
        ground_pressure: f32,
        mag_declination_deg: f32,
        mag_dip_deg: f32,
        tuning: EskfTuning,
    ) -> Self {
        let dip = (mag_dip_deg as Scalar).to_radians();
        let dec = (mag_declination_deg as Scalar).to_radians();

        // NED magnetic reference vector (unit length):
        //   North = cos(dip) * cos(dec)
        //   East  = cos(dip) * sin(dec)
        //   Down  = sin(dip)
        let mn = cos_lut(dip as f32) as Scalar * cos_lut(dec as f32) as Scalar;
        let me = cos_lut(dip as f32) as Scalar * sin_lut(dec as f32) as Scalar;
        let md = sin_lut(dip as f32) as Scalar;
        let mag_ref_ned = V3::new(mn, me, md).normalize();

        // Initial covariance — diagonal, units²
        let mut p = Cov15::zeros();
        p.fixed_view_mut::<3, 3>(0, 0).fill_diagonal(1.0); // pos  ±1 m
        p.fixed_view_mut::<3, 3>(3, 3).fill_diagonal(0.25); // vel  ±0.5 m/s
        p.fixed_view_mut::<3, 3>(6, 6).fill_diagonal(0.01); // att  ±5.7°
        p.fixed_view_mut::<3, 3>(9, 9).fill_diagonal(0.01); // b_a
        p.fixed_view_mut::<3, 3>(12, 12).fill_diagonal(1e-4); // b_g

        Self {
            state: NominalState::new(),
            p_cov: p,
            tuning,
            ground_pressure: ground_pressure as Scalar,
            mag_ref_ned,
            geo_ref: None,
            last_time_us: None,
            last_gps_time_us: None,
            history: [Snapshot::default(); HIST_LEN],
            history_head: 0,
            history_count: 0,
        }
    }

    /// Declare the geodetic home origin.  Must be called before `update_gps`.
    /// Resets `state.position` to zero.
    pub fn set_home(&mut self, lat_deg: f32, lon_deg: f32, alt_m: f32) {
        self.geo_ref = Some(GeoRef {
            lat0_rad: (lat_deg as Scalar).to_radians(),
            lon0_rad: (lon_deg as Scalar).to_radians(),
            alt0_m: alt_m as Scalar,
        });
        self.state.position = V3::zeros();
    }
}

// ─── Prediction ──────────────────────────────────────────────────────────────
impl RocketEsKf {
    /// Propagate the filter one step forward.
    ///
    /// At least one accelerometer channel should be present; gyro is optional
    /// (attitude integration skipped if `None`, covariance still inflated).
    /// When both channels are present the low-g channel is used until it
    /// saturates, at which point the high-g channel takes over.
    pub fn predict(
        &mut self,
        gyro: Option<Vector3<f32>>,
        accel_low: Option<Vector3<f32>>,
        accel_high: Option<Vector3<f32>>,
        time_us: u64,
    ) -> FilterStatus {
        let dt = self.advance_time(time_us);
        if dt < 1e-6 {
            return FilterStatus::SkippedSmallDt;
        }

        let accel = Self::select_accel(accel_low, accel_high);
        let gyro = gyro.map(f32v3_to_f64);

        match accel {
            Some(a_body) => self.predict_full(dt, a_body, gyro, time_us),
            None => self.predict_coast(dt, time_us),
        }
    }

    fn predict_full(
        &mut self,
        dt: Scalar,
        a_body: V3,
        w_body: Option<V3>,
        time_us: u64,
    ) -> FilterStatus {
        let a_ub = a_body - self.state.accel_bias;

        // Rotate specific force to NED and add gravity to recover true accel.
        // Accelerometer measures f = a_true - g_body, so:
        //   a_ned_true = R_bn * f_body + g_ned
        let r_bn = *self.state.orientation.to_rotation_matrix().matrix();
        let a_ned = r_bn * a_ub + V3::new(0.0, 0.0, G);

        // Euler kinematic integration (adequate for dt ≤ 10 ms)
        self.state.position += self.state.velocity * dt + 0.5 * a_ned * dt * dt;
        self.state.velocity += a_ned * dt;

        // Attitude integration
        let w_ub = w_body.unwrap_or(V3::zeros()) - self.state.gyro_bias;
        let angle = w_ub.norm() * dt;
        if angle > 1e-9 {
            let dq = Quat::from_axis_angle(&Unit::new_normalize(w_ub), angle);
            self.state.orientation *= dq;
            self.state.orientation.renormalize();
        }

        self.propagate_covariance(dt, a_ub, w_ub, &r_bn, w_body.is_some());
        self.push_history(time_us);
        FilterStatus::Updated
    }

    fn predict_coast(&mut self, dt: Scalar, time_us: u64) -> FilterStatus {
        self.state.position += self.state.velocity * dt;

        // Inflate covariance uniformly — we have no IMU data
        let q_coast = Cov15::identity() * (100.0 * dt);
        self.p_cov += q_coast;
        symmetrise(&mut self.p_cov);

        self.push_history(time_us);
        FilterStatus::Coasting
    }

    fn propagate_covariance(
        &mut self,
        dt: Scalar,
        a_ub: V3,  // unbiased accel in body frame
        w_ub: V3,  // unbiased angular rate in body frame
        r_bn: &M3, // body-to-NED rotation matrix
        has_gyro: bool,
    ) {
        // ── State-transition matrix F (continuous → discrete, first-order) ────
        //
        // δpos_dot  = δvel
        // δvel_dot  = -R [a]× δθ  - R δb_a
        // δθ_dot    = -[w]× δθ    - δb_g
        // δb_a_dot  = 0
        // δb_g_dot  = 0
        //
        // Discretised: F_d ≈ I + F_c·dt  (for small dt)

        let mut f = Cov15::identity();

        // ∂pos / ∂vel
        f.fixed_view_mut::<3, 3>(0, 3).fill_diagonal(dt);

        // ∂pos / ∂b_a  (0.5 * dt² term — often omitted, matters for slow IMU rates)
        let pos_ba = -r_bn * (0.5 * dt * dt);
        f.fixed_view_mut::<3, 3>(0, 9).copy_from(&pos_ba);

        // ∂vel / ∂θ  = -R [a]× dt
        let a_skew = skew(a_ub);
        let vel_att = -(r_bn * a_skew) * dt;
        f.fixed_view_mut::<3, 3>(3, 6).copy_from(&vel_att);

        // ∂vel / ∂b_a  = -R dt
        let vel_ba = -r_bn * dt;
        f.fixed_view_mut::<3, 3>(3, 9).copy_from(&vel_ba);

        if has_gyro {
            // ∂θ / ∂θ  = I - [w]× dt
            let w_skew = skew(w_ub);
            let att_att = M3::identity() - w_skew * dt;
            f.fixed_view_mut::<3, 3>(6, 6).copy_from(&att_att);

            // ∂θ / ∂b_g  = -I dt
            f.fixed_view_mut::<3, 3>(6, 12).fill_diagonal(-dt);
        }

        // ── Discrete process noise matrix Q ───────────────────────────────────
        //
        // For a random-walk noise with density σ (units/√Hz), the variance
        // accumulated over dt seconds is σ² · dt.  All entries are *variances*.
        let t = &self.tuning;
        let mut q = Cov15::zeros();
        q.fixed_view_mut::<3, 3>(0, 0)
            .fill_diagonal(t.pos_process_noise.powi(2) * dt);
        q.fixed_view_mut::<3, 3>(3, 3)
            .fill_diagonal(t.accel_noise_density.powi(2) * dt);
        q.fixed_view_mut::<3, 3>(6, 6)
            .fill_diagonal(t.gyro_noise_density.powi(2) * dt);
        q.fixed_view_mut::<3, 3>(9, 9)
            .fill_diagonal(t.accel_bias_instability.powi(2) * dt);
        q.fixed_view_mut::<3, 3>(12, 12)
            .fill_diagonal(t.gyro_bias_instability.powi(2) * dt);

        self.p_cov = f * self.p_cov * f.transpose() + q;
        symmetrise(&mut self.p_cov);
    }
}

// ─── Measurement updates ─────────────────────────────────────────────────────
impl RocketEsKf {
    /// Barometric pressure update.
    ///
    /// Converts pressure → AGL altitude → NED Down position.
    ///
    /// When a GPS fix has been received recently the barometer is deweighted by
    /// inflating its measurement noise.  This prevents the vertical channel from
    /// oscillating between two conflicting altitude references (barometric
    /// pressure, which drifts with weather, vs. geometric GPS altitude).
    /// The inhibit window is `GPS_BARO_INHIBIT_US` (default 2 s).
    pub fn update_baro(&mut self, pressure_pa: f32) -> FilterStatus {
        let mut h = SMatrix::<Scalar, 1, 15>::zeros();
        h[(0, 2)] = 1.0;

        let ratio = pressure_pa as Scalar / self.ground_pressure;
        let alt_agl = pressure_ratio_to_altitude_lut(ratio as f32) as Scalar;
        let z_meas = -alt_agl; // NED Down: above ground → negative

        // Deweight baro while GPS data is fresh.
        let r_val = match (self.last_gps_time_us, self.last_time_us) {
            (Some(t_gps), Some(t_now)) if t_now.saturating_sub(t_gps) < GPS_BARO_INHIBIT_US => {
                // GPS active: inflate baro variance so GPS dominates, but do not
                // discard baro entirely. Keeping some baro authority helps avoid
                // vertical drift when sparse/noisy GPS fixes are partially gated.
                self.tuning.r_baro * 10.0
            }
            _ => self.tuning.r_baro,
        };

        let inn = Vector1::new(z_meas - self.state.position.z);
        let r = Matrix1::new(r_val);
        self.apply_update_with_gate(&h, &inn, &r, VERTICAL_GATE_SIGMA)
    }

    /// Magnetometer update.
    ///
    /// `mag_body` is the raw (normalised) field vector in the body frame.
    /// The filter only cares about direction, so the magnitude is stripped.
    pub fn update_mag(&mut self, mag_body: Vector3<f32>) -> FilterStatus {
        let m_body = f32v3_to_f64(mag_body).normalize();
        // Predicted body-frame field: rotate NED reference into body
        let m_pred = self
            .state
            .orientation
            .inverse_transform_vector(&self.mag_ref_ned);
        let inn = m_body - m_pred;

        // H: linearisation of   h(x) = R_nb * m_ned   w.r.t. δθ
        //   ∂h/∂δθ = [m_pred]×
        let mut h = SMatrix::<Scalar, 3, 15>::zeros();
        h.fixed_view_mut::<3, 3>(0, 6).copy_from(&skew(m_pred));

        let r = M3::identity() * self.tuning.r_mag;
        self.apply_update(&h, &inn, &r)
    }

    /// GPS position + velocity update with latency compensation.
    ///
    /// `gps_time_us` is the timestamp *at which the fix was valid* (not arrival
    /// time).  The filter looks up its predicted state at that moment and forms
    /// the innovation there; the correction is then applied to the current state.
    pub fn update_gps(
        &mut self,
        lat_deg: f32,
        lon_deg: f32,
        alt_m: f32,
        vel_ned: Vector3<f32>,
        gps_time_us: u64,
    ) -> FilterStatus {
        let Some(home) = self.geo_ref else {
            return FilterStatus::SkippedOutdated;
        };
        let Some(snap) = self.lookup_history(gps_time_us) else {
            return FilterStatus::SkippedOutdated;
        };

        let pos_meas = geodetic_to_ned(lat_deg, lon_deg, alt_m, home);
        let vel_meas = f32v3_to_f64(vel_ned);

        let pos_inn = pos_meas - snap.position;
        let vel_inn = vel_meas - snap.velocity;

        let mut h_pos_xy = SMatrix::<Scalar, 2, 15>::zeros();
        h_pos_xy[(0, 0)] = 1.0;
        h_pos_xy[(1, 1)] = 1.0;

        let mut h_pos_z = SMatrix::<Scalar, 1, 15>::zeros();
        h_pos_z[(0, 2)] = 1.0;

        let mut h_vel_xy = SMatrix::<Scalar, 2, 15>::zeros();
        h_vel_xy[(0, 3)] = 1.0;
        h_vel_xy[(1, 4)] = 1.0;

        let mut h_vel_z = SMatrix::<Scalar, 1, 15>::zeros();
        h_vel_z[(0, 5)] = 1.0;

        let r_pos_xy = Matrix2::identity() * self.tuning.r_gps_pos;
        let r_pos_z = Matrix1::new(self.tuning.r_gps_pos);
        let r_vel_xy = Matrix2::identity() * self.tuning.r_gps_vel;
        let r_vel_z = Matrix1::new(self.tuning.r_gps_vel);

        // Keep baro deweighted while GPS measurements are actively arriving, even if
        // a particular GPS update gets innovation-gated. Otherwise a rejected GPS
        // fix can let the barometer immediately retake the vertical channel.
        self.last_gps_time_us = self.last_time_us;

        let pos_xy_result =
            self.apply_update(&h_pos_xy, &Vector2::new(pos_inn.x, pos_inn.y), &r_pos_xy);
        let pos_z_result = self.apply_update_with_gate(
            &h_pos_z,
            &Vector1::new(pos_inn.z),
            &r_pos_z,
            VERTICAL_GATE_SIGMA,
        );
        let vel_xy_result =
            self.apply_update(&h_vel_xy, &Vector2::new(vel_inn.x, vel_inn.y), &r_vel_xy);
        let vel_z_result = self.apply_update_with_gate(
            &h_vel_z,
            &Vector1::new(vel_inn.z),
            &r_vel_z,
            VERTICAL_GATE_SIGMA,
        );

        Self::combine_update_results(&[pos_xy_result, pos_z_result, vel_xy_result, vel_z_result])
    }
}

// ─── Core Kalman machinery ────────────────────────────────────────────────────
impl RocketEsKf {
    /// Joseph-form Kalman update with chi-squared innovation gate.
    ///
    /// The Joseph form  P ← (I−KH)P(I−KH)ᵀ + KRKᵀ  is unconditionally
    /// positive-semi-definite even when K is numerically imperfect, making it
    /// significantly more stable than the simpler  P ← (I−KH)P  form.
    fn apply_update<const D: usize>(
        &mut self,
        h: &SMatrix<Scalar, D, 15>,
        inn: &SVector<Scalar, D>,
        r: &SMatrix<Scalar, D, D>,
    ) -> FilterStatus {
        self.apply_update_with_gate(h, inn, r, GATE_SIGMA)
    }

    fn apply_update_with_gate<const D: usize>(
        &mut self,
        h: &SMatrix<Scalar, D, 15>,
        inn: &SVector<Scalar, D>,
        r: &SMatrix<Scalar, D, D>,
        gate_sigma: Scalar,
    ) -> FilterStatus {
        // Innovation covariance  S = H P Hᵀ + R
        let s = h * self.p_cov * h.transpose() + r;

        // ── Innovation gate ───────────────────────────────────────────────────
        // Normalise by the RMS diagonal of S (a rough but fast scale estimate).
        // A proper χ² gate would use  innᵀ S⁻¹ inn  but that requires an
        // inversion we're about to do anyway — this order avoids recomputing.
        let s_scale = s.diagonal().map(|v| v.abs().sqrt()).norm();
        if s_scale > 1e-10 && inn.norm() / s_scale > gate_sigma {
            return FilterStatus::RejectedInnovation(inn.norm() / s_scale);
        }

        // ── Kalman gain ───────────────────────────────────────────────────────
        let Some(s_inv) = s.try_inverse() else {
            return FilterStatus::SingularMatrix;
        };
        let k = self.p_cov * h.transpose() * s_inv;
        let dx = k * inn;

        // ── Apply error-state corrections to nominal state ────────────────────
        self.state.position += dx.fixed_rows::<3>(0).into_owned();
        self.state.velocity += dx.fixed_rows::<3>(3).into_owned();

        let dtheta = dx.fixed_rows::<3>(6).into_owned();
        let angle = dtheta.norm();
        if angle > 1e-10 {
            let dq = Quat::from_axis_angle(&Unit::new_normalize(dtheta), angle);
            self.state.orientation *= dq;
            self.state.orientation.renormalize();
        }

        self.state.accel_bias += dx.fixed_rows::<3>(9).into_owned();
        self.state.gyro_bias += dx.fixed_rows::<3>(12).into_owned();

        // ── Joseph-form covariance update ─────────────────────────────────────
        let i_kh = Cov15::identity() - k * h;
        self.p_cov = i_kh * self.p_cov * i_kh.transpose() + k * r * k.transpose();
        symmetrise(&mut self.p_cov);

        FilterStatus::Updated
    }

    fn combine_update_results(results: &[FilterStatus]) -> FilterStatus {
        let mut rejected_sigma: Option<Scalar> = None;

        for result in results {
            match result {
                FilterStatus::Updated => return FilterStatus::Updated,
                FilterStatus::RejectedInnovation(sigma) => {
                    rejected_sigma =
                        Some(rejected_sigma.map_or(*sigma, |current| current.max(*sigma)));
                }
                _ => {}
            }
        }

        if let Some(sigma) = rejected_sigma {
            return FilterStatus::RejectedInnovation(sigma);
        }

        if results
            .iter()
            .any(|result| matches!(result, FilterStatus::SingularMatrix))
        {
            return FilterStatus::SingularMatrix;
        }

        FilterStatus::SkippedOutdated
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────
impl RocketEsKf {
    /// Update `last_time_us` and return `dt` in seconds.
    /// On the first call returns a safe 1 ms default.
    fn advance_time(&mut self, time_us: u64) -> Scalar {
        let dt = match self.last_time_us {
            Some(prev) => time_us.saturating_sub(prev) as Scalar * 1e-6,
            None => 1e-3,
        };
        self.last_time_us = Some(time_us);
        dt
    }

    /// Choose the best accelerometer reading.
    ///
    /// Uses the low-g channel until it approaches saturation, then linearly
    /// blends in the high-g channel over a 16 m/s² crossover region.  A hard
    /// switch would inject a step into velocity integration equal to any bias
    /// offset between the two sensors; blending makes that transition smooth.
    ///
    /// ```text
    ///   |a|  <  140 m/s²  → 100 % low-g
    ///   140  ≤ |a| ≤ 156  → linear blend  (α = (|a|−140) / 16)
    ///   |a|  >  156 m/s²  → 100 % high-g
    /// ```
    fn select_accel(low: Option<Vector3<f32>>, high: Option<Vector3<f32>>) -> Option<V3> {
        const BLEND_START: Scalar = 140.0; // m/s², ~14 g
        const BLEND_END: Scalar = 156.0; // m/s², ~16 g  (old hard-switch point)
        const BLEND_WIDTH: Scalar = BLEND_END - BLEND_START;

        match (low, high) {
            (Some(l_f32), Some(h_f32)) => {
                let l = f32v3_to_f64(l_f32);
                let h = f32v3_to_f64(h_f32);
                let mag = l.norm();
                if mag <= BLEND_START {
                    Some(l)
                } else if mag >= BLEND_END {
                    Some(h)
                } else {
                    let alpha = (mag - BLEND_START) / BLEND_WIDTH;
                    Some(l * (1.0 - alpha) + h * alpha)
                }
            }
            (Some(l), None) => Some(f32v3_to_f64(l)),
            (None, Some(h)) => Some(f32v3_to_f64(h)),
            (None, None) => None,
        }
    }

    /// Append a position/velocity snapshot to the ring buffer.
    fn push_history(&mut self, time_us: u64) {
        self.history[self.history_head] = Snapshot {
            timestamp_us: time_us,
            position: self.state.position,
            velocity: self.state.velocity,
        };
        self.history_head = (self.history_head + 1) % HIST_LEN;
        if self.history_count < HIST_LEN {
            self.history_count += 1;
        }
    }

    /// Find the snapshot closest in time to `target_us`, within `GPS_WINDOW`.
    ///
    /// Searches newest-first so that it returns quickly when latency is small.
    fn lookup_history(&self, target_us: u64) -> Option<Snapshot> {
        if self.history_count == 0 {
            return None;
        }
        let mut best: Option<Snapshot> = None;
        let mut best_err: u64 = u64::MAX;
        for i in 0..self.history_count {
            let idx = (self.history_head + HIST_LEN - 1 - i) % HIST_LEN;
            let snap = self.history[idx];
            let err = snap.timestamp_us.abs_diff(target_us);
            if err < GPS_WINDOW && err < best_err {
                best_err = err;
                best = Some(snap);
            }
            // Snapshots are ordered newest→oldest; once we've gone past the
            // target window we won't find anything better.
            if snap.timestamp_us < target_us.saturating_sub(GPS_WINDOW) {
                break;
            }
        }
        best
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Convert a geodetic fix to a NED position relative to a home reference.
///
/// Uses a flat-Earth approximation (valid to ~50 km from home).
fn geodetic_to_ned(lat: f32, lon: f32, alt: f32, home: GeoRef) -> V3 {
    let lat_r = (lat as Scalar).to_radians();
    let lon_r = (lon as Scalar).to_radians();
    let north = (lat_r - home.lat0_rad) * R_EARTH;
    let east = (lon_r - home.lon0_rad) * R_EARTH * cos_lut(home.lat0_rad as f32) as Scalar;
    // Down = positive when vehicle is below home altitude
    let down = home.alt0_m - alt as Scalar;
    V3::new(north, east, down)
}

/// 3-vector cross-product matrix (`[v]×`).
#[inline]
fn skew(v: V3) -> M3 {
    M3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Enforce matrix symmetry in-place: M ← (M + Mᵀ) / 2.
/// Prevents small numerical asymmetries from accumulating into non-PSD P.
#[inline]
fn symmetrise(m: &mut Cov15) {
    let mt = m.transpose();
    *m = (*m + mt) * 0.5;
}

/// Losslessly widen an `f32` nalgebra vector to `f64`.
#[inline]
fn f32v3_to_f64(v: Vector3<f32>) -> V3 {
    V3::new(v.x as Scalar, v.y as Scalar, v.z as Scalar)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    fn default_tuning() -> EskfTuning {
        EskfTuning {
            accel_noise_density: 0.2236,
            gyro_noise_density: 0.03728,
            accel_bias_instability: 0.01,
            gyro_bias_instability: 3.728e-5,
            pos_process_noise: 1.0,
            r_gps_pos: 61.05,
            r_gps_vel: 0.07197,
            r_baro: 0.1,
            r_mag: 1.0,
        }
    }

    fn make_filter() -> RocketEsKf {
        RocketEsKf::new(101325.0, 0.0, 45.0, default_tuning())
    }

    #[test]
    fn test_filter_initialization() {
        let filter = make_filter();
        assert_eq!(filter.state.position, V3::zeros());
        assert_eq!(filter.state.velocity, V3::zeros());
    }

    #[test]
    fn test_set_home() {
        let mut filter = make_filter();
        filter.set_home(35.0, -106.0, 1500.0);
        // Should not panic, home location stored
    }

    #[test]
    fn test_predict_does_not_crash() {
        let mut filter = make_filter();
        let accel = Some(Vector3::new(0.0_f32, 0.0, -9.8));
        let gyro = Some(Vector3::zeros());
        let status = filter.predict(gyro, accel, None, 1000);
        assert!(matches!(status, FilterStatus::Updated));
    }

    #[test]
    fn test_update_baro_does_not_crash() {
        let mut filter = make_filter();
        let accel = Some(Vector3::new(0.0_f32, 0.0, -9.8));
        filter.predict(Some(Vector3::zeros()), accel, None, 10000);
        let status = filter.update_baro(100000.0); // ~100m altitude
                                                   // Should not crash; status depends on internal filter state
        let _ = status;
    }

    #[test]
    fn test_update_mag_does_not_crash() {
        let mut filter = make_filter();
        let accel = Some(Vector3::new(0.0_f32, 0.0, -9.8));
        filter.predict(Some(Vector3::zeros()), accel, None, 10000);
        let mag = Vector3::new(0.3_f32, 0.0, 0.5); // Earth's field in body frame
        let status = filter.update_mag(mag);
        // Could be Updated or InnovationRejected
        assert!(matches!(
            status,
            FilterStatus::Updated | FilterStatus::RejectedInnovation(..)
        ));
    }

    #[test]
    fn test_quaternion_remains_normalized() {
        let mut filter = make_filter();
        let accel = Some(Vector3::new(0.0_f32, 0.0, -50.0));
        let gyro = Some(Vector3::new(0.1, 0.05, 0.02));

        // Run many predictions
        let mut time_us: u64 = 0;
        for _ in 0..1000 {
            time_us += 1000; // 1ms per step
            filter.predict(gyro, accel, None, time_us);
        }

        let q = filter.state.orientation;
        let norm = (q.w * q.w + q.i * q.i + q.j * q.j + q.k * q.k).sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "Quaternion norm = {}", norm);
    }

    #[test]
    fn test_skew_matrix() {
        let v = V3::new(1.0, 2.0, 3.0);
        let s = skew(v);

        // Check cross product property: skew(v) * w = v × w
        let w = V3::new(4.0, 5.0, 6.0);
        let cross = v.cross(&w);
        let skew_result = s * w;
        assert!((cross - skew_result).norm() < 1e-10);
    }

    #[test]
    fn test_symmetrise() {
        let mut m = Cov15::from_element(1.0);
        // Add some asymmetry
        m[(0, 1)] = 2.0;
        m[(1, 0)] = 1.5;

        symmetrise(&mut m);

        assert!((m[(0, 1)] - 1.75).abs() < 1e-10);
        assert!((m[(1, 0)] - 1.75).abs() < 1e-10);
    }

    #[test]
    fn test_geodetic_to_ned_roundtrip() {
        let home = GeoRef {
            lat0_rad: 35.0_f64.to_radians(),
            lon0_rad: -106.0_f64.to_radians(),
            alt0_m: 1500.0,
        };

        // Home location should give zero offset
        let ned = geodetic_to_ned(35.0, -106.0, 1500.0, home);
        assert!(ned.norm() < 1.0); // Should be approximately zero

        // 1 degree north should be ~111 km
        let ned_north = geodetic_to_ned(36.0, -106.0, 1500.0, home);
        assert!(ned_north.x > 110000.0 && ned_north.x < 112000.0);
    }

    #[test]
    fn test_covariance_positive_definite() {
        let mut filter = make_filter();

        // Run prediction and updates
        let mut time_us: u64 = 0;
        for i in 0..100 {
            time_us += 10000;
            let accel = Some(Vector3::new(0.0, 0.0, -20.0));
            filter.predict(Some(Vector3::zeros()), accel, None, time_us);
            filter.update_baro(100000.0 - (i as f32) * 10.0);
        }

        // Covariance diagonal should be positive
        let p = filter.p_cov;
        for i in 0..15 {
            assert!(
                p[(i, i)] > 0.0,
                "Diagonal P[{},{}] = {} is not positive",
                i,
                i,
                p[(i, i)]
            );
        }
    }
}
