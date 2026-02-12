use crate::lut_data::{cos_lut, pressure_ratio_to_altitude_lut, sin_lut};
use nalgebra::{Matrix1, Matrix3, SMatrix, SVector, Unit, UnitQuaternion, Vector1, Vector3};

// ---------------------------------------------------------------------------
// CONFIGURATION
// ---------------------------------------------------------------------------
type Scalar = f64;

const GRAVITY: Scalar = 9.80665;
const EARTH_RADIUS: Scalar = 6_371_000.0;
const LOW_G_SATURATION_THRESHOLD: Scalar = 156.0;
const BUFFER_SIZE: usize = 256;

// Type Aliases
type Vector3r = Vector3<Scalar>;
type Matrix3r = Matrix3<Scalar>;
type UnitQuaternionr = UnitQuaternion<Scalar>;
type ErrorCovariance = SMatrix<Scalar, 15, 15>;

// ---------------------------------------------------------------------------
// STATUS & TUNING
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum FilterStatus {
    Updated,
    Coasting,
    SkippedSmallDt,
    SkippedOutdated,
    RejectedInnovation(Scalar),
    SingularMatrix,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EskfTuning {
    pub accel_noise_density: Scalar,
    pub gyro_noise_density: Scalar,
    pub accel_bias_instability: Scalar,
    pub gyro_bias_instability: Scalar,
    pub pos_process_noise: Scalar,
    pub r_gps_pos: Scalar,
    pub r_gps_vel: Scalar,
    pub r_baro: Scalar,
    pub r_mag: Scalar,
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

// ---------------------------------------------------------------------------
// STATE
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct NominalState {
    pub position: Vector3r,
    pub velocity: Vector3r,
    pub orientation: UnitQuaternionr,
    pub accel_bias: Vector3r,
    pub gyro_bias: Vector3r,
}

impl NominalState {
    pub fn new() -> Self {
        Self {
            position: Vector3r::zeros(),
            velocity: Vector3r::zeros(),
            orientation: UnitQuaternionr::identity(),
            accel_bias: Vector3r::zeros(),
            gyro_bias: Vector3r::zeros(),
        }
    }
}

impl Default for NominalState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GeoReference {
    lat0: Scalar,
    lon0: Scalar,
    alt0: Scalar,
}

#[derive(Clone, Copy)]
struct Snapshot {
    timestamp_us: u64,
    position: Vector3r,
    velocity: Vector3r,
}

impl Default for Snapshot {
    fn default() -> Self {
        Self {
            timestamp_us: 0,
            position: Vector3r::zeros(),
            velocity: Vector3r::zeros(),
        }
    }
}

// ---------------------------------------------------------------------------
// MAIN FILTER
// ---------------------------------------------------------------------------
pub struct RocketEsKf {
    pub state: NominalState,
    pub p_cov: ErrorCovariance,
    pub tuning: EskfTuning,

    ground_pressure: Scalar,
    mag_reference: Vector3r,
    geo_ref: Option<GeoReference>,

    last_time_us: Option<u64>,
    history: [Snapshot; BUFFER_SIZE],
    history_head: usize,
    history_count: usize,
}

impl RocketEsKf {
    pub fn new(ground_pressure: f32, mag_declination_deg: f32, tuning: EskfTuning) -> Self {
        let gp = ground_pressure as Scalar;
        let dip_rad = 60.0_f64.to_radians();
        let dec_rad = (mag_declination_deg as Scalar).to_radians();

        let mut p = ErrorCovariance::identity();
        p.fixed_view_mut::<3, 3>(0, 0).fill_diagonal(2.0);
        p.fixed_view_mut::<3, 3>(3, 3).fill_diagonal(0.5);
        p.fixed_view_mut::<3, 3>(6, 6).fill_diagonal(0.1);
        p.fixed_view_mut::<3, 3>(9, 9).fill_diagonal(0.2);
        p.fixed_view_mut::<3, 3>(12, 12).fill_diagonal(0.01);

        // Explicitly use f32 for LUT inputs, then cast result to f64 (Scalar)
        let mn = (cos_lut(dip_rad as f32) * cos_lut(dec_rad as f32)) as Scalar;
        let me = (cos_lut(dip_rad as f32) * sin_lut(dec_rad as f32)) as Scalar;
        let md = sin_lut(dip_rad as f32) as Scalar;
        let mag_ref = Vector3r::new(mn, me, md).normalize();

        Self {
            state: NominalState::new(),
            p_cov: p,
            tuning,
            ground_pressure: gp,
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
            lat0: (lat_deg as Scalar).to_radians(),
            lon0: (lon_deg as Scalar).to_radians(),
            alt0: alt_m as Scalar,
        });
        self.state.position.z = -(alt_m as Scalar);
    }

    // =====================================================================
    // PREDICT
    // =====================================================================
    pub fn predict(
        &mut self,
        gyro: Option<Vector3<f32>>,
        accel_low: Option<Vector3<f32>>,
        accel_high: Option<Vector3<f32>>,
        time_us: u64,
    ) -> FilterStatus {
        let dt = match self.last_time_us {
            Some(last) => (time_us.saturating_sub(last)) as Scalar * 1e-6,
            None => 0.001,
        };
        self.last_time_us = Some(time_us);

        if dt < 1e-6 {
            return FilterStatus::SkippedSmallDt;
        }

        let to_f64 = |v: Vector3<f32>| Vector3r::new(v.x as Scalar, v.y as Scalar, v.z as Scalar);

        let accel_meas = match (accel_low, accel_high) {
            (Some(l), Some(h)) => Some(self.blend_accels(to_f64(l), to_f64(h))),
            (Some(l), None) => Some(to_f64(l)),
            (None, Some(h)) => Some(to_f64(h)),
            (None, None) => None,
        };

        if let Some(accel) = accel_meas {
            let a_unbiased = accel - self.state.accel_bias;
            let q_rot = self.state.orientation.to_rotation_matrix();
            let a_ned = q_rot * a_unbiased + Vector3r::new(0.0, 0.0, GRAVITY);

            self.state.position += self.state.velocity * dt + 0.5 * a_ned * dt * dt;
            self.state.velocity += a_ned * dt;

            let w_unbiased = gyro.map(to_f64).unwrap_or(Vector3r::zeros()) - self.state.gyro_bias;
            let angle = w_unbiased.norm() * dt;

            if angle > 1e-8 {
                let axis = UnitQuaternionr::from_axis_angle(
                    &nalgebra::Unit::new_normalize(w_unbiased),
                    angle,
                );
                self.state.orientation *= axis;
            }

            self.propagate_cov(dt, a_unbiased, w_unbiased, q_rot.matrix(), gyro.is_some());

            self.push_history(time_us);
            FilterStatus::Updated
        } else {
            self.state.position += self.state.velocity * dt;
            let mut q_coast = ErrorCovariance::zeros();
            q_coast.fill_diagonal(100.0 * dt);
            self.p_cov += q_coast;

            self.push_history(time_us);
            FilterStatus::Coasting
        }
    }

    fn propagate_cov(
        &mut self,
        dt: Scalar,
        a_unbiased: Vector3r,
        w_unbiased: Vector3r,
        rot_mat: &Matrix3r,
        has_gyro: bool,
    ) {
        let mut f = ErrorCovariance::identity();
        f.fixed_view_mut::<3, 3>(0, 3).fill_diagonal(dt);

        let a_skew = skew_symmetric(a_unbiased);
        let vel_att = -(rot_mat * a_skew) * dt;
        f.fixed_view_mut::<3, 3>(3, 6).copy_from(&vel_att);

        let vel_ab = -rot_mat * dt;
        f.fixed_view_mut::<3, 3>(3, 9).copy_from(&vel_ab);

        if has_gyro {
            let w_skew = skew_symmetric(w_unbiased);
            let att_att = Matrix3r::identity() - w_skew * dt;
            f.fixed_view_mut::<3, 3>(6, 6).copy_from(&att_att);
            f.fixed_view_mut::<3, 3>(6, 12).fill_diagonal(-dt);
        }

        let mut q = ErrorCovariance::zeros();
        let t = &self.tuning;
        let and2 = t.accel_noise_density.powi(2);
        let gnd2 = t.gyro_noise_density.powi(2);

        q.fixed_view_mut::<3, 3>(0, 0)
            .fill_diagonal(t.pos_process_noise.powi(2) * dt);
        q.fixed_view_mut::<3, 3>(3, 3).fill_diagonal(and2 * dt);
        q.fixed_view_mut::<3, 3>(6, 6).fill_diagonal(gnd2 * dt);
        q.fixed_view_mut::<3, 3>(9, 9)
            .fill_diagonal(t.accel_bias_instability * dt);
        q.fixed_view_mut::<3, 3>(12, 12)
            .fill_diagonal(t.gyro_bias_instability * dt);

        self.p_cov = f * self.p_cov * f.transpose() + q;
        self.p_cov = (self.p_cov + self.p_cov.transpose()) * 0.5;
    }

    // =====================================================================
    // UPDATES
    // =====================================================================

    fn apply_correction<const D: usize>(
        &mut self,
        h: &SMatrix<Scalar, D, 15>,
        innovation: &SVector<Scalar, D>,
        r: &SMatrix<Scalar, D, D>,
    ) -> FilterStatus {
        let s = h * self.p_cov * h.transpose() + r;

        let s_diag_norm = s.diagonal().map(|v| v.sqrt()).norm();
        if s_diag_norm > 1e-10 {
            let normalized_inn = innovation.norm() / s_diag_norm;
            if normalized_inn > 5.0 {
                return FilterStatus::RejectedInnovation(normalized_inn);
            }
        }

        if let Some(s_inv) = s.try_inverse() {
            let k_gain = self.p_cov * h.transpose() * s_inv;
            let dx = k_gain * innovation;

            self.state.position += dx.fixed_rows::<3>(0).into_owned();
            self.state.velocity += dx.fixed_rows::<3>(3).into_owned();

            let theta = dx.fixed_rows::<3>(6).into_owned();
            if theta.norm() > 1e-9 {
                // Fix: Construct a Vector3 first, then wrap it in Unit::new_normalize
                let theta_vec = Vector3r::new(theta.x, theta.y, theta.z);
                let axis =
                    UnitQuaternionr::from_axis_angle(&Unit::new_normalize(theta_vec), theta.norm());
                self.state.orientation *= axis;
            }

            self.state.accel_bias += dx.fixed_rows::<3>(9).into_owned();
            self.state.gyro_bias += dx.fixed_rows::<3>(12).into_owned();

            let i = ErrorCovariance::identity();
            let kh = k_gain * h;
            let i_minus_kh = i - kh;

            self.p_cov =
                i_minus_kh * self.p_cov * i_minus_kh.transpose() + k_gain * r * k_gain.transpose();

            FilterStatus::Updated
        } else {
            FilterStatus::SingularMatrix
        }
    }

    pub fn update_baro(&mut self, pressure: f32) -> FilterStatus {
        let mut h = SMatrix::<Scalar, 1, 15>::zeros();
        h[(0, 2)] = 1.0;

        let pressure_ratio = (pressure as Scalar) / self.ground_pressure;
        let alt_f32 = pressure_ratio_to_altitude_lut(pressure_ratio as f32);
        let z_meas = -(alt_f32 as Scalar);

        // FIX: Define variables before call to avoid parser ambiguity
        let innovation_val = z_meas - self.state.position.z;
        let innovation = Vector1::<Scalar>::new(innovation_val);
        let r_cov = Matrix1::<Scalar>::new(self.tuning.r_baro);

        self.apply_correction(&h, &innovation, &r_cov)
    }

    pub fn update_mag(&mut self, mag_body: Vector3<f32>) -> FilterStatus {
        let mag_body_f64 = Vector3r::new(
            mag_body.x as Scalar,
            mag_body.y as Scalar,
            mag_body.z as Scalar,
        );
        let m_pred_body = self.state.orientation.inverse() * self.mag_reference;
        let innovation = mag_body_f64 - m_pred_body;

        let mut h = SMatrix::<Scalar, 3, 15>::zeros();
        let m_skew = skew_symmetric(m_pred_body);
        h.fixed_view_mut::<3, 3>(0, 6).copy_from(&m_skew);

        let r = Matrix3r::identity() * self.tuning.r_mag;
        self.apply_correction(&h, &innovation, &r)
    }

    pub fn update_gps(
        &mut self,
        lat_deg: f32,
        lon_deg: f32,
        alt_m: f32,
        vel_ned: Vector3<f32>,
        gps_time_us: u64,
    ) -> FilterStatus {
        let home = match self.geo_ref {
            Some(h) => h,
            None => return FilterStatus::SkippedOutdated,
        };

        let past_state = match self.get_historical_state(gps_time_us) {
            Some(s) => s,
            None => return FilterStatus::SkippedOutdated,
        };

        let vel_ned_f64 = Vector3r::new(
            vel_ned.x as Scalar,
            vel_ned.y as Scalar,
            vel_ned.z as Scalar,
        );
        let pos_ned_meas = Self::geodetic_to_ned(lat_deg, lon_deg, alt_m, home);

        let pos_inn = pos_ned_meas - past_state.position;
        let vel_inn = vel_ned_f64 - past_state.velocity;

        let mut innovation = SVector::<Scalar, 6>::zeros();
        innovation.fixed_rows_mut::<3>(0).copy_from(&pos_inn);
        innovation.fixed_rows_mut::<3>(3).copy_from(&vel_inn);

        let mut h = SMatrix::<Scalar, 6, 15>::zeros();
        h.fixed_view_mut::<3, 3>(0, 0).fill_diagonal(1.0);
        h.fixed_view_mut::<3, 3>(3, 3).fill_diagonal(1.0);

        let mut r = SMatrix::<Scalar, 6, 6>::zeros();
        r.fixed_view_mut::<3, 3>(0, 0)
            .fill_diagonal(self.tuning.r_gps_pos);
        r.fixed_view_mut::<3, 3>(3, 3)
            .fill_diagonal(self.tuning.r_gps_vel);

        self.apply_correction(&h, &innovation, &r)
    }

    // =====================================================================
    // HELPERS
    // =====================================================================
    fn blend_accels(&self, low: Vector3r, high: Vector3r) -> Vector3r {
        if low.norm() > LOW_G_SATURATION_THRESHOLD {
            high
        } else {
            low
        }
    }

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
            if snap.timestamp_us.abs_diff(target_us) < 2000 {
                return Some(*snap);
            }
        }
        None
    }

    fn geodetic_to_ned(lat: f32, lon: f32, alt: f32, ref_geo: GeoReference) -> Vector3r {
        let lat_rad = (lat as Scalar).to_radians();
        let lon_rad = (lon as Scalar).to_radians();
        let d_lat = lat_rad - ref_geo.lat0;
        let d_lon = lon_rad - ref_geo.lon0;
        let n = d_lat * EARTH_RADIUS;
        let e = d_lon * EARTH_RADIUS * cos_lut(ref_geo.lat0 as f32) as Scalar;
        let d = ref_geo.alt0 - (alt as Scalar);
        Vector3r::new(n, e, d)
    }
}

fn skew_symmetric(v: Vector3r) -> Matrix3r {
    Matrix3r::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}
