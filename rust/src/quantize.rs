use pyo3::prelude::*;

// GPS scaling: 1e7 → ~1.1 cm at equator
const GPS_SCALING: f32 = 10_000_000.0;

// =========================================================================
// Flight Data (20 bytes on wire)
// =========================================================================

/// Quantised flight telemetry packet.
#[pyclass(skip_from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct PyFlightData {
    #[pyo3(get)]
    pub pos_n_m: i16, // ±32 km, 1 m resolution
    #[pyo3(get)]
    pub pos_e_m: i16,
    #[pyo3(get)]
    pub alt_agl_cm: i32, // cm resolution
    #[pyo3(get)]
    pub vel_n_ds: i16, // 0.1 m/s resolution
    #[pyo3(get)]
    pub vel_e_ds: i16,
    #[pyo3(get)]
    pub vel_d_ds: i16,
    #[pyo3(get)]
    pub roll: u8, // 0..255 → 0..360°
    #[pyo3(get)]
    pub pitch: u8,
    #[pyo3(get)]
    pub yaw: u8,
    #[pyo3(get)]
    pub status: u8,
}

#[pymethods]
impl PyFlightData {
    #[new]
    fn new(
        pos_n_m: i16,
        pos_e_m: i16,
        alt_agl_cm: i32,
        vel_n_ds: i16,
        vel_e_ds: i16,
        vel_d_ds: i16,
        roll: u8,
        pitch: u8,
        yaw: u8,
        status: u8,
    ) -> Self {
        Self {
            pos_n_m,
            pos_e_m,
            alt_agl_cm,
            vel_n_ds,
            vel_e_ds,
            vel_d_ds,
            roll,
            pitch,
            yaw,
            status,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FlightData(pos=({},{},{}cm), vel=({},{},{} ds), rpy=({},{},{}))",
            self.pos_n_m,
            self.pos_e_m,
            self.alt_agl_cm,
            self.vel_n_ds,
            self.vel_e_ds,
            self.vel_d_ds,
            self.roll,
            self.pitch,
            self.yaw,
        )
    }
}

// =========================================================================
// Recovery Data (14 bytes on wire)
// =========================================================================

#[pyclass(skip_from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct PyRecoveryData {
    #[pyo3(get)]
    pub lat_i32: i32, // deg * 1e7
    #[pyo3(get)]
    pub lon_i32: i32,
    #[pyo3(get)]
    pub alt_msl_m: i16,
    #[pyo3(get)]
    pub sat_info: u8,
    #[pyo3(get)]
    pub batt_v: u8, // 0.1 V
}

#[pymethods]
impl PyRecoveryData {
    #[new]
    fn new(lat_i32: i32, lon_i32: i32, alt_msl_m: i16, sat_info: u8, batt_v: u8) -> Self {
        Self {
            lat_i32,
            lon_i32,
            alt_msl_m,
            sat_info,
            batt_v,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RecoveryData(lat={}, lon={}, alt={}m, sats={}, batt={})",
            self.lat_i32, self.lon_i32, self.alt_msl_m, self.sat_info, self.batt_v,
        )
    }
}

// =========================================================================
// Batch quantize / dequantize for columnar data
// =========================================================================

/// Quantize flight data arrays.
/// pos_n, pos_e in metres; alt_agl in metres; vel_n/e/d in m/s;
/// roll/pitch/yaw in degrees.
/// Returns a tuple of quantized column vectors.
#[pyfunction]
#[pyo3(signature = (pos_n, pos_e, alt_agl, vel_n, vel_e, vel_d, roll_deg, pitch_deg, yaw_deg))]
pub fn quantize_flight_array(
    pos_n: Vec<f32>,
    pos_e: Vec<f32>,
    alt_agl: Vec<f32>,
    vel_n: Vec<f32>,
    vel_e: Vec<f32>,
    vel_d: Vec<f32>,
    roll_deg: Vec<f32>,
    pitch_deg: Vec<f32>,
    yaw_deg: Vec<f32>,
) -> (
    Vec<i16>,
    Vec<i16>,
    Vec<i32>,
    Vec<i16>,
    Vec<i16>,
    Vec<i16>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
) {
    let n = pos_n.len();
    let mut q_pn = Vec::with_capacity(n);
    let mut q_pe = Vec::with_capacity(n);
    let mut q_alt = Vec::with_capacity(n);
    let mut q_vn = Vec::with_capacity(n);
    let mut q_ve = Vec::with_capacity(n);
    let mut q_vd = Vec::with_capacity(n);
    let mut q_roll = Vec::with_capacity(n);
    let mut q_pitch = Vec::with_capacity(n);
    let mut q_yaw = Vec::with_capacity(n);

    for i in 0..n {
        q_pn.push(clamp_i16(pos_n[i].round() as i32));
        q_pe.push(clamp_i16(pos_e[i].round() as i32));
        q_alt.push((alt_agl[i] * 100.0).round() as i32); // m → cm
        q_vn.push(clamp_i16((vel_n[i] * 10.0).round() as i32)); // m/s → 0.1 m/s
        q_ve.push(clamp_i16((vel_e[i] * 10.0).round() as i32));
        q_vd.push(clamp_i16((vel_d[i] * 10.0).round() as i32));
        q_roll.push(deg_to_u8(roll_deg[i]));
        q_pitch.push(deg_to_u8(pitch_deg[i]));
        q_yaw.push(deg_to_u8(yaw_deg[i]));
    }

    (q_pn, q_pe, q_alt, q_vn, q_ve, q_vd, q_roll, q_pitch, q_yaw)
}

/// Dequantize flight data back to f32 arrays.
/// Returns (pos_n, pos_e, alt_agl, vel_n, vel_e, vel_d, roll_deg, pitch_deg, yaw_deg).
#[pyfunction]
pub fn dequantize_flight_array(
    q_pn: Vec<i16>,
    q_pe: Vec<i16>,
    q_alt: Vec<i32>,
    q_vn: Vec<i16>,
    q_ve: Vec<i16>,
    q_vd: Vec<i16>,
    q_roll: Vec<u8>,
    q_pitch: Vec<u8>,
    q_yaw: Vec<u8>,
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
) {
    let n = q_pn.len();
    let mut pn = Vec::with_capacity(n);
    let mut pe = Vec::with_capacity(n);
    let mut alt = Vec::with_capacity(n);
    let mut vn = Vec::with_capacity(n);
    let mut ve = Vec::with_capacity(n);
    let mut vd = Vec::with_capacity(n);
    let mut roll = Vec::with_capacity(n);
    let mut pitch = Vec::with_capacity(n);
    let mut yaw = Vec::with_capacity(n);

    for i in 0..n {
        pn.push(q_pn[i] as f32); // 1 m resolution, no scaling
        pe.push(q_pe[i] as f32);
        alt.push(q_alt[i] as f32 / 100.0); // cm → m
        vn.push(q_vn[i] as f32 / 10.0); // 0.1 m/s → m/s
        ve.push(q_ve[i] as f32 / 10.0);
        vd.push(q_vd[i] as f32 / 10.0);
        roll.push(u8_to_deg(q_roll[i]));
        pitch.push(u8_to_deg(q_pitch[i]));
        yaw.push(u8_to_deg(q_yaw[i]));
    }

    (pn, pe, alt, vn, ve, vd, roll, pitch, yaw)
}

/// Quantize recovery (GPS) data arrays.
/// lat/lon in decimal degrees, alt in metres MSL.
#[pyfunction]
#[pyo3(signature = (lat_deg, lon_deg, alt_msl))]
pub fn quantize_recovery_array(
    lat_deg: Vec<f32>,
    lon_deg: Vec<f32>,
    alt_msl: Vec<f32>,
) -> (Vec<i32>, Vec<i32>, Vec<i16>) {
    let n = lat_deg.len();
    let mut qlat = Vec::with_capacity(n);
    let mut qlon = Vec::with_capacity(n);
    let mut qalt = Vec::with_capacity(n);

    for i in 0..n {
        qlat.push((lat_deg[i] * GPS_SCALING) as i32);
        qlon.push((lon_deg[i] * GPS_SCALING) as i32);
        qalt.push(clamp_i16(alt_msl[i].round() as i32));
    }

    (qlat, qlon, qalt)
}

/// Dequantize recovery data back to f32 arrays.
#[pyfunction]
pub fn dequantize_recovery_array(
    qlat: Vec<i32>,
    qlon: Vec<i32>,
    qalt: Vec<i16>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = qlat.len();
    let mut lat = Vec::with_capacity(n);
    let mut lon = Vec::with_capacity(n);
    let mut alt = Vec::with_capacity(n);

    for i in 0..n {
        lat.push(qlat[i] as f32 / GPS_SCALING);
        lon.push(qlon[i] as f32 / GPS_SCALING);
        alt.push(qalt[i] as f32);
    }

    (lat, lon, alt)
}

// =========================================================================
// Helpers
// =========================================================================

#[inline]
fn clamp_i16(v: i32) -> i16 {
    v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// Map degrees (0..360, wrapping) → u8 (0..255)
#[inline]
fn deg_to_u8(deg: f32) -> u8 {
    let normalised = ((deg % 360.0) + 360.0) % 360.0; // ensure positive
    ((normalised / 360.0) * 255.0).round() as u8
}

/// Map u8 (0..255) → degrees (0..360)
#[inline]
fn u8_to_deg(v: u8) -> f32 {
    (v as f32 / 255.0) * 360.0
}
