use zerocopy::IntoBytes;
// ---------------------------------------------------------------------------
// 1. Quantization Helpers (The missing piece)
// ---------------------------------------------------------------------------

/// Maps 0..360 degrees to 0..255 (u8).
///
/// We scale by 256/360 so that the full u8 range is used.
/// 0   -> 0
/// 180 -> 128
/// 360 -> 0 (Wraps)
#[inline]
pub fn deg_to_u8(deg: f32) -> u8 {
    // Normalize to 0..360 first to handle negative angles
    let rem = deg % 360.0;
    let pos = if rem < 0.0 { rem + 360.0 } else { rem };

    // Scale: [0, 360) -> [0, 256)
    (pos * (256.0 / 360.0)) as u8
}

/// Maps 0..255 (u8) back to 0..360 degrees.
#[inline]
pub fn u8_to_deg(v: u8) -> f32 {
    (v as f32 / 256.0) * 360.0
}

// ---------------------------------------------------------------------------
// 2. Packet Structs (Zero-Copy)
// ---------------------------------------------------------------------------

/// Full Resolution Flight Data (18 bytes).
/// Used as the "Keyframe" or "Header" in the compressed stream.
#[repr(C, packed)]
#[derive(zerocopy::IntoBytes, zerocopy::Immutable, Clone, Copy, Debug, PartialEq)]
pub struct FlightData {
    pub pos_n_m: i16,    // North Position (m)
    pub pos_e_m: i16,    // East Position (m)
    pub alt_agl_cm: i32, // Altitude AGL (cm) - High Res!
    pub vel_n_ds: i16,   // Velocity North (dm/s)
    pub vel_e_ds: i16,   // Velocity East (dm/s)
    pub vel_d_ds: i16,   // Velocity Down (dm/s)
    pub roll: u8,        // Quantized Roll (0-255)
    pub pitch: u8,       // Quantized Pitch (0-255)
    pub yaw: u8,         // Quantized Yaw (0-255)
    pub status: u8,      // Status Flags
}

impl FlightData {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pos_n: f64,
        pos_e: f64,
        alt: f64,
        vn: f64,
        ve: f64,
        vd: f64,
        r: f32,
        p: f32,
        y: f32,
        status: u8,
    ) -> Self {
        Self {
            pos_n_m: pos_n as i16,
            pos_e_m: pos_e as i16,
            alt_agl_cm: (alt * 100.0) as i32, // Convert m -> cm
            vel_n_ds: (vn * 10.0) as i16,     // Convert m/s -> dm/s
            vel_e_ds: (ve * 10.0) as i16,
            vel_d_ds: (vd * 10.0) as i16,
            roll: deg_to_u8(r),
            pitch: deg_to_u8(p),
            yaw: deg_to_u8(y),
            status,
        }
    }
}

/// Compressed Delta Frame (8 bytes).
/// Used for subsequent samples. Resolution is lowered to fit in i8.
#[repr(C, packed)]
#[derive(zerocopy::IntoBytes, zerocopy::Immutable, Clone, Copy, Debug)]
pub struct FlightDelta {
    pub d_pos_n: i8, // Delta North (m)
    pub d_pos_e: i8, // Delta East (m)
    pub d_alt_m: i8, // Delta Altitude (m) - Note: Lower res than FlightData!
    pub d_vel_n: i8, // Delta Vel North (dm/s)
    pub d_vel_e: i8, // Delta Vel East (dm/s)
    pub d_vel_d: i8, // Delta Vel Down (dm/s)
    pub roll: u8,    // Absolute Roll (0-255) - Rotates too fast to delta encode
    pub status: u8,  // Absolute Status
}

// ---------------------------------------------------------------------------
// 3. Batch Compressor
// ---------------------------------------------------------------------------

pub struct TelemetryBatcher {
    buffer: [u8; 250], // Max LoRa MTU
    offset: usize,
    last_state: Option<FlightData>,
}

impl Default for TelemetryBatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl TelemetryBatcher {
    pub fn new() -> Self {
        Self {
            buffer: [0u8; 250],
            offset: 0,
            last_state: None,
        }
    }

    pub fn push(&mut self, current: FlightData) -> bool {
        // A. First Sample -> Write Full Header
        if self.last_state.is_none() {
            if self.offset + 19 > self.buffer.len() {
                return false;
            }

            self.buffer[self.offset] = 0x00; // Tag: Full Frame
            self.offset += 1;

            self.buffer[self.offset..self.offset + 18].copy_from_slice(current.as_bytes());
            self.offset += 18;

            self.last_state = Some(current);
            return true;
        }

        // B. Subsequent Samples -> Try Delta Compression
        let prev = self.last_state.unwrap();

        // Calculate Deltas
        let d_pn = current.pos_n_m as i32 - prev.pos_n_m as i32;
        let d_pe = current.pos_e_m as i32 - prev.pos_e_m as i32;

        // Altitude Trick: Convert cm (prev) to m for delta check
        // We compare (current_cm / 100) - (prev_cm / 100)
        let d_alt = (current.alt_agl_cm / 100) - (prev.alt_agl_cm / 100);

        let d_vn = current.vel_n_ds as i32 - prev.vel_n_ds as i32;
        let d_ve = current.vel_e_ds as i32 - prev.vel_e_ds as i32;
        let d_vd = current.vel_d_ds as i32 - prev.vel_d_ds as i32;

        // Check if all deltas fit in i8 (±127)
        // We allow Roll/Status to change freely (they are absolute in Delta struct)
        if in_i8(d_pn) && in_i8(d_pe) && in_i8(d_alt) && in_i8(d_vn) && in_i8(d_ve) && in_i8(d_vd) {
            // Write Delta Frame (9 bytes including tag)
            if self.offset + 9 > self.buffer.len() {
                return false;
            }

            self.buffer[self.offset] = 0x01; // Tag: Delta Frame
            self.offset += 1;

            let delta = FlightDelta {
                d_pos_n: d_pn as i8,
                d_pos_e: d_pe as i8,
                d_alt_m: d_alt as i8,
                d_vel_n: d_vn as i8,
                d_vel_e: d_ve as i8,
                d_vel_d: d_vd as i8,
                roll: current.roll,     // Absolute!
                status: current.status, // Absolute!
            };

            self.buffer[self.offset..self.offset + 8].copy_from_slice(delta.as_bytes());
            self.offset += 8;
        } else {
            // Delta overflow (Rocket exploded/ejected?) -> Write Full Frame
            if self.offset + 19 > self.buffer.len() {
                return false;
            }

            self.buffer[self.offset] = 0x00; // Tag: Reset/Full
            self.offset += 1;

            self.buffer[self.offset..self.offset + 18].copy_from_slice(current.as_bytes());
            self.offset += 18;
        }

        self.last_state = Some(current);
        true
    }

    pub fn finalize(&mut self) -> &[u8] {
        &self.buffer[0..self.offset]
    }
}

fn in_i8(val: i32) -> bool {
    (-128..=127).contains(&val)
}
