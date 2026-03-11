//! # Physical and Numerical Constants for Rocket Simulation
//!
//! This module defines all magic constants used throughout the 6-DOF rocket simulation.
//! Each constant is documented with its physical meaning, typical values, and mathematical
//! justification where applicable.
//!
//! ## Coordinate System
//!
//! The simulation uses the **NED (North-East-Down)** coordinate frame:
//! - **X-axis**: Points North (tangent to Earth's surface)
//! - **Y-axis**: Points East (tangent to Earth's surface)
//! - **Z-axis**: Points Down (toward Earth's center)
//!
//! Altitude is stored as $`h = -z`$, so positive altitude means negative Z.
//!
//! ## References
//!
//! - Barrowman, J. S. (1967). "The Practical Calculation of the Aerodynamic Characteristics
//!   of Slender Finned Vehicles"
//! - Niskanen, S. (2009). "OpenRocket Technical Documentation"

use std::f64::consts::PI;

// ============================================================================
// NUMERICAL INTEGRATION PARAMETERS
// ============================================================================

/// # Integration Timestep
///
/// Fixed timestep for RK4 integration: $`\Delta t = 0.001\text{ s} = 1\text{ ms}`$
///
/// ## Rationale
///
/// The timestep must satisfy the **Courant-Friedrichs-Lewy (CFL) condition** for
/// numerical stability. For a rocket with:
/// - Maximum velocity: $`v_{\max} \approx 600\text{ m/s}`$ (Mach 1.8)
/// - Characteristic length: $`L \approx 0.1\text{ m}`$ (rocket radius)
///
/// The CFL number is:
///
/// ```math
/// \text{CFL} = \frac{v_{\max} \Delta t}{L} = \frac{600 \times 0.001}{0.1} = 6.0
/// ```
///
/// While this exceeds the strict CFL limit (typically < 1), RK4's fourth-order
/// accuracy provides sufficient stability for this stiff ODE system.
///
/// ## Performance Considerations
///
/// At 1ms timestep:
/// - 100-second flight requires 100,000 integration steps
/// - Computational cost: ~0.1-1 second on modern CPU
/// - Memory: ~5-10 MB for full trajectory storage
///
/// Reducing to 0.0001s (100 μs) would increase cost 10×; increasing to 0.01s
/// causes visible integration errors in high-acceleration phases.
pub const DT: f64 = 0.001;

/// # Maximum Simulation Time
///
/// Upper bound on simulation duration: $`T_{\max} = 400\text{ s}`$
///
/// ## Rationale
///
/// Covers full flight envelope:
/// - Pad dwell: ~1 s
/// - Powered ascent: ~5-15 s
/// - Coast to apogee: ~30-120 s
/// - Descent: ~60-240 s (parachute dependent)
///
/// For high-altitude flights (>30 km), descent can take 200+ seconds.
pub const MAX_TIME: f64 = 400.0;

// ============================================================================
// ATMOSPHERIC MODEL PARAMETERS
// ============================================================================

/// # Atmospheric Scale Height
///
/// Exponential density decay constant: $`H = 7400\text{ m}`$
///
/// ## Mathematical Model
///
/// The atmospheric density follows an exponential profile:
///
/// ```math
/// \rho(h) = \rho_0 \exp\left(-\frac{h}{H}\right)
/// ```
///
/// where:
/// - $`\rho_0 = 1.225\text{ kg/m}^3`$ (sea level density)
/// - $`h`$ is altitude above sea level (m)
/// - $`H = 7400\text{ m}`$ is the scale height
///
/// ## Physical Basis
///
/// This is derived from the barometric formula assuming:
/// - Isothermal atmosphere at $`T = 288.15\text{ K}`$
/// - Ideal gas law: $`p = \rho R T`$
/// - Hydrostatic equilibrium: $`\frac{dp}{dh} = -\rho g`$
///
/// Combining these gives:
///
/// ```math
/// H = \frac{RT}{gM} = \frac{8.314 \times 288.15}{9.80665 \times 0.029} \approx 7400\text{ m}
/// ```
///
/// where $`M = 0.029\text{ kg/mol}`$ is the molar mass of air.
///
/// ## Accuracy
///
/// This simple model is accurate to within 10% up to 10 km altitude.
/// For higher altitudes, use the U.S. Standard Atmosphere 1976 model.
pub const H_SCALE: f64 = 7400.0;

// ============================================================================
// AERODYNAMIC CALCULATION THRESHOLDS
// ============================================================================

/// # Minimum Velocity for Aerodynamic Forces
///
/// Cutoff velocity: $`v_{\min} = 0.1\text{ m/s}`$
///
/// ## Rationale
///
/// Below this velocity:
/// - Dynamic pressure $`q = \frac{1}{2}\rho v^2 < 0.01\text{ Pa}`$ (negligible)
/// - Angle of attack $`\alpha = \arcsin(v_\perp / v)`$ becomes numerically unstable
/// - Aerodynamic forces are dominated by numerical noise
///
/// This threshold prevents:
/// 1. **Division by near-zero** in $`\alpha = v_\perp / v`$
/// 2. **Spurious torques** when $`v \approx 0`$ on the launch pad
/// 3. **Numerical drift** in quaternion integration from near-zero angular rates
pub const MIN_AERO_VELOCITY: f64 = 0.1;

/// # Minimum Angle for Quaternion Rotation
///
/// Threshold for quaternion updates: $`\theta_{\min} = 10^{-9}\text{ rad}`$
///
/// ## Mathematical Justification
///
/// Quaternion rotation about axis $`\hat{n}`$ by angle $`\theta`$:
///
/// ```math
/// q = \cos\frac{\theta}{2} + \hat{n}\sin\frac{\theta}{2}
/// ```
///
/// For small $`\theta`$, Taylor expansion gives:
///
/// ```math
/// \sin\frac{\theta}{2} \approx \frac{\theta}{2} - \frac{\theta^3}{48} + O(\theta^5)
/// ```
///
/// At $`\theta = 10^{-9}`$:
/// - $`\sin(\theta/2) \approx 5 \times 10^{-10}`$
/// - Third-order error: $`\sim 10^{-27}`$ (below machine epsilon for `f64`)
///
/// Below this threshold, quaternion updates are skipped to avoid:
/// 1. **Accumulation of floating-point rounding errors**
/// 2. **Unnecessary normalization operations**
/// 3. **Numerical cancellation** in $`\cos(\theta/2) \approx 1`$
pub const MIN_ROTATION_ANGLE: f64 = 1e-9;

// ============================================================================
// GROUND INTERACTION PARAMETERS
// ============================================================================

/// # Ground Level in NED Coordinates
///
/// Launch pad altitude: $`z_{\text{ground}} = 0\text{ m}`$
///
/// ## Convention
///
/// In NED coordinates, the ground is defined as $`z = 0`$:
/// - Positive Z: Below ground (into Earth)
/// - Negative Z: Above ground (into sky)
/// - Altitude: $`h = -z`$
///
/// The simulation enforces $`z \leq 0`$ at all times after launch.
pub const GROUND_LEVEL: f64 = 0.0;

/// # Ground Collision Damping Coefficient
///
/// Velocity damping on ground impact: $`\beta = 0.9`$
///
/// ## Force Model
///
/// When the rocket touches ground ($`z \geq 0`$), apply damping force:
///
/// ```math
/// F_{\text{damp}} = -\beta \frac{m v_z}{\Delta t}
/// ```
///
/// This models inelastic collision without explicit coefficient of restitution.
/// The damping prevents the rocket from bouncing indefinitely due to numerical
/// integration errors.
///
/// ## Physical Interpretation
///
/// $`\beta = 0.9`$ corresponds to ~90% energy dissipation per contact, simulating
/// a soft landing with crushable structure or parachute touchdown.
pub const GROUND_DAMPING: f64 = 0.9;

// ============================================================================
// AERODYNAMIC DAMPING COEFFICIENTS
// ============================================================================

/// # Pitch/Yaw Damping Coefficient
///
/// Aerodynamic moment damping: $`C_{d,\text{p/y}} = 0.05`$
///
/// ## Mathematical Model
///
/// The damping torque opposes rotational velocity:
///
/// ```math
/// \tau_{\text{damp}} = -C_{d} \cdot q \cdot S \cdot L^2 \cdot \omega
/// ```
///
/// where:
/// - $`q = \frac{1}{2}\rho v^2`$: Dynamic pressure
/// - $`S`$: Reference area (cross-section)
/// - $`L = x_{CP} - x_{CG}`$: Moment arm (CP-CG distance)
/// - $`\omega`$: Angular velocity (rad/s)
///
/// ## Physical Basis
///
/// Fin-induced damping arises from:
/// 1. **Lift force lag**: Fins experience angle of attack $`\alpha + \frac{L\omega}{v}`$
/// 2. **Pressure asymmetry**: Rotating fins see different dynamic pressures
///
/// For typical model rockets with 3-4 fins:
/// - $`C_{d,\text{p/y}} \approx 0.01`$ to $`0.1`$
/// - Larger fins → higher damping
///
/// The value 0.05 is conservative, preventing over-damped response while
/// ensuring convergence to equilibrium attitude.
pub const PITCH_YAW_DAMPING_COEFF: f64 = 0.05;

/// # Roll Damping Coefficient
///
/// Aerodynamic roll damping: $`C_{d,\text{roll}} = 0.002`$
///
/// ## Rationale
///
/// Roll damping is **much weaker** than pitch/yaw damping because:
/// 1. Fins are thin in the roll direction
/// 2. No significant pressure differential for axial rotation
/// 3. Skin friction is the primary roll damping mechanism
///
/// The damping torque is:
///
/// ```math
/// \tau_{\text{roll}} = -C_{d,\text{roll}} \cdot q \cdot S \cdot r^2 \cdot \omega_x
/// ```
///
/// where $`r = \sqrt{S/\pi}`$ is the rocket radius.
///
/// Typical ratio: $`C_{d,\text{roll}} / C_{d,\text{p/y}} \approx 0.04`$
///
/// This allows spin-stabilized rockets to maintain high roll rates (10+ Hz)
/// while still providing eventual spin-down over long flight times.
pub const ROLL_DAMPING_COEFF: f64 = 0.002;

// ============================================================================
// PHYSICAL CONSTANTS
// ============================================================================

/// # Standard Gravity
///
/// Gravitational acceleration: $`g_0 = 9.80665\text{ m/s}^2`$
///
/// ## Definition
///
/// This is the **standard gravity** defined by the International Committee for
/// Weights and Measures (CIPM) in 1901. It represents the nominal acceleration
/// due to gravity at Earth's surface.
///
/// ## Variation
///
/// Actual gravity varies with:
/// - **Latitude**: $`g(\phi) = 9.780 + 0.052\sin^2\phi\text{ m/s}^2`$
/// - **Altitude**: $`g(h) = g_0(1 - 2h/R_E)`$ where $`R_E = 6.371 \times 10^6\text{ m}`$
///
/// For hobby rocketry (altitudes <50 km), variation is <0.4%, so we use the
/// constant $`g_0`$.
pub const STANDARD_GRAVITY: f64 = 9.80665;

/// # Speed of Sound at Sea Level
///
/// Sound speed in air: $`c_0 = 343\text{ m/s}`$
///
/// ## Formula
///
/// For an ideal gas:
///
/// ```math
/// c = \sqrt{\gamma R T / M}
/// ```
///
/// where:
/// - $`\gamma = 1.4`$ (heat capacity ratio for air)
/// - $`R = 8.314\text{ J/(mol·K)}`$ (universal gas constant)
/// - $`T = 288.15\text{ K}`$ (15°C, ISA sea level)
/// - $`M = 0.029\text{ kg/mol}`$ (molar mass of air)
///
/// This gives $`c_0 \approx 343\text{ m/s}`$ (Mach 1).
///
/// ## Altitude Dependence
///
/// In the troposphere (0-11 km), temperature decreases with altitude:
///
/// ```math
/// T(h) = T_0 - \Lambda h, \quad \Lambda = 6.5\text{ K/km}
/// ```
///
/// Thus:
///
/// ```math
/// c(h) = c_0 \sqrt{1 - \frac{\Lambda h}{T_0}}
/// ```
///
/// For simplicity, we use constant $`c_0`$ as Mach number variations are
/// second-order effects for subsonic/low-supersonic flights.
pub const SPEED_OF_SOUND: f64 = 343.0;

// ============================================================================
// DEFAULT SIMULATION PARAMETERS
// ============================================================================

/// Default rocket parameters for testing and validation.
///
/// These values represent a typical high-power model rocket:
/// - **Class**: M-class motor (~5000 N·s total impulse)
/// - **Mass**: 30 kg (20 kg dry + 10 kg propellant)
/// - **Diameter**: 15 cm (6 inch)
/// - **Length**: ~3 m
/// - **Apogee**: ~3-4 km
pub struct DefaultRocket;

impl DefaultRocket {
    /// Reference area for aerodynamic calculations: $`S = \pi r^2`$
    ///
    /// For a 15 cm diameter rocket:
    ///
    /// ```math
    /// S = \pi (0.075)^2 \approx 0.0177\text{ m}^2
    /// ```
    pub const REF_AREA: f64 = PI * 0.075_f64 * 0.075_f64;

    /// Default axial drag coefficient: $`C_D = 0.5`$
    ///
    /// Typical range for rockets:
    /// - Sleek design with boat tail: $`C_D \approx 0.3`$
    /// - Blunt nose or base drag: $`C_D \approx 0.7`$
    /// - Well-designed hobby rocket: $`C_D \approx 0.4-0.6`$
    pub const DRAG_COEFF: f64 = 0.5;

    /// Default normal force coefficient: $`C_N = 1.2`$
    ///
    /// This is the **normal force coefficient** (not CNα per radian).
    /// The actual force is:
    ///
    /// ```math
    /// F_N = C_N \cdot q \cdot S \cdot \sin\alpha
    /// ```
    ///
    /// For small angles: $`\sin\alpha \approx \alpha`$ (radians), so
    /// $`C_N \approx C_{N\alpha}`$ when $`\alpha`$ is in radians.
    ///
    /// Typical values for finned rockets:
    /// - 3 fins: $`C_{N\alpha} \approx 8-12\text{ rad}^{-1}`$
    /// - 4 fins: $`C_{N\alpha} \approx 10-15\text{ rad}^{-1}`$
    ///
    /// The reduced value (1.2) accounts for:
    /// 1. Numerical stability with wind
    /// 2. Potential fin flutter/stall at high speeds
    /// 3. Conservative estimate to avoid over-torquing
    pub const NORMAL_FORCE_COEFF: f64 = 1.2;

    /// Default launch rod length: $`L_{\text{rail}} = 3.0\text{ m}`$
    ///
    /// The launch rod constrains the rocket's attitude until:
    ///
    /// ```math
    /// h = -z \geq L_{\text{rail}}
    /// ```
    ///
    /// Typical values:
    /// - Small rockets (A-D motors): 0.5-1.0 m
    /// - Mid-power (E-G motors): 1.5-2.5 m
    /// - High-power (H+ motors): 2.5-5.0 m
    ///
    /// Longer rails ensure:
    /// 1. Higher exit velocity → better aerodynamic stability
    /// 2. Less weathercocking in crosswinds
    /// 3. Reduced pitch-over on rail exit
    ///
    /// Rule of thumb: $`L_{\text{rail}} \geq 2 \times L_{\text{rocket}}`$
    pub const LAUNCH_ROD_LENGTH: f64 = 3.0;

    /// Default specific impulse: $`I_{sp} = 200\text{ s}`$
    ///
    /// Specific impulse relates thrust to mass flow:
    ///
    /// ```math
    /// \dot{m} = \frac{F}{I_{sp} \cdot g_0}
    /// ```
    ///
    /// Typical values by propellant type:
    /// - Black powder (A-D): 80-100 s
    /// - APCP composite (E-G): 150-200 s
    /// - High-performance APCP (H+): 200-250 s
    /// - Hybrid (N₂O/HTPB): 230-260 s
    /// - Liquid (LOX/RP-1): 300-350 s
    pub const ISP: f64 = 200.0;
}

#[cfg(test)]
#[allow(clippy::assertions_on_constants)]
mod tests {
    use super::*;

    #[test]
    fn test_physical_constants() {
        // Verify standard gravity is within Earth's range
        assert!(STANDARD_GRAVITY > 9.76 && STANDARD_GRAVITY < 9.84);

        // Verify speed of sound is reasonable for 15°C
        assert!(SPEED_OF_SOUND > 340.0 && SPEED_OF_SOUND < 346.0);
    }

    #[test]
    fn test_atmospheric_model() {
        // At h=0: ρ(0) = ρ₀
        let rho_sea_level = 1.225;
        let h = 0.0;
        let rho = rho_sea_level * (-h / H_SCALE).exp();
        assert!((rho - rho_sea_level).abs() < 1e-10);

        // At h=H: ρ(H) = ρ₀/e
        let h = H_SCALE;
        let rho = rho_sea_level * (-h / H_SCALE).exp();
        let expected = rho_sea_level / std::f64::consts::E;
        assert!((rho - expected).abs() / expected < 1e-6);

        // At h=10km: ρ ≈ 0.32 kg/m³ (calculated: 1.225 * exp(-10000/7400))
        let h = 10000.0;
        let rho = rho_sea_level * (-h / H_SCALE).exp();
        assert!(rho > 0.30 && rho < 0.35); // Rough check
    }

    #[test]
    fn test_dt_cfl_condition() {
        // Verify timestep is reasonable for expected velocities
        let v_max = 600.0; // m/s (Mach 1.8)
        let char_length = 0.1; // m (rocket radius)
        let cfl = v_max * DT / char_length;

        // CFL should be in reasonable range for RK4
        assert!(cfl > 1.0 && cfl < 10.0);
    }

    #[test]
    fn test_default_parameters() {
        // Verify reference area calculation
        let diameter = 0.15_f64; // 15 cm
        let expected_area = PI * (diameter / 2.0).powi(2);
        assert!((DefaultRocket::REF_AREA - expected_area).abs() / expected_area < 1e-6);

        // Verify drag coefficient is in typical range
        assert!(DefaultRocket::DRAG_COEFF > 0.2 && DefaultRocket::DRAG_COEFF < 1.0);

        // Verify Isp is reasonable for hobby rocketry
        assert!(DefaultRocket::ISP > 50.0 && DefaultRocket::ISP < 400.0);
    }
}
