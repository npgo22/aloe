//! # Aloe Core
//!
//! Core rocket flight estimation library designed for `no_std` environments.
//! This crate contains only the code that needs to run on microcontrollers:
//! - Error-State Kalman Filter (ESKF)
//! - Telemetry quantization/dequantization
//! - Flight state machine
//! - Atmosphere lookup tables
//!
//! # Features
//! - `std`: Enable standard library support (for testing)
//! - Default: `no_std` with no allocations (bare metal embedded)

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

pub mod eskf;
pub mod lut_data;
pub mod quantize;
pub mod state_machine;

// Re-export core types
pub use eskf::{EskfTuning, NominalState, RocketEsKf};
pub use quantize::{deg_to_u8, u8_to_deg, FlightData};
pub use state_machine::{FlightState, StateMachine, NUM_STAGES};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
