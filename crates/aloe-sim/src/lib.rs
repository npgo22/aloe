//! Aloe Simulation Library
//!
//! Provides rocket flight simulation and sensor modeling capabilities.

pub mod filter;
pub mod params;
pub mod sensor;
pub mod sim;

// Re-export main types
pub use filter::{run_filter, run_filter_default, FilterConfig, FilterResult};
pub use params::*;
pub use sensor::{generate_sensor_data, SensorConfig, SensorData};
pub use sim::{simulate_6dof, RocketParams, SimResult};
