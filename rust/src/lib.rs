mod eskf;
mod lut_data;
mod quantize;

use pyo3::prelude::*;

#[pymodule]
fn aloe_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<eskf::PyRocketEsKf>()?;
    m.add_class::<quantize::PyFlightData>()?;
    m.add_class::<quantize::PyRecoveryData>()?;
    m.add_function(wrap_pyfunction!(eskf::run_eskf_on_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(quantize::quantize_flight_array, m)?)?;
    m.add_function(wrap_pyfunction!(quantize::quantize_recovery_array, m)?)?;
    m.add_function(wrap_pyfunction!(quantize::dequantize_flight_array, m)?)?;
    m.add_function(wrap_pyfunction!(quantize::dequantize_recovery_array, m)?)?;
    Ok(())
}
