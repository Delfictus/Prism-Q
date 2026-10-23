#![warn(clippy::all)]

#[path = "../../prism-nhs/src/dstw_dispatch/types.rs"]
pub mod nhs_types;

pub mod core;
pub mod ffi;
pub mod reactions;
pub mod scoring;

pub use nhs_types::{HitCount, ResidueIdx, VoxelIdx};

use pyo3::prelude::*;

#[pymodule]
fn prism_forge(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        ffi::kinematics_bridge::execute_3d_reaction,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        ffi::reward_bridge::compute_thermodynamic_reward_3d,
        module
    )?)?;
    Ok(())
}
