use super::ReducePlan;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ReduceCudaPlan {
    indices: CudaSlice<usize>,
}

impl ReduceCudaPlan {
    pub fn from_reduce_plan(
        reduce_plan: ReducePlan,
        dev: Arc<CudaDevice>,
    ) -> Result<ReduceCudaPlan, DriverError> {
        let slice = dev.htod_sync_copy(&reduce_plan.indices)?;

        Ok(ReduceCudaPlan { indices: slice })
    }
}
