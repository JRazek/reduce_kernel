use super::ReducePlan;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ReduceCudaPlan {
    indices: CudaSlice<usize>,
}

unsafe impl DeviceRepr for &ReduceCudaPlan {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&(self.indices)).as_kernel_param()
    }
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
