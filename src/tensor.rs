use cudarc::driver::{
    CudaDevice, CudaSlice, CudaView, DeviceRepr, DriverError, LaunchAsync, LaunchConfig,
};

use std::sync::Arc;

//very simplified. Should rather abstract shape to more generic - both for static and dynamic.
pub struct Tensor<T, const N: usize> {
    pub dev: Arc<CudaDevice>,
    pub data: CudaSlice<T>,
    pub shape: [usize; N],
}
