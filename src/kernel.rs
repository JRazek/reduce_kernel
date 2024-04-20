use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, CudaView, CudaViewMut, DevicePtrMut, DeviceRepr,
    DriverError, LaunchAsync, LaunchConfig, ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

pub(crate) fn load_and_get_kernel<K>(
    dev: &Arc<CudaDevice>,
    kernel: K,
) -> Result<CudaFunction, Box<dyn std::error::Error>>
where
    K: Kernel<f32>,
{
    if !dev.has_func(K::MODULE_NAME, K::FN_NAME) {
        let ptx = kernel.ptx();
        dev.load_ptx(ptx, K::MODULE_NAME, &[K::FN_NAME])?;
    }

    let function = dev
        .get_func(K::MODULE_NAME, K::FN_NAME)
        .ok_or("could not load kernel")?;

    Ok(function)
}

pub(crate) unsafe trait Kernel<T>
where
    T: DeviceRepr,
{
    const MODULE_NAME: &'static str;
    const FN_NAME: &'static str;

    fn ptx(&self) -> Ptx;
}
