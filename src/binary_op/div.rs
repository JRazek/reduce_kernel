use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, CudaView, CudaViewMut, DevicePtrMut, DeviceRepr,
    DriverError, LaunchAsync, LaunchConfig, ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use crate::kernel::{load_and_get_kernel, Kernel};

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/div.ptx"));

#[derive(Debug, Copy, Clone)]
pub(crate) struct DivOp;

unsafe impl Kernel<f32> for DivOp {
    const MODULE_NAME: &'static str = "kernel_ops";
    const FN_NAME: &'static str = "div_f32";

    fn ptx(&self) -> Ptx {
        todo!()
    }
}
