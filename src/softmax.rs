use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use super::tensor::{Shape, Tensor};

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/softmax.ptx"));

pub fn softmax(
    tensor: Tensor<f32>,
    dev: Arc<CudaDevice>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !dev.has_func("kernel_ops", "softmax_f32") {
        let ptx = Ptx::from_src(PTX_SRC);
        dev.load_ptx(ptx, "kernel_ops", &["softmax_f32"])?;
    }

    let softmax_kernel = dev
        .get_func("kernel_ops", "softmax_f32")
        .ok_or("could not load softmax kernel")?;

    Ok(())
}
