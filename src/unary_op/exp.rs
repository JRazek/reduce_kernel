use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use crate::kernel::{load_and_get_kernel, Kernel};
use crate::unary_op::UnaryOperator;

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/exp.ptx"));

#[derive(Debug, Copy, Clone)]
pub(crate) struct ExpOp;

unsafe impl Kernel<f32> for ExpOp {
    const MODULE_NAME: &'static str = "kernel_ops";
    const FN_NAME: &'static str = "exp_f32";

    fn ptx(&self) -> Ptx {
        Ptx::from_src(PTX_SRC)
    }
}

unsafe impl UnaryOperator<f32> for ExpOp {}

#[cfg(test)]
mod test {

    use cudarc::driver::{
        CudaDevice, CudaFunction, CudaSlice, CudaView, CudaViewMut, DevicePtrMut, DeviceRepr,
        DriverError, LaunchAsync, LaunchConfig, ValidAsZeroBits,
    };

    use crate::unary_op::apply_in_place;

    use super::ExpOp;

    #[test]
    fn test_exp() {
        let cuda_dev = CudaDevice::new(0).unwrap();
        let mut data = vec![1.0, 2.0, 3.0, 4.0];

        let mut input = cuda_dev.htod_sync_copy(&data).unwrap();

        unsafe {
            apply_in_place(&mut input, data.len(), &cuda_dev, ExpOp).unwrap();
        }

        let output = cuda_dev.dtoh_sync_copy(&input).unwrap();

        for (a, b) in data.iter().zip(output.iter()) {
            assert!((a.exp() - b).abs() < 1e-6);
        }
    }
}
