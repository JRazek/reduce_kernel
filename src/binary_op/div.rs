use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, CudaView, CudaViewMut, DevicePtrMut, DeviceRepr,
    DriverError, LaunchAsync, LaunchConfig, ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use crate::kernel::{load_and_get_kernel, Kernel};

use crate::binary_op::BinaryOperator;

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/div.ptx"));

#[derive(Debug, Copy, Clone)]
pub(crate) struct DivOp;

unsafe impl Kernel<f32> for DivOp {
    const MODULE_NAME: &'static str = "kernel_ops";
    const FN_NAME: &'static str = "div_f32";

    fn ptx(&self) -> Ptx {
        Ptx::from_src(PTX_SRC)
    }
}

unsafe impl BinaryOperator<f32> for DivOp {}

#[cfg(test)]
mod test {

    use cudarc::driver::{
        CudaDevice, CudaFunction, CudaSlice, CudaView, CudaViewMut, DevicePtrMut, DeviceRepr,
        DriverError, LaunchAsync, LaunchConfig, ValidAsZeroBits,
    };

    use crate::binary_op::apply_in_place;

    use super::DivOp;

    #[test]
    fn test_sub() {
        let cuda_dev = CudaDevice::new(0).unwrap();
        let mut lhs_host = vec![1.0, 2.0, 3.0, 4.0];
        let mut rhs_host = vec![2.1, 4.0, 3.0, 2.2];

        let mut lhs = cuda_dev.htod_sync_copy(&lhs_host).unwrap();
        let rhs = cuda_dev.htod_sync_copy(&rhs_host).unwrap();

        unsafe {
            apply_in_place(&mut lhs, &rhs, lhs_host.len(), &cuda_dev, DivOp).unwrap();
        }

        let output = cuda_dev.dtoh_sync_copy(&lhs).unwrap();

        let expected = lhs_host.iter().zip(rhs_host.iter()).map(|(a, b)| a / b);

        for (x, y) in output.iter().zip(expected) {
            assert!((x - y).abs() < 1e-6);
        }
    }
}
