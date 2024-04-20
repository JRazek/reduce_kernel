#[derive(Debug, Default, Copy, Clone)]
pub struct SumOp;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use crate::tensor::{Shape, Tensor};

use crate::reduce::reduce_cuda::ReduceCudaPlan;
use crate::reduce::reduce_cuda::ReduceOperator;

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/sum.ptx"));

impl ReduceOperator<f32> for SumOp {
    const MODULE_NAME: &'static str = "kernel_ops";
    const FN_NAME: &'static str = "sum_reduce_f32";

    fn ptx(&self) -> Ptx {
        Ptx::from_src(PTX_SRC)
    }
}

#[cfg(test)]
mod test {

    use super::*;

    use crate::reduce::reduce_cuda::reduce;

    #[test]
    pub fn test_sum_small() {
        return;
        let cuda_dev = CudaDevice::new(0).expect("could not create cuda device");

        let mut input_host = [0f32; 10];
        input_host[2] = 1.0;

        let input_dev = cuda_dev
            .htod_sync_copy(&input_host)
            .expect("could not alloc and copy");

        let mut output = cuda_dev.alloc_zeros(5).expect("could not alloc");

        unsafe {
            reduce(input_dev, 10, 2, &mut output, cuda_dev.clone(), SumOp)
                .expect("could not reduce");
        }

        let res = cuda_dev
            .dtoh_sync_copy(&mut output)
            .expect("could not copy to host");

        assert_eq!(res, vec![0.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    pub fn test_sum_large() {
        let cuda_dev = CudaDevice::new(0).expect("could not create cuda device");

        const N: usize = 100_000;
        let mut input_host = vec![0f32; N];
        input_host[N / 2 - 1] = 3f32;
        input_host[0] = 2f32;
        input_host[N - 1] = 4f32;

        let input_dev = cuda_dev
            .htod_sync_copy(&input_host)
            .expect("could not alloc and copy");

        let mut output = cuda_dev.alloc_zeros(2).expect("could not alloc");

        unsafe {
            reduce(input_dev, N, N / 2, &mut output, cuda_dev.clone(), SumOp)
                .expect("could not reduce");
        }

        let res = cuda_dev
            .dtoh_sync_copy(&mut output)
            .expect("could not copy to host");

        assert_eq!(res, vec![5f32, 4f32]);
    }

    #[test]
    pub fn test_sum1d() {
        let cuda_dev = CudaDevice::new(0).expect("could not create cuda device");

        const N: usize = 100_000;
        let mut input_host = vec![1f32; N];

        let input_dev = cuda_dev
            .htod_sync_copy(&input_host)
            .expect("could not alloc and copy");

        let mut output = cuda_dev.alloc_zeros(1).expect("could not alloc");

        unsafe {
            reduce(input_dev, N, N, &mut output, cuda_dev.clone(), SumOp)
                .expect("could not reduce");
        }

        let res = cuda_dev
            .dtoh_sync_copy(&mut output)
            .expect("could not copy to host");

        assert_eq!(res, vec![100_000f32]);
    }
}
