#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct MaxOp;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use crate::tensor::{Shape, Tensor};

use crate::reduce::reduce_cuda::ReduceCudaPlan;
use crate::reduce::reduce_cuda::ReduceOperator;

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/max.ptx"));

impl ReduceOperator<f32> for MaxOp {
    const MODULE_NAME: &'static str = "kernel_ops";
    const FN_NAME: &'static str = "max_reduce_f32";

    fn ptx(&self) -> Ptx {
        Ptx::from_src(PTX_SRC)
    }
}

#[cfg(test)]
mod test {

    use super::*;

    use crate::reduce::reduce_cuda::reduce;

    #[test]
    pub fn test_max_small() {
        let cuda_dev = CudaDevice::new(0).expect("could not create cuda device");

        let mut input_host = [0f32; 10];
        input_host[2] = 1.0;

        let input_dev = cuda_dev
            .htod_sync_copy(&input_host)
            .expect("could not alloc and copy");

        let mut output = cuda_dev.alloc_zeros(5).expect("could not alloc");

        unsafe {
            reduce(&input_dev, 10, 2, &mut output, cuda_dev.clone(), MaxOp)
                .expect("could not reduce");
        }

        let res = cuda_dev
            .dtoh_sync_copy(&mut output)
            .expect("could not copy to host");

        assert_eq!(res, vec![0.0, 1.0, 0.0, 0.0, 0.0]);

        //        let output_shape = Shape::new(vec![1, 3, 1]);
        //        let mut output_tensor: Tensor<f32> = Tensor::new(
        //            cuda_dev.clone(),
        //            cuda_dev
        //                .alloc_zeros(output_shape.elements_count() as usize)
        //                .expect("could not alloc"),
        //            output_shape,
        //        );
        //
        //        let reduce_cuda_plan: ReduceCudaPlan<f32> =
        //            ReduceCudaPlan::precompute(reduce_plan, cuda_dev.clone())
        //                .expect("could not create reduce cuda plan");
        //
        //        let tensor = max(
        //            &input_tensor,
        //            &mut output_tensor,
        //            cuda_dev.clone(),
        //            &reduce_cuda_plan,
        //        )
        //        .unwrap();
        //
        //        let res = tensor.as_vec().unwrap();
        //
        //        println!("{:?}", res);
        //
        panic!();
    }
}
