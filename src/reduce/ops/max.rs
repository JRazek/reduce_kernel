#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct MaxOp;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use crate::tensor::{Shape, Tensor};

use crate::reduce::reduce_cuda::ReduceCudaPlan;

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/max.ptx"));

pub fn max(
    input_tensor: &Tensor<f32>,
    output_tensor: &mut Tensor<f32>,
    dev: Arc<CudaDevice>,
    reduce_plan: &ReduceCudaPlan<f32>,
) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
    let tensor_len = input_tensor.shape.elements_count() as u32;

    assert!((1..32).contains(&tensor_len));

    if !dev.has_func("kernel_ops", "max_reduce_f32") {
        let ptx = Ptx::from_src(PTX_SRC);
        dev.load_ptx(ptx, "kernel_ops", &["max_reduce_f32"])?;
    }

    let max_kernel = dev
        .get_func("kernel_ops", "max_reduce_f32")
        .ok_or("could not load max kernel")?;

    //    let remainder_block_len = reduced_subtensor_len % block_len;

    todo!()
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::reduce::ReduceConfig;
    use crate::reduce::ReducePlan;

    #[test]
    pub fn test_max_small() {
        let cuda_dev = CudaDevice::new(0).expect("could not create cuda device");

        let mut data = [0f32; 24];

        data[0] = 1f32;
        data[12] = 1f32;

        data[4] = 2f32;
        data[15] = 2f32;
        let input_tensor = Tensor::new(
            cuda_dev.clone(),
            cuda_dev.htod_sync_copy(&data).unwrap(),
            Shape::new(vec![2, 3, 4]),
        );

//        let reduce_plan = ReducePlan::precompute(ReduceConfig {
//            tensor_shape: input_tensor.shape.clone(),
//            reduce_dims: vec![0, 2],
//        });
//
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
