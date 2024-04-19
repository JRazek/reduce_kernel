use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use super::tensor::{Shape, Tensor};

use super::reduce::reduce_cuda::ReduceCudaPlan;

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/softmax.ptx"));

pub trait SoftmaxKernelDtype: DeviceRepr {}

pub fn softmax(
    tensor: Tensor<f32>,
    dev: Arc<CudaDevice>,
    reduce_plan: &ReduceCudaPlan,
) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
    assert_eq!(tensor.shape, reduce_plan.reduce_plan.tensor_shape);

    let tensor_len = tensor.shape.elements_count() as u32;

    assert!((1..32).contains(&tensor_len));

    if !dev.has_func("kernel_ops", "softmax_f32") {
        let ptx = Ptx::from_src(PTX_SRC);
        dev.load_ptx(ptx, "kernel_ops", &["softmax_f32"])?;
    }

    let softmax_kernel = dev
        .get_func("kernel_ops", "softmax_f32")
        .ok_or("could not load softmax kernel")?;

    let reduced_subtensor_len = reduce_plan
        .reduce_plan
        .reduce_tensors_shape
        .elements_count() as u32;
    const MAX_BLOCK_LEN: u32 = 32;

    let block_len: u32 = MAX_BLOCK_LEN.min(reduced_subtensor_len);
    let grid_len = (tensor_len.div_ceil(block_len)) as u32; //should be no remained as
                                                            //for now

    println!("block_len: {}, grid_len: {}", block_len, grid_len);
    let type_len = std::mem::size_of::<f32>() as u32;

    //each tensor to reduce is of size
    let cfg = LaunchConfig {
        block_dim: (block_len, 1, 1),
        grid_dim: (grid_len, 1, 1),
        shared_mem_bytes: type_len * block_len,
    };

    let params = (&tensor.data, &tensor.data, reduce_plan);

    unsafe { softmax_kernel.launch(cfg, params)? };

    Ok(tensor)
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::reduce::ReduceConfig;
    use crate::reduce::ReducePlan;

    #[test]
    pub fn test_softmax_small() {
        let cuda_dev = CudaDevice::new(0).expect("could not create cuda device");

        let mut data = [0f32; 24];

        data[0] = 1f32;
        data[12] = 1f32;

        data[4] = 2f32;
        data[15] = 2f32;
        let tensor = Tensor::new(
            cuda_dev.clone(),
            cuda_dev.htod_sync_copy(&data).unwrap(),
            Shape::new(vec![2, 3, 4]),
        );

        let reduce_plan = ReducePlan::precompute(ReduceConfig {
            tensor_shape: tensor.shape.clone(),
            reduce_dims: vec![0, 2],
        });

        println!("indices: {:?}", reduce_plan.indices);

        let reduce_cuda_plan = ReduceCudaPlan::from_reduce_plan(reduce_plan, cuda_dev.clone())
            .expect("could not create reduce cuda plan");

        let tensor = softmax(tensor, cuda_dev.clone(), &reduce_cuda_plan).unwrap();

        let res = tensor.as_vec().unwrap();

        println!("{:?}", res);

        panic!();
    }

    //    #[test]
    pub fn test_softmax() {
        let cuda_dev = CudaDevice::new(0).expect("could not create cuda device");

        let mut data = [0f32; 64];
        data[31] = 100f32;
        let tensor = Tensor::new(
            cuda_dev.clone(),
            cuda_dev.htod_sync_copy(&data).unwrap(),
            Shape::new(vec![64]),
        );

        let reduce_plan = ReducePlan::precompute(ReduceConfig {
            tensor_shape: tensor.shape.clone(),
            reduce_dims: vec![0],
        });

        let reduce_cuda_plan = ReduceCudaPlan::from_reduce_plan(reduce_plan, cuda_dev.clone())
            .expect("could not create reduce cuda plan");

        let tensor = softmax(tensor, cuda_dev.clone(), &reduce_cuda_plan).unwrap();

        let res = tensor.as_vec().unwrap();

        println!("{:?}", res);

        panic!();
    }
}
