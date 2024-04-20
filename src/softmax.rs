use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig,
    ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use super::tensor::{compute_strided_index, Shape, Tensor};

use crate::exp::ExpOp;
use crate::map_kernel::map_offsets_in_place;

use super::reduce::ops::max::MaxOp;
use super::reduce::ops::sum::SumOp;
use super::reduce::reduce_cuda::reduce;
use super::reduce::reduce_cuda::ReduceOperator;

use crate::reduce::ReducePlan;

#[derive(Debug, Clone)]
struct SoftmaxCudaPlan<T>
where
    T: DeviceRepr,
{
    reduce_plan: ReducePlan,

    input_tensor_size_workspace: CudaSlice<T>,
    output_tensor_size_workspace: CudaSlice<T>,
}

pub fn softmax(
    tensor: Tensor<f32>,
    dev: Arc<CudaDevice>,
    softmax_cuda_plan: SoftmaxCudaPlan<f32>,
) -> Result<(Tensor<f32>, SoftmaxCudaPlan<f32>), Box<dyn std::error::Error>> {
    let reduce_plan = softmax_cuda_plan.reduce_plan;

    //algorithm:
    //1. apply mappings given by the plan to the tensor. This will make all elements in a reduced dimensions in consecutive memory locations.
    //2. given these apply reduce

    let tensor_data = tensor.data;
    //    let mut input_tensor_size_workspace = softmax_cuda_plan.input_tensor_size_workspace;
    //
    //    dev.dtod_copy(&tensor_data, &mut input_tensor_size_workspace)?;
    //
    //    let input_offsets_permutation = softmax_cuda_plan.idx_to_input_offsets;
    //
    //    let input_tensor_len = softmax_cuda_plan.input_tensor_shape.elements_count();
    //    unsafe {
    //        map_offsets_in_place(
    //            &mut input_tensor_size_workspace, //this is assumed to be of size as below
    //            input_tensor_len as u32,
    //            dev.clone(),
    //            &input_offsets_permutation,
    //        )?;
    //
    //        reduce(
    //            input_tensor_size_workspace,
    //            input_tensor_len as usize,
    //            1,
    //            &mut input_tensor_size_workspace,
    //            dev.clone(),
    //            SumOp,
    //        )?;
    //    };

    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
}
