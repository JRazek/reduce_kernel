pub mod ops;
pub mod reduce_cuda;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use super::tensor::{compute_strided_index, Shape, Tensor};

pub struct ReduceConfig {
    pub tensor_shape: Shape,
    pub reduce_dims: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct ReducePlan {
    pub(crate) tensor_shape: Shape,
    pub(crate) reduce_tensors_shape: Shape,
    pub(crate) non_reduce_tensor_shape: Shape,
    pub(crate) idx_to_input_offset: Vec<usize>,
    pub(crate) idx_to_output_offset_2nd_step: Vec<usize>,
}
