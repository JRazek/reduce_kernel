use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig,
    ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use super::tensor::{compute_strided_offset, Shape, Tensor};

use crate::map_kernel::map_offsets_in_place;

use super::reduce::ops::max::MaxOp;
use super::reduce::ops::sum::SumOp;
use super::reduce::reduce_cuda::reduce;
use super::reduce::reduce_cuda::ReduceOperator;

use crate::kernel::{load_and_get_kernel, Kernel};

pub mod div;
pub mod sub;

pub(crate) unsafe trait BinaryOperator<T>: Kernel<T>
where
    T: DeviceRepr,
{
}

pub(crate) unsafe fn apply_in_place<Op>(
    rhs: &mut CudaSlice<f32>,
    lhs: &CudaSlice<f32>,
    len: usize,
    dev: &Arc<CudaDevice>,
    op: Op,
) -> Result<(), Box<dyn std::error::Error>>
where
    Op: BinaryOperator<f32>,
{
    let kernel = load_and_get_kernel::<Op>(&dev, op)?;

    const MAX_BLOCK_SIZE: usize = 1024;

    let block_size = MAX_BLOCK_SIZE.min(len);

    let grid_size = (len.div_ceil(block_size)) as u32;

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let params = (rhs, lhs, len as u32);

    kernel.launch(cfg, params)?;

    Ok(())
}
