use super::{ReducePlan, Tensor};
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig,
    ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ReduceCudaPlan<T>
where
    T: DeviceRepr,
{
    pub(crate) idx_to_input_offset: CudaSlice<usize>,
    pub(crate) workspace_dev: CudaSlice<T>,
    pub(crate) reduce_plan: ReducePlan,

    //for initial reduction
    block_len: u32,
    blocks_per_reduce_tensor_count: u32,
    blocks_total: u32,
}

unsafe impl<T> DeviceRepr for &ReduceCudaPlan<T>
where
    T: DeviceRepr,
{
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&(self.idx_to_input_offset)).as_kernel_param()
    }
}

impl<T> ReduceCudaPlan<T>
where
    T: DeviceRepr + ValidAsZeroBits,
{
    pub fn from_reduce_plan(
        reduce_plan: ReducePlan,
        dev: Arc<CudaDevice>,
    ) -> Result<ReduceCudaPlan<T>, DriverError> {
        let slice = dev.htod_sync_copy(&reduce_plan.idx_to_input_offset)?;

        const MAX_GRID_LEN: u32 = 2 << 31 - 1;
        const MAX_BLOCK_LEN: u32 = 2 << 10;

        let reduced_subtensor_len = reduce_plan.reduce_tensors_shape.elements_count() as u32;

        let block_len: u32 = MAX_BLOCK_LEN.min(reduced_subtensor_len);
        let blocks_per_reduce_tensor_count = (reduced_subtensor_len.div_ceil(block_len)) as u32;

        let workspace_len = blocks_per_reduce_tensor_count
            * reduce_plan.non_reduce_tensor_shape.elements_count() as u32;

        //reduce will happen on 2 levels.
        //one for each block in a grid there will be a reduction to a single value.
        //second, for each block corresponding to the same reduced tensor, there will be a reduction to a single value.
        //for 1st level, we need to store intermediate results in memory. Thus the workspace.
        let workspace_dev = dev.alloc_zeros(workspace_len as usize)?;

        Ok(ReduceCudaPlan {
            idx_to_input_offset: slice,
            reduce_plan,
            workspace_dev,
            block_len,
            blocks_per_reduce_tensor_count,
            blocks_total: workspace_len,
        })
    }
}

pub trait ReduceOperator<T>: DeviceRepr {
    const MODULE_NAME: &'static str;
    const FN_NAME: &'static str;

    fn ptx(&self) -> Ptx;
}

fn load_and_get_kernel<Op>(
    dev: &Arc<CudaDevice>,
    operator: Op,
) -> Result<CudaFunction, Box<dyn std::error::Error>>
where
    Op: ReduceOperator<f32>,
{
    if !dev.has_func(Op::MODULE_NAME, Op::FN_NAME) {
        let ptx = operator.ptx();
        dev.load_ptx(ptx, Op::MODULE_NAME, &[Op::FN_NAME])?;
    }

    let function = dev
        .get_func(Op::MODULE_NAME, Op::FN_NAME)
        .ok_or("could not load kernel")?;

    Ok(function)
}

pub(crate) fn reduce<Op>(
    input_tensor: &Tensor<f32>,
    output_tensor: &mut Tensor<f32>,
    dev: Arc<CudaDevice>,
    reduce_plan: &mut ReduceCudaPlan<f32>,
    reduce_operator: Op,
) -> Result<Tensor<f32>, Box<dyn std::error::Error>>
where
    Op: ReduceOperator<f32>,
{
    assert_eq!(input_tensor.shape, reduce_plan.reduce_plan.tensor_shape);
    assert_eq!(
        output_tensor.shape,
        reduce_plan.reduce_plan.non_reduce_tensor_shape
    );
    assert_eq!(
        input_tensor.dev.ordinal(),
        reduce_plan.workspace_dev.device().ordinal()
    );

    let tensor_len = input_tensor.shape.elements_count() as u32;

    assert!((1..32).contains(&tensor_len));

    let kernel = load_and_get_kernel(&dev, reduce_operator)?;

    let blocks_total = reduce_plan.blocks_total;

    let type_len = std::mem::size_of::<f32>() as u32;

    let cfg1 = LaunchConfig {
        grid_dim: (blocks_total, 1, 1),
        block_dim: (reduce_plan.block_len, 1, 1),
        shared_mem_bytes: type_len * reduce_plan.block_len,
    };

    let input = &input_tensor.data;
    let workspace = &reduce_plan.workspace_dev;
    let global_indices = &reduce_plan.idx_to_input_offset;

    let params1 = (input, workspace, global_indices);

    //these 2 become now the new parameters for reduction.
    let non_reduce_tensor_count = reduce_plan
        .reduce_plan
        .non_reduce_tensor_shape
        .elements_count();
    let blocks_per_reduce_tensor_count = reduce_plan.blocks_per_reduce_tensor_count;

    let cfg2 = LaunchConfig {
        grid_dim: (non_reduce_tensor_count, 1, 1),
        block_dim: (blocks_per_reduce_tensor_count, 1, 1),
        shared_mem_bytes: type_len * reduce_plan.block_len,
    };

    let output_tensor = &output_tensor.data;

//    let params2 = (workspace, output_tensor, global_indices);
//
//    unsafe {
//        kernel.launch(cfg1, params1)?;
//        kernel.launch(cfg2, params2)?;
//    }

    todo!()
}
