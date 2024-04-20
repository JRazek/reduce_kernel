use crate::tensor::Tensor;
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
    pub(crate) workspace_dev: CudaSlice<T>,

    //for initial reduction
    block_len: u32,
    blocks_per_reduce_tensor_count: u32,
    blocks_total: u32,
}

impl<T> ReduceCudaPlan<T>
where
    T: DeviceRepr + ValidAsZeroBits,
{
    pub unsafe fn precompute(
        reduce_input: &CudaSlice<T>,
        reduce_input_len: usize,
        reduce_sub_input_len: usize, //how many subslices to reduce. Each one will have separate output.
        dev: Arc<CudaDevice>,
    ) -> Result<ReduceCudaPlan<T>, DriverError> {
        const MAX_GRID_LEN: u32 = 2 << 31 - 1;
        const MAX_BLOCK_LEN: u32 = 2 << 10;

        let block_len: usize = MAX_BLOCK_LEN.min(reduce_sub_input_len as u32) as usize;
        let blocks_per_reduce_tensor_count = (reduce_sub_input_len.div_ceil(block_len)) as u32;

        let ouput_len = reduce_input_len / reduce_sub_input_len;
        assert_eq!(reduce_input_len % reduce_sub_input_len, 0);

        //each sub_input will first be reduced to blocks_per_reduce_tensor_count values.
        //we need to store them somewhere and then reduce again to a single value.
        let workspace_len = blocks_per_reduce_tensor_count * ouput_len as u32;

        let workspace_dev = dev.alloc_zeros(workspace_len as usize)?;

        Ok(ReduceCudaPlan {
            workspace_dev,
            block_len: block_len as u32,
            blocks_per_reduce_tensor_count,
            blocks_total: workspace_len,
        })
    }
}

pub trait ReduceOperator<T>
where
    T: DeviceRepr,
{
    const MODULE_NAME: &'static str;
    const FN_NAME: &'static str;

    fn ptx(&self) -> Ptx;
}

//move to utility file.
pub(crate) fn load_and_get_kernel<Op>(
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

pub(crate) unsafe fn reduce<Op>(
    reduce_input: &CudaSlice<f32>,
    reduce_input_len: usize,
    reduce_subinput_len: usize,
    output: &mut CudaSlice<f32>, //requires output to be at least of size input_len / reduce_sub_input_len
    dev: Arc<CudaDevice>,
    reduce_operator: Op,
) -> Result<Tensor<f32>, Box<dyn std::error::Error>>
where
    Op: ReduceOperator<f32>,
{
    let subinputs = reduce_input_len / reduce_subinput_len;
    assert!(reduce_input_len % reduce_subinput_len == 0);

    let kernel = load_and_get_kernel(&dev, reduce_operator)?;

    let type_len = std::mem::size_of::<f32>() as u32;

    const MAX_BLOCK_LEN: u32 = 32;

    let first_step_block_len: u32 = MAX_BLOCK_LEN.min(reduce_subinput_len as u32);
    let blocks_per_reduce_subinput_count =
        (reduce_subinput_len.div_ceil(first_step_block_len as usize)) as u32;

    let step1_blocks_total = blocks_per_reduce_subinput_count * subinputs as u32;

    let step1_cfg = LaunchConfig {
        grid_dim: (step1_blocks_total, 1, 1),
        block_dim: (first_step_block_len, 1, 1),
        shared_mem_bytes: type_len * first_step_block_len,
    };

    let output_reborrow = &mut *output;

    kernel.launch(step1_cfg, (reduce_input, output_reborrow, reduce_input_len))?;

    let output_dev = dev.dtoh_sync_copy(output)?;

    println!("{:?}", output_dev);

    //    let input = &reduce_input.data;
    //    let workspace = &reduce_plan.workspace_dev;
    //    let global_indices = &reduce_plan.idx_to_input_offset;
    //
    //    let params1 = (input, workspace, global_indices);
    //
    //    //these 2 become now the new parameters for reduction.
    //    let non_reduce_tensor_count = reduce_plan
    //        .reduce_plan
    //        .non_reduce_tensor_shape
    //        .elements_count();
    //    let blocks_per_reduce_tensor_count = reduce_plan.blocks_per_reduce_tensor_count;
    //
    //    let cfg2 = LaunchConfig {
    //        grid_dim: (non_reduce_tensor_count, 1, 1),
    //        block_dim: (blocks_per_reduce_tensor_count, 1, 1),
    //        shared_mem_bytes: type_len * reduce_plan.block_len,
    //    };
    //
    //    let output_tensor = &output.data;

    //    let params2 = (workspace, output_tensor, global_indices);
    //
    //    unsafe {
    //        kernel.launch(cfg1, params1)?;
    //        kernel.launch(cfg2, params2)?;
    //    }

    todo!()
}

//tests only through other implementations - max, sum etc.
