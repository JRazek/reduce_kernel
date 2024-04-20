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
        reduce_input: &mut CudaSlice<T>,
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

pub trait ReduceOperator<T>: DeviceRepr {
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

//pub(crate) fn reduce<Op>(
//    input_tensor: &Tensor<f32>,
//    output_tensor: &mut Tensor<f32>,
//    dev: Arc<CudaDevice>,
//    reduce_plan: &mut ReduceCudaPlan<f32>,
//    reduce_operator: Op,
//) -> Result<Tensor<f32>, Box<dyn std::error::Error>>
//where
//    Op: ReduceOperator<f32>,
//{
//    assert_eq!(input_tensor.shape, reduce_plan.reduce_plan.tensor_shape);
//    assert_eq!(
//        output_tensor.shape,
//        reduce_plan.reduce_plan.non_reduce_tensor_shape
//    );
//    assert_eq!(
//        input_tensor.dev.ordinal(),
//        reduce_plan.workspace_dev.device().ordinal()
//    );
//
//    let tensor_len = input_tensor.shape.elements_count() as u32;
//
//    assert!((1..32).contains(&tensor_len));
//
//    let kernel = load_and_get_kernel(&dev, reduce_operator)?;
//
//    let blocks_total = reduce_plan.blocks_total;
//
//    let type_len = std::mem::size_of::<f32>() as u32;
//
//    let cfg1 = LaunchConfig {
//        grid_dim: (blocks_total, 1, 1),
//        block_dim: (reduce_plan.block_len, 1, 1),
//        shared_mem_bytes: type_len * reduce_plan.block_len,
//    };
//
//    let input = &input_tensor.data;
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
//    let output_tensor = &output_tensor.data;
//
//    //    let params2 = (workspace, output_tensor, global_indices);
//    //
//    //    unsafe {
//    //        kernel.launch(cfg1, params1)?;
//    //        kernel.launch(cfg2, params2)?;
//    //    }
//
//    todo!()
//}
