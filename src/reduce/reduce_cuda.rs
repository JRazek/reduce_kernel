use crate::tensor::Tensor;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig,
    ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
struct ReduceStep {
    block_size: u32,
    total_blocks_count: u32,
}

#[derive(Debug, Clone)]
pub struct ReduceCudaPlan<T>
where
    T: DeviceRepr,
{
    workspace_dev: CudaSlice<T>,

    steps: Vec<ReduceStep>,
}

const MAX_GRID_LEN: u32 = 1 << 31 - 1;
const MAX_BLOCK_LEN: u32 = 1 << 10;

fn make_steps(
    mut reduce_input_len: usize,
    mut reduce_subinput_len: usize,
) -> Result<Vec<ReduceStep>, Box<dyn std::error::Error>> {
    if reduce_input_len % reduce_subinput_len != 0 {
        return Err("reduce_input_len must be divisible by reduce_subinput_len!".into());
    }

    let mut steps = Vec::new();

    let output_len = reduce_input_len / reduce_subinput_len;

    loop {
        let subinputs_in_input = reduce_input_len / reduce_subinput_len;
        assert_eq!(reduce_input_len % reduce_subinput_len, 0); //this should always be true by
                                                               //induction.

        let block_size = MAX_BLOCK_LEN.min(reduce_subinput_len as u32);

        let blocks_per_subinput = reduce_subinput_len.div_ceil(block_size as usize) as u32;

        let total_blocks_count = blocks_per_subinput * subinputs_in_input as u32;

        steps.push(ReduceStep {
            block_size,
            total_blocks_count,
        });

        //TODO: prove correctness:
        //total_blocks_count_{i+1} < total_blocks_count_{i}
        if total_blocks_count == output_len as u32 {
            assert_eq!(blocks_per_subinput, 1);
            break;
        }

        assert!(total_blocks_count > output_len as u32);

        reduce_input_len = total_blocks_count as usize;
        reduce_subinput_len = blocks_per_subinput as usize;
    }

    Ok(steps)
}

impl<T> ReduceCudaPlan<T>
where
    T: DeviceRepr + ValidAsZeroBits,
{
    pub fn precompute(
        reduce_input: &CudaSlice<T>,
        mut reduce_input_len: usize,
        mut reduce_subinput_len: usize, //how many subslices to reduce. Each one will have separate output.
        dev: Arc<CudaDevice>,
    ) -> Result<ReduceCudaPlan<T>, Box<dyn std::error::Error>> {
        let steps = make_steps(reduce_input_len, reduce_subinput_len)?;
        let workspace_dev = dev.alloc_zeros(reduce_input_len)?;

        let plan = ReduceCudaPlan {
            workspace_dev,
            steps,
        };

        Ok(plan)
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
    workspace: &mut CudaSlice<f32>, //requires workspace to be at least of size input_len / reduce_sub_input_len
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

    //    kernel.launch(step1_cfg, (reduce_input, workspace, reduce_input_len))?;
    //
    //    let output_dev = dev.dtoh_sync_copy(output)?;
    //
    //    println!("{:?}", output_dev);

    //now repeat for all results from each block corresponding to a subinput.

    //    let cfg2 = LaunchConfig {
    //        grid_dim: (reduce_input_len / reduce_subinput_len, 1, 1),
    //        block_dim: (blocks_per_reduce_tensor_count, 1, 1),
    //        shared_mem_bytes: type_len * reduce_plan.block_len,
    //    };
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

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_steps_creation() {
        let input_len = 10_000_000;
        let subinput_len = 2_000;

        let output_len = input_len / subinput_len;

        match make_steps(input_len, subinput_len).unwrap().as_slice() {
            [ReduceStep {
                block_size: 1024, //MAX_BLOCK_LEN
                total_blocks_count: 10_000,
            }, ReduceStep {
                block_size: 2,
                total_blocks_count: output_len,
            }] => {}
            _ => panic!(),
        }
    }
}
