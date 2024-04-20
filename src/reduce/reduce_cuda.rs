use crate::tensor::Tensor;
use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig,
    ValidAsZeroBits,
};

use crate::kernel::{load_and_get_kernel, Kernel};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
struct ReduceStep {
    block_size_x: u32,
    grid_dim: (u32, u32, u32),
    reduce_subinput_len: u32,
}

const MAX_GRID_LEN: u32 = 1 << 31 - 1;
const MAX_BLOCK_LEN: u32 = 1024;

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

        let optimal_subinput_len = {
            if reduce_subinput_len.is_power_of_two() {
                reduce_subinput_len
            } else {
                reduce_subinput_len.next_power_of_two()
            }
        } as u32;

        let block_size = MAX_BLOCK_LEN.min(optimal_subinput_len);
        assert!(block_size.is_power_of_two());

        let blocks_per_subinput = reduce_subinput_len.div_ceil(block_size as usize) as u32;

        let total_blocks_count = blocks_per_subinput * subinputs_in_input as u32;

        steps.push(ReduceStep {
            block_size_x: block_size,
            grid_dim: (blocks_per_subinput as u32, subinputs_in_input as u32, 1),
            reduce_subinput_len: reduce_subinput_len as u32,
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

    println!("steps: {:?}", steps);
    Ok(steps)
}

pub(crate) unsafe trait ReduceOperator<T>: Kernel<T>
where
    T: DeviceRepr,
{
}

pub(crate) unsafe fn reduce<Op>(
    reduce_input: CudaSlice<f32>, //acts as workspace as well.
    reduce_input_len: usize,
    reduce_subinput_len: usize,
    output: &mut CudaSlice<f32>, //requires output to be at least of size input_len / reduce_sub_input_len
    dev: Arc<CudaDevice>,
    reduce_operator: Op,
) -> Result<(), Box<dyn std::error::Error>>
where
    Op: ReduceOperator<f32>,
{
    //currently only for 2^n inputs.

    //    assert!(reduce_input_len.is_power_of_two());
    let workspace = reduce_input;
    let kernel = load_and_get_kernel(&dev, reduce_operator)?;

    let type_len = std::mem::size_of::<f32>() as u32;

    //TODO this should probably also be in a plan
    let steps = make_steps(reduce_input_len, reduce_subinput_len)?;

    let last_step_idx = steps.len() - 1;
    for (i, step) in steps.into_iter().enumerate() {
        let cfg = LaunchConfig {
            grid_dim: step.grid_dim,
            block_dim: (step.block_size_x, 1, 1),
            shared_mem_bytes: type_len * step.block_size_x,
        };

        let out = {
            if i == last_step_idx {
                &*output
            } else {
                &workspace
            }
        };

        println!("launching kernel with cfg: {:?}", cfg);
        println!("input_addr: {:?}", (&workspace).as_kernel_param());
        println!("output_addr: {:?}", out.as_kernel_param());

        let params = (&workspace, out, step.reduce_subinput_len);

        unsafe {
            kernel.clone().launch(cfg, params)?;
        }

        dev.synchronize()?;
        println!("next step\n\n\n");
    }

    Ok(())
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_steps_creation() {
        let input_len = 10_000_000;
        let subinput_len = 2_000;

        let output_len = input_len / subinput_len;

        let steps = make_steps(input_len, subinput_len).unwrap();
        match steps.as_slice() {
            [ReduceStep {
                block_size_x: 1024, //MAX_BLOCK_LE
                grid_dim: (2, 5000, 1),
                reduce_subinput_len: 2_000,
            }, ReduceStep {
                block_size_x: 2,
                grid_dim: (1, 5000, 1),
                reduce_subinput_len: 2,
            }] => {}
            _ => panic!("incorrect steps: {:?}", steps),
        }
    }
}
