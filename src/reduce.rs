pub mod ops;
pub mod reduce_cuda;

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

#[derive(Debug, Clone)]
pub struct ReducePlan {
    pub(crate) input_tensor_shape: Shape,
    pub(crate) output_tensor_shape: Shape,

    pub(crate) idx_to_input_offsets: Vec<usize>, //for initial mapping
    pub(crate) idx_to_output_offsets: Vec<usize>, //for final mapping
}

impl ReducePlan {
    pub fn check_compatible_shape(&self, shape: &Shape) -> bool {
        (*shape) == self.input_tensor_shape
    }

    pub fn precompute(input_tensor_shape: &Shape, mut reduce_dims: Vec<usize>) -> ReducePlan {
        reduce_dims.sort();
        reduce_dims.dedup();

        let reduce_mask = make_reduce_mask(input_tensor_shape, &reduce_dims);
        let input_tensor_strides = input_tensor_shape.compute_strides();

        //projected onto dimension of input tensor.
        //e.g. if input is of shape (2, 3, 4) and reduce_dims is [0, 2], then
        //output_tensor_shape is (1, 3, 1)
        let (output_tensor_shape, reduce_tensors_shape) =
            output_reduce_tensors_shapes(input_tensor_shape, &reduce_dims);

        let output_tensor_strides = output_tensor_shape.compute_strides();
        let reduce_tensors_strides = reduce_tensors_shape.compute_strides();

        let mut idx_to_input_offset_vec = vec![0; input_tensor_shape.elements_count() as usize];
        let mut idx_to_output_offset_vec = vec![0; output_tensor_shape.elements_count() as usize];

        //each element in output tensor corresponds to a block of elements in input tensor.
        //this describes the mapping from output tensor to input tensor (first elements of each block)

        let output_elements_strides_in_input_tensor =
            element_mul(reduce_tensors_strides.clone(), &output_tensor_strides); //Check correctness here!

        let mut input_tensor_idx = 0;
        for output_tensor_idx in 0..output_tensor_shape.elements_count() {
            let output_tensor_strided_index =
                compute_strided_index(output_tensor_idx as usize, &output_tensor_strides);

            idx_to_output_offset_vec[output_tensor_idx as usize] =
                dot(&output_tensor_strided_index, &output_tensor_strides);

            for reduce_tensor_idx in 0..reduce_tensors_shape.elements_count() {
                let input_tensor_offset = dot(
                    &output_tensor_strided_index,
                    &output_elements_strides_in_input_tensor,
                ) + dot(
                    &compute_strided_index(reduce_tensor_idx as usize, &reduce_tensors_strides),
                    &input_tensor_strides,
                );

                idx_to_input_offset_vec[input_tensor_idx] = input_tensor_offset;
                input_tensor_idx += 1;
            }
        }

        let plan = ReducePlan {
            idx_to_input_offsets: idx_to_input_offset_vec,
            idx_to_output_offsets: idx_to_output_offset_vec,
            input_tensor_shape: input_tensor_shape.clone(),
            output_tensor_shape,
        };

        plan
    }
}

fn dot(lhs: &Vec<usize>, rhs: &Vec<usize>) -> usize {
    lhs.iter().zip(rhs.iter()).map(|(&l, &r)| l * r).sum()
}

fn element_mul(lhs: Vec<usize>, rhs: &Vec<usize>) -> Vec<usize> {
    lhs.into_iter()
        .zip(rhs.iter())
        .map(|(l, &r)| l * r)
        .collect()
}

fn reduce_mul_elements(v: &Vec<usize>) -> usize {
    v.iter().map(|&e| e).reduce(|acc, e| acc * e).unwrap()
}

fn output_reduce_tensors_shapes(
    input_tensor_shape: &Shape,
    reduce_dims: &Vec<usize>,
) -> (Shape, Shape) {
    let reduce_mask = make_reduce_mask(input_tensor_shape, reduce_dims);

    let output_tensor_shape = Shape::new(
        input_tensor_shape
            .shape
            .iter()
            .zip(reduce_mask.iter())
            .map(|(&s_i, &is_reduced)| if is_reduced { 1 } else { s_i })
            .collect::<Vec<usize>>(),
    );

    let reduce_tensors_shape = Shape::new(
        input_tensor_shape
            .shape
            .iter()
            .zip(reduce_mask.iter())
            .map(|(&s_i, &is_reduced)| if !is_reduced { 1 } else { s_i })
            .collect::<Vec<usize>>(),
    );

    (output_tensor_shape, reduce_tensors_shape)
}

fn make_reduce_mask(tensor_shape: &Shape, reduce_dims: &Vec<usize>) -> Vec<bool> {
    (0..tensor_shape.shape.len())
        .map(|i| reduce_dims.binary_search(&i).is_ok())
        .collect::<Vec<bool>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan01() {
        let tensor_shape = Shape::from([2, 3, 4]);
        let ReducePlan {
            idx_to_input_offsets,
            ..
        } = ReducePlan::precompute(&tensor_shape, vec![0, 2]);

        assert_eq!(idx_to_input_offsets[0], 0);
        assert_eq!(idx_to_input_offsets[1], 1);
        assert_eq!(idx_to_input_offsets[2], 2);
        assert_eq!(idx_to_input_offsets[3], 3);

        assert_eq!(idx_to_input_offsets[4], 12);
        assert_eq!(idx_to_input_offsets[5], 13);
        assert_eq!(idx_to_input_offsets[6], 14);
        assert_eq!(idx_to_input_offsets[7], 15);

        assert_eq!(idx_to_input_offsets[8], 4);
        assert_eq!(idx_to_input_offsets[9], 5);
        assert_eq!(idx_to_input_offsets[10], 6);
        assert_eq!(idx_to_input_offsets[11], 7);

        assert_eq!(idx_to_input_offsets[12], 16);
        assert_eq!(idx_to_input_offsets[13], 17);
        assert_eq!(idx_to_input_offsets[14], 18);
        assert_eq!(idx_to_input_offsets[15], 19);

        assert_eq!(idx_to_input_offsets[16], 8);
        assert_eq!(idx_to_input_offsets[17], 9);
        assert_eq!(idx_to_input_offsets[18], 10);
        assert_eq!(idx_to_input_offsets[19], 11);

        assert_eq!(idx_to_input_offsets[20], 20);
        assert_eq!(idx_to_input_offsets[21], 21);
        assert_eq!(idx_to_input_offsets[22], 22);
        assert_eq!(idx_to_input_offsets[23], 23);
    }
    #[test]
    fn test_plan02() {
        let tensor_shape = Shape::from([2, 3, 4]);
        let tensor_shape = Shape::from([2, 3, 4]);
        let ReducePlan {
            idx_to_input_offsets,
            ..
        } = ReducePlan::precompute(&tensor_shape, vec![1]);

        assert_eq!(idx_to_input_offsets[0], 0);
        assert_eq!(idx_to_input_offsets[1], 4);
        assert_eq!(idx_to_input_offsets[2], 8);

        assert_eq!(idx_to_input_offsets[3], 1);
        assert_eq!(idx_to_input_offsets[4], 5);
        assert_eq!(idx_to_input_offsets[5], 9);

        assert_eq!(idx_to_input_offsets[6], 2);
        assert_eq!(idx_to_input_offsets[7], 6);
        assert_eq!(idx_to_input_offsets[8], 10);

        assert_eq!(idx_to_input_offsets[9], 3);
        assert_eq!(idx_to_input_offsets[10], 7);
        assert_eq!(idx_to_input_offsets[11], 11);

        assert_eq!(idx_to_input_offsets[12], 12);
        assert_eq!(idx_to_input_offsets[13], 16);
        assert_eq!(idx_to_input_offsets[14], 20);

        assert_eq!(idx_to_input_offsets[15], 13);
        assert_eq!(idx_to_input_offsets[16], 17);
        assert_eq!(idx_to_input_offsets[17], 21);

        assert_eq!(idx_to_input_offsets[18], 14);
        assert_eq!(idx_to_input_offsets[19], 18);
        assert_eq!(idx_to_input_offsets[20], 22);

        assert_eq!(idx_to_input_offsets[21], 15);
        assert_eq!(idx_to_input_offsets[22], 19);
        assert_eq!(idx_to_input_offsets[23], 23);
    }

    #[test]
    fn test_compute_strided_index01() {
        let strides = vec![4, 4, 1];
        let el_id = 4;
        let strided_index = compute_strided_index(el_id, &strides);

        assert_eq!(strided_index, vec![1, 0, 0]);
    }

    #[test]
    fn test_compute_strided_index02() {
        let strides = vec![3, 1, 1];
        let el_id = 1;
        let strided_index = compute_strided_index(el_id, &strides);

        assert_eq!(strided_index, vec![0, 1, 0]);
    }
}
