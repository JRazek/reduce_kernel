pub mod ops;
pub mod reduce_cuda;

use super::tensor::{compute_strided_offset, Shape};

use crate::map_kernel::map_offsets_in_place;

#[derive(Debug, Clone)]
pub struct ReducePlan {
    pub(crate) input_tensor_shape: Shape,
    pub(crate) output_tensor_shape: Shape,

    //after the reduction, when in context of an input tensor-shaped buffer, there is no need to
    //store indices. If you run each block corresponding to a separate sub-tensor in a different grid's y component, you can access
    //blockIdx.y element in the reduced buffer.
    pub(crate) idx_to_input_offsets: Vec<usize>, //for initial mapping
}

impl ReducePlan {
    pub fn check_compatible_shape(&self, shape: &Shape) -> bool {
        (*shape) == self.input_tensor_shape
    }

    pub fn precompute(input_tensor_shape: &Shape, mut reduce_dims: Vec<usize>) -> ReducePlan {
        reduce_dims.sort();
        reduce_dims.dedup();

        todo!()
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
}

#[derive(Debug, Clone)]
pub struct NewReducePlan {
    shape: Shape,
}

impl NewReducePlan {
    pub fn new_precompute(input_tensor_shape: Shape, mut reduce_dims: Vec<usize>) -> NewReducePlan {
        //descending order
        reduce_dims.sort();
        reduce_dims.dedup();

        let n = input_tensor_shape.len();
        let k = reduce_dims.len();

        dbg!(&reduce_dims);

        let mut permutation: Vec<_> = (0..n).into_iter().collect();

        for i in 0..k {
            let lhs_idx = n - k + i;

            permutation.swap(lhs_idx, reduce_dims[i]);
            dbg!(&permutation);
        }

        let mut shape = input_tensor_shape;

        shape.permute(&permutation);

        NewReducePlan { shape }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    #[test]
    fn test_precompute_01() {
        let shape = Shape::new(vec![2, 3, 4]);
        let reduce_dims = vec![0, 2];

        let NewReducePlan { shape } = NewReducePlan::new_precompute(shape, reduce_dims);
        assert_eq!(shape.as_slice(), &[3, 2, 4]);

        assert_eq!(shape.get_offset(0).unwrap(), 0);
        assert_eq!(shape.get_offset(1).unwrap(), 1);
        assert_eq!(shape.get_offset(2).unwrap(), 2);
        assert_eq!(shape.get_offset(3).unwrap(), 3);
        assert_eq!(shape.get_offset(4).unwrap(), 12);
        assert_eq!(shape.get_offset(5).unwrap(), 13);
        assert_eq!(shape.get_offset(6).unwrap(), 14);
        assert_eq!(shape.get_offset(7).unwrap(), 15);

        assert_eq!(shape.get_offset(8).unwrap(), 4);
        assert_eq!(shape.get_offset(9).unwrap(), 5);
        assert_eq!(shape.get_offset(10).unwrap(), 6);
        assert_eq!(shape.get_offset(11).unwrap(), 7);
        assert_eq!(shape.get_offset(12).unwrap(), 16);
        assert_eq!(shape.get_offset(13).unwrap(), 17);
        assert_eq!(shape.get_offset(14).unwrap(), 18);
        assert_eq!(shape.get_offset(15).unwrap(), 19);

        assert_eq!(shape.get_offset(16).unwrap(), 8);
        assert_eq!(shape.get_offset(17).unwrap(), 9);
        assert_eq!(shape.get_offset(18).unwrap(), 10);
        assert_eq!(shape.get_offset(19).unwrap(), 11);
        assert_eq!(shape.get_offset(20).unwrap(), 20);
        assert_eq!(shape.get_offset(21).unwrap(), 21);
        assert_eq!(shape.get_offset(22).unwrap(), 22);
        assert_eq!(shape.get_offset(23).unwrap(), 23);
    }

    #[test]
    fn test_precompute_02() {
        let shape = Shape::new(vec![2, 3, 4]);
        let reduce_dims = vec![1];

        let NewReducePlan { shape } = NewReducePlan::new_precompute(shape, reduce_dims);
        assert_eq!(shape.as_slice(), &[2, 4, 3]);

        assert_eq!(shape.get_offset(0).unwrap(), 0);
        assert_eq!(shape.get_offset(1).unwrap(), 4);
        assert_eq!(shape.get_offset(2).unwrap(), 8);

        assert_eq!(shape.get_offset(3).unwrap(), 1);
        assert_eq!(shape.get_offset(4).unwrap(), 5);
        assert_eq!(shape.get_offset(5).unwrap(), 9);

        assert_eq!(shape.get_offset(6).unwrap(), 2);
        assert_eq!(shape.get_offset(7).unwrap(), 6);
        assert_eq!(shape.get_offset(8).unwrap(), 10);

        assert_eq!(shape.get_offset(9).unwrap(), 3);
        assert_eq!(shape.get_offset(10).unwrap(), 7);
        assert_eq!(shape.get_offset(11).unwrap(), 11);

        assert_eq!(shape.get_offset(12).unwrap(), 12);
        assert_eq!(shape.get_offset(13).unwrap(), 16);
        assert_eq!(shape.get_offset(14).unwrap(), 20);

        assert_eq!(shape.get_offset(15).unwrap(), 13);
        assert_eq!(shape.get_offset(16).unwrap(), 17);
        assert_eq!(shape.get_offset(17).unwrap(), 21);

        assert_eq!(shape.get_offset(18).unwrap(), 14);
        assert_eq!(shape.get_offset(19).unwrap(), 18);
        assert_eq!(shape.get_offset(20).unwrap(), 22);

        assert_eq!(shape.get_offset(21).unwrap(), 15);
        assert_eq!(shape.get_offset(22).unwrap(), 19);
        assert_eq!(shape.get_offset(23).unwrap(), 23);
    }
}
