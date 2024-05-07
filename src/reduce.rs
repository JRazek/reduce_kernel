pub mod ops;
pub mod reduce_cuda;

use super::tensor::{compute_strided_offset, Shape};

#[derive(Debug, Clone)]
pub struct NewReducePlan {
    shape_permuted: Shape,
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

        shape.permute(permutation);

        NewReducePlan {
            shape_permuted: shape,
        }
    }
    pub fn check_compatible_shape(&self, shape: &Shape) -> bool {
        self.shape_permuted.elements_count() == shape.elements_count()
    }
}

#[cfg(test)]
mod test {

    use super::*;
    #[test]
    fn test_precompute_01() {
        let shape = Shape::new(vec![2, 3, 4]);
        let reduce_dims = vec![0, 2];

        let NewReducePlan {
            shape_permuted: shape,
        } = NewReducePlan::new_precompute(shape, reduce_dims);
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

        let NewReducePlan {
            shape_permuted: shape,
        } = NewReducePlan::new_precompute(shape, reduce_dims);
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
