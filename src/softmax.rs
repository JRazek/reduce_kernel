use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use super::tensor::{compute_strided_index, Shape, Tensor};

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/softmax.ptx"));

//unsafe impl DeviceRepr for SoftmaxOp {}

pub struct ReduceOpConfig {
    pub reduce_dims: Vec<usize>,
}

pub struct ReducePlan {
    indices: Vec<usize>,
}

pub fn precompute_global_offsets(
    reduce_op_config: ReduceOpConfig,
    tensor_shape: Shape,
) -> ReducePlan {
    println!("tensor_shape: {:?}", tensor_shape.shape);
    let mut reduce_dims = reduce_op_config.reduce_dims;
    reduce_dims.sort();
    reduce_dims.dedup();

    println!("reduce_dims: {:?}", reduce_dims);

    let reduce_mask = (0..tensor_shape.shape.len())
        .map(|i| reduce_dims.binary_search(&i).is_ok())
        .collect::<Vec<bool>>();

    println!("reduce_mask: {:?}", reduce_mask);

    let strides = tensor_shape.compute_strides();

    let non_reduce_tensors_shapes = Shape::new(
        tensor_shape
            .shape
            .iter()
            .zip(reduce_mask.iter())
            .map(|(&s_i, &is_reduced)| if is_reduced { 1 } else { s_i })
            .collect::<Vec<usize>>(),
    );
    println!(
        "non_reduce_tensors_shapes: {:?}",
        non_reduce_tensors_shapes.shape
    );

    let non_reduce_strides = non_reduce_tensors_shapes.compute_strides();
    println!("non_reduce_strides: {:?}", non_reduce_strides);

    let reduce_tensors_shapes = Shape::new(
        tensor_shape
            .shape
            .iter()
            .zip(reduce_mask.iter())
            .map(|(&s_i, &is_reduced)| if !is_reduced { 1 } else { s_i })
            .collect::<Vec<usize>>(),
    );
    println!("reduce_tensors_shapes: {:?}", reduce_tensors_shapes.shape);

    let reduce_strides = reduce_tensors_shapes.compute_strides();
    println!("reduce_strides: {:?}", reduce_strides);

    let global_non_reduce_strides = element_mul(reduce_strides.clone(), &non_reduce_strides);

    // for idx = blockDim.x * blockId.x + threadId.x, "idx-th" element corresponds to an index from which it has to fetch data in initial tensor.
    let mut indices = Vec::with_capacity(non_reduce_strides[0] * reduce_strides[0]);

    let non_reduce_tensor_len = reduce_mul_elements(&non_reduce_tensors_shapes.shape);
    let reduce_tensor_len = reduce_mul_elements(&reduce_tensors_shapes.shape);
    println!("non_reduce_tensor_len: {:?}", non_reduce_tensor_len);
    println!("reduce_tensor_len: {:?}", reduce_tensor_len);

    for non_reduce_tensor_index in 0..non_reduce_tensor_len {
        for reduce_tensor_index in 0..reduce_tensor_len {
            let global_tensor_idx = dot(
                &compute_strided_index(reduce_tensor_index, &reduce_strides),
                &reduce_strides, //mul with (1, 1, .., 1) - identity
            );

            indices.push(global_tensor_idx);
        }
    }

    let plan = ReducePlan { indices };

    plan
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

pub fn softmax(
    tensor: Tensor<f32>,
    dev: Arc<CudaDevice>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !dev.has_func("kernel_ops", "softmax_f32") {
        let ptx = Ptx::from_src(PTX_SRC);
        dev.load_ptx(ptx, "kernel_ops", &["softmax_f32"])?;
    }

    let softmax_kernel = dev
        .get_func("kernel_ops", "softmax_f32")
        .ok_or("could not load softmax kernel")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    //    #[test]
    //    fn test_compute_strided_index() {
    //        let shape = Shape::from([2, 3, 4]);
    //        let reduce_plan = precompute_global_offsets(
    //            ReduceOpConfig {
    //                reduce_dims: vec![0, 2],
    //            },
    //            shape,
    //        );
    //    }

    //    #[test]
    //    fn test_compute_strided_index21() {
    //        let shape = Shape::from([2, 3, 4]);
    //        let ReducePlan { indices } = precompute_global_offsets(
    //            ReduceOpConfig {
    //                reduce_dims: vec![0, 2],
    //            },
    //            shape,
    //        );
    //        println!("indices: {:?}", indices);
    //
    //        assert_eq!(0, 0);
    //        assert_eq!(1, 1);
    //        assert_eq!(2, 2);
    //        assert_eq!(3, 3);
    //
    //        assert_eq!(4, 12);
    //        assert_eq!(5, 13);
    //        assert_eq!(6, 14);
    //        assert_eq!(7, 15);
    //    }
    #[test]
    fn test_compute_strided_index21() {
        let strides = vec![4, 4, 1];
        let el_id = 4;
        let strided_index = compute_strided_index(el_id, &strides);

        assert_eq!(strided_index, vec![1, 0, 0]);
    }
    #[test]
    fn test_compute_strided_index22() {
        let strides = vec![3, 1, 1];
        let el_id = 1;
        let strided_index = compute_strided_index(el_id, &strides);

        assert_eq!(strided_index, vec![0, 1, 0]);
    }
}
