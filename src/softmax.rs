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

    // for idx = blockDim.x * blockId.x + threadId.x, "idx-th" element corresponds to an index from which it has to fetch data in initial tensor.
    let mut indices = Vec::with_capacity(non_reduce_strides[0] * reduce_strides[0]);

    let non_reduce_tensor_len = reduce_mul_elements(&non_reduce_tensors_shapes.shape);
    let reduce_tensor_len = reduce_mul_elements(&reduce_tensors_shapes.shape);
    println!("non_reduce_tensor_len: {:?}", non_reduce_tensor_len);
    println!("reduce_tensor_len: {:?}", reduce_tensor_len);

    let global_non_reduce_strides = element_mul(reduce_strides.clone(), &non_reduce_strides);

    for non_reduce_tensor_index in 0..non_reduce_tensor_len {
        for reduce_tensor_index in 0..reduce_tensor_len {
            let global_tensor_idx = dot(
                &compute_strided_index(non_reduce_tensor_index, &non_reduce_strides),
                &global_non_reduce_strides,
            ) + dot(
                &compute_strided_index(reduce_tensor_index, &reduce_strides),
                &strides, //mul with (1, 1, .., 1) - identity
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
    #[test]
    fn test_plan01() {
        let shape = Shape::from([2, 3, 4]);
        let ReducePlan { indices } = precompute_global_offsets(
            ReduceOpConfig {
                reduce_dims: vec![0, 2],
            },
            shape,
        );
        println!("indices: {:?}", indices);

        assert_eq!(indices[0], 0);
        assert_eq!(indices[1], 1);
        assert_eq!(indices[2], 2);
        assert_eq!(indices[3], 3);

        assert_eq!(indices[4], 12);
        assert_eq!(indices[5], 13);
        assert_eq!(indices[6], 14);
        assert_eq!(indices[7], 15);

        assert_eq!(indices[8], 4);
        assert_eq!(indices[9], 5);
        assert_eq!(indices[10], 6);
        assert_eq!(indices[11], 7);

        assert_eq!(indices[12], 16);
        assert_eq!(indices[13], 17);
        assert_eq!(indices[14], 18);
        assert_eq!(indices[15], 19);

        assert_eq!(indices[16], 8);
        assert_eq!(indices[17], 9);
        assert_eq!(indices[18], 10);
        assert_eq!(indices[19], 11);

        assert_eq!(indices[20], 20);
        assert_eq!(indices[21], 21);
        assert_eq!(indices[22], 22);
        assert_eq!(indices[23], 23);
    }
    #[test]
    fn test_plan02() {
        let shape = Shape::from([2, 3, 4]);
        let ReducePlan { indices } = precompute_global_offsets(
            ReduceOpConfig {
                reduce_dims: vec![1],
            },
            shape,
        );
        println!("indices: {:?}", indices);

        assert_eq!(indices[0], 0);
        assert_eq!(indices[1], 4);
        assert_eq!(indices[2], 8);

        assert_eq!(indices[3], 1);
        assert_eq!(indices[4], 5);
        assert_eq!(indices[5], 9);

        assert_eq!(indices[6], 2);
        assert_eq!(indices[7], 6);
        assert_eq!(indices[8], 10);

        assert_eq!(indices[9], 3);
        assert_eq!(indices[10], 7);
        assert_eq!(indices[11], 11);

        assert_eq!(indices[12], 12);
        assert_eq!(indices[13], 16);
        assert_eq!(indices[14], 20);

        assert_eq!(indices[15], 13);
        assert_eq!(indices[16], 17);
        assert_eq!(indices[17], 21);

        assert_eq!(indices[18], 14);
        assert_eq!(indices[19], 18);
        assert_eq!(indices[20], 22);

        assert_eq!(indices[21], 15);
        assert_eq!(indices[22], 19);
        assert_eq!(indices[23], 23);
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
