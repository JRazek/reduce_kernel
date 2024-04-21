# Softmax Kernel

Note, this is only a sketch. It would be still needed to test these solutions extensively in practice. There may be breaking corner cases etc.

## Problem
Biggest issue is the fact that dimensions of reduction AND dimensions of a tensor are arbitary.
In order to somehow apply any operations on a tensor, we need to know somehow know the mapping between coordinates and contiguous buffer.
Another thing is that for reduce operations, we'd like to have buffers corresponding to each reduction subtensor to be contiguous.

Thus:
Each, i-th launched thread in kernel needs to know to which coordinate it corresponds to and that it will reduce on a contiguous buffer.

Usually the same operation is applied many times - there is no sense in copying the data for one operation.
I assume precomputing is accepted as long as it may be reused for each operation (same input tensor shape, same reduce dimensions).

Since computing such mapping in each kernel may take some time, I see it as reasonable to precompute it on host and copy it to device buffer. 
This mapping obviously will be a permutation of a buffer dimensions.

Mapping may be performed in-place with the following kernel:
```cpp
template <typename T>
__device__ auto map_offsets_in_place(T *data, const std::size_t *idx_to_offsets)
    -> void {
  auto tid = threadIdx.x;
  auto idx = blockIdx.x * blockDim.x + threadIdx.x; // # in grid.

  auto tensor_idx = idx_to_offsets[idx]; // offset in data.
  auto input = data[tensor_idx];

  auto group = cooperative_groups::this_grid();

  group.sync(); //this is needed to ensure that all threads in a grid have read the data.

  data[idx] = input;
}
```

but... how to compute `idx_to_offsets`?

For testing I used mostly `Rust` and `cudarc` crate. For computation of such idx_to_offsets I used the following code:
```rust
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
```

It may be probably simpified and optimized but its just a CPU side precomputation.
`ReducePlan` struct holds the initial permutation - before reduction and the final permutation - after reduction.
Tests show how it works for a multidimensional reduce shape.

Output shape in this case after reduction is a shape with "projected" dimensions of input tensor onto non-reduced dimensions.
e.g. if input tensor is of shape (2, 3, 4) and reduce_dims is [0, 2], then output shape is (1, 3, 1).
Each of indices in output tensor corresponds to a block of elements in input tensor that was reduced. 

e.g. if our tensor is of shape (4, 3) with reduce dim [0] and our reduce operator is max(), then we get:
```
[1, 2, 3, 4]
[5, 6, 7, 8]
[9, 10, 11, 12]
```
```
[max(1, 2, 3, 4)]
[max(5, 6, 7, 8)]
[max(9, 10, 11, 12)]
```
with a shape (1, 3).


One still however needs to actually implement kernel and its execution. In general, there's no hard limit on the size of a reduced tensor. 
Start with a tensor (2, 3, 4) and [0, 2] reduce dims.
Shape of each tensor for reduction is (2, 1, 4) and its size is 8.
In the output we will have the following after mapping:

```
[0,  4,  8,  1,  5,  9,  2,  6,  10,  3,  7,  11, 12, 16, 20, 13, 17, 21,  14, 18, 22, 15,  19, 23]
[___________(0,0,0)_____________][___________(0,1,0)_____________][___________(0,2,0)_____________]
```

Each of these slices may be divided into blocks of size given by 

```rust
let optimal_subinput_len = {
    if reduce_subinput_len.is_power_of_two() {
        reduce_subinput_len
    } else {
        reduce_subinput_len.next_power_of_two()
    }
} as u32;

let block_size = MAX_BLOCK_LEN.min(optimal_subinput_len);
assert!(block_size.is_power_of_two());
```
Assume for simplicity that MAX_BLOCK_LEN=4. Each of these boxes will need to be reduced with 2 separate blocks of size 4.
The problem is each of such "block level" reduction will only be capable of reducing 4 elements at once.
Thus we will be again left with 2 elements to reduce for each box.

### Solution 1
use the following planning algorithm
```rust
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
```

It divides our reduction into separate steps and in the end gives us a list of steps to perform.
This kernel may be used along with the given planning algorithm.
```cpp

template <typename T, typename Op>
__device__ auto reduce(const T *in, T *out, std::uint32_t reduce_input_len,
                       Op reduce_op) -> void {
  auto subinput_id =
      threadIdx.x + blockIdx.x * blockDim.x; // in particular subinput

  auto grid_id = subinput_id + reduce_input_len * blockIdx.y; // in entire input

  // in each gridDim.x, it may happen, that for gridId.y==n and gridId.y==n+1,
  // ending and starting grid_id will coincide. Also to make sure that no out of
  // bound access happens, this is used.
  // This will lead to branch divergence only on boundaries.
  if (subinput_id >= reduce_input_len) {
    return;
  }

  auto tid = threadIdx.x; // in block

  extern __shared__ T shared[];
  shared[tid] = in[grid_id];

  __syncthreads(); //after each of our threads in a block has written data into shared memory, wait on a barrier.

  for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && subinput_id + s < reduce_input_len) {
      auto lhs = shared[tid];
      auto rhs = shared[tid + s];
      auto res = reduce_op(lhs, rhs);

      shared[tid] = res;
    }

    __syncthreads();
  }

  if (tid == 0) {
    auto out_id = blockIdx.x + gridDim.x * blockIdx.y;
    out[out_id] = shared[0];
  }
}
```
for the first iteration in the given example (2, 3, 4) [0, 2], 
it requires us to launch blocks in the following grid
```
[block0, block1] 
[block2, block3]
[block4, block5]
```
each row corresponds to subinput and column to a specific block.
e.g. block3 will handle data mapped as [10,  3,  7,  11].
Each of these blocks will of course consist of 4 threads.

### Solution 2
Use cooperative groups.


# Concrete reduction algorithms
Using this template we can obviously implement many reduction algorithms:

```cpp

template <typename T> struct MaxOp {
  __device__ auto operator()(T a, T b) const -> T { return a > b ? a : b; }
};

template <typename T>
__device__ auto max_reduce(const T *in, T *out, std::uint32_t reduce_input_len)
    -> void {
  reduce<T, MaxOp<T>>(in, out, reduce_input_len, MaxOp<T>{});
}


template <typename T> struct SumOp {
  __device__ auto operator()(T a, T b) const -> T { return a + b; }
};

template <typename T>
__device__ auto sum_reduce(const T *in, T *out, std::uint32_t reduce_input_len)
    -> void {
  reduce<T, SumOp<T>>(in, out, reduce_input_len, SumOp<T>{});
}
```
