# Softmax Kernel

This is a very first sketch of a solution. It would be still very important to test these solutions extensively in practice. There may be breaking corner cases and obviously tons of optimizations/simplifications to be made.

## Problem
IMO biggest issue is the fact that dimensions of reduction AND dimensions of a tensor are arbitary.
In order to apply any operations on a tensor, we need to know the mapping between coordinates and contiguous buffer.
Another thing is that for reduce operations, we'd like to have buffers corresponding to each reduction subtensor to be contiguous.

Thus:
Each, i-th launched thread in kernel needs to know to which coordinate it corresponds to and that it will reduce on a contiguous buffer.

Usually the same operation is applied many times - there is no sense in copying the data for one operation.
I assume precomputing is accepted as long as it may be reused for each operation (same input tensor shape, same reduce dimensions).

Since computing such mapping in each kernel may take some time, I see it as reasonable to precompute it on host and copy it to device buffer. 
This mapping obviously will be a permutation of an initial buffer indices.

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
Though one could try to do it without actually permuting data while reading to shared memory in later steps. This however makes it more "modular" and easier to explain here :).

but... how to compute `idx_to_offsets`?

For testing I used mostly `Rust` and `cudarc` crate. For computation of such idx_to_offsets I used the following code:
```rust
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

        //each element in output tensor corresponds to a block of elements in input tensor.
        //this describes the mapping from output tensor to input tensor (first elements of each block)

        let output_elements_strides_in_input_tensor =
            element_mul(reduce_tensors_strides.clone(), &output_tensor_strides); //Check correctness here!

        let mut input_tensor_idx = 0;
        for output_tensor_idx in 0..output_tensor_shape.elements_count() {
            let output_tensor_strided_index =
                compute_strided_index(output_tensor_idx as usize, &output_tensor_strides);

            let idx_to_idx_output_el = dot(
                &output_tensor_strided_index,
                &output_elements_strides_in_input_tensor,
            );

            for reduce_tensor_idx in 0..reduce_tensors_shape.elements_count() {
                //inside the reduced subtensor. Assuming its first element is of index 0 (locally).
                let offset_from_first_element_of_subtensor = dot(
                    &compute_strided_index(reduce_tensor_idx as usize, &reduce_tensors_strides),
                    &input_tensor_strides,
                );

                let input_tensor_offset =
                    idx_to_idx_output_el + offset_from_first_element_of_subtensor;

                idx_to_input_offset_vec[input_tensor_idx] = input_tensor_offset;

                input_tensor_idx += 1;
            }
        }

        let plan = ReducePlan {
            idx_to_input_offsets: idx_to_input_offset_vec,
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

`ReducePlan` struct holds the initial permutation (mapping, not actual mapped data) - before reduction, and the final permutation - after reduction.
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

Each of these slices may be divided into thread blocks of size given by 

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
Power of 2 block size is used to ensure that reduce kernel will not skip any element while iterating over strides.

Assume for simplicity that MAX_BLOCK_LEN=4. Each of these boxes will need to be reduced with 2 separate blocks of size 4.
The problem is each of such "block level" reduction will only be capable of reducing 4 elements at once.
Thus we will be again left with 2 elements to reduce for each box.

### Solution 1

use the following planning algorithm
```rust
struct ReduceStep {
    block_size_x: u32,
    grid_dim: (u32, u32, u32),
    reduce_subinput_len: u32,
}

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


  auto group = cooperative_groups::this_grid();
  group.sync(); //if one uses input as new output in the iterations, this might be important to ensure that no datarace happen.

  if (tid == 0) {
    auto out_id = blockIdx.x + gridDim.x * blockIdx.y;
    out[out_id] = shared[0];
  }
}
```

And its launch on host side is:
```rust
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


        // it may use use aliasing workspace/out. It should be ok. 
        // Threads only use data within their block and they're synchronized with barriers on that level.

        let params = (&workspace, out, step.reduce_subinput_len);

        unsafe {
            kernel.clone().launch(cfg, params)?;
        }

        dev.synchronize()?;
        println!("next step\n\n\n");
    }

    Ok(())
}

```
each block needs sizeof(T) * blockDim.x bytes of shared memory. Indexing in `shared[tid]` is modulo blockDim.x.
This kernel will work well for sizeof(T)==4. Each consecutive 4 bytes in shared memory correspond to a bank.
Shared memory is fast, but if more then 2 threads try access memory (even if it doesnt overlap) within the same memory bank, it will need to be serialized into multiple cycles. 
The reason why kernel will work well is that each index will correspond to a separate memory bank. i.e. memory bank conflict may occur iff 2 threads explicity try to access the same cell.

Given planning algorithm for the first iteration in the given example (2, 3, 4) [0, 2], 
it requires us to launch blocks in the following grid
```
[block_0, block_1] 
[block_2, block_3]
[block_4, block_5]
```
each row corresponds to subinput and column to a specific block.
e.g. block_3 will handle data mapped as [10,  3,  7,  11].
Each of these blocks will of course consist of 4 threads.

For this specific iteration step, in the end, block_i will write to exactly ith memory cell in the output buffer.
This is relevant, as again in the next iteration, output buffer may be reused as a new input buffer.

With the given planning and execution, this should generalize to any number of steps.

### Solution 2
Use cooperative groups, I would need to actually research this more though.

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

sample setup for max reduce kernel:
```rust
#[test]
pub fn test_max_large() {
    let cuda_dev = CudaDevice::new(0).expect("could not create cuda device");

    const N: usize = 100_000;
    let mut input_host = vec![0f32; N];
    input_host[N / 2 - 1] = 3f32;
    input_host[0] = 2f32;
    input_host[N - 1] = 4f32;

    let input_dev = cuda_dev
        .htod_sync_copy(&input_host)
        .expect("could not alloc and copy");

    let mut output = cuda_dev.alloc_zeros(2).expect("could not alloc");

    unsafe {
        //N/2 is a subinput len.
        reduce(input_dev, N, N / 2, &mut output, cuda_dev.clone(), MaxOp)
            .expect("could not reduce");
    }

    let res = cuda_dev
        .dtoh_sync_copy(&mut output)
        .expect("could not copy to host");

    assert_eq!(res, vec![3f32, 4f32]);
}
```

Softmax itself may be implemented as a host-side module that calls multiple kernels one after another.
1) apply the permutation mapping.
2) max reduce on given reduce dimensions.
3) To subtract data, we may again launch a grid of dim (reduced_subtensor_size.div_ceil(block_size), number_of_reduced_subtensors, 1). Each thread will have a single memory cell to handle. No device level sync is needed. Each block, will be able to access the "maximal x" for its subtensor by accessing blockIdx.y element of a buffer given from reduce operation.

4) numeric, element wise exponent kernel - no need to split the execution in any significant way. Just apply for all elements of a buffer.
5) repeat 2-3, but with sum operator on an element-wise exponentiated buffer and apply numeric division instead of subtraction.
6) element wise multiply by normalization parameter alpha.

To map buffer back to the shape we initially had, just reverse the initial permutation and again apply mapping kernel.

AFAIK if one used cooperative groups for reduction it would be possible to pack all of these into a single kernel with one host-sided launch.
