#include <concepts>
#include <cuda.h>
#include <cuda_runtime_api.h>

// template <typename >
// concept ReduceOp = requires(T a, T b) {
//   { T(a, b) } -> std::same_as<T>;
// };

template <typename T, typename Op>
__device__ auto reduce(T in, const Op reduce_op) -> T {
  auto tid = threadIdx.x;                           // # in block.
  auto idx = blockIdx.x * blockDim.x + threadIdx.x; // # in grid.

  extern __shared__ T shared[];
  shared[tid] = in;

  __syncthreads();

  for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] = reduce_op(shared[tid], shared[tid + s]);
    }
    __syncthreads();
  }

  // broadcast should happen from the first thread of a block.

  auto first_block_thread = blockIdx.x * blockDim.x;

  printf("first_block_thread: %d\n", first_block_thread);
  return shared[first_block_thread]; // is broadcast happening implicitly here
                                     // or I need to use sth else?
}
