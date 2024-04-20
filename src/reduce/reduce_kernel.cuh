#include <concepts>
#include <cuda.h>
#include <cuda_runtime_api.h>

template <typename T, typename Op>
__device__ auto reduce(const T *in, T *out, const std::size_t *global_offsets,
                       Op reduce_op) -> void {
  auto tid = threadIdx.x;                           // # in block.
  auto idx = blockIdx.x * blockDim.x + threadIdx.x; // # in grid.

  // get rid of global offsets here, apply mapping kernel before calling reduce
  // kernel (host level).
  auto tensor_idx = global_offsets[idx]; // offset in data.

  extern __shared__ T shared[];
  shared[tid] = in[tensor_idx];

  __syncthreads();

  for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] = reduce_op(shared[tid], shared[tid + s]);
    }
    __syncthreads();
  }

  out[blockIdx.x] = shared[0];

  // when done with the second step, apply mapping kernel again in reverse
  // direction (again host side).
}
