#include <concepts>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>

template <typename T, typename Op>
__device__ auto reduce(const T *in, T *out, std::uint32_t reduce_input_len,
                       Op reduce_op) -> void {
  auto tid = threadIdx.x;                           // # in block.
  auto idx = blockIdx.x * blockDim.x + threadIdx.x; // # in grid.

  if (idx >= reduce_input_len) {
    return;
  }

  // YOU STILL NEED TO CHECK FOR OUT OF BOUNDS KERNELS!

  extern __shared__ T shared[];
  shared[tid] = in[idx];

  __syncthreads();

  for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] = reduce_op(shared[tid], shared[tid + s]);
    }
    __syncthreads();
  }

  out[blockIdx.x] = shared[0];
}
