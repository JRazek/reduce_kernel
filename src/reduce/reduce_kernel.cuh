#include <concepts>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>

template <typename T, typename Op>
__device__ auto reduce(const T *in, T *out, std::uint32_t reduce_input_len,
                       Op reduce_op) -> void {
  auto tid = threadIdx.x;                           // # in block.
  auto idx = blockIdx.x * blockDim.x + threadIdx.x; // # in grid.

  // this is incorrect.
  if (idx >= reduce_input_len) {
    return;
  }

  printf("in[%d]: %f\n", idx, in[idx]);

  extern __shared__ T shared[];
  shared[tid] = in[idx];
  printf("shared[tid]: %d\n", shared[tid]);

  __syncthreads();

  for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      auto res = reduce_op(shared[tid], shared[tid + s]);
      printf("reduce(shared[%d], shared[%d]): %f\n", tid, tid + s, res);
      shared[tid] = res;
    }
    __syncthreads();
  }

  out[blockIdx.x] = shared[0];
}
