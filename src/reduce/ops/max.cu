#include <cuda.h>
#include <cuda_runtime_api.h>

#include "reduce_kernel.cuh"

// TODO - research "atomic max"
template <typename T> struct MaxOp {
  __device__ auto operator()(T a, T b) const -> T { return a > b ? a : b; }
};

// calculates maximum in each block and stores it in out[blockIdx.x]
template <typename T>
__device__ auto max_reduce(const T *in, T *out,
                           const std::size_t *global_offsets) -> void {

  auto idx = blockIdx.x * blockDim.x + threadIdx.x; // # in grid.
  auto tensor_idx = global_offsets[idx];            // offset in data.

  auto x = in[tensor_idx];

  auto max_x = reduce(x, MaxOp<T>{});

  out[blockIdx.x] = max_x;
}

#define EXTERN(T, SUFFIX)                                                      \
  extern "C" __global__ void max_reduce_##SUFFIX(                              \
      T *in, T *out, const std::size_t *global_offsets) {                      \
    max(in, out, global_offsets);                                              \
  }

// actually they are not always f32/f64 by cpp standard but for simplicity -
// assume that yes.
// EXTERN(float, f32) // currently lets keep it and make it f32 only.
