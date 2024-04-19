#include <cuda.h>
#include <cuda_runtime_api.h>

#include "reduce_kernel.cuh"

template <typename T> struct Plus {
  __device__ auto operator()(T a, T b) const -> T { return a + b; }
};

// TODO - research "atomic max"
template <typename T> struct Max {
  __device__ auto operator()(T a, T b) const -> T { return a > b ? a : b; }
};

template <typename T>
__device__ auto softmax(const T *in, T *out, const std::size_t *global_offsets)
    -> void {

  auto tid = threadIdx.x;                           // # in block.
  auto idx = blockIdx.x * blockDim.x + threadIdx.x; // # in grid.
  auto tensor_idx = global_offsets[idx];            // offset in data.

  auto x = in[tensor_idx];

  //  auto max_x = reduce(x, Max<T>{});

  //  auto e_xi = std::exp(in[idx] - max_x);
  auto sum = reduce(x, Plus<T>{});

  //  auto y_i = e_xi / sum;

  out[tensor_idx] = sum;
}

#define EXTERN(T, SUFFIX)                                                      \
  extern "C" __global__ void softmax_##SUFFIX(                                 \
      T *in, T *out, const std::size_t *global_offsets) {                      \
    softmax(in, out, global_offsets);                                          \
  }

// actually they are not always f32/f64 by cpp standard but for simplicity -
// assume that yes.
EXTERN(float, f32) // currently lets keep it and make it f32 only.
