#include <cuda.h>
#include <cuda_runtime_api.h>

#include <functional>

#include "reduce_kernel.cuh"

template <typename T> struct Plus {
  __device__ auto operator()(T a, T b) const -> T { return a + b; }
};

// TODO - research "atomic max"
template <typename T> struct Max {
  __device__ auto operator()(T a, T b) const -> T { return a + b; }
};

template <typename T>
/**
 * @param global_offsets
         preprocessed on CPU.
         for each consecutive idx in kernel, one needs to associate appropriate
         offsets in data. why? We want to have data corresponding to the same
         reduced tensor in the consecutive memory cells.
 * @return
 */
__device__ auto softmax(const T *in, T *out, const std::size_t *global_offsets)
    -> void {

  auto tid = threadIdx.x;                           // # in block.
  auto idx = blockIdx.x * blockDim.x + threadIdx.x; // # in grid.
  auto tensor_idx = global_offsets[idx];            // offset in data.

  auto x = in[tensor_idx];

  auto max_x = reduce(x, Max<T>{});

  auto e_xi = std::exp(in[idx] - max_x);
  auto sum = reduce(e_xi, Plus<T>{});

  auto y_i = e_xi / sum;

  out[tensor_idx] = y_i;
}

#define EXTERN(T, SUFFIX)                                                      \
  __global__ void softmax_##SUFFIX(T *in, T *out, T mul) {                     \
    printf("softmax_\n");                                                      \
  }

// actually they are not always f32/f64 by cpp standard but for simplicity -
// assume that yes.
extern "C" {
__global__ void softmax_f32(float *in, float *out, float mul,
                            std::size_t *global_offsets) {
  softmax(in, out, global_offsets);
}
}
