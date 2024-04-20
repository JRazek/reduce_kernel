#include <cuda.h>
#include <cuda_runtime_api.h>

template <typename T> __device__ auto exp_kernel(const T *in, T *out) -> void {
  auto tid = threadIdx.x;
  auto idx = blockIdx.x * blockDim.x + threadIdx.x; // # in grid.

  out[idx] = std::exp(in[idx]);
}

#define EXTERN(T, SUFFIX)                                                      \
  extern "C" __global__ void exp_kernel_##SUFFIX(const T *in, T *out) {        \
    exp_kernel(in, out);                                                       \
  }

// actually they are not always f32/f64 by cpp standard but for simplicity -
// assume that yes.
EXTERN(float, f32) // currently lets keep it and make it f32 only.
