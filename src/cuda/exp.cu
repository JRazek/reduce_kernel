#include <cuda.h>
#include <cuda_runtime_api.h>

template <typename T> __device__ auto exp(T *in, T *out) -> void {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;

  out[idx] = in[idx];
}

#define EXTERN(T, SUFFIX)                                                      \
  __global__ void softmax_##SUFFIX(T *in, T *out, T mul) {                     \
    printf("softmax_\n");                                                      \
  }

// actually they are not always f32/f64 by cpp standard but for simplicity -
// assume that yes.
extern "C" {
__global__ void softmax_f32(float *in, float *out, float mul) {}
}
