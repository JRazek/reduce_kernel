#include <array>
#include <cuda.h>
#include <cuda_runtime_api.h>

struct ReduceOp {};

template <typename T>
/**
 * @param global_offsets
         preprocessed on CPU.
         for each consecutive idx in kernel, one needs to associate appropriate
         offsets in data. why? We want to have data corresponding to the same
         reduced tensor in the consecutive memory cells.
 * @return
 */
__device__ auto softmax(T *in, T *out, std::size_t *global_offsets,
                        const ReduceOp *reduce_op) -> void {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto global_offset = global_offsets[idx];

  auto exp = std::exp(in[global_offset]);
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
  softmax(in, out, global_offsets, nullptr);
}
}
