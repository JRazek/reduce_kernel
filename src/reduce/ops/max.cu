#include <cooperative_groups/reduce.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "reduce_kernel.cuh"

template <typename T> struct MaxOp {
  __device__ auto operator()(T a, T b) const -> T { return a > b ? a : b; }
};

// calculates maximum in each block and stores it in out[blockIdx.x]
template <typename T>
__device__ auto max_reduce(const T *in, T *out,
                           std::uint32_t reduce_subinput_len) -> void {
  reduce<T, MaxOp<T>>(in, out, reduce_subinput_len, MaxOp<T>{});
}

#define EXTERN(T, SUFFIX)                                                      \
  extern "C" __global__ void max_reduce_##SUFFIX(                              \
      T *in, T *out, std::uint32_t reduce_subinput_len) {                      \
    max_reduce(in, out, reduce_subinput_len);                                  \
  }

// actually they are not always f32/f64 by cpp standard but for simplicity -
// assume that yes.
EXTERN(float, f32) // currently lets keep it and make it f32 only.
