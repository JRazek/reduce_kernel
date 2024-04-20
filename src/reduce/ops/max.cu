#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "reduce_kernel.cuh"

// TODO - research "atomic max"
template <typename T> struct MaxOp {
  __device__ auto operator()(T a, T b) const -> T { return a > b ? a : b; }
};

// calculates maximum in each block and stores it in out[blockIdx.x]
template <typename T> __device__ auto max_reduce(const T *in, T *out) -> void {
  reduce<T, MaxOp<T>>(in, out, MaxOp<T>{});
}

#define EXTERN(T, SUFFIX)                                                      \
  extern "C" __global__ void max_reduce_##SUFFIX(T *in, T *out) {              \
    max_reduce(in, out);                                                       \
  }

// actually they are not always f32/f64 by cpp standard but for simplicity -
// assume that yes.
EXTERN(float, f32) // currently lets keep it and make it f32 only.
