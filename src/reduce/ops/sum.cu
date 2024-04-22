#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "reduce_kernel.cuh"

template <typename T> struct SumOp {
  __device__ auto operator()(T a, T b) const -> T { return a + b; }
};

// calculates maximum in each block and stores it in out[blockIdx.x]
template <typename T>
__device__ auto sum_reduce(const T *in, T *out,
                           std::uint32_t reduce_subinput_len) -> void {
  reduce<T, SumOp<T>>(in, out, reduce_subinput_len, SumOp<T>{});
}

#define EXTERN(T, SUFFIX)                                                      \
  extern "C" __global__ void sum_reduce_##SUFFIX(                              \
      T *in, T *out, std::uint32_t reduce_subinput_len) {                      \
    sum_reduce(in, out, reduce_subinput_len);                                  \
  }

// actually they are not always f32/f64 by cpp standard but for simplicity -
// assume that yes.
EXTERN(float, f32) // currently lets keep it and make it f32 only.
