#include "unary.cuh"
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>

template <typename T> struct Exp {
  __device__ auto operator()(T a) const -> T { return std::exp(a); }
};

template <typename T> __device__ auto exp(T *lhs, std::uint32_t size) -> void {
  unary_op<T, Exp<T>>(lhs, size, Exp<T>{});
}
#define EXTERN(T, SUFFIX)                                                      \
  extern "C" __global__ void exp_##SUFFIX(T *lhs, const T *rhs,                \
                                          std::uint32_t size) {                \
    exp(lhs, size);                                                            \
  }

EXTERN(float, f32)
