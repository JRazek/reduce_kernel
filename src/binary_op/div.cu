#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "binary.cuh"

template <typename T> struct Div {
  __device__ auto operator()(T a, T b) const -> T { return a / b; }
};

template <typename T>
__device__ auto div(T *lhs, const T *rhs, std::uint32_t size) -> void {
  binary_op<T, Div<T>>(lhs, rhs, size, Div<T>{});
}

#define EXTERN(T, SUFFIX)                                                      \
  extern "C" __global__ void div_##SUFFIX(T *lhs, const T *rhs,                \
                                          std::uint32_t size) {                \
    div(lhs, rhs, size);                                                       \
  }

EXTERN(float, f32)
