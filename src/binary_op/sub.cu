#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "binary.cuh"

template <typename T> struct Sub {
  __device__ auto operator()(T a, T b) const -> T { return a - b; }
};

template <typename T>
__device__ auto sub(T *lhs, const T *rhs, std::uint32_t size) -> void {
  binary_op<T, Sub<T>>(lhs, rhs, size, Sub<T>{});
}
#define EXTERN(T, SUFFIX)                                                      \
  extern "C" __global__ void sub_##SUFFIX(T *lhs, const T *rhs,                \
                                          std::uint32_t size) {                \
    sub(lhs, rhs, size);                                                       \
  }

EXTERN(float, f32)
