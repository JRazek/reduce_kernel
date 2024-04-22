#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

template <typename T, typename Op>
__device__ auto binary_op(T *lhs, const T *rhs, std::size_t size, const Op op)
    -> void {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    lhs[idx] = op(lhs[idx], rhs[idx]);
  }
}
