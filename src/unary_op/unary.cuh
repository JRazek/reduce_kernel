#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

template <typename T, typename Op>
__device__ auto unary_op(T *data, std::size_t size, const Op op) -> void {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    data[idx] = op(data[idx]);
  }
}
