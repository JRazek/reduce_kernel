#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>

constexpr auto Debug = false;

template <typename T, typename Op>
__device__ auto reduce(const T *in, T *out, std::uint32_t reduce_input_len,
                       Op reduce_op) -> void {
  auto subinput_id =
      threadIdx.x + blockIdx.x * blockDim.x; // in particular subinput

  auto grid_id = subinput_id + reduce_input_len * blockIdx.y; // in entire input

  if constexpr (Debug) {
    if (grid_id == 0) {
      printf("input_addr: %p\n", in);
      printf("output_addr: %p\n", out);
    }
  }

  // in each gridDim.x, it may happen, that for gridId.y==n and gridId.y==n+1,
  // ending and starting grid_id will coincide. Also to make sure that no out of
  // bound access happens, this is used.
  // This will lead to branch divergence only on boundaries.
  if (subinput_id >= reduce_input_len) {
    if constexpr (Debug) {
      printf(
          "thread with subinput_id: %d, on blockIdx.x: %d, and blockIdy.y: %d "
          "exitted\n",
          subinput_id, blockIdx.x, blockIdx.y);
    }
    return;
  }

  auto tid = threadIdx.x; // in block

  extern __shared__ T shared[];
  shared[tid] = in[grid_id];

  __syncthreads();

  for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && subinput_id + s < reduce_input_len) {
      auto lhs = shared[tid];
      auto rhs = shared[tid + s];
      auto res = reduce_op(lhs, rhs);

      shared[tid] = res;

      if constexpr (Debug) {
        printf("stride: %d, blockId.y: %d, blockIdx.x: %d, subinput_id: %d, "
               "reduce(shared[%d], "
               "shared[%d]) = reduce(%f, %f): %f\n",
               s, blockIdx.y, blockIdx.x, grid_id, tid, tid + s, lhs, rhs, res);
      }
    }

    __syncthreads();
  }

  if (tid == 0) {
    auto out_id = blockIdx.x + gridDim.x * blockIdx.y;
    out[out_id] = shared[0];

    if constexpr (Debug) {
      printf("saving in out[%d]: %f\n", out_id, shared[0]);
    }
  }
}


