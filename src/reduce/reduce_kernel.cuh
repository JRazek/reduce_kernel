#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <cooperative_groups.h>

constexpr auto Debug = false;

template <typename T, typename Op>
__device__ auto reduce(const T *in, T *out, std::uint32_t reduce_input_len,
                       Op reduce_op) -> void {
  auto subinput_id =
      threadIdx.x + blockIdx.x * blockDim.x; // in particular subinput

  auto grid_id = subinput_id + reduce_input_len * blockIdx.y; // in entire input

  auto tid = threadIdx.x; // in block

  auto grid = cooperative_groups::this_grid();

  extern __shared__ T shared[];

  auto input = in[grid_id];

  grid.sync();//if output and input are same, then in order to avoid data race, we need to first store and sync.

  // in each gridDim.x, it may happen, that for gridId.y==n and gridId.y==n+1,
  // ending and starting grid_id will coincide. Also to make sure that no out of
  // bound access happens, this is used.
  // This will lead to branch divergence only on boundaries.
  // I cannot return from the thread in the beginning, as it will lead to a deadlock

  if (subinput_id < reduce_input_len) {
  	shared[tid] = input;
  }

  __syncthreads();

  for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && subinput_id + s < reduce_input_len) {
      auto lhs = shared[tid];
      auto rhs = shared[tid + s];
      auto res = reduce_op(lhs, rhs);

      shared[tid] = res;
    }

    __syncthreads();
  }

  if (tid == 0) {
    auto out_id = blockIdx.x + gridDim.x * blockIdx.y;
    out[out_id] = shared[0];
  }
}


