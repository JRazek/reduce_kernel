#pragma once

#include <cooperative_groups.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>

constexpr auto Debug = false;

template <typename T, typename Op>
__device__ auto reduce(const T *in, T *out, std::uint32_t reduce_subinput_len,
                       Op reduce_op) -> void {
  auto tid_in_subinput =
      threadIdx.x + blockIdx.x * blockDim.x; // in particular subinput

  auto tid_in_grid =
      tid_in_subinput + reduce_subinput_len * blockIdx.y; // in entire input

  auto tid_in_block = threadIdx.x; // in block

  auto grid = cooperative_groups::this_grid();

  auto is_thread_valid = tid_in_subinput < reduce_subinput_len;

  extern __shared__ T shared[];

  if (is_thread_valid) {
    shared[tid_in_block] = in[tid_in_grid];
  }

  // after read to local memory only block sync is needed - no writes to global
  // before the final grid sync.
  __syncthreads();

  for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid_in_block < s && tid_in_subinput + s < reduce_subinput_len) {
      assert(is_thread_valid);

      auto lhs = shared[tid_in_block];
      auto rhs = shared[tid_in_block + s];
      auto res = reduce_op(lhs, rhs);

      shared[tid_in_block] = res;
    }

    __syncthreads();
  }

  grid.sync(); // if output and input are same, then in order to avoid data
               // race, we need to first store and sync.

  if (tid_in_block == 0) {
    assert(is_thread_valid);
    auto out_id = blockIdx.x + gridDim.x * blockIdx.y;

    out[out_id] = shared[0];
  }
}
