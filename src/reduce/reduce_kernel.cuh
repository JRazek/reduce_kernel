#include <concepts>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>

template <typename T, typename Op>
__device__ auto reduce(const T *in, T *out, std::uint32_t reduce_input_len,
                       Op reduce_op) -> void {
  auto tid = threadIdx.x; // in block
  auto subinput_id =
      threadIdx.x + blockIdx.x * blockDim.x; // in particular subinput

  auto grid_id = subinput_id + reduce_input_len * blockIdx.y; // in entire input

  // in each gridDim.x, it may happen, that for gridId.y==n and gridId.y==n+1,
  // ending and starting grid_id will coincide. Also to make sure that no out of
  // bound access happens, this is used.
  // This will lead to branch divergence only on boundaries.
  if (subinput_id >= reduce_input_len) {
    //    printf("thread with subinput_id: %d, on blockIdx.x: %d, and blockIdy.y
    //    %d, "
    //           "exitted\n",
    //           subinput_id, blockIdx.x, blockIdx.y);
    return;
  }

  //  printf("blockDim.x: %d\n", blockDim.x);
  //  printf("thread with subinput_id: %d, on blockIdx.x: %d, and blockIdy.y %d,
  //  "
  //         "passed\n",
  //         subinput_id, blockIdx.x, blockIdx.y);

  extern __shared__ T shared[];
  shared[tid] = in[grid_id];
  //  printf("in[%d]: %f\n", grid_id, in[grid_id]);

  __syncthreads();

  for (auto s =
           static_cast<std::uint32_t>(ceil(static_cast<float>(blockDim.x) / 2));
       s > 0; s >>= 1) {
    if (tid < s && subinput_id + s < reduce_input_len) {
      auto res = reduce_op(shared[tid], shared[tid + s]);
      shared[tid] = res;

      //      printf("reduce(shared[%d], shared[%d]): %f\n", tid, tid + s, res);
    }

    __syncthreads();
  }

  // https://stackoverflow.com/a/9587804/14508019
  // its undefined which thread actually writes to this cell, but regardless of
  // that, the value will be the same.
  out[blockIdx.y] = shared[0];
  //  if (blockIdx.x == 0) {
  //    printf("out[%d]: %f\n", blockIdx.y, out[blockIdx.y]);
  //  }
}
