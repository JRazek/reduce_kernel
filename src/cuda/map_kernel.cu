#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

template <typename T>
__device__ auto map_offsets_in_place(T *data, const std::size_t *idx_to_offsets,
                                     std::size_t size) -> void {
  auto tid = threadIdx.x;
  auto idx = blockIdx.x * blockDim.x + threadIdx.x; // # in grid.

  if (idx >= size) {
    return;
  }

  auto tensor_idx = idx_to_offsets[idx]; // offset in data.
  auto input = data[tensor_idx];

  auto group = cooperative_groups::this_grid();

  group.sync();

  data[idx] = input;
}

#define EXTERN(T, SUFFIX)                                                      \
  extern "C" __global__ void map_offsets_##SUFFIX(                             \
      T *in, const std::size_t *offsets, std::size_t size) {                   \
    map_offsets_in_place(in, offsets, size);                                   \
  }

// actually they are not always f32/f64 by cpp standard but for simplicity -
// assume that yes.
EXTERN(float, f32) // currently lets keep it and make it f32 only.
