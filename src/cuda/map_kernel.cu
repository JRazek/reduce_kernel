#include <concepts>
#include <cuda.h>
#include <cuda_runtime_api.h>

template <typename T>
__device__ auto map_offsets_in_place(const T *in, T *out,
                                     const std::size_t *idx_to_offsets)
    -> void {
  auto tid = threadIdx.x;
  auto idx = blockIdx.x * blockDim.x + threadIdx.x; // # in grid.
  // YOU STILL NEED TO CHECK FOR OUT OF BOUNDS KERNELS!!!!!!!
  // HERE AS WELL!!!

  extern __shared__ T shared[];

  auto tensor_idx = idx_to_offsets[idx]; // offset in data.
  shared[tid] = in[tensor_idx];

  // in case when in/out are the same (which will be the case for most cases).
  // first read to memory, cache it, then potentially overwrite input with
  // mapped data.
  __syncthreads();

  out[idx] = shared[tid];
}

#define EXTERN(T, SUFFIX)                                                      \
  extern "C" __global__ void map_offsets_in_place_##SUFFIX(                    \
      const T *in, T *out, const std::size_t *offsets) {                       \
    map_offsets_in_place(in, out, offsets);                                    \
  }

// actually they are not always f32/f64 by cpp standard but for simplicity -
// assume that yes.
EXTERN(float, f32) // currently lets keep it and make it f32 only.
