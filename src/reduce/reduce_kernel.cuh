#include <concepts>
#include <cuda.h>
#include <cuda_runtime_api.h>

template <typename T>
concept ReduceOp = requires(T a, T b) {
  { a + b } -> std::same_as<T>;
};

template <typename T, ReduceOp Op>
__device__ auto reduce(T *in, T *out, const std::size_t *global_offsets,
                       const Op reduce_op) -> void {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto tensor_index = global_offsets[idx];

  extern __shared__ T shared[];
}
