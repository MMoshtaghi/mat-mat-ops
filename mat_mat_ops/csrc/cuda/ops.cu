#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace mat_mat_ops {

#define CEIL_DIV(M, size) (((M) + (size)-1) / (size))
// Threads per CTA dimension
#define NUM_THREADS_PER_DIM_PER_BLOCK 32 // it should be multiple of 32 as they will be translated to number of warps

// Shared Memory is private for each Block (SM), and shared among the threads and warps (SP)
// we set the SHMEM to have 1 Register or element for each thread in that Block 
#define SHMEM_SIZE 32*32

/////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void muladd_kernel(int numel, const float* a, const float* b, float c, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] * b[idx] + c;
}

at::Tensor mymuladd_cuda(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();

  int numel = a_contig.numel();
  muladd_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, c, result_ptr);
  return result;
}

__global__ void mul_kernel(int numel, const float* a, const float* b, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] * b[idx];
}


at::Tensor mymul_cuda(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  int numel = a_contig.numel();
  mul_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, result_ptr);
  return result;
}

///////////////////////////////////////////////////////////////////////////////////

__global__ void add_kernel(int numel, const float* a, const float* b, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] + b[idx];
}

void myadd_out_cuda(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(b.sizes() == out.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(out.dtype() == at::kFloat);
  TORCH_CHECK(out.is_contiguous());
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = out.data_ptr<float>();
  int numel = a_contig.numel();
  add_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, result_ptr);
}

///////////////////////////////////////////////////////////////////////////////////
// ! Mat Mul
__global__ void cachedTiledBatchedMatMatMul(const int M, const int N, const int K,
                                          const float *a, const float *b, float *result) {
  // cols:x:n , rows:y:m
  // Compute each thread's global row and column index
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  int bs = blockIdx.z;

  int tile_size = blockDim.x;

  // Statically allocated shared memory
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  // Accumulate in temporary variable
  float tmp = 0;

  // Sweep tile across matrix
  for (int k = 0; k < K; k += tile_size) {
    // Load in elements for this tile
    if (k+threadIdx.x < K) s_a[threadIdx.y * tile_size + threadIdx.x] = a[bs*M*K + m*K + (k+threadIdx.x)];
    else s_a[threadIdx.y * tile_size + threadIdx.x] = 0.0;
    if (k+threadIdx.y < K) s_b[threadIdx.y * tile_size + threadIdx.x] = b[bs*K*N + (k+threadIdx.y) * N + n];
    else s_b[threadIdx.y * tile_size + threadIdx.x] = 0.0;

    // Wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // Do matrix multiplication on the small matrix (2D tile) , BATCH_SIZE is computed  by 3D Blocks
    for (int t = 0; t < tile_size; t++) {
      tmp += s_a[threadIdx.y * tile_size + t] * s_b[t * tile_size + threadIdx.x];
    }

    // Wait for all threads to finish using current tiles before loading in new
    // ones
    __syncthreads();
  }

  // Write back results with Boundary check 
  if (m<M && n<N) result[bs*M*N + m*N + n] = tmp;
}



at::Tensor mat_mat_mul_cuda(const at::Tensor& mat1, const at::Tensor& mat2) {
  // mat1.dim == 3
  // mat2.dim == 3
  TORCH_CHECK(mat1.size(0) == mat2.size(0));
  TORCH_CHECK(mat1.size(2) == mat2.size(1));

  TORCH_CHECK(mat1.dtype() == at::kFloat);
  TORCH_CHECK(mat2.dtype() == at::kFloat);

  TORCH_INTERNAL_ASSERT(mat1.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(mat2.device().type() == at::DeviceType::CUDA);

  at::Tensor mat1_contig = mat1.contiguous();
  at::Tensor mat2_contig = mat2.contiguous();
  at::Tensor result = torch::empty( {mat1.size(0), mat1.size(1), mat2.size(2)} , mat1_contig.options());

  const float* mat1_ptr = mat1_contig.data_ptr<float>();
  const float* mat2_ptr = mat2_contig.data_ptr<float>();
  
  float* result_ptr = result.data_ptr<float>();

  const int M = mat1.size(1); // minimum 32
  const int N = mat2.size(2); // minimum 32
  const int K = mat1.size(2); // minimum 32

  const int BATCH_SIZE = mat1.size(0);
  
  dim3 threads_per_block(NUM_THREADS_PER_DIM_PER_BLOCK, NUM_THREADS_PER_DIM_PER_BLOCK, 1);
  // cols:x:n , rows:y:m
  int BLOCKS_X = CEIL_DIV(N , NUM_THREADS_PER_DIM_PER_BLOCK);
  int BLOCKS_Y = CEIL_DIV(M , NUM_THREADS_PER_DIM_PER_BLOCK);
  dim3 blocks_per_grid( BLOCKS_X , BLOCKS_Y , BATCH_SIZE);

  cachedTiledBatchedMatMatMul<<<blocks_per_grid, threads_per_block>>>(M, N, K,
                                                                    mat1_ptr, mat2_ptr, result_ptr);
  
  return result;
}


///////////////////////////////////////////////////////////////////////////////////
// ! Mat L1 fw
__global__ void cachedTiledBatchedMatMatL1(const int M, const int N, const int K,
                                          const float *a, const float *b, float *result) {
  // cols:x:n , rows:y:m
  // Compute each thread's global row and column index
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  int bs = blockIdx.z;

  int tile_size = blockDim.x;

  // Statically allocated shared memory
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  // Accumulate in temporary variable
  float tmp = 0;

  // Sweep tile across matrix
  for (int k = 0; k < K; k += tile_size) {
    // Load in elements for this tile
    if (k+threadIdx.x < K) s_a[threadIdx.y * tile_size + threadIdx.x] = a[bs*M*K + m*K + (k+threadIdx.x)];
    else s_a[threadIdx.y * tile_size + threadIdx.x] = 0.0;
    if (k+threadIdx.y < K) s_b[threadIdx.y * tile_size + threadIdx.x] = b[bs*K*N + (k+threadIdx.y) * N + n];
    else s_b[threadIdx.y * tile_size + threadIdx.x] = 0.0;

    // Wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // Do matrix multiplication on the small matrix (2D tile) , BATCH_SIZE is computed  by 3D Blocks
    for (int t = 0; t < tile_size; t++) {
      tmp += fabsf( s_a[threadIdx.y * tile_size + t] - s_b[t * tile_size + threadIdx.x] ) ;
    }

    // Wait for all threads to finish using current tiles before loading in new
    // ones
    __syncthreads();
  }

  // Write back results with Boundary check 
  if (m<M && n<N) result[bs*M*N + m*N + n] = tmp;
}



at::Tensor mat_mat_l1_cuda(const at::Tensor& mat1, const at::Tensor& mat2) {
  // mat1.dim == 3
  // mat2.dim == 3
  TORCH_CHECK(mat1.size(0) == mat2.size(0));
  TORCH_CHECK(mat1.size(2) == mat2.size(1));

  TORCH_CHECK(mat1.dtype() == at::kFloat);
  TORCH_CHECK(mat2.dtype() == at::kFloat);

  TORCH_INTERNAL_ASSERT(mat1.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(mat2.device().type() == at::DeviceType::CUDA);

  at::Tensor mat1_contig = mat1.contiguous();
  at::Tensor mat2_contig = mat2.contiguous();
  at::Tensor result = torch::empty( {mat1.size(0), mat1.size(1), mat2.size(2)} , mat1_contig.options());

  const float* mat1_ptr = mat1_contig.data_ptr<float>();
  const float* mat2_ptr = mat2_contig.data_ptr<float>();
  
  float* result_ptr = result.data_ptr<float>();

  const int M = mat1.size(1); // minimum 32
  const int N = mat2.size(2); // minimum 32
  const int K = mat1.size(2); // minimum 32

  const int BATCH_SIZE = mat1.size(0);
  
  dim3 threads_per_block(NUM_THREADS_PER_DIM_PER_BLOCK, NUM_THREADS_PER_DIM_PER_BLOCK, 1);
  // cols:x:n , rows:y:m
  int BLOCKS_X = CEIL_DIV(N , NUM_THREADS_PER_DIM_PER_BLOCK);
  int BLOCKS_Y = CEIL_DIV(M , NUM_THREADS_PER_DIM_PER_BLOCK);
  dim3 blocks_per_grid( BLOCKS_X , BLOCKS_Y , BATCH_SIZE);

  cachedTiledBatchedMatMatL1<<<blocks_per_grid, threads_per_block>>>(M, N, K,
                                                                    mat1_ptr, mat2_ptr, result_ptr);
  
  return result;
}


///////////////////////////////////////////////////////////////////////////////////
// ! Mat L1 bw
__global__ void cachedTiledBatchedMatMatL1_AGrad(const int M, const int N, const int K,
                                                const float *mat1, const float *mat2, const float *mat3, float *mat3_grad) {
  // cols:x:n , rows:y:m
  // Compute each thread's global row and column index
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  int bs = blockIdx.z;

  int tile_size = blockDim.x;

  // Statically allocated shared memory
  __shared__ float s_mat1[SHMEM_SIZE];
  __shared__ float s_mat2[SHMEM_SIZE];
  __shared__ float s_mat3[SHMEM_SIZE];

  // Accumulate in temporary variable
  float tmp = 0;

  // Sweep tile across matrix
  for (int k = 0; k < K; k += tile_size) {
    // Load in elements for this tile
    if (k+threadIdx.x < K) s_mat1[threadIdx.y * tile_size + threadIdx.x] = mat1[bs*M*K + m*K + (k+threadIdx.x)];
    else s_mat1[threadIdx.y * tile_size + threadIdx.x] = 0.0;
    if (k+threadIdx.y < K) s_mat2[threadIdx.y * tile_size + threadIdx.x] = mat2[bs*K*N + (k+threadIdx.y) * N + n];
    else s_mat2[threadIdx.y * tile_size + threadIdx.x] = 0.0;

    if (k == 0) s_mat3[threadIdx.y*tile_size + threadIdx.x] = mat3[bs*M*N + m*N + n];

    // Wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // Do matrix multiplication on the small matrix (2D tile) , BATCH_SIZE is computed  by 3D Blocks
    for (int t = 0; t < tile_size; t++) {
      float cmp = -1;
      if ( s_mat3[threadIdx.y*tile_size + threadIdx.x] > s_mat2[t * tile_size + threadIdx.x] ) cmp = 1;
      else if ( s_mat3[threadIdx.y*tile_size + threadIdx.x] == s_mat2[t * tile_size + threadIdx.x] ) cmp = 0;
      
      tmp += s_mat1[threadIdx.y * tile_size + t] * cmp ;
    }

    // Wait for all threads to finish using current tiles before loading in new
    // ones
    __syncthreads();
  }

  // Write back results with Boundary check 
  if (m<M && n<N) mat3_grad[bs*M*N + m*N + n] = tmp;
  // ! here we divide by K to normalize the range, but based on the chain rule, we must divide by N here bc of the division by K in forward pass.
  // ! So the test only works if you manually multiply the reference_output by ( K/N with K,N difinitions here or N/K with difinitions in reference)
}

__global__ void cachedTiledBatchedMatMatL1_BGrad(const int M, const int N, const int K,
                                                const float *mat1, const float *mat2, const float *mat3, float *mat3_grad) {
  // cols:x:n , rows:y:m
  // Compute each thread's global row and column index
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  int bs = blockIdx.z;

  int tile_size = blockDim.x;

  // Statically allocated shared memory
  __shared__ float s_mat1[SHMEM_SIZE];
  __shared__ float s_mat2[SHMEM_SIZE];
  __shared__ float s_mat3[SHMEM_SIZE];

  // Accumulate in temporary variable
  float tmp = 0;

  // Sweep tile across matrix
  for (int k = 0; k < K; k += tile_size) {
    // Load in elements for this tile
    if (k+threadIdx.x < K) s_mat1[threadIdx.y * tile_size + threadIdx.x] = mat1[bs*M*K + m*K + (k+threadIdx.x)];
    else s_mat1[threadIdx.y * tile_size + threadIdx.x] = 0.0;
    if (k+threadIdx.y < K) s_mat2[threadIdx.y * tile_size + threadIdx.x] = mat2[bs*K*N + (k+threadIdx.y) * N + n];
    else s_mat2[threadIdx.y * tile_size + threadIdx.x] = 0.0;

    if (k == 0) s_mat3[threadIdx.y*tile_size + threadIdx.x] = mat3[bs*M*N + m*N + n];

    // Wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // Do matrix multiplication on the small matrix (2D tile) , BATCH_SIZE is computed  by 3D Blocks
    for (int t = 0; t < tile_size; t++) {
      float cmp = 1;
      if ( s_mat1[threadIdx.y * tile_size + t] > s_mat3[threadIdx.y*tile_size + threadIdx.x] ) cmp = -1;
      else if ( s_mat1[threadIdx.y * tile_size + t] == s_mat3[threadIdx.y*tile_size + threadIdx.x] ) cmp = 0;
      
      tmp += cmp * s_mat2[t * tile_size + threadIdx.x] ;
    }

    // Wait for all threads to finish using current tiles before loading in new
    // ones
    __syncthreads();
  }

  // Write back results with Boundary check 
  if (m<M && n<N) mat3_grad[bs*M*N + m*N + n] = tmp;
  // ! here we divide by K to normalize the range, but based on the chain rule, we must divide by M here bc of the division by K in forward pass. 
  // ! So the test only works if you manually multiply the reference_output by ( K/M with K , M difinitions here or M/K with difinitions in reference)
}


at::Tensor mat_mat_l1_grad_cuda(const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& mat3, const bool grad_for_A) {
  // mat1.dim == 3
  // mat2.dim == 3
  TORCH_CHECK(mat1.size(0) == mat2.size(0));
  TORCH_CHECK(mat1.size(2) == mat2.size(1));

  TORCH_CHECK(mat1.size(1) == mat3.size(1));
  TORCH_CHECK(mat2.size(2) == mat3.size(2));

  TORCH_CHECK(mat1.dtype() == at::kFloat);
  TORCH_CHECK(mat2.dtype() == at::kFloat);

  TORCH_INTERNAL_ASSERT(mat1.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(mat2.device().type() == at::DeviceType::CUDA);

  at::Tensor mat1_contig = mat1.contiguous();
  at::Tensor mat2_contig = mat2.contiguous();
  at::Tensor mat3_contig = mat3.contiguous();

  at::Tensor mat3_grad = torch::empty( mat3_contig.sizes() , mat3_contig.options());

  const float* mat1_ptr = mat1_contig.data_ptr<float>();
  const float* mat2_ptr = mat2_contig.data_ptr<float>();
  const float* mat3_ptr = mat3_contig.data_ptr<float>();
  
  float* mat3_grad_ptr = mat3_grad.data_ptr<float>();

  const int M = mat1.size(1); // minimum 32
  const int N = mat2.size(2); // minimum 32
  const int K = mat1.size(2); // minimum 32

  const int BATCH_SIZE = mat1.size(0);

  dim3 threads_per_block(NUM_THREADS_PER_DIM_PER_BLOCK, NUM_THREADS_PER_DIM_PER_BLOCK, 1);

  int BLOCKS_X = CEIL_DIV(N , NUM_THREADS_PER_DIM_PER_BLOCK);
  int BLOCKS_Y = CEIL_DIV(M , NUM_THREADS_PER_DIM_PER_BLOCK);
  dim3 blocks_per_grid( BLOCKS_X , BLOCKS_Y , BATCH_SIZE);

  if (grad_for_A == true) cachedTiledBatchedMatMatL1_AGrad<<<blocks_per_grid, threads_per_block>>>(M, N, K,
                                                                                                  mat1_ptr, mat2_ptr, mat3_ptr, mat3_grad_ptr);
  else cachedTiledBatchedMatMatL1_BGrad<<<blocks_per_grid, threads_per_block>>>(M, N, K,
                                                                                mat1_ptr, mat2_ptr, mat3_ptr, mat3_grad_ptr);

  return mat3_grad;
}

///////////////////////////////////////////////////////////////////////////////////

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(mat_mat_ops, CUDA, m) {
  m.impl("mymuladd", &mymuladd_cuda);
  m.impl("mymul", &mymul_cuda);
  m.impl("myadd_out", &myadd_out_cuda);

  m.impl("mat_mat_mul", &mat_mat_mul_cuda);

  m.impl("mat_mat_l1", &mat_mat_l1_cuda);
  m.impl("mat_mat_l1_grad", &mat_mat_l1_grad_cuda);
}

}
