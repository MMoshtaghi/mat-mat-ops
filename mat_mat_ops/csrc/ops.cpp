#include <torch/extension.h>
#include <vector>

namespace mat_mat_ops {

at::Tensor mymuladd_cpu(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
  }
  return result;
}

at::Tensor mymul_cpu(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i];
  }
  return result;
}

// An example of an operator that mutates one of its inputs.
void myadd_out_cpu(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(b.sizes() == out.sizes());

  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(out.dtype() == at::kFloat);
  
  TORCH_CHECK(out.is_contiguous());
  
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CPU);
  
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = out.data_ptr<float>();

  for (int64_t i = 0; i < out.numel(); i++) {
    result_ptr[i] = a_ptr[i] + b_ptr[i];
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

at::Tensor mat_mat_mul_cpu(const at::Tensor& mat1, const at::Tensor& mat2) {
  // mat1.dim == 3
  // mat2.dim == 3
  TORCH_CHECK(mat1.size(0) == mat2.size(0));
  TORCH_CHECK(mat1.size(2) == mat2.size(1));

  TORCH_CHECK(mat1.dtype() == at::kFloat);
  TORCH_CHECK(mat2.dtype() == at::kFloat);

  TORCH_INTERNAL_ASSERT(mat1.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(mat2.device().type() == at::DeviceType::CPU);

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
  
  for (int64_t bs = 0; bs < BATCH_SIZE; bs++) {
    // For every row of a and result ...
    for (int64_t m = 0; m < M; m++) {
      // For every column of b and result ...
      for (int64_t n = 0; n < N; n++) {
        // For every element in the row of a and column of b
        float tmp = 0;
        for (int64_t k = 0; k < K; k++) {
          // Accumulate the partial results
          tmp += mat1_ptr[bs*M*K + m*K + k] * mat2_ptr[bs*K*N + k*N + n];
        }
        result_ptr[bs*M*N + m*N + n] = tmp;
      }
    }
  }
  
  return result;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////

at::Tensor mat_mat_l1_cpu(const at::Tensor& mat1, const at::Tensor& mat2) {
  // mat1.dim == 3
  // mat2.dim == 3
  TORCH_CHECK(mat1.size(0) == mat2.size(0));
  TORCH_CHECK(mat1.size(2) == mat2.size(1));

  TORCH_CHECK(mat1.dtype() == at::kFloat);
  TORCH_CHECK(mat2.dtype() == at::kFloat);

  TORCH_INTERNAL_ASSERT(mat1.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(mat2.device().type() == at::DeviceType::CPU);

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
  
  for (int64_t bs = 0; bs < BATCH_SIZE; bs++) {
    // For every row of a and result ...
    for (int64_t m = 0; m < M; m++) {
      // For every column of b and result ...
      for (int64_t n = 0; n < N; n++) {
        // For every element in the row of a and column of b
        float tmp = 0;
        for (int64_t k = 0; k < K; k++) {
          // Accumulate the partial results
          tmp += std::fabs( mat1_ptr[bs*M*K + m*K + k] - mat2_ptr[bs*K*N + k*N + n] );
        }
        result_ptr[bs*M*N + m*N + n] = tmp;
      }
    }
  }
  
  return result;
}


at::Tensor mat_mat_l1_grad_cpu(const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& mat3, const bool grad_for_A) {
  // mat1.dim == 3
  // mat2.dim == 3
  TORCH_CHECK(mat1.size(0) == mat2.size(0));
  TORCH_CHECK(mat1.size(2) == mat2.size(1));

  TORCH_CHECK(mat1.size(1) == mat3.size(1));
  TORCH_CHECK(mat2.size(2) == mat3.size(2));

  TORCH_CHECK(mat1.dtype() == at::kFloat);
  TORCH_CHECK(mat2.dtype() == at::kFloat);

  TORCH_INTERNAL_ASSERT(mat1.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(mat2.device().type() == at::DeviceType::CPU);

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
  
  if (grad_for_A) {
    for (int64_t bs = 0; bs < BATCH_SIZE; bs++) {
      // For every row of a and result ...
      for (int64_t m = 0; m < M; m++) {
        // For every column of b and result ...
        for (int64_t n = 0; n < N; n++) {
          // For every element in the row of a and column of b
          float tmp = 0;
          for (int64_t k = 0; k < K; k++) {
            // Accumulate the partial results
            float cmp = -1;
            if ( mat3_ptr[bs*M*N + m*N + n] > mat2_ptr[bs*K*N + k*N + n] ) cmp = 1;
            else if ( mat3_ptr[bs*M*N + m*N + n] == mat2_ptr[bs*K*N + k*N + n] ) cmp = 0;
            tmp += mat1_ptr[bs*M*K + m*K + k] * cmp;
          }
          mat3_grad_ptr[bs*M*N + m*N + n] = tmp;
          // ! here we divide by K to normalize the range, but based on the chain rule, we must divide by N here bc of the division by K in forward pass.
          // ! So the test only works if you manually multiply the reference_output by ( K/N with K,N difinitions here or N/K with difinitions in reference)
        }
      }
    }
  }
  else {
    for (int64_t bs = 0; bs < BATCH_SIZE; bs++) {
      // For every row of a and result ...
      for (int64_t m = 0; m < M; m++) {
        // For every column of b and result ...
        for (int64_t n = 0; n < N; n++) {
          // For every element in the row of a and column of b
          float tmp = 0;
          for (int64_t k = 0; k < K; k++) {
            // Accumulate the partial results
            float cmp = 1;
            if ( mat1_ptr[bs*M*K + m*K + k] > mat3_ptr[bs*M*N + m*N + n] ) cmp = -1;
            else if ( mat1_ptr[bs*M*K + m*K + k] == mat3_ptr[bs*M*N + m*N + n] ) cmp = 0;
            tmp += cmp * mat2_ptr[bs*K*N + k*N + n] ;
          }
          mat3_grad_ptr[bs*M*N + m*N + n] = tmp;
          // ! here we divide by K to normalize the range, but based on the chain rule, we must divide by M here bc of the division by K in forward pass.
          // ! So the test only works if you manually multiply the reference_output by ( K/M with K,M difinitions here or M/K with difinitions in reference)
        }
      }
    }
  }

  
  return mat3_grad;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(mat_mat_ops, m) {
  m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
  m.def("mymul(Tensor a, Tensor b) -> Tensor");
  m.def("myadd_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");

  m.def("mat_mat_mul(Tensor mat1, Tensor mat2) -> Tensor");

  m.def("mat_mat_l1(Tensor mat1, Tensor mat2) -> Tensor");
  m.def("mat_mat_l1_grad(Tensor mat1, Tensor mat2, Tensor mat3, bool grad_for_A) -> Tensor");
}

// Registers CPU implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(mat_mat_ops, CPU, m) {
  m.impl("mymuladd", &mymuladd_cpu);
  m.impl("mymul", &mymul_cpu);
  m.impl("myadd_out", &myadd_out_cpu);

  m.impl("mat_mat_mul", &mat_mat_mul_cpu);

  m.impl("mat_mat_l1", &mat_mat_l1_cpu);
  m.impl("mat_mat_l1_grad", &mat_mat_l1_grad_cpu);
}

}
