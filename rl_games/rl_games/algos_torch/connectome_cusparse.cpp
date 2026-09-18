// Shared-structure FP32 SpMM and sampled edge gradients, CUDA 12 generic API.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cusparse.h>

#define CS(call) TORCH_CHECK((call) == CUSPARSE_STATUS_SUCCESS, "cuSPARSE failed: ", #call)

class Plan {
  torch::Tensor row_, col_, workspace_;
  int64_t n_, b_;
  bool column_, ready_ = false;
  cusparseHandle_t handle_ = nullptr;
  cusparseSpMatDescr_t a_ = nullptr;
  cusparseDnMatDescr_t x_ = nullptr, y_ = nullptr;
  cusparseSpMMAlg_t alg_;

  void stream() {
    CS(cusparseSetStream(handle_, at::cuda::getCurrentCUDAStream(row_.get_device())));
  }
  void check(const torch::Tensor& values, const torch::Tensor& x) {
    TORCH_CHECK(values.is_cuda() && x.is_cuda() && values.device() == row_.device() && x.device() == row_.device(), "cuSPARSE device mismatch");
    TORCH_CHECK(values.scalar_type() == torch::kFloat32 && x.scalar_type() == torch::kFloat32, "cuSPARSE requires FP32");
    TORCH_CHECK(values.is_contiguous() && values.numel() == col_.numel(), "invalid edge values");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == n_ && x.size(1) == b_, "invalid dense shape");
  }
 public:
  Plan(torch::Tensor row, torch::Tensor col, int64_t b, int alg, bool column)
      : row_(row), col_(col), n_(row.numel()-1), b_(b), column_(column) {
    TORCH_CHECK(row.is_cuda() && col.device() == row.device() && row.is_contiguous() && col.is_contiguous(), "invalid CSR device/storage");
    TORCH_CHECK(row.scalar_type() == torch::kInt64 && col.scalar_type() == torch::kInt64, "CSR requires int64 indices");
    c10::cuda::CUDAGuard guard(row.device());
    CS(cusparseCreate(&handle_));
    alg_ = alg == 1 ? CUSPARSE_SPMM_CSR_ALG1 : (alg == 2 ? CUSPARSE_SPMM_CSR_ALG2 : CUSPARSE_SPMM_CSR_ALG3);
  }
  ~Plan() {
    if (x_) cusparseDestroyDnMat(x_);
    if (y_) cusparseDestroyDnMat(y_);
    if (a_) cusparseDestroySpMat(a_);
    if (handle_) cusparseDestroy(handle_);
  }
  torch::Tensor mm(torch::Tensor values, torch::Tensor x) {
    check(values, x);
    TORCH_CHECK(column_ ? x.t().is_contiguous() : x.is_contiguous(), "dense layout mismatch");
    c10::cuda::CUDAGuard guard(row_.device());
    stream();
    auto y = column_ ? torch::empty({b_, n_}, x.options()).t() : torch::empty({n_, b_}, x.options());
    const float alpha = 1, beta = 0;
    if (!ready_) {
      CS(cusparseCreateCsr(&a_, n_, n_, col_.numel(), row_.data_ptr(), col_.data_ptr(), values.data_ptr(), CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
      auto order = column_ ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW;
      int64_t ld = column_ ? n_ : b_;
      CS(cusparseCreateDnMat(&x_, n_, b_, ld, x.data_ptr(), CUDA_R_32F, order));
      CS(cusparseCreateDnMat(&y_, n_, b_, ld, y.data_ptr(), CUDA_R_32F, order));
      size_t bytes = 0;
      CS(cusparseSpMM_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, a_, x_, &beta, y_, CUDA_R_32F, alg_, &bytes));
      workspace_ = torch::empty({static_cast<int64_t>(bytes)}, x.options().dtype(torch::kUInt8));
      CS(cusparseSpMM_preprocess(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, a_, x_, &beta, y_, CUDA_R_32F, alg_, workspace_.data_ptr()));
      ready_ = true;
    } else {
      CS(cusparseSpMatSetValues(a_, values.data_ptr()));
      CS(cusparseDnMatSetValues(x_, x.data_ptr()));
      CS(cusparseDnMatSetValues(y_, y.data_ptr()));
    }
    CS(cusparseSpMM(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, a_, x_, &beta, y_, CUDA_R_32F, alg_, workspace_.data_ptr()));
    return y;
  }
  torch::Tensor edge_grad(torch::Tensor grad, torch::Tensor x) {
    c10::cuda::CUDAGuard guard(row_.device());
    stream();
    auto result = torch::zeros({col_.numel()}, x.options());
    check(result, x);
    check(result, grad);
    TORCH_CHECK(x.is_contiguous() && grad.is_contiguous(), "edge gradient requires row-major inputs");
    // dW[i,j] = sum_b dY[i,b] X[j,b], sampled on the original CSR support.
    struct Descriptors {
      cusparseSpMatDescr_t c = nullptr;
      cusparseDnMatDescr_t g = nullptr, x = nullptr;
      ~Descriptors() { if(c) cusparseDestroySpMat(c); if(g) cusparseDestroyDnMat(g); if(x) cusparseDestroyDnMat(x); }
    } d;
    CS(cusparseCreateCsr(&d.c, n_, n_, col_.numel(), row_.data_ptr(), col_.data_ptr(), result.data_ptr(), CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CS(cusparseCreateDnMat(&d.g, n_, b_, b_, grad.data_ptr(), CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CS(cusparseCreateDnMat(&d.x, n_, b_, b_, x.data_ptr(), CUDA_R_32F, CUSPARSE_ORDER_ROW));
    const float alpha = 1, beta = 0;
    size_t bytes = 0;
    CS(cusparseSDDMM_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, &alpha, d.g, d.x, &beta, d.c, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, &bytes));
    auto buffer = torch::empty({static_cast<int64_t>(bytes)}, x.options().dtype(torch::kUInt8));
    CS(cusparseSDDMM(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, &alpha, d.g, d.x, &beta, d.c, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, buffer.data_ptr()));
    return result;
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<Plan>(m, "Plan")
    .def(pybind11::init<torch::Tensor, torch::Tensor, int64_t, int, bool>())
    .def("mm", &Plan::mm).def("edge_grad", &Plan::edge_grad);
}
