#include "benchmark_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#  include <cublasLt.h>
#endif

#include "ctranslate2/ops/ops.h"

using namespace ctranslate2;

#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
#define CUBLASLT_CHECK(ans)                                             \
  do {                                                                  \
    cublasStatus_t status = (ans);                                      \
    if (status != CUBLAS_STATUS_SUCCESS)                                \
      throw std::runtime_error("cuBLASLt failed with status " + std::to_string(status)); \
  } while (false)
#endif

#if defined(CT2_WITH_CUDA) && !defined(CT2_USE_HIP)
int get_algo_config(cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr) {
  int value = 0;
  size_t written = 0;
  CUBLASLT_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&algo,
                                                      attr,
                                                      &value,
                                                      sizeof(value),
                                                      &written));
  return value;
}
#endif

std::vector<float> small_rand_vector(dim_t size) {
  std::vector<float> vec(size);
  for (dim_t i = 0; i < size; ++i)
    vec[i] = static_cast<float>(static_cast<int>(i % 23) - 11) / 23.f;
  return vec;
}

StorageView make_float_storage(Shape shape, DataType dtype, Device device) {
  const dim_t size = compute_size(shape);
  StorageView storage(std::move(shape), small_rand_vector(size), device);
  if (dtype != DataType::FLOAT32)
    storage = storage.to(dtype);
  return storage;
}

std::vector<float> to_float_vector(const StorageView& storage) {
  return storage.to(DataType::FLOAT32).to_vector<float>();
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double diff = 0;
  for (size_t i = 0; i < a.size(); ++i)
    diff = std::max(diff, static_cast<double>(std::abs(a[i] - b[i])));
  return diff;
}

double mean_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double diff = 0;
  for (size_t i = 0; i < a.size(); ++i)
    diff += static_cast<double>(std::abs(a[i] - b[i]));
  return diff / a.size();
}

double max_abs_value(const std::vector<float>& a) {
  double value = 0;
  for (float x : a)
    value = std::max(value, static_cast<double>(std::abs(x)));
  return value;
}

double validation_tolerance(DataType dtype, double ref_max) {
  switch (dtype) {
  case DataType::BFLOAT16:
    return std::max(1e-2, ref_max * 4e-3);
  case DataType::FLOAT16:
    return std::max(1e-3, ref_max * 1e-3);
  default:
    return 1e-5;
  }
}

void benchmark_gather(Device device) {
  StorageView data({512, 512}, DataType::FLOAT32, device);
  std::vector<int32_t> input_v(250);
  std::iota(input_v.begin(), input_v.end(), 0);
  StorageView input({static_cast<dim_t>(input_v.size())}, input_v, device);
  StorageView output(device);
  const ops::Gather gather_op;
  BENCHMARK(gather_op(data, input, output), 100000);
}

void benchmark_transpose(Device device) {
  StorageView x({64, 48, 8, 64}, DataType::FLOAT32, device);
  StorageView y(device);
  const ops::Transpose transpose_op({0, 2, 1, 3});
  BENCHMARK(transpose_op(x, y), 1000);
}

void benchmark_split(Device device) {
  StorageView x({64, 512*3}, DataType::FLOAT32, device);
  StorageView a(device);
  StorageView b(device);
  StorageView c(device);
  const ops::Split split_op(-1);
  BENCHMARK(split_op(x, a, b, c), 10000);
}

void benchmark_layer_norm(Device device) {
  std::vector<float> gamma_ = rand_vector(512);
  std::vector<float> beta_ = rand_vector(512);
  std::vector<float> x_ = rand_vector(100 * 512);

  StorageView gamma({512}, gamma_, device);
  StorageView beta({512}, beta_, device);
  StorageView x({100, 512}, x_, device);
  StorageView y(x.device());
  const ops::LayerNorm layer_norm_op{};
  BENCHMARK(layer_norm_op(beta, gamma, x, y), 10000);
}

void benchmark_softmax(Device device) {
  std::vector<float> x_ = rand_vector(100 * 512);
  StorageView x({100, 512}, x_, device);
  StorageView y(x.device());
  const ops::SoftMax softmax_op{};
  BENCHMARK(softmax_op(x, y), 10000);
}

void benchmark_masked_softmax(Device device) {
  const dim_t batch_size = 32;
  const dim_t num_heads = 8;
  const dim_t max_source = 24;
  const dim_t max_target = 36;
  const dim_t rows = batch_size * num_heads * max_source;
  StorageView lengths(
      {rows},
      std::vector<int32_t>(rows, max_target - 5),
      device);
  StorageView x({batch_size, num_heads, max_source, max_target},
      rand_vector(batch_size * num_heads * max_source * max_target),
      device);
  StorageView y(x.device());
  const ops::SoftMax softmax_op{};
  BENCHMARK(softmax_op(x, lengths, y), 10000);
}

void benchmark_topk(Device device) {
  const size_t k = 4;
  const size_t batch_size = 8;
  const size_t vocab_size = 32000;
  std::vector<float> x = rand_vector(batch_size * k * vocab_size);
  StorageView input({batch_size, k * vocab_size}, x, device);
  StorageView values(input.dtype(), device);
  StorageView indices(DataType::INT32,  device);
  const ops::TopK op(k);
  BENCHMARK(op(input, values, indices), 2000);
}

void benchmark_gemm(Device device, DataType dtype,
                    dim_t m = 32 * 32, dim_t n = 2048, dim_t k = 512,
                    size_t samples = 1000) {
  DataType output_dtype = (dtype == DataType::INT8 || dtype == DataType::INT16
                           ? DataType::INT32
                           : dtype);
  StorageView a = make_float_storage({m, k}, dtype, device);
  StorageView b = make_float_storage({n, k}, dtype, device);
  StorageView c(output_dtype, device);
  const ops::Gemm gemm_op(1, 0, false, true);
  std::cerr << "shape " << m << "x" << k << " * " << n << "x" << k << "^T" << std::endl;
  BENCHMARK(gemm_op(a, b, c), samples);
}

void benchmark_gemm_lt(Device device, DataType dtype,
                       dim_t m = 32 * 32, dim_t n = 2048, dim_t k = 512,
                       size_t samples = 1000) {
#if !defined(CT2_WITH_CUDA) || defined(CT2_USE_HIP)
  (void)device;
  (void)dtype;
  (void)m;
  (void)n;
  (void)k;
  (void)samples;
  throw std::runtime_error("cuBLASLt benchmark requires CUDA");
#else
  if (device != Device::CUDA)
    throw std::runtime_error("gemm_lt only supports CUDA");
  if (dtype != DataType::FLOAT16 && dtype != DataType::BFLOAT16 && dtype != DataType::FLOAT32)
    throw std::runtime_error("gemm_lt only supports float32, float16, and bfloat16");

  cudaDataType_t cuda_dtype = CUDA_R_32F;
  if (dtype == DataType::FLOAT16)
    cuda_dtype = CUDA_R_16F;
  else if (dtype == DataType::BFLOAT16)
    cuda_dtype = CUDA_R_16BF;

  StorageView a = make_float_storage({m, k}, dtype, device);
  StorageView b = make_float_storage({n, k}, dtype, device);
  StorageView c({m, n}, dtype, device);
  StorageView ref(dtype, device);
  const ops::Gemm gemm_op(1, 0, false, true);
  gemm_op(a, b, ref);
  const auto ref_values = to_float_vector(ref);
  const double ref_max = max_abs_value(ref_values);
  const double tolerance = validation_tolerance(dtype, ref_max);

  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  CUBLASLT_CHECK(cublasLtCreate(&handle));
  CUBLASLT_CHECK(cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  const cublasOperation_t transa = CUBLAS_OP_N;
  const cublasOperation_t transb = CUBLAS_OP_T;
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
    operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
    operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&a_desc, cuda_dtype, m, k, k));
  CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&b_desc, cuda_dtype, n, k, k));
  CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&c_desc, cuda_dtype, m, n, n));

  const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
    a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
    b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
    c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

  CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  const uint64_t workspace_size = 32 * 1024 * 1024;
  CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
    preference,
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
    &workspace_size,
    sizeof(workspace_size)));

  StorageView workspace({static_cast<dim_t>(workspace_size)}, DataType::INT8, device);

  constexpr int requested_algo_count = 16;
  cublasLtMatmulHeuristicResult_t results[requested_algo_count];
  int returned_algo_count = 0;
  CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(handle,
                                                operation_desc,
                                                a_desc,
                                                b_desc,
                                                c_desc,
                                                c_desc,
                                                preference,
                                                requested_algo_count,
                                                results,
                                                &returned_algo_count));

  const float alpha = 1;
  const float beta = 0;
  std::cerr << "shape " << m << "x" << k << " * " << n << "x" << k
            << "^T, " << returned_algo_count << " cuBLASLt algos" << std::endl;

  double best_ms = std::numeric_limits<double>::infinity();
  int best_algo = -1;
  for (int i = 0; i < returned_algo_count; ++i) {
    if (results[i].state != CUBLAS_STATUS_SUCCESS)
      continue;

    const int algo_id = get_algo_config(results[i].algo, CUBLASLT_ALGO_CONFIG_ID);
    const int split_k = get_algo_config(results[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
    const int reduction = get_algo_config(results[i].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);

    auto run = [&] {
      CUBLASLT_CHECK(cublasLtMatmul(handle,
                                    operation_desc,
                                    &alpha,
                                    a.buffer(),
                                    a_desc,
                                    b.buffer(),
                                    b_desc,
                                    &beta,
                                    c.buffer(),
                                    c_desc,
                                    c.buffer(),
                                    c_desc,
                                    &results[i].algo,
                                    workspace.buffer(),
                                    results[i].workspaceSize,
                                    0));
    };

    run();
    SYNCHRONIZE;
    const auto values = to_float_vector(c);
    const double diff = max_abs_diff(values, ref_values);
    const double mean_diff = mean_abs_diff(values, ref_values);
    if (diff > tolerance) {
      std::cerr << "algo " << i << " failed validation:"
                << " id " << algo_id
                << " split_k " << split_k
                << " reduction " << reduction
                << " max_diff " << diff
                << " tolerance " << tolerance
                << " mean_diff " << mean_diff
                << " ref_max " << ref_max
                << std::endl;
      continue;
    }

    for (size_t j = 0; j < 10; ++j)
      run();
    SYNCHRONIZE;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t j = 0; j < samples; ++j)
      run();
    SYNCHRONIZE;
    auto t2 = std::chrono::high_resolution_clock::now();

    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const double ms = static_cast<double>(duration) / (samples * 1000);
    std::cerr << "algo " << i << " avg " << ms << " ms"
              << " id " << algo_id
              << " split_k " << split_k
              << " reduction " << reduction
              << " max_diff " << diff
              << " tolerance " << tolerance
              << " mean_diff " << mean_diff
              << " workspace " << results[i].workspaceSize
              << " waves " << results[i].wavesCount << std::endl;

    if (ms < best_ms) {
      best_ms = ms;
      best_algo = i;
    }
  }

  std::cerr << "best  algo " << best_algo << " avg " << best_ms << " ms" << std::endl;

  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatrixLayoutDestroy(c_desc);
  cublasLtMatrixLayoutDestroy(b_desc);
  cublasLtMatrixLayoutDestroy(a_desc);
  cublasLtMatmulDescDestroy(operation_desc);
  cublasLtDestroy(handle);
#endif
}

void benchmark_quantize(Device device, DataType dtype) {
  StorageView x({32, 512}, rand_vector(32 * 512), device);
  StorageView y(dtype, device);
  StorageView scale(DataType::FLOAT32, device);
  const ops::Quantize quantize_op;
  BENCHMARK(quantize_op(x, y, scale), 10000);
}

void benchmark_dequantize(Device device) {
  StorageView x({64, 8192}, DataType::INT32, device);
  StorageView input_scale({32}, DataType::FLOAT32, device);
  StorageView weight_scale({8192}, DataType::FLOAT32, device);
  StorageView bias({8192}, DataType::FLOAT32, device);
  StorageView y(device);
  const ops::ActivationType activation_type = ops::ActivationType::ReLU;
  const ops::Dequantize dequantize_op(&activation_type);
  BENCHMARK(dequantize_op(x, input_scale, weight_scale, false, true, y, &bias), 10000);
}

void benchmark_conv1d(Device device) {
  StorageView x({1, 768, 3000}, DataType::FLOAT32, device);
  StorageView weight({768, 768, 3}, DataType::FLOAT32, device);
  StorageView bias({768}, DataType::FLOAT32, device);
  StorageView y(device);
  const ops::Conv1D conv_op{2, 1};
  BENCHMARK(conv_op(x, weight, bias, y), 100);
}

void benchmark_median_filter(Device device) {
  const dim_t width = 5;
  std::vector<float> x_ = rand_vector(100 * 512);
  StorageView x({100, 512}, x_, device);
  StorageView y(device);
  const ops::MedianFilter median_filter_op(width);
  BENCHMARK(median_filter_op(x, y), 10000);
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " op device [dtype] [m n k [samples]]" << std::endl;
    return 1;
  }

  std::string op = argv[1];
  Device device = std::string(argv[2]) == "cuda" ? Device::CUDA : Device::CPU;
  std::string dtype_str = argc > 3 ? argv[3] : "float32";
  DataType dtype = DataType::FLOAT32;
  if (dtype_str == "int16")
    dtype = DataType::INT16;
  else if (dtype_str == "int8")
    dtype = DataType::INT8;
  else if (dtype_str == "float16")
    dtype = DataType::FLOAT16;
  else if (dtype_str == "bfloat16")
    dtype = DataType::BFLOAT16;
  else if (dtype_str != "float32") {
    std::cerr << "unsupported dtype: " << dtype_str << std::endl;
    return 1;
  }

  if (op == "gather")
    benchmark_gather(device);
  else if (op == "transpose")
    benchmark_transpose(device);
  else if (op == "split")
    benchmark_split(device);
  else if (op == "layer_norm")
    benchmark_layer_norm(device);
  else if (op == "softmax")
    benchmark_softmax(device);
  else if (op == "masked_softmax")
    benchmark_masked_softmax(device);
  else if (op == "topk")
    benchmark_topk(device);
  else if (op == "gemm" || op == "gemm_lt") {
    dim_t m = 32 * 32;
    dim_t n = 2048;
    dim_t k = 512;
    size_t samples = 1000;

    if (argc >= 7) {
      m = static_cast<dim_t>(std::stoll(argv[4]));
      n = static_cast<dim_t>(std::stoll(argv[5]));
      k = static_cast<dim_t>(std::stoll(argv[6]));
      if (argc >= 8)
        samples = static_cast<size_t>(std::stoull(argv[7]));
    } else if (argc > 4) {
      std::cerr << "gemm expects either no shape or: m n k [samples]" << std::endl;
      return 1;
    }

    if (m <= 0 || n <= 0 || k <= 0 || samples == 0) {
      std::cerr << "gemm shape and samples must be positive" << std::endl;
      return 1;
    }

    if (op == "gemm")
      benchmark_gemm(device, dtype, m, n, k, samples);
    else
      benchmark_gemm_lt(device, dtype, m, n, k, samples);
  }
  else if (op == "quantize")
    benchmark_quantize(device, dtype);
  else if (op == "dequantize")
    benchmark_dequantize(device);
  else if (op == "conv1d")
    benchmark_conv1d(device);
  else if (op == "median_filter")
    benchmark_median_filter(device);

  return 0;
}
