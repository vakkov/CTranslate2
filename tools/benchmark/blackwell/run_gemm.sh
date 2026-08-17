#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-12.0+PTX}"
ARCH_TAG="$(printf '%s' "${CUDA_ARCH_LIST}" | tr -c '[:alnum:]' '_')"
BUILD_DIR="${BUILD_DIR:-/tmp/ctranslate2-blackwell-${ARCH_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/ctranslate2-blackwell-results/$(date -u +%Y%m%dT%H%M%SZ)-${ARCH_TAG}}"
JOBS="${JOBS:-$(nproc)}"
RUNS="${RUNS:-3}"
SAMPLES="${SAMPLES:-300}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
OPENMP_RUNTIME="${OPENMP_RUNTIME:-COMP}"

if [[ -z "${CUDA_TOOLKIT_ROOT_DIR:-}" && -x /usr/local/cuda/bin/nvcc ]]; then
  CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
fi

mkdir -p "${RESULTS_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

if [[ -n "${CUDA_TOOLKIT_ROOT_DIR:-}" ]]; then
  export PATH="${CUDA_TOOLKIT_ROOT_DIR}/bin:${PATH}"
fi

{
  echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git -C "${ROOT_DIR}" rev-parse HEAD)"
  echo "git_branch=$(git -C "${ROOT_DIR}" branch --show-current)"
  echo "cuda_arch_list=${CUDA_ARCH_LIST}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "openmp_runtime=${OPENMP_RUNTIME}"
  echo "build_dir=${BUILD_DIR}"
  command -v nvcc >/dev/null && nvcc --version || true
  command -v cmake >/dev/null && cmake --version || true
  command -v nvidia-smi >/dev/null && nvidia-smi || true
  command -v nvidia-smi >/dev/null && \
    nvidia-smi --query-gpu=name,uuid,driver_version,pci.bus_id,compute_cap,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem \
      --format=csv,noheader || true
} 2>&1 | tee "${RESULTS_DIR}/metadata.txt"

cmake_args=(
  -S "${ROOT_DIR}"
  -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}/install"
  -DBUILD_CLI=OFF
  -DBUILD_TESTS=ON
  -DWITH_CUDA=ON
  -DWITH_CUDNN=OFF
  -DWITH_DNNL=OFF
  -DWITH_MKL=OFF
  "-DOPENMP_RUNTIME=${OPENMP_RUNTIME}"
  -DCUDA_DYNAMIC_LOADING=OFF
  "-DCUDA_ARCH_LIST=${CUDA_ARCH_LIST}"
)

if [[ -n "${CUDA_TOOLKIT_ROOT_DIR:-}" ]]; then
  cmake_args+=("-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}")
fi

cmake "${cmake_args[@]}" 2>&1 | tee "${RESULTS_DIR}/configure.log"
cmake --build "${BUILD_DIR}" --target benchmark_ops -j"${JOBS}" 2>&1 \
  | tee "${RESULTS_DIR}/build.log"

benchmark="${BUILD_DIR}/tests/benchmark_ops"
library="$(find "${BUILD_DIR}" -maxdepth 1 -type f -name 'libctranslate2.so.*' | sort | tail -n 1)"
if [[ ! -x "${benchmark}" || -z "${library}" ]]; then
  echo "Benchmark executable or CTranslate2 library was not built" >&2
  exit 1
fi

if command -v cuobjdump >/dev/null; then
  cuobjdump --list-elf "${library}" 2>&1 | tee "${RESULTS_DIR}/cuda-images.txt"
  if [[ "${CUDA_ARCH_LIST}" == *"12.0"* ]] \
      && ! grep -q 'sm_120' "${RESULTS_DIR}/cuda-images.txt"; then
    echo "Native sm_120 image was not found in ${library}" >&2
    exit 1
  fi
fi

has_cublaslt=0
if grep -a -q 'CT2_CUDA_USE_CUBLASLT_BF16_GEMM' "${library}"; then
  has_cublaslt=1
fi

shapes=(
  "1024 3072 768"
  "1500 1280 1280"
  "1 51865 1280"
)

results_file="${RESULTS_DIR}/gemm.txt"
for dtype in float16 bfloat16; do
  modes=("cublas:0")
  if [[ "${dtype}" == "bfloat16" && "${has_cublaslt}" == 1 ]]; then
    modes+=("cublaslt:1")
  fi

  for shape in "${shapes[@]}"; do
    read -r m n k <<<"${shape}"
    for mode in "${modes[@]}"; do
      label="${mode%%:*}"
      enabled="${mode##*:}"
      for run in $(seq 1 "${RUNS}"); do
        {
          echo "case dtype=${dtype} m=${m} n=${n} k=${k} samples=${SAMPLES} mode=${label} run=${run}"
          CT2_CUDA_USE_CUBLASLT_BF16_GEMM="${enabled}" \
            "${benchmark}" gemm cuda "${dtype}" "${m}" "${n}" "${k}" "${SAMPLES}"
        } 2>&1 | tee -a "${results_file}"
      done
    done
  done
done

if [[ "${has_cublaslt}" == 1 ]]; then
  algo_results_file="${RESULTS_DIR}/gemm-lt-algos.txt"
  for shape in "${shapes[@]}"; do
    read -r m n k <<<"${shape}"
    {
      echo "case dtype=bfloat16 m=${m} n=${n} k=${k} samples=${SAMPLES} mode=cublaslt-algorithms"
      "${benchmark}" gemm_lt cuda bfloat16 "${m}" "${n}" "${k}" "${SAMPLES}"
    } 2>&1 | tee -a "${algo_results_file}"
  done
fi

echo "Results written to ${RESULTS_DIR}"
