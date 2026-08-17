# Blackwell experiment handoff

This handoff targets NVIDIA RTX PRO 6000 Blackwell GPUs (compute capability
12.0) with CUDA Toolkit 12.8 or newer.

## Upstream audit

Upstream `origin/master` was checked at commit `6419e4c1` on 2026-08-17.
There is no upstream BF16 GEMM optimization at that revision. BF16 still uses
`cublasGemmEx`, `CUBLAS_COMPUTE_32F`, and `CUBLAS_GEMM_DEFAULT`. Upstream has
previous Blackwell runtime fixes, but its CMake 3.22 `FindCUDA` path does not
emit native `sm_120` images.

## Experiment branches

* `blackwell-sm120`: lite-Whisper changes, native `sm_120` build support, and
  benchmark tooling. Runtime GEMM behavior is unchanged.
* `blackwell-sm120-cublaslt`: the same base plus the opt-in BF16 cuBLASLt
  implementation.
* `bf16-cublaslt-gemm`: the cuBLASLt implementation without the Blackwell
  build-system and handoff commits.

The experiment matrix should use the same source branch and vary one setting
at a time:

| Build | `CUDA_ARCH_LIST` | BF16 mode |
| --- | --- | --- |
| PTX baseline | `8.6+PTX` | cuBLAS |
| Native baseline | `12.0+PTX` | cuBLAS |
| Native optimized | `12.0+PTX` | cuBLASLt |

## Prepare the machine

Use CUDA Toolkit 12.8 or newer and a driver that supports the installed
Blackwell GPU. CUDA 12.5 can run older PTX through driver JIT but cannot build
native `sm_120` code.

```bash
git clone --recursive git@github.com:vakkov/CTranslate2.git
cd CTranslate2
git fetch origin
git switch blackwell-sm120
git submodule update --init --recursive
```

If the experiment branches are not available from the fork, transfer them as
a self-contained Git bundle. Create and verify the bundle on this machine:

```bash
git bundle create /tmp/CTranslate2-blackwell-experiments.bundle \
  blackwell-sm120 blackwell-sm120-cublaslt
git bundle verify /tmp/CTranslate2-blackwell-experiments.bundle
scp /tmp/CTranslate2-blackwell-experiments.bundle BLACKWELL_HOST:/tmp/
```

Then clone it on the Blackwell machine and restore the upstream remote:

```bash
git clone -b blackwell-sm120 \
  /tmp/CTranslate2-blackwell-experiments.bundle CTranslate2
cd CTranslate2
git remote rename origin experiment-bundle
git remote add origin https://github.com/OpenNMT/CTranslate2
git fetch origin
git submodule update --init --recursive
```

The local converted model is not tracked by Git. Transfer it separately:

```bash
rsync -ah --info=progress2 \
  /media/hdd/CTranslate2/lite-whisper-large-v3-turbo-acc-ct2/ \
  BLACKWELL_HOST:/path/to/CTranslate2/lite-whisper-large-v3-turbo-acc-ct2/
```

## Run GEMM experiments

Run these on an otherwise idle GPU. Keep the driver, power limit, clocks, and
`CUDA_VISIBLE_DEVICES` unchanged between runs.

```bash
# Forward-compatible PTX baseline.
CUDA_ARCH_LIST=8.6+PTX \
BUILD_DIR=/tmp/ct2-blackwell-ptx \
RESULTS_DIR=/tmp/ct2-results/ptx \
tools/benchmark/blackwell/run_gemm.sh

# Native sm_120 baseline.
CUDA_ARCH_LIST=12.0+PTX \
BUILD_DIR=/tmp/ct2-blackwell-native \
RESULTS_DIR=/tmp/ct2-results/native \
tools/benchmark/blackwell/run_gemm.sh

# Native sm_120 with the opt-in cuBLASLt path.
git switch blackwell-sm120-cublaslt
CUDA_ARCH_LIST=12.0+PTX \
BUILD_DIR=/tmp/ct2-blackwell-cublaslt \
RESULTS_DIR=/tmp/ct2-results/cublaslt \
tools/benchmark/blackwell/run_gemm.sh
```

On the optimized branch the runner records both the legacy cuBLAS path and the
cuBLASLt path for BF16. It also writes `gemm-lt-algos.txt`, which compares and
validates each heuristic candidate against the legacy result. The first runtime
call for each unique shape includes algorithm tuning; the benchmark macro
performs warmup calls before measuring steady state.

The native build must list an `sm_120` image in `cuda-images.txt`. Also verify
the PTX fallback independently:

```bash
CUDA_FORCE_PTX_JIT=1 /tmp/ct2-blackwell-ptx/tests/benchmark_ops \
  gemm cuda bfloat16 1024 3072 768 300
unset CUDA_FORCE_PTX_JIT
```

## Run tests

```bash
cmake --build /tmp/ct2-blackwell-native --target ctranslate2_test -j"$(nproc)"
ctest --test-dir /tmp/ct2-blackwell-native --output-on-failure
```

Run at least FP16 and BF16. Test INT8 separately because Blackwell INT8 support
has changed during upstream development.

## Run lite-Whisper end to end

Install the library and build a matching Python extension. Do not combine a
new C++ library with an older Python extension.

```bash
cmake --install /tmp/ct2-blackwell-native
python3 -m venv /tmp/ct2-blackwell-venv
/tmp/ct2-blackwell-venv/bin/pip install -U pip setuptools wheel pybind11
/tmp/ct2-blackwell-venv/bin/pip install numpy transformers tokenizers
CTRANSLATE2_ROOT=/tmp/ct2-blackwell-native/install \
  /tmp/ct2-blackwell-venv/bin/pip install ./python

LD_LIBRARY_PATH=/tmp/ct2-blackwell-native/install/lib \
  /tmp/ct2-blackwell-venv/bin/python \
  tools/benchmark/blackwell/benchmark_whisper.py \
  lite-whisper-large-v3-turbo-acc-ct2 \
  --device cuda --compute-type bfloat16 \
  | tee /tmp/ct2-results/native/whisper-bfloat16.json
```

Repeat with `--compute-type float16` and on the cuBLASLt branch. The JFK sample
should detect English and produce the sentence beginning "And so, my fellow
Americans". Preserve all result directories for comparison.

## Acceptance criteria

* CUDA 12.8+ configures `12.0` and the library contains native `sm_120` code.
* The C++ test suite passes on the Blackwell machine.
* FP16 and BF16 GEMM benchmarks complete with stable repeated timings.
* PTX-JIT, native cuBLAS, and native cuBLASLt results are recorded separately.
* Lite-Whisper detects English and returns the expected JFK transcription.
* Any performance claim includes machine metadata and at least three runs.
