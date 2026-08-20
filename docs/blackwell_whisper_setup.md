# Building CTranslate2 for Whisper on NVIDIA Blackwell

This guide builds this CTranslate2 fork with native `sm_120` CUDA code, installs
its Python extension in the same virtual environment as faster-whisper, and
converts standard Whisper or Lite Whisper checkpoints to CTranslate2 format.

The commands target Linux, an NVIDIA RTX PRO 6000 Blackwell GPU, CUDA Toolkit
12.8 or newer, and an x86-64 host. They also work on an AMD CPU: Intel OpenMP is
not required.

## Choose the branch

Use `blackwell-sm120` for normal inference testing. It contains the Lite Whisper
runtime changes and native `sm_120` build support while retaining the normal
cuBLAS GEMM path.

`blackwell-sm120-cublaslt` adds the experimental BF16 cuBLASLt implementation.
Use it only for the controlled BF16 experiment described in
[`blackwell_experiments.md`](blackwell_experiments.md), not as the default
faster-whisper build.

```bash
git fetch --all --prune
git switch blackwell-sm120
git submodule update --init --recursive
```

## Install build dependencies

Install CUDA Toolkit 12.8 or newer separately, then install the host build
tools. With GCC, `OPENMP_RUNTIME=COMP` selects GCC's `libgomp`, regardless of
whether the CPU is made by AMD or Intel.

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build \
  python3-dev python3-venv \
  libgomp1 libopenblas-dev

nvcc --version
nvidia-smi
```

If multiple CUDA toolkits are installed, point CMake at the intended one:

```bash
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8
export PATH="$CUDA_TOOLKIT_ROOT_DIR/bin:$PATH"
```

## Build and install the C++ library

Run these commands from the CTranslate2 repository root:

```bash
export CT2_SOURCE="$PWD"
export CT2_BUILD=/tmp/ct2-blackwell-native
export CT2_PREFIX="$CT2_BUILD/install"

cmake -S "$CT2_SOURCE" -B "$CT2_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CT2_PREFIX" \
  -DBUILD_CLI=OFF \
  -DBUILD_TESTS=ON \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=OFF \
  -DWITH_FLASH_ATTN=OFF \
  -DWITH_MKL=OFF \
  -DWITH_DNNL=OFF \
  -DWITH_OPENBLAS=ON \
  -DOPENMP_RUNTIME=COMP \
  -DCUDA_DYNAMIC_LOADING=OFF \
  -DCUDA_ARCH_LIST=12.0+PTX \
  ${CUDA_TOOLKIT_ROOT_DIR:+-DCUDA_TOOLKIT_ROOT_DIR="$CUDA_TOOLKIT_ROOT_DIR"}

cmake --build "$CT2_BUILD" -j"$(nproc)"
ctest --test-dir "$CT2_BUILD" --output-on-failure
cmake --install "$CT2_BUILD"
```

The CMake deprecation messages about compatibility with CMake older than 3.10
are warnings and do not indicate a failed build.

Check that the installed library contains native Blackwell code:

```bash
CT2_LIBRARY=$(find "$CT2_PREFIX" -type f -name 'libctranslate2.so.*' \
  | sort | tail -n 1)
cuobjdump --list-elf "$CT2_LIBRARY" | grep sm_120
```

At least one `sm_120` entry should be printed. The `+PTX` part of
`CUDA_ARCH_LIST` also keeps a forward-compatible PTX image in the library.

### Optional build choices

* Set `WITH_CUDNN=ON` only when a compatible cuDNN installation is available.
  It changes the Whisper convolution implementation, so benchmark it before
  adopting it.
* Keep `WITH_FLASH_ATTN=OFF` for the initial Blackwell baseline. The bundled
  FlashAttention code has not been established as the best Blackwell path in
  these experiments.
* Set `BUILD_TESTS=OFF` after validation when producing a smaller, faster
  production build.

## Install faster-whisper with the local CTranslate2 build

Use one virtual environment for faster-whisper and the locally built Python
extension. Installing faster-whisper normally also installs CTranslate2 from
PyPI, so install the local CTranslate2 package last.

```bash
python3 -m venv "$HOME/.venvs/ct2-blackwell"
source "$HOME/.venvs/ct2-blackwell/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install faster-whisper transformers

# Replace the PyPI CTranslate2 package pulled in by faster-whisper.
python -m pip uninstall -y ctranslate2
export LD_LIBRARY_PATH="$CT2_PREFIX/lib:$CT2_PREFIX/lib64:${LD_LIBRARY_PATH:-}"
CTRANSLATE2_ROOT="$CT2_PREFIX" \
  CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" \
  python -m pip install --no-cache-dir --no-deps ./python
```

`LD_LIBRARY_PATH` must also be set whenever Python runs. For a persistent
installation, add the export to the service environment or the virtual
environment activation script instead of relying on the interactive shell.

Verify both the Python package and the shared library it loads:

```bash
python - <<'PY'
import ctranslate2
import ctranslate2._ext

print("CTranslate2 version:", ctranslate2.__version__)
print("Python extension:", ctranslate2._ext.__file__)
print("CUDA devices:", ctranslate2.get_cuda_device_count())
PY

CT2_EXTENSION=$(python -c \
  'import ctranslate2._ext; print(ctranslate2._ext.__file__)')
ldd "$CT2_EXTENSION" | grep ctranslate2
```

The `ldd` result must resolve `libctranslate2` under `$CT2_PREFIX`, not under
`site-packages` or another system installation. Re-run the local installation
after any later `pip install --upgrade faster-whisper` operation that replaces
CTranslate2.

## Convert a standard Whisper model

The CTranslate2 converter changes the serialization and optional weight type;
it does not alter the Whisper architecture. Install PyTorch only on the machine
performing the conversion. PyTorch is not needed for faster-whisper inference.

```bash
python -m pip install torch huggingface-hub

ct2-transformers-converter \
  --model openai/whisper-large-v3-turbo \
  --output_dir whisper-large-v3-turbo-ct2 \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization float16
```

The output directory can be passed directly to faster-whisper. The model can
also be downloaded and converted from a local Hugging Face directory by using
that directory as the `--model` value.

## Convert a published Lite Whisper model

Lite Whisper is not produced by adding a converter flag to a standard Whisper
model. LiteASR first replaces selected encoder linear layers with two low-rank
factors. The CTranslate2 converter then serializes those existing factors.

This fork recognizes Hugging Face checkpoints whose configuration class is
`LiteWhisperConfig`. The checkpoint uses custom model code, so
`--trust_remote_code` is required. Review or pin that remote code before using
it in a production build.

```bash
ct2-transformers-converter \
  --model efficient-speech/lite-whisper-large-v3-turbo-acc \
  --output_dir lite-whisper-large-v3-turbo-acc-ct2 \
  --quantization float16 \
  --trust_remote_code
```

On this branch the converter automatically finds the corresponding OpenAI
tokenizer/preprocessor when the Lite Whisper repository does not contain them,
and writes both `tokenizer.json` and `preprocessor_config.json` to the result.
Check the output before deployment:

```bash
ls -lh lite-whisper-large-v3-turbo-acc-ct2/
```

Use `float16` for the first validated Lite Whisper deployment. The current
low-rank runtime path does not support quantized GEMM, so an INT8 Lite Whisper
conversion is not a drop-in alternative.

### Create a new Lite Whisper checkpoint

The upstream [LiteASR project](https://github.com/efeslab/LiteASR) provides the
compression procedure. Its documented example calibrates Whisper Turbo on
speech data, applies PCA-based low-rank approximation, and saves the factors:

```bash
git clone https://github.com/efeslab/LiteASR.git
cd LiteASR
python -m pip install -r requirements.txt
python src/compress.py \
  --base_model turbo \
  --low_rank \
  --rank_threshold 0.99:0.999 \
  --save_weight
```

`rank_threshold` controls the speed/size/accuracy trade-off and must be
validated on representative audio. The script also downloads calibration
datasets and requires substantial GPU memory and local storage.

At present this command saves an OpenAI Whisper-style PyTorch state dictionary,
not a complete Hugging Face `LiteWhisperConfig` directory. Therefore the
supported end-to-end route in this fork is to convert one of the published
`efficient-speech/lite-whisper-*` Hugging Face checkpoints. A newly compressed
custom checkpoint still needs a separate export into the same Hugging Face
Lite Whisper structure before `ct2-transformers-converter` can consume it.
CTranslate2 does not currently provide that export step.

## Run a 30-second smoke test

The low-level benchmark accepts a mono 16 kHz NumPy array and intentionally
processes a single Whisper window. Convert an audio file to that format with
faster-whisper's decoder:

```bash
python - <<'PY'
import numpy as np
from faster_whisper.audio import decode_audio

audio = decode_audio("recording.wav", sampling_rate=16000)
np.save("/tmp/audio.npy", audio[: 30 * 16000])
PY

python tools/benchmark/blackwell/benchmark_whisper.py \
  lite-whisper-large-v3-turbo-acc-ct2 \
  --audio /tmp/audio.npy \
  --device cuda \
  --compute-type float16
```

This benchmark measures the CTranslate2 Whisper API directly. It does not
implement faster-whisper's long-recording segmentation.

## Transcribe a complete recording

Use faster-whisper for full files. It decodes the input, splits long recordings
into Whisper windows, and returns a lazy segment iterator:

```python
from faster_whisper import WhisperModel

model = WhisperModel(
    "lite-whisper-large-v3-turbo-acc-ct2",
    device="cuda",
    compute_type="float16",
)

segments, info = model.transcribe(
    "recording.wav",
    beam_size=1,
    vad_filter=True,
)

print(f"language={info.language} probability={info.language_probability:.3f}")
for segment in segments:
    print(f"[{segment.start:8.2f} -> {segment.end:8.2f}] {segment.text}")
```

To transcribe only the first 30 seconds while retaining the normal
faster-whisper pipeline, create a clipped input first:

```bash
ffmpeg -y -i recording.wav -t 30 -ac 1 -ar 16000 /tmp/recording-first-30s.wav
```

Then pass `/tmp/recording-first-30s.wav` to `model.transcribe`.

## Troubleshooting

### `Intel OpenMP runtime libiomp5 not found`

Delete the old CMake cache or configure a new build directory, then pass
`-DOPENMP_RUNTIME=COMP`. CMake caches the old `INTEL` value.

### faster-whisper still uses the PyPI CTranslate2 package

Install the local Python package after faster-whisper and inspect the extension
with `ldd` as shown above. Matching version strings are not sufficient proof,
because the local and PyPI builds can report the same CTranslate2 version.

### `libctranslate2.so` cannot be found

Export both possible install library directories before starting Python:

```bash
export LD_LIBRARY_PATH="$CT2_PREFIX/lib:$CT2_PREFIX/lib64:${LD_LIBRARY_PATH:-}"
```

For a system service, configure this in the service definition so it is present
in the service process rather than only in an interactive terminal.

### No `sm_120` entry is reported

Confirm that `nvcc --version` reports CUDA 12.8 or newer, remove the build
directory, and configure again with `-DCUDA_ARCH_LIST=12.0+PTX`. CUDA 12.5 can
run a compatible PTX build through driver JIT, but it cannot produce this
branch's native `sm_120` image.
