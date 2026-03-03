# Docker Usage

This project can be run as a CLI container with the same subcommands and flags as `mkv-match`.

## Important Notes

- The Linux container is intended to use the `whispercpp` transcriber.
- `parakeet-mlx` is macOS-only and is not supported in Linux containers.
- `whisper-cli` is built from `whisper.cpp` source in a Docker build stage (default ref: `v1.8.2`) and installed into the final runtime image.
- The Docker build exposes `whisper.cpp` CMake tuning args for CPU backends (`GGML_BLAS`, `GGML_OPENMP`, SIMD flags, etc.).
- The application scheduler submits whispercpp work in micro-batches by default; each batch is executed as one `whisper-cli`
  invocation with repeated `-f`/`-of` arguments.
- On `linux/amd64` with Python 3.10+, dependency locking resolves `torch` from the PyTorch CPU wheel index (`download.pytorch.org/whl/cpu`) to avoid CUDA/NVIDIA package downloads in CPU-only deployments.
- The image includes compiler toolchain packages because `annoy` and `hnswlib` are built from source for `linux/amd64` + Python 3.12.
- Python dependencies are installed with `uv sync --frozen` from `uv.lock` for reproducible versions.
- Dockerfile layering installs dependencies before source code, so editing application files does not trigger full dependency reinstalls.
- Whisper models are **not** baked into the image. Pre-download them and mount them at runtime.

## Build Image Locally

Build an amd64 image locally and load it into your Docker daemon:

```bash
docker buildx build \
  --platform linux/amd64 \
  --load \
  -t mkv-episode-matcher:local \
  .
```

Ryzen-focused build (OpenBLAS + AVX2/FMA/BMI2 + LTO):

```bash
docker buildx build \
  --platform linux/amd64 \
  --build-arg WHISPERCPP_ENABLE_BLAS=ON \
  --build-arg WHISPERCPP_BLAS_VENDOR=OpenBLAS \
  --build-arg WHISPERCPP_ENABLE_OPENMP=ON \
  --build-arg WHISPERCPP_ENABLE_LTO=ON \
  --build-arg WHISPERCPP_FORCE_AVX2=ON \
  --build-arg WHISPERCPP_ENABLE_NATIVE=OFF \
  --load \
  -t mkv-episode-matcher:local \
  .
```

If you build directly on the target Ryzen host (not a generic/shared builder), you can experiment with:

```bash
--build-arg WHISPERCPP_ENABLE_NATIVE=ON
```

`WHISPERCPP_ENABLE_NATIVE=ON` may improve performance on that exact CPU, but reduces portability of the image across other x86_64 machines.

## Build and Push for x64 (amd64)

Create/use a buildx builder and push an amd64 image:

```bash
docker buildx create --name mkv-matcher-builder --use --bootstrap

docker buildx build \
  --platform linux/amd64 \
  --build-arg WHISPERCPP_ENABLE_BLAS=ON \
  --build-arg WHISPERCPP_BLAS_VENDOR=OpenBLAS \
  --build-arg WHISPERCPP_ENABLE_OPENMP=ON \
  --build-arg WHISPERCPP_ENABLE_LTO=ON \
  --build-arg WHISPERCPP_FORCE_AVX2=ON \
  --build-arg WHISPERCPP_ENABLE_NATIVE=OFF \
  -t <registry>/<namespace>/mkv-episode-matcher:<tag> \
  --push \
  .
```

### Whisper.cpp Build Args

- `WHISPERCPP_ENABLE_BLAS` (`ON`/`OFF`, default `ON`): enables BLAS kernels.
- `WHISPERCPP_BLAS_VENDOR` (default `OpenBLAS`): BLAS backend vendor passed to CMake.
- `WHISPERCPP_ENABLE_OPENMP` (`ON`/`OFF`, default `ON`): OpenMP parallelism support.
- `WHISPERCPP_ENABLE_LTO` (`ON`/`OFF`, default `ON`): link-time optimization.
- `WHISPERCPP_ENABLE_NATIVE` (`ON`/`OFF`, default `OFF`): `-march=native` style host-targeted build.
- `WHISPERCPP_FORCE_AVX2` (`ON`/`OFF`, default `ON`): forces x86 SIMD toggles (`SSE4.2`, `AVX`, `AVX2`, `FMA`, `F16C`, `BMI2`) when building `linux/amd64`.

## Pre-Download + Mount Whisper Models

Download the model file once on the host (example model name shown):

```bash
mkdir -p "$HOME/.cache/whisper.cpp"
# Place model files such as ggml-small.en.bin in this directory
```

At runtime, mount this directory and set `WHISPER_CPP_MODELS_DIR=/models`.

Because the model directory is mounted from host storage (or a named volume), models are reused across container runs.

## Run Locally as a CLI Container

Basic help/version checks:

```bash
docker run --rm --platform linux/amd64 mkv-episode-matcher:local --help
docker run --rm --platform linux/amd64 mkv-episode-matcher:local --version
```

Run `match` with mounted series/video paths, persistent app data, and mounted models:

```bash
docker run --rm \
  --platform linux/amd64 \
  -e WHISPER_CPP_MODELS_DIR=/models \
  -v "$HOME/.mkv-episode-matcher:/home/appuser/.mkv-episode-matcher" \
  -v "$HOME/.cache/whisper.cpp:/models:ro" \
  -v /absolute/path/to/series:/data/series \
  -v /absolute/path/to/videos:/data/videos \
  mkv-episode-matcher:local \
  match /data/series /data/videos --whispercpp
```

## Run on Remote x64 Host

Pull the amd64 image on the x64 host and run the same way:

```bash
docker pull <registry>/<namespace>/mkv-episode-matcher:<tag>

docker run --rm \
  -e WHISPER_CPP_MODELS_DIR=/models \
  -v /home/<user>/.mkv-episode-matcher:/home/appuser/.mkv-episode-matcher \
  -v /home/<user>/.cache/whisper.cpp:/models:ro \
  -v /srv/media/series:/data/series \
  -v /srv/media/incoming:/data/videos \
  <registry>/<namespace>/mkv-episode-matcher:<tag> \
  match /data/series /data/videos --whispercpp
```

## Other Subcommands

Because the image entrypoint is `mkv-match`, pass any subcommand/flags directly:

```bash
# Config bootstrap
docker run --rm -it \
  --platform linux/amd64 \
  -v "$HOME/.mkv-episode-matcher:/home/appuser/.mkv-episode-matcher" \
  mkv-episode-matcher:local config

# Initialize series
docker run --rm \
  --platform linux/amd64 \
  -v "$HOME/.mkv-episode-matcher:/home/appuser/.mkv-episode-matcher" \
  -v /absolute/path/to/series:/data/series \
  mkv-episode-matcher:local init-series /data/series

# Fetch subtitles
docker run --rm \
  --platform linux/amd64 \
  -v "$HOME/.mkv-episode-matcher:/home/appuser/.mkv-episode-matcher" \
  -v /absolute/path/to/series:/data/series \
  mkv-episode-matcher:local fetch-subs /data/series
```

## Verification Checklist

1. `docker run --rm <image> --version` prints version.
2. `docker run --rm <image> --help` prints CLI help.
3. `docker run ... match ... --whispercpp` works when models are mounted and `WHISPER_CPP_MODELS_DIR` is set.
4. Repeated runs reuse mounted model files without re-downloading.
5. Config/log/data persist across runs when app-data directory is mounted.
6. Build fails early with a clear message if `whisper-cli` is missing or has unresolved shared-library dependencies.

## Troubleshooting: Unexpected NVIDIA/CUDA Downloads

If you still see `nvidia-*` or `cuda-*` downloads in Docker for `linux/amd64`, check:

1. You are building with Python 3.10+ (the project Docker image uses Python 3.12).
2. `uv.lock` is current and includes `torch==...+cpu` from `https://download.pytorch.org/whl/cpu` for non-Darwin Python 3.10+ markers.
3. The Docker build uses `uv sync --frozen` against the committed lockfile.
