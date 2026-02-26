# Docker Usage

This project can be run as a CLI container with the same subcommands and flags as `mkv-match`.

## Important Notes

- The Linux container is intended to use the `whispercpp` transcriber.
- `parakeet-mlx` is macOS-only and is not supported in Linux containers.
- `whisper-cli` is built from `whisper.cpp` source in a Docker build stage (default ref: `v1.8.2`) and installed into the final runtime image.
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

## Build and Push for x64 (amd64)

Create/use a buildx builder and push an amd64 image:

```bash
docker buildx create --name mkv-matcher-builder --use --bootstrap

docker buildx build \
  --platform linux/amd64 \
  -t <registry>/<namespace>/mkv-episode-matcher:<tag> \
  --push \
  .
```

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
