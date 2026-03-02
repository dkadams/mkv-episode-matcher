FROM python:3.12-slim-bookworm AS whispercpp_build

ARG WHISPER_CPP_REF=v1.8.2
ARG TARGETARCH
ARG WHISPERCPP_ENABLE_BLAS=ON
ARG WHISPERCPP_BLAS_VENDOR=OpenBLAS
ARG WHISPERCPP_ENABLE_LTO=ON
ARG WHISPERCPP_ENABLE_OPENMP=ON
ARG WHISPERCPP_ENABLE_NATIVE=OFF
ARG WHISPERCPP_FORCE_AVX2=ON

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libopenblas-dev \
        pkg-config \
    ; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

RUN set -eux; \
    cmake_args=( \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/whispercpp-install \
        -DWHISPER_BUILD_EXAMPLES=ON \
        -DWHISPER_BUILD_SERVER=OFF \
        -DWHISPER_BUILD_TESTS=OFF \
        -DGGML_NATIVE="${WHISPERCPP_ENABLE_NATIVE}" \
        -DGGML_BLAS="${WHISPERCPP_ENABLE_BLAS}" \
        -DGGML_BLAS_VENDOR="${WHISPERCPP_BLAS_VENDOR}" \
        -DGGML_OPENMP="${WHISPERCPP_ENABLE_OPENMP}" \
        -DGGML_LTO="${WHISPERCPP_ENABLE_LTO}" \
    ); \
    if [ "${TARGETARCH:-}" = "arm64" ]; then \
        cmake_args+=( -DGGML_CPU_ARM_ARCH=armv8-a ); \
    elif [ "${TARGETARCH:-}" = "amd64" ] && [ "${WHISPERCPP_FORCE_AVX2}" = "ON" ]; then \
        cmake_args+=( \
            -DGGML_SSE42=ON \
            -DGGML_AVX=ON \
            -DGGML_AVX2=ON \
            -DGGML_FMA=ON \
            -DGGML_F16C=ON \
            -DGGML_BMI2=ON \
        ); \
    fi; \
    git clone --depth 1 --branch "${WHISPER_CPP_REF}" https://github.com/ggml-org/whisper.cpp.git whisper.cpp; \
    cmake -S whisper.cpp -B whisper.cpp/build "${cmake_args[@]}"; \
    cmake --build whisper.cpp/build --target install -j"$(nproc)"

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    PATH="/app/.venv/bin:${PATH}"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ffmpeg \
        libgomp1 \
        libopenblas0-pthread \
        libstdc++6 \
    ; \
    rm -rf /var/lib/apt/lists/*

COPY --from=whispercpp_build /opt/whispercpp-install/ /usr/local/

RUN pip install uv

RUN set -eux; \
    ldconfig; \
    command -v whisper-cli; \
    ldd /usr/local/bin/whisper-cli; \
    if ldd /usr/local/bin/whisper-cli | grep -q "not found"; then \
        echo "ERROR: whisper-cli has missing shared libraries in final image." >&2; \
        exit 1; \
    fi

WORKDIR /app

COPY pyproject.toml uv.lock setup.py setup.cfg README.md LICENSE ./

RUN uv sync --frozen --no-dev --no-editable --no-install-project

COPY mkv_episode_matcher ./mkv_episode_matcher

RUN uv sync --frozen --no-dev --no-editable

RUN useradd --create-home --shell /bin/bash appuser
USER appuser
WORKDIR /home/appuser

ENTRYPOINT ["mkv-match"]
CMD ["--help"]
