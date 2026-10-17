# Slim, public, secret-free base image for vast.ai runs (docs/M1-SPEC.md section 7, PLAN section 11).
# It holds no project code and no secrets: the boot script downloads the release binaries for one
# commit from R2, checks their SHA-256 against the release manifest, and runs `runner run`.
# Built with the repository root as context (image.yml): `docker build -f deploy/docker/Dockerfile .`
# Pin the base by digest when building releases (image.yml resolves and records it).
FROM nvidia/cuda:12.6.3-base-ubuntu24.04

ARG RCLONE_PIN=1.75.2
ARG VASTAI_VERSION=1.8.3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl python3 python3-pip unzip zstd \
 && rm -rf /var/lib/apt/lists/*

# rclone from the official release, checked against the published SHA256SUMS.
RUN set -eu; cd /tmp; \
    f="rclone-v${RCLONE_PIN}-linux-amd64.zip"; \
    curl -fsSLO "https://downloads.rclone.org/v${RCLONE_PIN}/${f}"; \
    curl -fsSL "https://downloads.rclone.org/v${RCLONE_PIN}/SHA256SUMS" | grep " ${f}\$" | sha256sum -c -; \
    unzip -q "$f"; install -m 0755 "rclone-v${RCLONE_PIN}-linux-amd64/rclone" /usr/local/bin/rclone; \
    rm -rf /tmp/rclone-*; rclone version

RUN pip install "vastai==${VASTAI_VERSION}" && vastai --help >/dev/null

# uv, pinned and checked against the SHA-256 release-binaries.yml uses for the same version.
ARG UV_VERSION=0.13.0
ARG UV_SHA256=1468ebd5a5541121837c5a2817b9972ba6090fa6caa3d142620850a47fb75154
RUN set -eu; cd /tmp; \
    curl -fsSLo uv.tar.gz "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"; \
    echo "${UV_SHA256}  uv.tar.gz" | sha256sum -c -; \
    tar -xzf uv.tar.gz; install -m 0755 uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv; \
    rm -rf /tmp/uv*; uv --version

RUN useradd --create-home --shell /bin/bash app

# Pre-populated uv cache (PLAN section 11.2): Python 3.14 and every package of python/nn's uv.lock
# cu126 group (torch 2.14.1+cu126, about 3.7 GB of wheels) AND of its cu130 group (torch 2.14.1+cu130,
# about 3.0 GB of wheels, 2.75 GB of them not shared with cu126) are fetched once here, as the normal
# user, by hash, so the runner's `uv sync --locked --no-default-groups --group <cu126|cu130> --python 3.14`
# on the machine finds everything locally. Size: the cu126 cache alone was 6.3 GB on disk (wheels plus
# their unpacked copies, build of 2026-10-16); the same ratio on the 2.75 GB of new wheels is about
# 4.7 GB more on disk, so expect roughly 11 GB of cache and a compressed image several GB larger than
# before (an estimate from the lock's wheel sizes, no image was built for it; the `du` line below and
# image.yml's `du` of the cache give the real numbers). The runner picks the group by the first GPU's
# compute capability: cu130 from 10.0 (Blackwell: RTX 50-series, B200), cu126 for every card below it
# (CUDA 13 drops the cards older than Turing, so cu126 stays). Same PyTorch 2.14.1 in both; the image
# holds both builds, so one image serves every card. The cache and the managed Python live at fixed paths set by ENV (not
# $HOME), and the runner's child process inherits them. If the run's uv.lock differs from this
# one, uv simply downloads what is missing. The throwaway environments are deleted; only the cache
# stays. No project code and no secrets are in the image: only the two lock inputs.
ENV UV_CACHE_DIR=/home/app/.cache/uv \
    UV_PYTHON_INSTALL_DIR=/home/app/.local/share/uv/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_NO_PROGRESS=1
COPY --chown=app:app python/nn/pyproject.toml python/nn/uv.lock /tmp/nnlock/
USER app
RUN set -eu; \
    for build in cu126 cu130; do \
        UV_PROJECT_ENVIRONMENT=/tmp/nnenv uv sync --locked --no-default-groups --group "$build" \
            --python 3.14 --project /tmp/nnlock; \
        /tmp/nnenv/bin/python -c 'import sys, torch; want = "2.14.1+" + sys.argv[1]; assert torch.__version__.startswith(want), (torch.__version__, want)' "$build"; \
        rm -rf /tmp/nnenv; \
    done; \
    rm -rf /tmp/nnlock; \
    du -sh /home/app/.cache/uv /home/app/.local/share/uv/python

USER root
COPY deploy/docker/boot.sh /usr/local/bin/41bots-boot
RUN chmod 0755 /usr/local/bin/41bots-boot

USER app
WORKDIR /home/app
# vast.ai must be launched in entrypoint ("args") mode with --entrypoint pointing here; its ssh/jupyter
# modes replace ENTRYPOINT/CMD with their own init and this script would never run.
ENTRYPOINT ["/usr/local/bin/41bots-boot"]
