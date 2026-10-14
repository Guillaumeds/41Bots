# Slim, public, secret-free base image for vast.ai runs (docs/M1-SPEC.md section 7, PLAN section 11).
# It holds no project code and no secrets: the boot script downloads the release binaries for one
# commit from R2, checks their SHA-256 against the release manifest, and runs `runner run`.
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

COPY boot.sh /usr/local/bin/41bots-boot
RUN chmod 0755 /usr/local/bin/41bots-boot \
 && useradd --create-home --shell /bin/bash app

USER app
WORKDIR /home/app
CMD ["/usr/local/bin/41bots-boot"]
