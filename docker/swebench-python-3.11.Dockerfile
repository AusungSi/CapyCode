FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential gcc g++ make pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Keep baseline build requirements in the immutable image.  Installing these
# in each per-instance workspace can consume a tool-call timeout before the
# agent has started to diagnose the task.
RUN python -m pip install --no-cache-dir \
    "setuptools<70" wheel cython pytest extension-helpers pyerfa numpy

COPY docker/swebench-entrypoint.sh /usr/local/bin/capycode-entrypoint
RUN chmod +x /usr/local/bin/capycode-entrypoint

ENTRYPOINT ["/usr/local/bin/capycode-entrypoint"]
WORKDIR /workspace
