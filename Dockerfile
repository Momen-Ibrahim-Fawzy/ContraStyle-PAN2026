# PAN 2026 — Multi-Author Writing Style Analysis
# SCL-DeBERTa + LightGBM + SSPC ensemble with per-difficulty calibration
#
# FULLY OFFLINE BUILD — no internet access needed.
#
# Stage 1 (pkgsrc): borrows all Python packages from the local pan26-ai-detector image
#   (torch, transformers, lightgbm, sentence-transformers, scikit-learn, etc.)
# Stage 2 (final):  plain ubuntu:22.04 + Python 3.11 + packages from stage 1
#   Using ubuntu:22.04 instead of nvcr.io/nvidia/cuda:* saves ~3.3 GB because:
#     - nvcr.io base is 4.5 GB; ubuntu:22.04 is 78 MB  (-4.4 GB)
#     - With no system CUDA, we keep all nvidia-* wheel packages  (+1.1 GB)
#   GPU access at runtime is provided by the NVIDIA Container Toolkit (--gpus=all),
#   not by the base image. CUDA libs come entirely from the torch wheel packages.
#   SBERT model files are injected via --build-context sbert_models=./sbert-cache
#
# One-time setup (populate sbert-cache/ from mifawzy's HF cache):
#   sudo cp -r /home/mifawzy/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2  sbert-cache/
#   sudo cp -r /home/mifawzy/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2 sbert-cache/
#   sudo chown -R $(whoami) sbert-cache/
#
# Build command:
#   docker build --build-context sbert_models=./sbert-cache -t pan26-style-change:v2 .
#
# Quick local test (no TIRA needed):
#   mkdir -p test-output
#   docker run --rm --gpus=all \
#     -v /home/mifawzy/PAN-Multi/DATA/19068843/mawsa26-pan-zenodo/mawsa26-pan-zenodo:/input:ro \
#     -v $(pwd)/test-output:/output \
#     pan26-style-change:v2 -i /input -o /output
#
# TIRA smoke-test:
#   tira-run \
#     --input-dataset multi-author-writing-style-analysis-2026/smoketest-20260330-training \
#     --image pan26-style-change:v2 \
#     --command 'python3.11 /app/predict.py -i $inputDataset -o $outputDir'

# ── Stage 1: package source ───────────────────────────────────────────────────
FROM pan26-ai-detector:latest AS pkgsrc

# ── Stage 2: final runtime image ──────────────────────────────────────────────
# Plain Ubuntu — no bundled CUDA. GPU access comes from NVIDIA Container Toolkit
# at runtime; all CUDA .so files come from the nvidia-* torch wheel packages below.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps + Python 3.11 + rsync (used below for selective package copy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    rsync \
    curl \
    git \
    build-essential \
    ca-certificates \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Copy all required Python packages from pkgsrc.
# Using --mount=type=bind (no intermediate layer) so the full 9.6 GB pkgsrc never
# lands in any image layer — only the rsync result is stored.
#
# nvidia/* — keep EVERYTHING. No system CUDA in this base image, so all CUDA
#   shared libraries (libcudart, libcublas, libcudnn, etc.) come from the wheel.
#   This avoids the cuDNN version-mismatch issue seen with nvcr.io (9.7 vs 9.19).
#
# Excluded (~400 MB of inference-irrelevant packages):
#   pyarrow (149 MB), pandas (76 MB), dash (31 MB), nltk (14 MB),
#   datasets (5 MB), accelerate (3.5 MB), tensorboard, plotly, matplotlib, seaborn
RUN --mount=type=bind,from=pkgsrc,source=/usr/local/lib/python3.11/dist-packages,target=/mnt/pkgsrc \
    rsync -a \
      --exclude='/pyarrow/'                    --exclude='/pyarrow-*.dist-info/' \
      --exclude='/pandas/'                     --exclude='/pandas-*.dist-info/' \
      --exclude='/pandas_stubs/'               --exclude='/pandas_stubs-*.dist-info/' \
      --exclude='/dash/'                       --exclude='/dash-*.dist-info/' \
      --exclude='/flask_compress/'             --exclude='/flask_compress-*.dist-info/' \
      --exclude='/nltk/'                       --exclude='/nltk-*.dist-info/' \
      --exclude='/datasets/'                   --exclude='/datasets-*.dist-info/' \
      --exclude='/accelerate/'                 --exclude='/accelerate-*.dist-info/' \
      --exclude='/tensorboard/'                --exclude='/tensorboard-*.dist-info/' \
      --exclude='/tensorboard_data_server/'    --exclude='/tensorboard_data_server-*.dist-info/' \
      --exclude='/plotly/'                     --exclude='/plotly-*.dist-info/' \
      --exclude='/matplotlib/'                 --exclude='/matplotlib-*.dist-info/' \
      --exclude='/matplotlib_inline/'          --exclude='/matplotlib_inline-*.dist-info/' \
      --exclude='/seaborn/'                    --exclude='/seaborn-*.dist-info/' \
      /mnt/pkgsrc/ /usr/local/lib/python3.11/dist-packages/

# All CUDA libs come from nvidia-* wheel packages (no system CUDA in ubuntu:22.04).
# List covers every nvidia/ subdirectory found in the pkgsrc image.
ENV LD_LIBRARY_PATH=\
/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cufft/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/curand/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cusolver/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cusparse/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/nvjitlink/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/nccl/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/nvtx/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cuda_cupti/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cuda_nvrtc/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cufile/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cusparselt/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem/lib

# Inject SBERT models from sbert-cache/ (--build-context sbert_models=./sbert-cache).
# sbert-cache/ is excluded from /app via .dockerignore so there is no duplication.
# all-MiniLM-L6-v2  → stylometric pair features (features.py)
# all-mpnet-base-v2 → SSPC sentence encoder (sspc_model.py)
COPY --from=sbert_models models--sentence-transformers--all-MiniLM-L6-v2/ \
     /root/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/
COPY --from=sbert_models models--sentence-transformers--all-mpnet-base-v2/ \
     /root/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/

# Copy source code and models
# (data_prepared/, logs/, deberta_*_latest/ excluded via .dockerignore)
COPY . /app
WORKDIR /app

# Smoke-test: verify SBERT and torch load correctly
RUN python3.11 -c "\
import torch; \
print(f'torch {torch.__version__}  CUDA={torch.cuda.is_available()}'); \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
SentenceTransformer('all-mpnet-base-v2'); \
print('SBERT models OK')"

# Verify all required model files are present
RUN python3.11 - <<'EOF'
from pathlib import Path
models = Path('/app/models')
required = []
for diff in ['easy', 'medium', 'hard']:
    required.append(models / f'deberta_{diff}' / 'config.json')
    required.append(models / f'lgbm_{diff}.pkl')
    required.append(models / f'sspc_{diff}.pt')
required.append(models / 'ensemble_config.pkl')

missing = [str(p) for p in required if not p.exists()]
if missing:
    print("WARNING: Missing model files:")
    for m in missing:
        print(f"  {m}")
else:
    print(f"All {len(required)} model files verified OK.")
EOF

# Block all HuggingFace / internet downloads at inference time
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

ENTRYPOINT ["python3.11", "/app/predict.py"]
