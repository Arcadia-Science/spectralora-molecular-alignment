FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY apps/model/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

# Model source code (detanet_model, train entrypoint, vendored ELoRA e3nn)
COPY capsule-3259363/code/detanet_model /app/detanet_model
COPY train/train_detanet.py /app/train/train_detanet.py
COPY third_party/ELoRA /app/third_party/ELoRA

# Service application code
COPY apps/model/app /app/app

# Trained weights
COPY artifacts/spectra_queue/prodq-depolar-a100x8-20260219-044935/latest_depolar.pth /weights/depolar.pth
COPY artifacts/hi/prod-hi-a10080x8-clean-20260224-182057/latest_Hi.pth               /weights/Hi.pth
COPY artifacts/hij/prod-hij-a10080x8-2ep-20260224-232300/latest_Hij.pth               /weights/Hij.pth
COPY ramanchembl_pipeline/alignment_results/refinement_v9/es_step1400.pth             /weights/refnet.pth
COPY artifacts/spectra_queue/prodq-depolar-a100x8-20260219-044935/config.json         /weights/config.json

ENV WEIGHTS_DIR=/weights
ENV MODEL_DEVICE=cpu
ENV WEB_CONCURRENCY=5

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers ${WEB_CONCURRENCY}"]
