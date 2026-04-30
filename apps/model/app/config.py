import os
import sys

WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "/weights")
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")

for _p in ["/app", "/app/third_party/ELoRA"]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
