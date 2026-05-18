"""
Upload SpectraLoRA model weights to a HuggingFace Hub model repo.

Usage:
    pip install huggingface_hub
    huggingface-cli login          # paste your HF token (write access)
    python inference/upload_weights_hf.py

The repo is created automatically if it doesn't exist.
After running, update HF_REPO in colab_inference.ipynb to match HF_REPO below.
"""
from pathlib import Path
from huggingface_hub import HfApi

REPO_ROOT  = Path(__file__).resolve().parent.parent
REPO_NAME  = "spectralora-weights"   # repo name on HF Hub

WEIGHT_FILES = {
    "Hi.pth":       REPO_ROOT / "artifacts" / "hi"           / "prod-hi-a10080x8-clean-20260224-182057"        / "latest_Hi.pth",
    "Hij.pth":      REPO_ROOT / "artifacts" / "hij"          / "prod-hij-a10080x8-2ep-20260224-232300"          / "latest_Hij.pth",
    "depolar.pth":  REPO_ROOT / "artifacts" / "spectra_queue" / "prodq-depolar-a100x8-20260219-044935"          / "latest_depolar.pth",
    "refnet.pth":   REPO_ROOT / "ramanchembl_pipeline" / "alignment_results" / "refinement_v9"                 / "es_step1400.pth",
    "config.json":  REPO_ROOT / "artifacts" / "spectra_queue" / "prodq-depolar-a100x8-20260219-044935"          / "config.json",
}


def main() -> None:
    api     = HfApi()
    user    = api.whoami()["name"]
    hf_repo = f"{user}/{REPO_NAME}"

    api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True, private=False)
    print(f"Repo: https://huggingface.co/{hf_repo}\n")

    for remote_name, local_path in WEIGHT_FILES.items():
        if not local_path.exists():
            print(f"MISSING  {local_path}")
            continue
        size_mb = local_path.stat().st_size / 1e6
        print(f"Uploading {remote_name:15s} ({size_mb:.1f} MB) ...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_name,
            repo_id=hf_repo,
            repo_type="model",
        )
        print("done")

    print(f"\nAll weights uploaded to https://huggingface.co/{hf_repo}")
    print(f"Update colab_inference.ipynb Cell 2:  HF_REPO = \"{hf_repo}\"")


if __name__ == "__main__":
    main()