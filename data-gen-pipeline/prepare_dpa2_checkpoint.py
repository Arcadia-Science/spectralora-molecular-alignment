import argparse
from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(state, dict) or "_extra_state" not in state:
        raise RuntimeError("Checkpoint missing _extra_state; expected DeepMD training checkpoint")
    extra = state.get("_extra_state", {})
    model_params = extra.get("model_params")
    if not isinstance(model_params, dict):
        raise RuntimeError("model_params missing or invalid in checkpoint")
    return state, model_params


def select_head(state: dict[str, Any], model_params: dict[str, Any], head: str | None):
    if "model_dict" not in model_params:
        return state, model_params
    if head is None:
        raise RuntimeError("Head is required for multitask checkpoints")
    if head not in model_params["model_dict"]:
        raise RuntimeError(
            f"Head {head} not found; available: {list(model_params['model_dict'].keys())}"
        )
    head_params = model_params["model_dict"][head]
    state_head: dict[str, Any] = {"_extra_state": state.get("_extra_state", {})}
    prefix = f"model.{head}."
    for key, value in state.items():
        if key.startswith(prefix):
            state_head[key.replace(prefix, "model.Default.")] = value
    return state_head, head_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a DPA2 checkpoint into a TorchScript .pth model.")
    parser.add_argument("--input", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", required=True, help="Output .pth path")
    parser.add_argument("--head", default=None, help="Model branch head (e.g. Domains_Drug)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict state_dict loading (default: relaxed)",
    )
    args = parser.parse_args()

    state, model_params = load_checkpoint(Path(args.input))
    state, model_params = select_head(state, model_params, args.head)

    from deepmd.pt.model.model import get_model
    from deepmd.pt.train.wrapper import ModelWrapper
    from deepmd.pt.utils.env import DEVICE

    model_params = dict(model_params)
    model_params.pop("hessian_mode", None)
    model = get_model(model_params).to(DEVICE)
    wrapper = ModelWrapper(model)
    incompatible = wrapper.load_state_dict(state, strict=args.strict)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("load_state_dict results:")
        if incompatible.missing_keys:
            print("  missing:", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            print("  unexpected:", incompatible.unexpected_keys)

    wrapper.eval()
    base_model = wrapper.model["Default"]
    base_model.eval()
    scripted = torch.jit.script(base_model)
    torch.jit.save(scripted, args.output)
    print(f"Saved frozen model to {args.output}")


if __name__ == "__main__":
    main()
