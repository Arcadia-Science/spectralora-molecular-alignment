from __future__ import annotations

from typing import Iterable, List, Optional

from torch import nn


def collect_adalora_targets(
    model: nn.Module,
    include_scalar_heads: bool = True,
    include_attention: bool = True,
    include_all_linears: bool = False,
) -> List[str]:
    targets: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if include_all_linears:
            targets.append(name)
            continue
        if include_scalar_heads and name.startswith("sout."):
            targets.append(name)
        if include_attention and ".Attention." in name and name.endswith(("lq", "lk", "lv", "la")):
            targets.append(name)
    return sorted(set(targets))


def apply_adalora(
    model: nn.Module,
    adalora_config,
    target_modules: Optional[Iterable[str]] = None,
    freeze_base: bool = True,
):
    try:
        from peft import AdaLoraConfig, TaskType, inject_adapter_in_model
    except Exception as exc:
        raise RuntimeError("peft is required for AdaLoRA adapters.") from exc

    if isinstance(adalora_config, dict):
        config_dict = dict(adalora_config)
        if "task_type" not in config_dict:
            config_dict["task_type"] = TaskType.FEATURE_EXTRACTION
        adalora_config = AdaLoraConfig(**config_dict)
    else:
        if getattr(adalora_config, "task_type", None) is None:
            try:
                adalora_config.task_type = TaskType.FEATURE_EXTRACTION
            except Exception:
                pass

    if target_modules is not None:
        adalora_config.target_modules = list(target_modules)

    if not getattr(adalora_config, "target_modules", None):
        raise ValueError("AdaLoRA target_modules is empty. Provide adalora_targets or enable defaults.")

    inject_adapter_in_model(adalora_config, model)

    if freeze_base:
        bias_setting = getattr(adalora_config, "bias", "none")
        for name, param in model.named_parameters():
            lower_name = name.lower()
            if "lora" in lower_name:
                param.requires_grad = True
            elif bias_setting == "all" and lower_name.endswith("bias"):
                param.requires_grad = True
            else:
                param.requires_grad = False

    return adalora_config
