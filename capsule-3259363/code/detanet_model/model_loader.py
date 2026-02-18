import torch
from pathlib import Path
from .detanet import DetaNet

BACKUP_DIRS = ("qm9spectra", "qm9nmr", "qm7x")

BASE_MODEL_CONFIG = {
    "num_features": 128,
    "act": "swish",
    "maxl": 3,
    "num_block": 3,
    "radial_type": "trainable_bessel",
    "num_radial": 32,
    "attention_head": 8,
    "rc": 5.0,
    "dropout": 0.0,
    "use_cutoff": False,
    "max_atomic_number": 118,
    "atom_ref": None,
    "scale": 1.0,
    "norm": False,
}

TASK_CONFIGS = {
    "energy": dict(scalar_outsize=1, irreps_out=None, summation=True, out_type="scalar", grad_type=None),
    "force": dict(scalar_outsize=1, irreps_out=None, summation=True, out_type="scalar", grad_type="force"),
    "tran_energy": dict(scalar_outsize=10, irreps_out=None, summation=True, out_type="scalar", grad_type=None),
    "npacharge": dict(scalar_outsize=1, irreps_out=None, summation=False, out_type="scalar", grad_type=None),
    "dipole": dict(scalar_outsize=1, irreps_out="1o", summation=True, out_type="dipole", grad_type=None),
    "tran_dipole": dict(scalar_outsize=10, irreps_out="1o", summation=True, out_type="dipole", grad_type=None),
    "polar": dict(scalar_outsize=2, irreps_out="2e", summation=True, out_type="2_tensor", grad_type=None),
    "quadrupole": dict(scalar_outsize=2, irreps_out="2e", summation=True, out_type="2_tensor", grad_type=None),
    "hyperpolar": dict(scalar_outsize=2, irreps_out="1o+3o", summation=True, out_type="3_tensor", grad_type=None),
    "octapole": dict(scalar_outsize=2, irreps_out="1o+3o", summation=True, out_type="3_tensor", grad_type=None),
    "Hi": dict(scalar_outsize=1, irreps_out=None, summation=False, out_type="scalar", grad_type="Hi"),
    "Hij": dict(scalar_outsize=1, irreps_out=None, summation=False, out_type="scalar", grad_type="Hij"),
    "dedipole": dict(scalar_outsize=1, irreps_out="1o", summation=False, out_type="dipole", grad_type="dipole"),
    "depolar": dict(scalar_outsize=2, irreps_out="2e", summation=False, out_type="2_tensor", grad_type="polar"),
    "borden_os": dict(scalar_outsize=240, irreps_out=None, summation=True, out_type="scalar", grad_type=None),
    "shield_iso_c": dict(scalar_outsize=1, irreps_out=None, summation=False, out_type="scalar", grad_type=None),
    "shield_iso_h": dict(scalar_outsize=1, irreps_out=None, summation=False, out_type="scalar", grad_type=None),
}

def get_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def _normalize_task_name(task_name):
    if task_name.startswith("latest_"):
        task_name = task_name[len("latest_"):]
    if task_name.endswith("_latest"):
        task_name = task_name[: -len("_latest")]
    return task_name

def default_weight_path(task_name, base_dir="trained_param"):
    return str(Path(base_dir) / "latest" / f"latest_{task_name}.pth")

def resolve_weight_path(task_name, params_override=None, base_dir="trained_param"):
    """Find a checkpoint path for a task.

    Order:
      1) explicit params_override (if it exists)
      2) trained_param/latest/(latest_{task} | {task} | {task}_latest)
      3) backups in trained_param/qm9spectra|qm9nmr|qm7x
      4) fall back to params_override or latest_{task} path for a clear error
    """
    task_name = _normalize_task_name(task_name)
    override_path = Path(params_override) if params_override else None
    if override_path is not None and override_path.exists():
        return str(override_path)

    base_dir = Path(base_dir)
    latest_dir = base_dir / "latest"
    if latest_dir.is_dir():
        for name in (f"latest_{task_name}.pth", f"{task_name}.pth", f"{task_name}_latest.pth"):
            candidate = latest_dir / name
            if candidate.exists():
                return str(candidate)

    for backup_dir in BACKUP_DIRS:
        candidate = base_dir / backup_dir / f"{task_name}.pth"
        if candidate.exists():
            return str(candidate)

    if override_path is not None:
        return str(override_path)

    return default_weight_path(task_name, base_dir=base_dir)

def _resolve_params(params, task_name=None):
    # Compatibility shim: resolve based on task name + optional explicit path.
    if params is None:
        return params
    default_path = Path(params)
    if task_name is None:
        task_name = default_path.stem
    base_dir = default_path.parent
    if base_dir.name != "trained_param" and base_dir.parent.name == "trained_param":
        base_dir = base_dir.parent
    return resolve_weight_path(task_name, params_override=params, base_dir=base_dir)

def _pad_tensor(source, target_shape):
    if source.shape == target_shape:
        return source
    result = source.new_empty(target_shape)
    if result.dim() >= 2:
        torch.nn.init.xavier_uniform_(result)
    else:
        result.uniform_(-0.05, 0.05)
    slices = tuple(slice(0, min(s, t)) for s, t in zip(source.shape, target_shape))
    result[slices] = source[slices]
    return result

def _load_state_dict(model, state_dict):
    model_state = model.state_dict()
    for key in ("Embedding.nuclare_emb.weight", "Embedding.elec_emb.weight"):
        if key in state_dict and key in model_state:
            if state_dict[key].shape != model_state[key].shape:
                state_dict[key] = _pad_tensor(state_dict[key], model_state[key].shape)
    model.load_state_dict(state_dict=state_dict)

def build_model(task_name, device=None, max_number=None, overrides=None):
    # Build a DetaNet instance for the requested task config.
    task_name = _normalize_task_name(task_name)
    if task_name not in TASK_CONFIGS:
        raise KeyError(f"Unknown task '{task_name}'. Available: {sorted(TASK_CONFIGS)}")
    device = get_device(device)
    config = dict(BASE_MODEL_CONFIG)
    config.update(TASK_CONFIGS[task_name])
    if max_number is not None:
        config["max_atomic_number"] = max_number
    if overrides:
        config.update(overrides)
    return DetaNet(**config, device=device)

def load_model(task_name, device=None, params_override=None, max_number=None, map_location=None):
    # Resolve device and checkpoint, then build and load the model weights.
    device = get_device(device)
    params = resolve_weight_path(task_name, params_override=params_override)
    if map_location is None and device.type == "cpu":
        map_location = torch.device("cpu")
    state_dict = torch.load(params, map_location=map_location)
    model = build_model(task_name, device=device, max_number=max_number)
    _load_state_dict(model, state_dict)
    return model

def resolve_weight_paths(tasks=None, overrides=None, base_dir="trained_param"):
    tasks = tasks or list(TASK_CONFIGS.keys())
    overrides = overrides or {}
    return {
        name: resolve_weight_path(name, params_override=overrides.get(name), base_dir=base_dir)
        for name in tasks
    }

def load_all_models(device=None, tasks=None, overrides=None, max_number=None, map_location=None):
    device = get_device(device)
    tasks = tasks or list(TASK_CONFIGS.keys())
    overrides = overrides or {}
    models = {}
    for task in tasks:
        models[task] = load_model(
            task,
            device=device,
            params_override=overrides.get(task),
            max_number=max_number,
            map_location=map_location,
        )
    return models

def scalar_model(device, params=None, max_number=118):
    return load_model("energy", device=device, params_override=params, max_number=max_number)

def force_model(device, params=None):
    return load_model("force", device=device, params_override=params)

def charge_model(device, params=None):
    return load_model("npacharge", device=device, params_override=params)

def dipole_model(device, params=None):
    return load_model("dipole", device=device, params_override=params)

def polar_model(device, params=None):
    return load_model("polar", device=device, params_override=params)

def quadrupole_model(device, params=None):
    return load_model("quadrupole", device=device, params_override=params)

def hyperpolar_model(device, params=None):
    return load_model("hyperpolar", device=device, params_override=params)

def octapole_model(device, params=None):
    return load_model("octapole", device=device, params_override=params)

def Hi_model(device, params=None):
    return load_model("Hi", device=device, params_override=params)

def Hij_model(device, params=None):
    return load_model("Hij", device=device, params_override=params)

def dedipole_model(device, params=None):
    return load_model("dedipole", device=device, params_override=params)

def depolar_model(device, params=None):
    return load_model("depolar", device=device, params_override=params)

def nmr_model(device, params=None, task_name=None):
    if task_name is None:
        if params is None:
            raise ValueError("nmr_model requires params or task_name")
        task_name = Path(params).stem
    task_name = _normalize_task_name(task_name)
    return load_model(task_name, device=device, params_override=params)

def uv_model(device, params=None):
    return load_model("borden_os", device=device, params_override=params)
