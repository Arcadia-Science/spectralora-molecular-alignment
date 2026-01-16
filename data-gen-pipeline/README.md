# data-gen-pipeline

Generate PyG Data objects for DetaNet training from SMILES strings.

## Dependencies
- torch + torch_geometric (install in your environment)
- RDKit, ASE
- MACE or DeepMD (for ML energy/forces)
- Psi4 (optional DFT fallback)

DeepMD install and API references:
- https://docs.deepmodeling.com/projects/deepmd/en/v3.0.0b2/getting-started/install.html#install-python-interface-with-pip
- https://docs.deepmodeling.com/projects/deepmd/en/v3.0.0b2/autoapi/deepmd/index.html

## DeepMD DPA2 usage
Download a DPA2 .pt model locally (e.g., from AIS Square) and set the head/branch.
List branches with:

```
dp --pt show /path/to/model.pt model-branch
```

If the checkpoint is a training `.pt`, freeze it to `.pth` first (TorchScript):

```
python data-gen-pipeline/prepare_dpa2_checkpoint.py \
  --input data-gen-pipeline/checkpoints/DPA2_medium_28_10M_beta4.pt \
  --head Domains_Drug \
  --output data-gen-pipeline/checkpoints/DPA2_medium_28_10M_beta4_Domains_Drug.pth
```

Example pipeline run:

```
python -m data-gen-pipeline \
  --output-dir data/gen \
  --smiles "CC(=O)N" \
  --deepmd-pot-model data-gen-pipeline/checkpoints/DPA2_medium_28_10M_beta4_Domains_Drug.pth \
  --deepmd-head Domains_Drug \
  --deepmd-dipole-model /path/to/dipole.pt \
  --deepmd-polar-model /path/to/polar.pt \
  --deepmd-atomic-energy
```

Notes:
- Psi4 is used for small molecules if enabled (see --dft-atom-cutoff).
- For larger molecules, dipole/polar require DeepMD models unless --allow-missing-polar is set.
- Use --deepmd-branch as an alias for --deepmd-head.
- If DeepMD custom ops fail to load (ABI mismatch), reinstall deepmd-kit against your torch:
  `DP_ENABLE_PYTORCH=1 pip install --no-binary deepmd-kit deepmd-kit==3.1.2`
