# Step 10000 model backup, 2026-10-26

This directory records the completed RoboTwin clean object-video model and its
recovery dependencies. Local preparation alone is not an off-server backup;
remote upload status is recorded separately under `runs/checkpoint_backup_20261026/`.

## Contents

- The project Git repository contains the current adapter, model, dataset,
  training/inference/evaluation scripts, tests, prompts and configurations.
- `vpp2-source.tar.gz` contains all 226 tracked upstream files as they exist
  locally, including the modified trainer and upstream license notices. It does
  not contain `.git`, weights, a virtual environment or datasets.
- `upstream.patch` records the changes relative to upstream commit
  `72c17acf022494ff63617be86079bb92e92f5bf8`.
- `metadata/` contains the actual completed run configuration and log, current
  train/test prompt manifests, data validation summaries, and dependency records.
- `packages_current.json` records installed package versions. The original
  environment freeze and audit are also retained under `metadata/`.
- `checkpoint.json` identifies the separate weight artifact and its SHA-256.
- The three source/metadata manifests permit byte-level verification.

The checkpoint has format `wan21_video_v1`, step 10000, and a complete `dit`
state dictionary (1,303 BF16 tensors). It is not a LoRA or a full training-state
checkpoint. It does not include VAE, CLIP, T5, optimizer, scheduler or RNG state.
The local full optimizer checkpoint is approximately 231.3 GB and is outside
this requested code-and-model backup.

## Restore the model code

Clone the backed-up project repository. In its root, restore the included
upstream source into an empty destination:

```bash
test ! -e third_party/vpp2
tar -xzf backups/step_010000_20261026/vpp2-source.tar.gz
```

The archive already includes the local changes. Do not apply `upstream.patch`
again. Recreate `.venv-train` using the upstream requirements and saved package
versions; CUDA/PyTorch/DeepSpeed binaries must match the target host. This
backup does not include the virtual environment itself.

The original project location is `/data2/niurc/vpp2-obj-generation`. If restoring
elsewhere, update absolute paths in `env_train.sh`, configurations, manifests
and embedding contracts. Their checksums/signatures also need to be regenerated
consistently. Merely setting `VPP2_PROJECT_ROOT` does not rewrite all artifacts.

## Restore the weights

Download the private Hugging Face artifact after its remote location has been
recorded. Restore `step_010000.pt` and verify its SHA-256 against
`checkpoint.json`. The original local location is:

```
runs/obj_video_train_v1/checkpoints/weights/step_010000.pt
```

The frozen components can be downloaded from the pinned public repository
`Haodong082399/VPP2`, revision
`04824ddb3aae11f13db50984ed7d540c3b77dd9f`, under
`checkpoints/Wan2.1-I2V-14B-480P/`:

- `config.json` and `Wan2.1_VAE.pth`;
- `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`;
- `models_t5_umt5-xxl-enc-bf16.pth` and the `google/umt5-xxl/` tokenizer if
  generating new text embeddings.

Their previously verified hashes are retained in
`metadata/runs/baseline_setup_20261015/shared_weights_verified.json`.
Public download availability is a dependency, not a claim that these frozen
files have been uploaded to the new private backup.

For inference, `python -m vpp2_obj.infer --checkpoint /path/to/step_010000.pt ...`
overrides the Stage-2 initializer. The original 65.6 GB Stage-2 weights are not
needed to load this completed model. For new training from these weights,
explicitly use the 10k initializer and a fresh optimizer/schedule; do not claim
an exact optimizer-state resume from this weight file.

## Data scope

The backed-up train/test JSONL files retain the exact prompts and references,
but the corresponding RGB/object videos, frame-index NPZ files and text
embedding tensors are not included in the code repository. Those original
data artifacts or a separately prepared inference input are still required.
The manifests are not a backup of their referenced media.

This snapshot reflects the current working code on 2026-10-26. It includes the
post-training evaluation and DROID preparation scripts; it is not presented as
an independently archived source tree captured at training launch.

Historical evaluation selection plans, replay records and generated videos under
`runs/` are also outside this backup. Retaining evaluation code does not by itself
make every past evaluation directly reproducible.
