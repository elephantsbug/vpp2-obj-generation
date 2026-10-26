# vpp2-obj-generation

## Current training prompts: chronological object-video instructions, 128-token context

The current prompt revision is [prompt_obj_chronological_128_v1](runs/prompt_obj_chronological_128_v1/exports/): **2395 episodes across 50 tasks**, with the previous 105 flagged episodes excluded. A fixed short prefix specifies full-scene RGB first-frame/history conditions and future object-only video on black, observer-left / ego-right. The body follows each episode's reviewed action order, retains simultaneous pickups and handovers, and omits unrelated static-arm clauses. Physical arm identities are unchanged: observer image-left is the actual right arm.

All complete prompts, including the prefix and special tokens, are **57–120 UMT5 tokens** (median 90), with zero truncation at 128. The previous selected prompts had 866 entries over 128 and a maximum of 377. Validation used the actual VPP2 whitespace-cleaning/tokenization path and the saved tokenizer snapshot.

- [VPP2 prompt list](runs/prompt_obj_chronological_128_v1/exports/text_prompt_enhanced.txt): `prompt@@images/<task>/<episode>.png`.
- [Per-task files](runs/prompt_obj_chronological_128_v1/exports/by_task/): paths remain relative to the exports or RGB dataset root.
- [Current RGB/obj prompt source](runs/prompt_obj_chronological_128_v1/exports/paired_training.jsonl): 2395 entries with the new prompt and unchanged RGB conditions/obj targets. This is the unsplit source catalogue; use the frozen train/test manifests below for training and evaluation.
- [Old/new comparison](runs/prompt_obj_chronological_128_v1/exports/comparison.csv), [token lengths](runs/prompt_obj_chronological_128_v1/exports/token_lengths.csv), and [validation](runs/prompt_obj_chronological_128_v1/exports/validation.json).
- [Prompt/tokenizer contract](runs/prompt_obj_chronological_128_v1/prompt_contract.json) and [tokenizer snapshot](runs/prompt_obj_chronological_128_v1/tokenizer/).

Use the full prompt exactly once; do not add the old view suffix, another task wrapper, or the `@@image` path to the encoded text. Use the revision-specific [text cache](data/text_embeddings/prompt_obj_chronological_128_v1_20261019/) through each split row's `text_embedding_path`; existing long-prompt embeddings must not be reused. The cache stores frozen Wan UMT5 outputs as BF16 `[128,4096]` context with a real boolean padding mask and exact token IDs. The 2395 episodes contain 1382 unique texts, so identical prompts share a cache file. Source long prompts and original paired datasets are preserved. The rewrite reuses the existing video review rather than claiming a new visual inspection.

## RGB-conditioned object video target

Conversion and pairing are complete: **2500 episodes, 1,046,361 frames, zero failures**. All RGB/obj media hashes and simulation-timestamp files passed final checks. The paired manifests contain 2395 episodes without unresolved prompt flags and 105 requiring prompt review. See [validation](data/robotwin_observer_ego_obj_full_vpp2/validation.json) and a [three-frame alignment preview](runs/convert_obj_full_preview/alignment_preview.png).

The confirmed learning task uses **full-scene RGB episode-first/history frames as conditions** and **`video_object_full` as the predicted and supervised video**. Both use observer-left / ego-right 224×224 panels, total 448×224. The separate converter is `bash scripts/convert_observer_ego_obj_full.sh --workers 16`; add `--dry-run` to check inputs. It reuses the existing upstream pixel composition and verifies RGB/obj timing, geometry and source hashes.

Targets and paired manifests are saved under `data/robotwin_observer_ego_obj_full_vpp2/`. `paired_all.jsonl` explicitly separates `condition_video_path` (RGB), `condition_episode_first_frame_path` (RGB), and `target_video_path` (obj); `paired_no_flags.jsonl` excludes unresolved prompt flags. Target PNGs under this new directory are diagnostic obj frames, not RGB conditions. See [data notes](docs/obj_video_data.txt) and the [data contract](configs/obj_video_data_contract.json).

The frozen 90%/10% split excludes all 105 prompt-review episodes: [train](data/robotwin_observer_ego_obj_splits_v1/train.jsonl) contains 2155 episodes and [test](data/robotwin_observer_ego_obj_splits_v1/test.jsonl) contains 240. Seed 42 and task stratification keep all 50 tasks in both sets; each episode and all of its windows stay in one set. Refreshing prompts preserves episode membership, row order, media references, and dedup indices. Use these split manifests in the training loader, and keep the test episodes out of optimization. The data contract's `current_training_manifest` points only to `train.jsonl`. [Split provenance and counts](data/robotwin_observer_ego_obj_splits_v1/summary.json), [current validation](data/robotwin_observer_ego_obj_splits_v1/validation.json), and the [excluded list](data/robotwin_observer_ego_obj_splits_v1/excluded_prompt_review.jsonl) are retained.

`python3 scripts/split_obj_video_dataset.py --test-fraction 0.10 --seed 42 --dry-run` reproduces the original allocation with its historical long prompts. The revision update is implemented in [refresh_obj_split_prompts.py](scripts/refresh_obj_split_prompts.py), which defaults to a read-only check; `--commit` requires a completed cache and makes a [backup](data/robotwin_observer_ego_obj_splits_v1/revisions/before_chronological_128_20261019/) before replacement. The cache generator is [cache_obj_prompt_embeddings.py](scripts/cache_obj_prompt_embeddings.py); run it with the isolated training environment and `CUDA_VISIBLE_DEVICES=4`. Add `--resume` to validate and continue the existing cache with its identical contract. Its [summary](data/text_embeddings/prompt_obj_chronological_128_v1_20261019/summary.json) records token limits, checksums, tensor validation, and generation settings. Reusing frozen text features across splits does not fit anything to the held-out episodes.

The first baseline is implemented in [src/vpp2_obj/](src/vpp2_obj/), with a [review guide](docs/obj_video_baseline_v1.md) and [training configuration](configs/obj_video_baseline_v1.yaml). It reads RGB conditions and obj targets from separate paired videos using shared deduplicated indices, retains `condition_latent_loss_weight=1.0` and the upstream loss normalization, and logs RGB-prefix and object-future losses separately. Inference retains joint denoising of all seven latent positions, decodes 25 frames and exports only `[9:25]`. CPU checks and one real frozen-VAE window check passed; complete 14B GPU training/inference and distributed training remain unverified. Training has not started.

## Inference review display

Use [scripts/build_obj_video_review.py](scripts/build_obj_video_review.py) for new inference reviews. The default comparison shows **synchronized future full-scene RGB on the left**, including the background and robots, and **predicted object-only video on the right**. Each panel retains observer-left / ego-right composition. Reference frames come from the exact future source indices recorded for the prediction window, rather than from playback FPS. Future RGB is used only for this post-inference display; the model still receives only the episode-first/history RGB and action-plan text, and numerical metrics still compare predictions against black-background object GT.

```bash
CUDA_VISIBLE_DEVICES='' .venv-train/bin/python scripts/build_obj_video_review.py \
  --run runs/obj_video_step4000_motion_start_v1
```

The CPU renderer writes `full_scene_review_v1/index.html`, comparison videos, contact sheets, an overview and a provenance summary under the completed run. It preserves existing predictions and reports and refuses to overwrite an existing review directory; use `--output` for a new revision. For an object-GT display, explicitly select `--left-source object` (default output directory: `object_review_v1`).

## Video-grounded prompts: completed 50-task run

The full-length source prompt result is [runs/prompt_all_video_agent_v1/exports/](runs/prompt_all_video_agent_v1/exports/): **50 tasks × 50 episodes = 2500 nonempty prompts**. Agents wrote the prompts from each episode's task and timestamped video evidence. All clips were decoded; visual review covered samples throughout each clip, normally every second plus endpoints, with additional evidence where needed. This was not a visual inspection of every decoded frame. The 50 RGB-ranking captions reuse the previous completed video review after verifying identical source hashes.

| Output | Purpose |
| --- | --- |
| [text_video_candidates_all.txt](runs/prompt_all_video_agent_v1/exports/text_video_candidates_all.txt) | All 2500 candidate prompts, including flagged records. |
| [text_video_prompt_enhanced.txt](runs/prompt_all_video_agent_v1/exports/text_video_prompt_enhanced.txt) | 2395 prompts without unresolved review flags. |
| [by_task/](runs/prompt_all_video_agent_v1/exports/by_task/) | All 50 tasks saved separately in the same format. |
| [review.html](runs/prompt_all_video_agent_v1/exports/review.html) | Video, initial image, chronological observations, evidence, and editable human review. Filter priority P1 for the 105 flagged episodes. |
| [needs_review.csv](runs/prompt_all_video_agent_v1/exports/needs_review.csv) | The 105 episodes with task/video differences or remaining uncertainties. |
| [video_sequence.csv](runs/prompt_all_video_agent_v1/exports/video_sequence.csv) | Actual participating arms, order/overlap, and approximate observed time intervals for every episode. |
| [records.jsonl](runs/prompt_all_video_agent_v1/exports/records.jsonl) | All captions, evidence paths, source hashes, observations, and review flags. |

Each text line is `English prompt@@images/<task>/<episode>.png`. Image paths are relative to the **exports directory or the converted dataset root**, including when the text file is under `by_task/`. The video layout is observer on the left and robot ego on the right, each 224×224. Captions describe the actual right arm first (observer image-left), then the actual left arm after `, while`; an unused arm says `keeps static`. Explicit temporal clauses distinguish sequential actions, parallel actions, waiting, and handovers.

Final verification passed: all 2500 unique episodes, per-task text files, TXT/CSV/JSON consistency, 10986 media/evidence links, embedded review data, and JavaScript syntax. Browser rendering was not tested. See [final_export_validation.json](runs/prompt_all_video_agent_v1/progress/final_export_validation.json). The exporter freezes this completed run before any later record writes; changes should use a new run. Read-only export consistency checks can be rerun with `python scripts/check_all_video_prompt_exports.py` (only its validation report is refreshed).

No unresolved flag is not the same as human approval. Review decisions start pending. The original videos, instructions, historical prompt runs, and original conversion manifests remain unchanged; the current split manifests use the prompt revision described above. Training has not started.

## Earlier workflow: two-stage first-frame generation

The earlier workflow generates one prompt per episode for the 448×224
observer/ego dataset at `data/robotwin_observer_ego_vpp2/`. The user has authorized
the full 50-task × 50-episode run after the current sample checks; this run does
not require another approval step. Training has not started.

Both stages use the local BF16 `Qwen2.5-VL-72B-Instruct` model at
`/data2/models/Qwen2.5-VL-72B-Instruct`. Two independent replicas use H200 GPU
pairs **4–5** and **6–7**, with Transformers balanced layer placement, SDPA
attention, greedy decoding, and a default batch size of **4 episodes per
replica**. GPU 0–3 are not selected by these scripts.

1. **Locate objects.** Split the composed first-frame PNG into its observer and
   ego panels. Resize observer to 896×896 and ego to 448×448. The model returns
   absolute pixel coordinates in the observer image; ego only assists object
   recognition. The program then converts observer coordinates to a 0–1000
   scale and derives a side-based arm preference. This coordinate convention
   follows [Qwen2.5-VL's image-size coordinate representation](https://qwenlm.github.io/blog/qwen2.5-vl/).
   Resizing does not create new image detail.
2. **Plan actions.** A text-only request contains the original episode task,
   the object locations, and explicit task constraints. It produces ordered
   actions for each arm and English verb phrases. It receives no image, so the
   ego view cannot directly change arm assignment at this stage.
3. **Check and render.** Check explicit single-arm constraints, arm assignment,
   tool ownership, handovers, grasping parts, task requirements, and output
   structure. The program renders the actual right arm first, then the actual
   left arm, joined by `, while`, and appends the view description and image path.

The fixed mapping is **observer image-left → actual right arm**, and
**observer image-right → actual left arm**. Task-specified arms take priority
over the position heuristic. Object directions such as `right of` and `forward`
are not reversed. An arm that does not participate says `keeps static`.
The renderer uses this suffix:

```text
This is a composed robot video with the observer view on the left and the robot ego view on the right.
```

Only the first frame and the original episode instruction (`seen[0]`) supply
task evidence. Later video frames, task folder names, and scene metadata are
not supplied to the model. Source videos, images, and original instructions
remain unchanged. The templates are
[`ground_observer.txt`](prompts/two_step/ground_observer.txt) and
[`plan_actions.txt`](prompts/two_step/plan_actions.txt).

## Run and monitor prompt generation

Use the existing Python environment from the project directory:

```bash
cd /data2/niurc/vpp2-obj-generation

# Inspect the full selection and configuration without loading the model or writing outputs.
/opt/anaconda3/envs/starvla_env/bin/python scripts/enhance_prompts_two_step_batch.py \
  --all --batch-size 4 --output-root runs/prompt_two_step_full_v1 --dry-run
```

The full-run entry point is `python scripts/run_prompt_full.py`. The supervisor
itself must run detached from the interactive terminal; it starts and monitors
the batch process, resumes identical completed records, and records bounded
failure recovery. For a new run or a resumable stopped run, launch it as follows:

```bash
mkdir -p runs/prompt_two_step_full_v1_control
nohup /opt/anaconda3/envs/starvla_env/bin/python -u scripts/run_prompt_full.py \
  --output-root runs/prompt_two_step_full_v1 --batch-size 4 --max-restarts 2 \
  > runs/prompt_two_step_full_v1_control/supervisor.log 2>&1 < /dev/null &
```

An equivalent launcher may use `subprocess.Popen(..., start_new_session=True)`
with standard input set to `DEVNULL` and output redirected to a log. Check the
existing control status before launching a second supervisor. Locks prevent
concurrent writers; configuration and source fingerprints must match when
resuming. A changed template, model, or generation implementation requires a
new output directory.

The supervisor allows at most two fatal-process restarts by default and one
repair pass for records with no renderable caption, preserving the previous
empty records in an archive. Nonempty flagged captions are retained for review.
Retry counters persist across supervisor restarts. It does not start training.

Full-run artifacts are under `runs/prompt_two_step_full_v1/`:

| File or directory | Purpose |
| --- | --- |
| `progress.json`, `summary.json` | Current processing state, completed/pending counts, generated captions, automatic-check counts, and counts per task. |
| `index.html`, `review/<task>.html` | Image, original task, candidate caption, grounding, plan, and review flags, generated at report export. |
| `results.jsonl`, `records/<task>/<episode>.json` | Completed episode records, raw model responses and retries, derived plans, provenance, and diagnostics. |
| `needs_review.jsonl` | Records with review flags or no renderable caption. |
| `pending.jsonl` | Episodes that do not yet have terminal records. |
| `text_original.txt` | Original instructions for the selected episodes. |
| `text_candidates_all.txt` | All nonempty candidate captions, including flagged captions. |
| `text_prompt_enhanced.txt` | Nonempty captions that passed the automatic screening rules. |
| `images`, `views/` | Link to the original composed PNGs and the per-view crops used for generation. |
| `run_config.json`, `source_snapshot/`, `prompts/`, `worker_*.log` | Configuration, source/template snapshots, and replica logs. |

All three text lists use `instruction@@images/<task>/<episode>.png`; screened
lists can contain fewer entries than the original list. During processing,
result exports contain only the records already completed.

Supervisor artifacts are in the sibling directory
`runs/prompt_two_step_full_v1_control/`: `status.json` tracks the supervisor and
batch progress, `supervisor.log` and `batch_attempt_*.log` retain logs, and
`completion.json` records terminal completion counts when the run finishes.
For example:

```bash
cat runs/prompt_two_step_full_v1_control/status.json
cat runs/prompt_two_step_full_v1/progress.json
```

**Passing automatic screening does not establish semantic correctness.**
Object localization can be wrong even when the model reports high confidence;
checks also cannot prove that every required action and relationship was
preserved. All candidates and screened captions remain separate, and the
reports retain issues for review. Generation completion does not imply that
all 2500 captions pass checks or are approved training captions. These VPP2
`text@@image` lists still need separate integration into the Stage-2 training
dataset.

The earlier single-stage experiment is retained for comparison in
[`runs/prompt_preview_v2/`](runs/prompt_preview_v2/), including its
[`annotated review`](runs/prompt_preview_v2/review_annotated.html). It is
historical; the full-run entry point above uses the two-stage batch workflow.

## Observer + ego: 448×224 horizontal video

For the second dataset variant, run:

```bash
cd /data2/niurc/vpp2-obj-generation
bash scripts/convert_observer_ego.sh --dry-run
bash scripts/convert_observer_ego.sh --workers 16
```

Output is separate at `data/robotwin_observer_ego_vpp2/`. Each frame has the
**observer on the left (224×224)** and **ego on the right (224×224)**. Both full
source views are resized directly, with no flip or center crop. The converter
calls the upstream `vpp2.libero.eval_helpers.compose_libero_camera_views` with
`preprocess_mode="training_exact"`: tensor bilinear interpolation with
`antialias=True`, matching the LIBERO training resize. Its generic `image` and
`wrist_image` input slots are mapped to observer and ego respectively; the right
panel represents the ego camera, not a wrist camera. Float RGB is rounded to
uint8 before saving PNG or encoding H.264.

The two videos are paired by task and episode. Conversion rejects missing views,
unequal frame counts/FPS, inconsistent relative video PTS or mismatched simulation
timestamps. Both streams retain their original order and encoded cadence; a
shared timestamp CSV is copied from observer after equality checks. Metadata
records both video sources, view order and alignment checks. First-frame PNGs,
records, the manifest and original instruction list follow the structure below.

This mode reuses the original converter's utility functions while keeping its
416×240 output separate. It additionally needs PyTorch and torchvision; the
existing environment has `torch 2.8.0+cu128` and `torchvision 0.23.0+cu128`.
Tensor resizing runs on CPU with one Torch thread per worker. No VLM or training
job is started by this command.

For a single-episode preview:

```bash
bash scripts/convert_observer_ego.sh --task adjust_bottle --limit 1 \
  --output-root runs/convert_observer_ego_sample
```

Run its integration tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /opt/anaconda3/envs/starvla_env/bin/python -m unittest discover \
  -s tests -p 'test_convert_observer_ego.py' -v
```

Prepare RoboTwin observer videos for VPP2, then expand episode instructions and
adapt the VPP2 Stage-2 video model. Video conversion is complete. The current video-grounded prompt run is complete as described above; model training is not implemented or launched.

## Upstream source and geometry

The official checkout is `third_party/vpp2`. Its pinned revision and source paths
are recorded in [`upstream.json`](upstream.json).

The two upstream `prepare.sh` scripts are entry points for checking/indexing
benchmark datasets, not interchangeable image compositors:

| Official code | Behavior |
| --- | --- |
| `scripts/libero/prepare.sh` | Runs LIBERO `data-check`; its training config uses 448×224 with two horizontal views. |
| `scripts/robodojo/prepare.sh` | Prepares RoboDojo metadata and indices. |
| `scripts/robodojo/convert.sh` → `src/vpp2/convert_robodojo.py` | Encodes a three-view 880×480 video at the RoboDojo-specific 25 Hz; its training loader resizes to 416×240. |
| `src/vpp2/utils/conditioning.py::prepare_image` | Defines the 416×240 robot input in the video prediction/prompt tutorial. This project calls it directly for every frame. |

Our selected layout follows the tutorial's **robot** input, with dimensions in
**width × height**:

1. Read the observer RGB frame without rotating or flipping it.
2. Pass `[observer_image, None, None]` to upstream `prepare_image(..., "robot")`.
3. Upstream center-crops the main view to 4:3 and resizes it to 320×240 using PIL bicubic.
4. It places the main view on the left of a 440×240 black canvas. The upper and
   lower right wrist slots (120×120 each) stay black.
5. It resizes the whole canvas to 416×240 using PIL bicubic.

The frame is encoded with the RoboDojo converter's H.264 settings: `libx264`,
`veryfast`, CRF 20, `yuv420p`, `+faststart`. Geometry matches the upstream image
function; lossy H.264 output is not pixel-identical to a PNG. Bicubic filtering
also mixes a few pixels at the main-view/black-panel boundary. The wrist panels'
interior remains black.

## Input and output

Default source (the asterisk is a literal character in the directory name):

```text
/data2/yaozhongbo/robotwin_clean_50*50/demo_clean/
  <task>/aloha_agilex/
    observer/video/episode_XXXXXXX.mp4
    observer/video/episode_XXXXXXX.csv
    instruction/episode_XXXXXXX.json
```

Output under `data/robotwin_observer_vpp2/`:

```text
videos/<task>/<episode>.mp4       # composed 416×240 video, all source frames
images/<task>/<episode>.png       # lossless composition of frame 0, before encoding
timestamps/<task>/<episode>.csv   # original CSV copied unchanged
records/<task>/<episode>.json     # provenance, geometry, timing and original instruction
manifest.jsonl                   # successful episodes in the current command's selection
text_original.txt                # seen[0]@@images/<task>/<episode>.png, not VLM-enhanced
summary.json                     # selection, successes, skips and failures
```

The original video files are not modified. Source encoded FPS, frame count and
order are retained. The CSV carries nonuniform simulation timestamps, which are
different from the MP4's constant-rate presentation timeline. Keep these CSVs
when constructing training windows. This conversion does not resample to 25/6
FPS, subsample frames, or cut 17-frame clips.

`text_original.txt` is a VPP2 inference prompt manifest. These artifacts are not
yet a LeRobot/Parquet training dataset. Prompt enhancement and the object-only
training targets are subsequent steps.

## Run

On this machine the wrapper defaults to
`/opt/anaconda3/envs/starvla_env/bin/python`, which already has PyAV, NumPy and PIL.
Set `PYTHON_BIN` to use another environment. System `ffmpeg` with `libx264` and
`ffprobe` are required. No CUDA device is needed for preparation.

The available environment was checked with PyAV 12.3.0, NumPy 1.24.4 and Pillow
12.2.0. A two-frame NVENC probe on GPU 4 returned `unsupported device` on this
H200 host, so this pipeline uses parallel CPU workers and the upstream H.264
encoder instead of NVENC.

```bash
cd /data2/niurc/vpp2-obj-generation

# Validate discovery without creating conversion outputs.
bash scripts/convert.sh --dry-run

# One real episode, isolated from the full output manifest.
bash scripts/convert.sh --task adjust_bottle --limit 1 \
  --output-root runs/convert_sample

# All 50 tasks × 50 episodes; rerun the same command to resume.
bash scripts/convert.sh --workers 16
```

Use `--source-root '/data2/yaozhongbo/robotwin_clean_50*50/demo_clean'` to override
the default explicitly. Quote the path in shell commands. `--task` may be
repeated; `--limit` selects the first N episodes in sorted order. A filtered run
rebuilds its root manifest for that selection, so use a separate `--output-root`
for experiments. `--overwrite` permits replacement of existing derived outputs.

Completed episodes have a record written last. Resume checks fingerprints and
output integrity before skipping; failures do not enter the success manifest.
The output directory is locked against concurrent conversion commands. Each
worker runs in a separate spawned process; the default is 16 CPU workers.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/starvla_env/bin/python \
  -m unittest discover -s tests -v
```

Integration checks cover wide/square/portrait inputs, exact first-frame agreement
with the official compositor, all-frame count/order, FPS, black panels, copied
timestamps, repeat-run resume, no-write dry runs, and rejection of mismatched
timestamps. No VPP2 model weights are required.

Official references:
- [VPP2 video and prompt preparation](https://github.com/roboterax/video-prediction-policy-2/blob/main/docs/video_prediction.md)
- [RoboDojo converter](https://github.com/roboterax/video-prediction-policy-2/blob/main/src/vpp2/convert_robodojo.py)
- [LIBERO video configuration](https://github.com/roboterax/video-prediction-policy-2/blob/main/configs/libero/train_video.yaml)
