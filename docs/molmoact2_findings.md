# MolmoAct2 findings relevant to the SO-101 world model

Notes gathered from the MolmoAct2 paper, released code, and published artifacts,
filtered to what actually affects our pipeline. MolmoAct2 trains a VLA, not a
world model, but it consumes the *same* SO-100/101 community corpus, so its data
handling is directly reusable.

Sources: [arXiv:2605.02881](https://arxiv.org/html/2605.02881v1) ·
[github.com/allenai/molmoact2](https://github.com/allenai/molmoact2) ·
[allenai/MolmoAct2-SO100_101](https://huggingface.co/allenai/MolmoAct2-SO100_101) ·
[MolmoAct2-SO100_101-Dataset](https://huggingface.co/datasets/allenai/MolmoAct2-SO100_101-Dataset) ·
[LeRobot MolmoAct2 docs](https://huggingface.co/docs/lerobot/main/en/molmoact2)

**Caveat:** the paper's appendix does not exist in arXiv v1 — several details it
defers there (exact re-annotation prompt, sampling params) are genuinely
unavailable.

---

## 1. Calibration convention — the most important finding

The corpus was recorded under the **pre-LeRobot-PR-#777 (v2.1) joint
convention**, which differs from LeRobot >= 0.5.0 (v3.0). The published fix, from
`lerobot/MolmoAct2-SO100_101-LeRobot/config.json`:

```
joint_signs   = [1.0, -1.0, 1.0, 1.0, 1.0, 1.0]   # flips shoulder_lift
joint_offsets = [0.0, 90.0, 90.0, 0.0, 0.0, 0.0]  # shifts shoulder_lift, elbow_flex
```

equivalently

```python
action["shoulder_lift.pos"] = -(action["shoulder_lift.pos"] - 90)
action["elbow_flex.pos"]   -= 90
```

LeRobot's docs warn that without it "the arm may move in the wrong direction";
a community report notes the arm "will slam hard into the table on startup".

**For us:** this only matters if we ever drive a real arm from model output. For
training a *video* world model the convention is self-consistent within the
corpus, so it is not a blocker — but it explains the calibration offset we
measured (per-dataset joint means differing 18-56 deg).

## 2. Action space — confirms our measurements

Published `norm_stats.json`, tag `so100_so101_molmoact2`, over **19,619,650 frames**:

| stat | pan | lift | elbow | wrist_flex | wrist_roll | gripper |
| --- | --- | --- | --- | --- | --- | --- |
| q01 | -42.1 | 45.2 | 35.4 | 4.9 | -65.6 | -0.3 |
| q99 | 48.6 | 186.1 | 173.6 | 93.4 | 43.5 | 44.7 |
| mean | 3.3 | 125.8 | 120.2 | 55.9 | -11.5 | 11.3 |
| std | 28.9 | 52.3 | 47.9 | 36.0 | **69.4** | 17.1 |

* **Units are degrees**, matching our own scan of all 1,215 datasets (100% degrees).
* The **wrist_roll std of 69.4** is the fingerprint of inconsistent per-user
  calibration — independently matching the ~56 deg cross-dataset spread we measured.
* Normalization is **q01-q99 -> [-1,1], clipped** (`--norm_mode q01_q99`).
* **`normalize_gripper=True` for SO-100/101** — the opposite of their YAM/DROID/
  LIBERO settings. We already do this.
* Their stats cover far more frames than we will hold locally, so
  `norm_stats.json` is usable directly instead of computing our own.

## 3. Language re-annotation — done, and downloadable

Community task strings are junk (`"lerobot_test"`, `"Test run"`). MolmoAct2
re-captioned every episode with an open VLM (**Qwen3.5-27B** — a real Feb-2026
model, not a typo), prompting it with sampled frames plus the original
instruction and asking for a description of *roughly N words*, N randomly
sampled, to force diversity.

* Global effect: unique instructions 22% -> 46%.
* On a 40-repo sample of *this* corpus: **39 of 40 datasets had exactly one
  unique task string**; 2.7% -> 32.4% unique after re-annotation (~12x).
* Download: `https://huggingface.co/datasets/allenai/MolmoAct2-SO100_101-Dataset/resolve/main/language_annotations/<user>/<repo>/tasks_annotated.parquet`
  — 1,220 files, ~3.5 MB total, indexed by `episode_index` with a `task` column.
* **`<ALL FROZEN FRAMES>` sentinel**: 154 rows (0.41%) mark episodes where the
  VLM saw no motion. Their loader resamples rather than training on them. This is
  an independent cross-check on our `no_motion` QC flag.

**For us:** irrelevant to an action-only v1, but this is the ready-made source if
we ever add language conditioning.

## 4. Visual augmentation

From `olmo/preprocessing/image_preprocessor.py::_apply_augmentation`
(`--img_aug` default `full`):

| Augmentation | photometric | full | parameters |
| --- | --- | --- | --- |
| random crop -> resize back | no | yes | fixed 95% of H and W, random offset |
| random rotation | no | yes | uniform(-5, +5) degrees |
| color jitter | yes | yes | brightness 0.2, contrast/saturation 0.8-1.2, hue 0.05 |
| Gaussian blur | yes | yes | kernel 5, sigma 0.1-1.0, p=0.2 |
| **horizontal flip** | **no** | **no** | never — chirality matters for manipulation |
| pixel noise / image dropout / resolution jitter | no | no | `image_dropout_rate = 0.0` everywhere |

* Each camera view is augmented **independently**, not jointly.
* Color jitter and blur draw from torch's global RNG, so augmentation is not
  bit-reproducible from the dataloader seed alone.

**Camera-order randomization** (`--random_camera_order=episode`) is used for
SO-100/101 because "camera layouts are diverse and inconsistent" — they refuse to
impose a fixed view-naming convention. This independently validates our decision
to treat each verified-fixed view as its own stream rather than trusting names.

## 5. Training tricks

* **Optimizer:** AdamW, betas `[0.9, 0.95]`, eps 1e-6, **weight decay 0.0**,
  grad clip 1.0, bf16 autocast + fp32 master weights, **no EMA**.
* **Per-group LR:** ViT/connector 5e-6, LLM 1e-5, action expert 5e-5. Cosine with
  200-step warmup, floor at 10% of peak (`alpha_f = 0.1`).
* **Loss token weighting** `root_subsegments_root_tokens`: scale by
  `1/sqrt(n_subsegments)` and `2/sqrt(n_loss_tokens)` so long action sequences do
  not dominate short text examples.
* **Action chunk:** 30 steps @ 30 Hz (1 s). Actions padded to 32 dims with
  `mask_action_dim_padding=True` so padded dims contribute no loss.
* **State:** discretized to **256 bins**. Action vocab: 2048 OpenFAST tokens.
* **Flow-matching expert:** 36 layers, per-layer KV-cache conditioning from the
  VLM (ablation: 95.9% vs 94.0% for hidden-state conditioning), K=4 flow samples
  at post-training / K=8 at fine-tuning (K=1 -> 94.15%, K=8 -> 95.90%).
* **Curriculum:** embodied pre-train (200K steps) -> attach flow expert (100K) ->
  embodiment fine-tune (100K, 32xH100, ~1,150 GPU-h).
* **Co-training:** 90% robot / 10% multimodal. Robot mix: YAM 0.30,
  **SO-100/101 0.30**, DROID 0.30, remainder 0.10.
* **Inference:** caching + CUDA graphs took 23 -> 55.8 Hz (2.4x).

## 6. Curation recipe (what produced the 1,220 list)

Four gates: structural validity -> remove eval-style captures -> license/codebase
eligibility -> **TOPReward** quality gate (a video-VLM scores task completion via
the log-prob of the "True" token; keep datasets whose mean over the last 3
sampled episodes beats a human-audited threshold).

Realized as a static allowlist in `data_constants.py`:
`SO100_SO101_MOLMOACT2_V1` (1,660 repos) minus `SO100_SO101_FILTERED` (440) =
**1,220 repos / 377 users**. The paper says 1,222; the code and `repo_list.json`
say 1,220 — use 1,220.

Feature names are canonicalized by stripping `main_`/`left_`/`right_` prefixes
before matching the canonical 6-DoF order.

## 7. What we adopt, and where we differ

| MolmoAct2 (VLA) | Our world model |
| --- | --- |
| q01-q99 normalization, gripper included | **same** (can reuse their `norm_stats.json`) |
| Camera order randomized, no fixed naming | **same idea** — we verify views from pixels, each fixed view is its own stream |
| Re-annotated language instructions | not used in v1 (action-only); available later |
| Absolute joint pose, 30-step chunk @ 30 Hz | absolute joint pose, continuous MLP conditioning |
| Aug: 95% crop, +-5 deg rotation, jitter, blur, no flip | **adopt as-is** — sensible starting recipe |
| Discrete action tokens + flow expert | continuous 6-DoF vector, no action tokenizer |
| TOPReward dataset gate | inherited via the 1,220 list; not re-run |
