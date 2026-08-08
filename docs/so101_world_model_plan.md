# Training a Visionary World Model on SO-101 community data

Branch: `so101`. This plan retargets the repo's Dreamer‑4–style world model
(currently trained on Atari) to the **SO-100/SO-101 robot arm**, using the
community-collected LeRobot datasets curated by **MolmoAct2**.

Everything in §1–§4 has been **verified against real data and a working pilot**
(see `scripts/so101/`). §5 onward is the execution plan.

### Status (confirmed scope: action-only · TPU+GCS · start at Tier-2)

**Done & validated on the `so101` branch:**

* Survey + curation + ingestion tools (`scripts/so101/`), full metadata sweep of
  all 1,220 datasets, tiered manifests.
* Ingestion piloted end-to-end: LeRobot(AV1)→NPZ→tokenizer ArrayRecords.
* **Code changes for continuous 6-DoF actions are implemented and unit-tested**
  (forward + backprop for both modes; Atari discrete path unchanged) — §6.
* `compute_action_stats.py` + q01–q99 normalization in `save_dynamics_dataset.py`,
  validated on real data. Landscape `--resize` path validated (10 MB vs 60 MB NPZ).
* Cloud job specs `cloud/so101_{tokenizer,dynamics}.yaml` (TPU v6e-8).

**Remaining (your hands on the cloud — cost + credentials):** provision the GCS
bucket + data disks, run Tier-2 ingestion to GCS, train tokenizer then dynamics
on TPU. Exact commands in §12.

---

## 0. TL;DR

* **Data source is solved.** MolmoAct2's community list is a single public
  manifest of **1,220 LeRobot datasets** (377 users): `repo_list.json`. That
  file is *already the post-curation allowlist* — MolmoAct started from ~1,660
  candidates and dropped ~27% via structural checks + a TOPReward quality gate.
  We reuse the result; we do **not** need to re-run TOPReward.
  Saved locally: `scripts/so101/molmoact2_repo_list.json`.
* **Compatibility is excellent where it matters and messy only in cameras.**
  Full metadata survey of all 1,215 reachable datasets:
  * **Action & state space: 100% homogeneous** — every dataset is `(6,)`
    absolute joint pose with identical joint names
    (`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`).
  * **LeRobot codebase v2.1** everywhere; **fps mostly 30** (91%); **resolution
    mostly 480×640** (99% of cameras); **video codec AV1** (yuv420p).
  * Heterogeneity is concentrated in **camera key names / count** (1–4 cams,
    user-chosen names) and a long tail of odd fps/resolutions.
  * **482 of 1,215 datasets have <5 episodes** (mostly `*_test` junk) → filter.
* **The core code change is small and localized:** the model is currently
  hard-wired to **discrete integer actions** (`nn.Embed`). SO-101 needs
  **continuous 6-DoF conditioning** (an MLP action embedding). Three files, all
  identified below. The video tokenizer needs **no structural change** — only a
  resize-shape config for landscape frames.
* **The ingestion path works today.** `scripts/so101/lerobot_to_npz.py` was
  piloted on `hawnsoung/so101_test_coff_5`: AV1 decode + parquet action
  extraction + alignment → NPZ → the *unmodified* `save_tokenizer_dataset.py`
  produced valid ArrayRecord shards. See §4.

---

## 1. What the repo is, and its current data contract

Small (~7M param) Dreamer‑4–style latent world model in JAX/Flax, trained on
Atari, exported to ONNX, played in the browser.

Pipeline (5 stages):

| Stage | Script | Input → Output |
| --- | --- | --- |
| A. Collect | `scripts/dream-arcade/collect_rollouts.py` | ALE env → per-episode `*.npz` |
| B. Tokenizer data | `scripts/data/save_tokenizer_dataset.py` | raw `*.npz` → `*.arecord` (frames) |
| C. Train tokenizer | `scripts/train_tokenizer.py` | frames → video tokenizer ckpt |
| D. Dynamics data | `scripts/data/save_dynamics_dataset.py` | raw `*.npz` + tokenizer → `*.arecord` (latents + actions) |
| E. Train dynamics | `scripts/train_dynamics.py` | latents + actions → action-conditioned dynamics ckpt |

**Raw NPZ schema** (verified from `data/space_invaders_raw/...`):
`frames (T,210,160,3) uint8`, `actions (T,) int32`, `rewards (T,) float32`,
`terminations/truncations (T,) bool`.

**The action assumption is discrete and lives in exactly three places:**

1. `visionary/dynamics.py` → `ActionEmbedding`: `nn.Embed(num_embeddings=num_actions)`
   on int actions; `-1` is the "invalid/first-frame" sentinel.
2. `visionary/dynamics.py` → `DynamicsModel.loss`: `actions = jnp.asarray(batch["actions"], dtype=jnp.int32)`.
3. `scripts/train_dynamics.py:709` (video eval): `actions = jnp.asarray(..., dtype=jnp.int32)`.

The **tokenizer is video-only** (no actions) — so it transfers to robot video
untouched except image size. `rewards` are loaded by
`DynamicsDataSource.__getitem__` but **not used** in the dynamics loss.

Data format details that matter:

* `TokenizerPreprocessor.preprocess_video` resizes with `cv2.resize(frame, (W, H))`
  where `resize_shape = (H, W)`. Atari uses `[128, 96]` (portrait). Robot frames
  are landscape → needs a landscape resize shape.
* Actions are aligned to frames by `align_actions_to_frames` (aligned[t] = action
  that led to frame t; first frame uses `prev_action`, default fill `-1`).

---

## 2. Target and the gap

**Goal:** an action-conditioned video world model of an SO-101 arm — given a
short context of frames + the stream of 6-DoF joint commands, imagine future
frames. Same architecture, same training objective (shortcut/flow-matching in
latent space), different embodiment.

Gap vs. current code:

| Aspect | Atari (now) | SO-101 (target) | Change size |
| --- | --- | --- | --- |
| Action space | discrete int (≤18) via `nn.Embed` | continuous `(6,)` float, MLP embed | **core** |
| Action units | index | **degrees**, per-dataset ranges → normalize | new (norm stats) |
| Frames | 210×160 portrait, deterministic | 480×640 landscape, AV1, real-world | config + decode |
| fps | fixed | 30 (mostly), tail of 15–60 | resample |
| Reward | used for nothing | none | drop/zeros |
| Data origin | self-collected RL rollouts | 1,220 heterogeneous community repos | **new pipeline** |
| Language | none | optional (re-annotated tasks available) | optional extension |

---

## 3. How MolmoAct2 handled this community data (and what we reuse)

MolmoAct2 (arXiv:2605.02881) trains a **VLA**, not a world model, but its
**data-curation methodology is exactly what we need**. Verified from their
released code (`allenai/molmoact2`, `experiments/launch_scripts/`,
`olmo/data/`):

1. **Allowlist/denylist curation.** Candidate pool `SO100_SO101_MOLMOACT2_V1`
   (1,660) minus `SO100_SO101_FILTERED` (440) = final **1,220**. Four gates:
   structural validity → remove eval-style captures → license/codebase
   eligibility → **TOPReward** quality gate (a video-VLM scores task
   completion via the log-prob of the "True" token; per-episode min-max; keep
   datasets whose mean over the last 3 episodes beats a human-audited threshold).
   → **We inherit the 1,220 list directly.** TOPReward re-run is optional.
2. **Joint canonicalization.** Strip `main_/left_/right_` prefixes → fixed 6-DoF
   order. (Our survey confirms the names are already near-uniform; this is a
   cheap safety net.)
3. **Continuous normalization.** `q01–q99` percentile → map to [−1,1] → **clip**
   to [−1,1]; stats **merged count-weighted across all datasets**; gripper is a
   continuous joint on SO-101 so it **is** normalized (unlike binary grippers).
4. **Cameras: no fixed schema.** They *randomize camera order* and *resize a
   single crop* (no tiling). Consume each dataset at its native fps but train on
   a 30 Hz / 30-step absolute-joint-pose action chunk.
5. **Task re-annotation.** A VLM re-captions each episode (junk like
   `"lerobot_test"` → real descriptions); unique instructions ~doubled. The
   re-annotations ship in the manifest repo under
   `language_annotations/<user>/<repo>/tasks_annotated.parquet`.

**What we adopt vs. differ:**

| MolmoAct2 (VLA) | Our world model |
| --- | --- |
| OpenFAST → 2,048 discrete action tokens + flow-matching action *expert* | condition dynamics on the **continuous 6-DoF vector** via MLP (no action tokenizer) |
| Multi-camera, randomized order | **single primary external camera** for v1 (multi-cam as augmentation later) |
| Language-conditioned | **action-only** for v1; language token optional later |
| q01–q99 normalization, gripper included | **same** |
| Native fps, 30-step chunk | **resample to 30 fps**, use repo's frame-length crops |
| Allowlist of 1,220 | **same**, plus a WM-specific filter (drop <5-episode and camera-poor sets) |

---

## 4. Data compatibility analysis — results (the answer to "are they compatible?")

Tooling built and run (all in `scripts/so101/`, outputs in `artifacts/so101/`):

* `survey_metadata.py` — fetched `meta/info.json` for **all 1,220** datasets
  (metadata only, no video). 1,215 OK, 5 auth-gated (401).
* `download_datasets.py` — dataset filtering (>=5 episodes) folded into the fetch.
* `lerobot_to_npz.py` — the real ingestion converter (piloted, see below).

### Survey results (1,215 datasets, 37,247 episodes, 19.5M frames)

| Dimension | Distribution |
| --- | --- |
| codebase_version | **v2.1: 1,215 (100%)** |
| robot_type | so100 834 · so101 298 · so100MovellaDot 75 · misc SO-variants 8 |
| **action_shape** | **(6,): 1,215 (100%)** |
| **state_shape** | **(6,): 1,215 (100%)** |
| fps | 30: 1,107 · 20: 78 · 25: 18 · 60: 8 · 50: 2 · 24/15: 2 |
| n_cameras | 2: 832 · 1: 228 · 3: 149 · 4: 6 |
| camera resolution | 480×640: 2,282 cams · 1080×1920: 27 · 720×1280: 18 · long tail |
| camera codec | AV1 / yuv420p (spot-checked) |
| camera key names | laptop 548 · phone 496 · wrist 164 · side 132 · above 93 · top 83 · front 51 · … |

### Curation results

* **Primary-camera selection** (fixed external preferred, wrist/eye-in-hand
  avoided): only **61 datasets (5%)** have *no* external camera; **2** have a
  weird aspect ratio. Chosen-camera distribution: laptop 548, above 92, top 65,
  phone 53, front 51, …
* **Episode-count distribution:** 1–4 eps: **482** · 5–19: 204 · 20–49: 226 ·
  50–99: 227 · 100+: 76. → The <5 tier is mostly throwaway `*_test` uploads.
* **Tiers emitted** (`tier{1,2,3}_*.json`):

| Tier | Filter | Datasets | Episodes | Frames |
| --- | --- | --- | --- | --- |
| **1 — pilot** | so101, 30fps, 480×640, external cam, ≥5 eps | 168 | 6,104 | 3.96M |
| **2 — clean main** | so100+so101, 30fps, 480×640, external cam, ≥5 eps | 654 | 33,964 | 17.5M |
| **3 — full** | all fps/robots, resample, ≥5 eps | 733 | 36,320 | 18.8M |

**Verdict: the datasets are compatible.** The action/state space is identical
across all of them; the only real work is (a) picking one camera per dataset,
(b) normalizing joint values to a shared scale, (c) standardizing image size and
fps. No embodiment mismatch, no action-dimension mismatch.

### Pilot validation (end-to-end, real data)

`lerobot_to_npz.py --repo hawnsoung/so101_test_coff_5 --max-episodes 3`:

* AV1 mp4 decoded via system `ffmpeg` (dav1d) → `frames (289,480,640,3) uint8`
  (verified non-black, mean≈126).
* parquet → `actions (289,6) float32` + `state (289,6)`; `mean|action−state|`
  ≈ 2–5° per joint (confirms action=commanded pose, state=achieved — alignment
  correct).
* NPZ fed to the **unmodified** `save_tokenizer_dataset.py` → 15 ArrayRecord
  chunks that round-trip via `grain.ArrayRecordDataSource` with keys
  `frames/actions/state/rewards`. **Existing tokenizer pipeline consumes robot
  data as-is.**

### 4b. Follow-up: action-range compatibility & viewpoint (measured)

**Action ranges DO mismatch across datasets — and it matters.** Probing 250
datasets: the per-dataset *mean* joint value varies
a lot across datasets (across-dataset std ≈ 18°/33°/23°/25°/**56°**/12° for the
6 joints; some datasets' means are a full revolution apart on wrist_roll). Values
are absolute joint angles in **degrees**, but each user calibrates their arm
differently, so the **same absolute value means a different physical/visual pose
across datasets**.

* **What global q01–q99 normalization does:** aligns the *distributions* to
  [−1,1] and bounds outliers. Necessary, but it does **not** remove per-arm
  calibration offset — so absolute-pose actions are only *partially* compatible.
* **Why a world model can still learn from absolute actions:** it is conditioned
  on the video *context*, which grounds the current pose; the action then mainly
  supplies the control signal relative to what's visible.
* **The robust fix — delta actions** `a[t]−a[t−1]`: the constant calibration
  offset cancels, so "move shoulder_pan +5°" is consistent across arms (SO-100/101
  share degree units + gearing). Demonstrated on two datasets: absolute means
  differ by 30–65° per joint, but **delta means are ~0 for both**. Implemented as
  `--action-repr {absolute,delta}` in `compute_action_stats.py` +
  `save_dynamics_dataset.py` (the continuous MLP embedding is unchanged). q01–q99
  clipping also tames wrist_roll wrap-around jumps. Engineering recommendation is
  delta, but the **chosen v1 path is absolute** (§10); delta stays a ready fallback.

**Viewpoint / top-only.** The world model is single-view, and viewpoint
consistency helps a lot. My default curation picks the best *fixed external*
camera (most often `laptop`, a front view) — it does **not** restrict to
top-down. Genuine top-down views (excluding the `laptop` false-positive) exist in
only **278/1,215** datasets. A top-only clean tier (30 fps, ≥5 eps):

| Top-only slice | Datasets | Episodes |
| --- | --- | --- |
| SO-100 + SO-101 | 140 | 10,541 |
| SO-101 only | 21 | 1,236 |

A top-only variant was explored but not adopted; the shipped pipeline keeps
every verified-fixed view as its own stream. Trade-off: top-only gives a consistent bird's-eye action→pixel mapping
(and, being overhead, partly mitigates the calibration issue since XY is directly
visible) but cuts the training set ~5× (654 → 140). Exact camera height/angle
still varies per user even among "top" views.

---

## 5. End-to-end pipeline

Data stays encoded end to end: episodes are packed as mp4 bytes inside
ArrayRecord shards and decoded into clips inside grain workers, so no stage ever
materializes frames to disk.

```
molmoact2_repo_list.json (1,220)
   |
   +-- survey_metadata.py ------> survey.json          (metadata only, no video)
   +-- download_datasets.py ----> data/so101/hf/       (filters + fetches all cameras)
   |
   +-- analyze.py --------------> camera_views.json    (--cameras: fixed vs moving, from pixels)
   |                              episode_bounds.json  (--episodes: trim bounds + flags)
   |
   +-- pack_arecord.py ---------> data/so101/shards/   (trim + transcode + shuffle + pack)
   +-- compute_action_stats.py -> norm_stats.json      (q01-q99, AFTER trimming)
   |
   +-- train_tokenizer.py         (VideoBytesDataSource decodes clips on the fly)
   +-- save_dynamics_dataset.py   (frames -> latents, actions normalized)
   +-- train_dynamics.py          (continuous 6-DoF conditioning)
```

| Script | Purpose |
| --- | --- |
| `survey_metadata.py` | fetch every dataset's `meta/info.json`; compatibility survey |
| `download_datasets.py` | filter + bulk/probe fetch, resumable, verifies outcomes |
| `analyze.py` | camera classification (pixels) + episode QC (parquet) |
| `pack_arecord.py` | trim, transcode, shuffle, pack into ArrayRecord shards |
| `compute_action_stats.py` | pooled q01/q99 normalization stats |
| `sample_clips.py` | render annotated samples to audit the automated verdicts |
| `purge_moving_cameras.py` | delete video for verified-moving cameras |
| `lerobot_to_npz.py` | per-episode NPZ for the NPZ-based dynamics stage |

See [`so101_status.md`](so101_status.md) for the record schema and current state.

## 6. Codebase changes — **IMPLEMENTED** on branch `so101`

All changes below are done and gated on `action_mode` (default `discrete`, so the
Atari path is byte-for-byte unchanged). Validated by a forward+backprop smoke
test on both modes.

### 6.1 Continuous action embedding — `visionary/dynamics.py`

`ActionEmbedding` gains a continuous mode, selected by `action_mode`. The
input width is inferred from the action array, so no separate dim field is
needed. In `__call__`:

```python
if self.action_mode == "continuous":
    # actions: (B, T, action_dim) float, normalized to ~[-1, 1].
    # First-frame/unknown action is the zero vector (neutral) -> base_token only.
    if actions is None:
        return jnp.broadcast_to(base_token, (batch_size, seq_len, self.model_dim))
    a = jnp.asarray(actions, dtype=self.dtype)
    h = nn.Dense(self.model_dim, dtype=self.dtype, name="action_in")(a)
    h = nn.gelu(h)
    h = nn.Dense(self.model_dim, dtype=self.dtype, name="action_out")(h)
    return h + base_token
# else: existing discrete nn.Embed path
```

Wire `action_mode`/`action_dim` from config through `DynamicsModel.setup` →
`ActionEmbedding`. Keep `num_actions` for the Atari/discrete path.

### 6.2 Action dtype — remove int casts for continuous

* `DynamicsModel.loss`: `actions = jnp.asarray(batch["actions"],
  dtype=jnp.float32 if continuous else jnp.int32)`.
* `scripts/train_dynamics.py:709`: same conditional dtype.
* `generate_next`/`generate_rollout` pass `actions` straight through — fine.

### 6.3 First-frame / prev_action fill — `visionary/dataset.py` + `save_dynamics_dataset.py`

The `-1` sentinel means "invalid discrete action". For continuous, `-1×(6,)` is
a *valid-looking* action → wrong. Use **zeros** (normalized neutral):

* `align_actions_to_frames`: default `prev_action` fill → `0` for continuous.
* `save_dynamics_dataset.encode_record`: `np.full(value.shape[1:], -1)` → zeros
  when `action_mode==continuous`.
* Make `DynamicsDataSource.__getitem__` tolerate missing `rewards` (default
  zeros) — robot data has none. (Our NPZ already writes zeros, so optional.)

### 6.4 Action normalization — `save_dynamics_dataset.py`

Add `--action_stats norm_stats.json`. When building the dynamics ArrayRecord,
apply per-joint `q01–q99 → [-1,1]` + clip (gripper included). Store the stats
path in `metadata.json`. The model then only ever sees normalized actions;
de-normalization is only needed if we ever drive a real arm.

### 6.5 Config — `scripts/config/dynamics.yaml`

```yaml
dynamics:
  action_mode: continuous     # NEW  (discrete for Atari)
  # num_actions kept for the discrete path
video_eval:
  fps: 30                     # was 15
dataset:
  action_stats: data/so101/norm_stats.json   # NEW
```

### 6.6 Config — `scripts/config/tokenizer.yaml`

```yaml
tokenizer:
  resize_shape: [128, 160]    # was [128, 96]  (H, W) landscape 4:3
  # patch_size 8 -> y_len 16, x_len 20 -> 320 patches/frame
  # num_latents: 64 -> consider 96 for richer real-world scenes
```

Robot scenes are more detailed than Atari sprites; start at `[128,160]` and
consider bumping `num_latents` 64→96 and `channel_dim` 16→ (retune) after the
first reconstruction eval. Store raw NPZ frames at the tokenizer input size (or
2×) to bound disk (see §8).

### 6.7 Cloud configs

Add `cloud/so101_tokenizer.yaml` + `cloud/so101_dynamics.yaml` mirroring the
per-game pairs, pointing `dataset.train_dir/eval_dir` at the GCS staged data and
`checkpoint.manager.directory` at `gs://visionary-uc1/so101/...`.

---

## 7. Data cleanup / QC rules (world-model-specific)

The MolmoAct allowlist already removed low-quality *datasets*. On top of that,
for a **world model** we add cheap **episode/frame**-level hygiene during
`lerobot_to_npz` and `compute_action_stats`:

1. **Drop tiny datasets** (<5 episodes) — already excluded by the tiers.
2. **Camera choice — measured, not named.** Camera *names are unreliable*: many
   `laptop`/`phone`/`side` cameras are actually wrist-mounted. In a local sample
   **37% (7/19) of the name-picked primary cameras were moving cameras**
   (visually confirmed: `00ri/so100_battery`'s "laptop" shows gripper fingers and
   a swinging viewpoint). `analyze.py` classifies each camera from
   pixels instead:
   * `static_frac` — fraction of pixels with ~zero temporal std.
     Fixed **0.46–0.85**, wrist **0.00–0.11**.
   * `shift_p90` — 90th-pct global phase-correlation egomotion.
     Fixed **≤0.40**, wrist **0.65–45**.
   * Rule `static_frac < 0.30 and shift_p90 > 0.5`. Both are required: a fixed
     close-up camera where the arm fills the frame scored `static_frac` 0.16 but
     showed no egomotion. Validated **8/8** on cameras named `wrist` and **4/4**
     against visually inspected contact sheets.
   Pipeline: probe one episode of every camera → detect → keep the best verified
   *fixed* camera per dataset (name priority only breaks ties) → drop datasets
   with no fixed view (76 Tier-2 datasets have a single camera, so some are
   unusable) → only then bulk-download.
3. **Static-episode filter**: drop episodes where joint motion is negligible
   (`std(action) < ε` across the episode) — teleop idles / mis-records.
4. **Length filter**: drop episodes shorter than the dynamics `batch_length`
   (e.g. <64 frames after resample) — can't form a training crop.
5. **Corruption guard**: decode failures and all-black/frozen videos are skipped
   (the converter already catches decode exceptions; add a frozen-frame check:
   drop if consecutive-frame diff ≈ 0 for the whole clip).
6. **fps standardization**: `--fps-target 30` downsamples 50/60 fps; 20/25 fps
   left as-is (upsampling robot trajectories is unsafe) or dropped from tier 2.
7. **Aspect/resolution**: resize to the tokenizer input; drop the 2 stitched
   ultrawide sets (or center-crop).
8. **Action normalization**: global q01–q99 clip to [−1,1], gripper included,
   stats merged count-weighted (matches MolmoAct).

---

## 8. Compute & storage plan

**Reality check on "download all the data":** raw decoded frames are huge.
19.5M frames × 480×640×3 ≈ **18 TB uncompressed**. Even at a 128×160 tokenizer
size that's ~1.2 TB uint8. So we do **not** materialize the full raw set on the
laptop (239 GB free). Instead:

* **Laptop (now):** metadata survey (done, all 1,220) + develop on **Tier-1
  pilot** or a subset. A ~20-dataset pilot ≈ a few GB of NPZ.
* **Cloud (scale):** stream-download videos on CPU workers → decode+resize →
  write NPZ/ArrayRecord straight to **GCS** (`gs://visionary-uc1/so101/`). The
  *dynamics* dataset stores **latents**, not frames: 64×16 float16 ≈ 2 KB/frame
  → **~40 GB for all 19.5M frames**. That's the only artifact the dynamics
  trainer streams.
* **Training:** JAX on **TPU** via `cloud/starter.sh --accelerator tpu`
  (existing FSDP path, `data_axis_size 8`). Tokenizer first, then dynamics.
* **Order of magnitude:** tokenizer ~7M params + dynamics 24-layer/128-dim are
  small; the bottleneck is I/O + tokenizing 19.5M frames once (embarrassingly
  parallel across CPU workers writing to GCS).

Disk sizing to store frames for **tokenizer** training (needs pixels, not
latents): store at tokenizer input size to bound it. Tier-1 (3.96M frames @
128×160 uint8 ≈ 61 KB) ≈ **240 GB uncompressed / ~120 GB compressed** → do Tier-1
tokenizer on GCS too, or a subsample (tokenizer needs diversity, not all frames —
a few hundred K frames is plenty).

---

## 9. Phased execution

* **Phase 0 — Setup (done).** Branch `so101`; survey + curate + ingest tools;
  pilot validated; this plan.
* **Phase 1 — Ingest pilot.** `lerobot_to_npz.py --manifest tier1_pilot.json`
  for ~10–20 datasets (`--resize 128,160`), `compute_action_stats.py` →
  `norm_stats.json`. Deliverable: a few GB of clean NPZ + global stats.
* **Phase 2 — Code changes.** Implement §6.1–6.6 behind `action_mode`. Unit-test
  the continuous `ActionEmbedding` shape/grad; keep Atari path green.
* **Phase 3 — Train tokenizer (pilot).** `train_tokenizer.py` on pilot frames;
  eval reconstructions (`scripts/analysis/diagnose_tokenizer.py --mode recon`). Tune
  `resize_shape`/`num_latents` until robot frames reconstruct cleanly.
* **Phase 4 — Train dynamics (pilot).** `save_dynamics_dataset.py` (with
  normalization) → `train_dynamics.py`. Success = action-conditioned rollouts
  that track the held-out video (rising PSNR/SSIM vs. an action-shuffled
  control).
* **Phase 5 — Scale.** Move ingestion to cloud/GCS over Tier-2 (654 datasets);
  retrain tokenizer + dynamics at scale on TPU.
* **Phase 6 — (optional) extensions.** Language conditioning from the
  re-annotated task parquets; multi-camera augmentation; ONNX export + browser
  demo ("Dream SO-101").

---

## 10. Decisions (confirmed)

1. **v1 scope:** action-conditioned **only**. (Language is a later extension via
   the re-annotated task parquets in the manifest repo.)
2. **Compute:** **TPU + GCS** (existing `cloud/starter.sh` path).
3. **Start slice:** **Tier-2** (654 SO-100+SO-101 datasets, 34k episodes, **any
   external view**) built directly on GCS — keep the validated local pilot as the
   pipeline smoke test.
4. **Action representation:** **absolute joint pose + global q01–q99 normalization**
   (`--action-repr absolute`, the default). We accept the residual per-arm
   calibration-offset noise (§4b): the video context grounds absolute pose, so the
   action mainly supplies control. `--action-repr delta` is implemented and stays
   the fallback if rollouts show the model struggling to bind actions to motion.
5. **Camera policy:** single primary **external** cam, `--camera-policy external`
   (default) — mixed viewpoints, so the model must be viewpoint-robust. Top-only
   tooling (`--camera-policy top`, 140 clean datasets) kept as an alternative.
6. **Tokenizer resolution:** `[128,160]` to start; revisit after reconstruction eval.

## 11. Risks

* **Camera semantics vary** even within "external" (overhead vs. front vs. side)
  → the WM must be viewpoint-robust; mitigated by single-view v1 + later
  multi-view. Some `laptop`/`phone` cams are hand-held/unstable in a few repos.
* **AV1 decode throughput** is the ingestion bottleneck; parallelize across CPU
  workers; consider caching decoded+resized frames.
* **Action–video sync**: LeRobot guarantees per-frame alignment; the converter
  clips to `min(len)` (≤1-frame slack observed). Verified on the pilot.
* **fps mixing** if Tier-3 is used without care → prefer resample-to-30 or keep
  Tier-2 (uniform 30 fps).
* **Domain shift** across 377 users (lighting, table, objects) is a *feature*
  for a robust WM but raises tokenizer capacity needs.

## 12. Concrete next commands

```bash
# 0. (once) provision GCS and authenticate to HF. A token matters here for the
#    many small parquet requests, not for video bandwidth.
hf auth login --token <token>

# 1. Survey every dataset's metadata (no video downloaded).
uv run python scripts/so101/survey_metadata.py

# 2. Download. Filtering happens inline: >=5 episodes, all cameras, since camera
#    names cannot be trusted and the verdict needs pixels.
uv run python scripts/so101/download_datasets.py \
  --manifest artifacts/so101/survey.json --out-dir data/so101/hf --dry-run
uv run python scripts/so101/download_datasets.py \
  --manifest artifacts/so101/survey.json --out-dir data/so101/hf

# 3. Measure: classify cameras from pixels, then compute episode trim bounds.
uv run python scripts/so101/analyze.py
uv run python scripts/so101/sample_clips.py      # eyeball what the thresholds selected
uv run python scripts/so101/purge_moving_cameras.py --dry-run   # then --yes

# 4. Pack: trim, transcode, shuffle, write ArrayRecord shards.
uv run python scripts/so101/pack_arecord.py --out-dir data/so101/shards --dry-run
uv run python scripts/so101/pack_arecord.py --out-dir data/so101/shards

# 5. Train the tokenizer directly off the shards.
uv run python scripts/train_tokenizer.py --config-name tokenizer exp_name=so101_tok \
  dataset.source._target_=visionary.dataset.VideoBytesDataSource \
  dataset.clip_transform._target_=visionary.dataset.DecodeRandomVideoClip \
  dataset.train_dir=data/so101/shards/train dataset.eval_dir=data/so101/shards/eval

# 6. Dynamics: per-episode NPZ for the latent precompute, stats computed on the
#    TRIMMED actions, then the dynamics dataset and training run.
uv run python scripts/so101/lerobot_to_npz.py --manifest artifacts/so101/survey.json \
  --out-dir data/so101/raw --fps-target 30
uv run python scripts/so101/compute_action_stats.py \
  --raw-dir data/so101/raw --out data/so101/norm_stats.json
uv run python scripts/data/save_dynamics_dataset.py --checkpoint_dir <so101_tok_ckpt> \
  --input_dir data/so101/raw --output_dir data/so101/dyn --frame_length 64 \
  --action_mode continuous --action_stats data/so101/norm_stats.json
```

Normalization stats must be computed **after** trimming: removing the parked pose
widens the percentiles, so MolmoAct's published stats are not interchangeable
with ours (§6 of [`so101_status.md`](so101_status.md)).
