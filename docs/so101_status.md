# SO-101 world model — working status

Durable record of what has been built, decided, and measured on branch `so101`.
Companion docs: [`so101_world_model_plan.md`](so101_world_model_plan.md) (the plan)
and [`molmoact2_findings.md`](molmoact2_findings.md) (upstream research).

---

## 1. Where the data comes from

MolmoAct2 published a curated allowlist of community LeRobot SO-100/101 datasets:
**1,220 repos from 377 users** (`scripts/so101/molmoact2_repo_list.json`). It is
already post-curation — they started from 1,660 candidates and dropped ~27% via
structural checks plus a TOPReward quality gate. We inherit that result.

Of the 1,220, **1,215 are reachable** (5 are auth-gated). Applying the only
dataset-level filter we kept — at least 5 episodes, which removes 482 abandoned
`*_test` uploads — leaves **733 datasets, 36,320 episodes, 174.8 h**. All of it is
downloaded (~339 GB after purging moving-camera video).

## 2. Measured facts about the corpus

Established by scanning, not assumption:

| Property | Finding |
| --- | --- |
| Action / state space | **`(6,)` joint pose, identical names, 100% of datasets** |
| Units | **degrees, 100% of 1,215** — no radians, no normalized |
| LeRobot version | v2.1 everywhere |
| fps | 30 (95.8% of streams), tail of 20–60 |
| Resolution | 640×480 (97.9% of streams), tail incl. 16:9 and one portrait |
| Codec | ~87% AV1, ~13% H.264, always in mp4 |
| Rewards | **none — zero datasets have any reward field** (teleop demos) |
| Cameras | 1–4 per dataset, **names that do not describe the view** |

### Camera names are unreliable — measured

Classifying all **1,336 cameras** from pixels: **~30% are moving** (wrist/egocentric).

| Name | Actually moving |
| --- | --- |
| `on_robot` | 92% |
| `wrist` | 89% (so 11% of "wrist" cameras are *fixed*) |
| `base` | 62% |
| `laptop` | 26% |
| `phone` | 19% |
| `top` | 13% |
| `side_view` | 0% |

Names mislead in **both** directions, so the verdict comes only from pixels:
`static_frac` (fraction of near-static pixels) plus `shift_p90` (global
phase-correlation egomotion). Both conditions are required — a fixed close-up
camera can have few static pixels, but only a moving camera also translates.

### Calibration differs per user

Per-dataset joint means differ by 18–56 deg (wrist_roll worst). MolmoAct's own
published stats show the same signature (wrist_roll std 69.4). Units are
consistent; *calibration* is not. See `molmoact2_findings.md` §1 for the exact
correction they apply when driving a real arm.

## 3. Decisions made (and why)

| Decision | Rationale |
| --- | --- |
| **Action-only v1** (no language) | smallest correct first target; MolmoAct's re-annotations are available later |
| **TPU + GCS** for training | matches the repo's existing `cloud/starter.sh` path |
| **Absolute joint pose** + q01–q99 normalization | matches MolmoAct; video context grounds absolute pose. Delta actions were considered, then removed as unused |
| **Keep video encoded**, decode clips on the fly | measured 2,400 fps/core random-access decode, so no frame dump is needed |
| **Download all cameras**, filter later from pixels | names are unreliable; filtering after download costs only disk |
| **Each fixed view = its own stream** | multi-view datasets become several single-camera streams sharing one action trajectory: **1.47× more training video** |
| **Junk filter only for download** (≥5 episodes) | every other property is better measured after download |
| **Trim stale head/tail physically** | ~14% of duration carries no action-to-pixel signal; approved after reviewing `outputs/trim_audit/` |
| **Drop off-rate and off-aspect streams** | LeRobot stores one action row per frame, so resampling desyncs actions; scaling 16:9 into 4:3 distorts geometry. Costs 5.7% of frames |
| **Shuffle records before packing** | grain shards by contiguous blocks, so dataset-ordered records would give each host a disjoint subset |
| **No mid-episode cuts** | would break the temporal continuity a world model depends on |

## 4. Pipeline

```
molmoact2_repo_list.json (1,220)
   |
   +-- survey_metadata.py ------> survey.json          (metadata only, no video)
   +-- download_datasets.py ----> data/so101/hf/       (filters + fetches)
   |
   +-- analyze.py --------------> camera_views.json    (--cameras: fixed vs moving, from pixels)
   |                              episode_bounds.json  (--episodes: trim bounds + flags)
   |
   +-- pack_arecord.py ---------> data/so101/shards/   (trim + transcode + shuffle + pack)
   +-- compute_action_stats.py -> norm_stats.json      (q01-q99, AFTER trimming)
   |
   +-- train_tokenizer.py                    (VideoBytesDataSource decodes clips on the fly)
   +-- save_dynamics_dataset_from_shards.py  (shards -> latents, multi-device, actions normalized)
   +-- train_dynamics.py                     (continuous 6-DoF conditioning)
```

### Record schema

One record per (episode, fixed camera). A dataset with top and side views yields
two independent streams sharing one action trajectory.

```
video   : encoded mp4 bytes, already trimmed to the non-idle span
length  : frame count
actions : (T, 6) float32, trimmed identically
state   : (T, 6) float32
repo, episode, camera
```

Video and actions are cut with the same bounds and the decoded frame count is
asserted equal to the action rows — a desync would poison training without
raising anywhere downstream.

### Tools in `scripts/so101/`

| Script | Purpose |
| --- | --- |
| `survey_metadata.py` | fetch every dataset's `meta/info.json`; compatibility survey |
| `download_datasets.py` | filter + bulk/probe fetch, resumable, verifies outcomes, disk guard |
| `analyze.py` | camera classification (pixels) + episode QC (parquet), independently runnable |
| `pack_arecord.py` | trim, transcode, shuffle, pack into ArrayRecord shards |
| `compute_action_stats.py` | pooled q01/q99 normalization stats |
| `sample_clips.py` | render annotated samples so the automated verdicts can be eyeballed |
| `purge_moving_cameras.py` | delete video for verified-moving cameras (reclaimed 87 GB) |
| `lerobot_to_npz.py` | per-episode NPZ; obsolete since the dynamics stage reads the packed shards |
| `metadata/camera_views.json` | checked in — the moving video it describes is deleted, so it cannot be re-derived |

## 5. Code changes to the repo

Behaviour is gated so the **Atari recipe is unchanged** (`action_mode=discrete`,
the arecord frame source stays the default, no warmup or clipping). The one
exception is the final RMSNorm, which changes the parameter tree for every
model: Atari checkpoints from before it need `abe92ca`, as the README says.

* `visionary/dynamics.py` — `ActionEmbedding` gains a continuous MLP branch
  selected by `action_mode`; unknown modes raise.
* `visionary/transformer.py` — `SpatioTemporalTransformer.remat` wraps each block
  in `nn.remat`, with `remat_policy` naming a `jax.checkpoint_policies` attribute.
  Threaded through `Tokenizer` and `DynamicsModel`, off by default, on for SO-101.
  The parameter tree is identical either way, so a checkpoint moves between
  settings. Measured cost at the configured shapes, per v6e chip:

  | policy | tokenizer | dynamics |
  | --- | --- | --- |
  | off | 1.00× flops, 73.5 GB — will not fit | 1.00×, 19.6 GB |
  | `nothing_saveable` | 1.25×, 1.3 GB | 1.25×, 0.5 GB |
  | `dots_with_no_batch_dims_saveable` | **1.02×, 9.9 GB** | 1.01×, 3.5 GB |
  | `dots_saveable` | 1.00×, 22.0 GB | **1.00×, 5.4 GB** |

  Bold is what each config uses. Full remat costs 25% of compute to save memory
  neither model needs; keeping the matmuls buys almost all of it back. Measured
  before the final norm and fp32 head; the tokenizer now holds 21.3 GB of
  activations per chip, 23.7 GB with optimizer state, of 32 GB.
* `visionary/transformer.py` — `SpatioTemporalTransformer` now ends in an
  RMSNorm. Pre-norm never normalizes the residual stream, so every consumer
  inherited a scale that grows with depth. This killed the first tokenizer run
  (§7). Changes the parameter tree, so pre-`abe92ca` checkpoints no longer load.
* `visionary/tokenizer.py` — the decoder's output projection and sigmoid run in
  fp32. In bf16 a logit past ~8 rounds sigmoid to exactly 1, making the gradient
  zero rather than small, which is unrecoverable.
* `scripts/config/*.yaml` — the optimizer is a `_target_` tree instantiated with
  hydra, so the schedule and any gradient transforms are config, not code. SO-101
  chains `clip_by_global_norm(1.0)` with adam on a 1000-step linear warmup; the
  Atari configs are a bare `optax.adam`, unchanged.
* `visionary/sigreg.py` — SIGReg (LeJEPA), ported from le-wm's official module
  (MIT) and verified to 1e-7 against it on shared inputs. Replaces the
  tokenizer's bottleneck RMSNorm: the encoder ends in a bare `Dense(16)` and the
  loss (`sigreg_weight`, le-wm's 0.09) holds the pooled latents at N(0, I).
  Rematted, so it adds no activation memory; `Tokenizer.reconstruct` now also
  returns the latent, and `compute_loss_metrics` logs per-channel latent
  mean/std so the distribution can be watched during training.
* `visionary/dataset.py` — `decode_video_window` (accepts a path or mp4 bytes),
  `VideoBytesDataSource`, `DecodeRandomVideoClip`. Removed dead
  `PreprocessAndPatchify`.
* `scripts/train_tokenizer.py` — data source and clip transform are instantiated
  from config, so storage formats are selected without a branch.
* `scripts/train_dynamics.py` — eval action dtype follows `action_mode`.
* `scripts/data/save_dynamics_dataset.py` — `--action_mode`, `--action_stats`;
  q01–q99 normalization with clipping; shape-aware `prev_action`.
* `scripts/data/save_dynamics_dataset_from_shards.py` — the SO-101 dynamics
  stage. Reads the packed video shards directly (no NPZ detour), encodes every
  frame with the tokenizer export in 32-frame windows sharded over all local
  devices, and writes the chunked latent records `train_dynamics.py` reads.
  Preserves the input train/eval split; computes q01–q99 action stats from the
  shards themselves (deduped per (repo, episode) so multi-view streams do not
  double-count) unless `--action_stats` is given, resolving the stale-stats
  open item. `--encode_window_overlap` re-encodes warm-up frames so kept
  latents see that much temporal context; `--augment_copies` writes extra
  augmented encodings per train stream (photometric + crop, one draw per
  episode, eval stays clean). Latents default to float16 — SIGReg holds them
  near N(0, 1), so half precision costs nothing and halves the dataset.
* `scripts/config/{tokenizer,dynamics}.yaml` — new keys, Atari defaults preserved.
* `scripts/config/so101_{tokenizer,dynamics}.yaml` — standalone, no `defaults`
  list, so an Atari change cannot silently move the SO-101 run.
* `cloud/so101_{tokenizer,dynamics}.yaml` — TPU v6e-8 job specs. They name the
  standalone configs and override only mount paths and the names the watcher
  keys off; everything else lives in the config.
* `pyproject.toml` — `av`, `pyarrow`, `huggingface-hub`.

## 6. Open items

1. **The first tokenizer run is dead and its checkpoints are worthless** (§7).
   Restart from scratch; the parameter tree changed anyway.
2. **Verify the latent distribution after training.** The bottleneck is now a
   bare `Dense(16)` with a SIGReg loss (LeJEPA, ported from le-wm and verified
   against it numerically) pulling the pooled latents toward N(0, I) — the
   distribution the flow-matching dynamics interpolates against, removing the
   need for the per-channel standardization stage open-dreamer bolts on.
   Context: Dreamer 4 and its reimplementations use `linear -> tanh` with no
   norm; their tanh survives because trained latent std lands at ~0.1 (deep in
   tanh's linear region), which is exactly why open-dreamer must standardize
   per channel afterward. Our earlier tanh died because bf16 tanh past |z| ≈ 4
   rounds to ±1 with exactly zero gradient — same trap as §7. SIGReg is a soft
   regularizer, not a guarantee: after training, measure per-channel mean/std
   over the tokenized corpus; if they are off, fall back to standardizing in
   `save_dynamics_dataset.py` (stats file alongside the records, normalize
   entering dynamics training, invert before `Tokenizer.decode`).
3. **163 borderline cameras** classified but not visually verified.
4. **Mid-episode pauses** (264) — flagged, currently kept.

## 7. Gotchas learned the hard way

* **A saturated sigmoid in bf16 gives exactly zero gradient, not a small one.**
  The first tokenizer run sat at mse 0.31 (= E[x²] of the data, the score of an
  input-independent output) for 17k steps. Diagnosed from the step-15000
  checkpoint: decoder logits had rms 628 against 3.6 at init, 48% above +8.3
  where bf16 rounds σ to 1.0 and 50% below −87 where it underflows to 0. With
  σ′ = 0 everywhere the whole model was frozen — median parameter movement over
  15k steps was 4.9%, cosine to init ≥ 0.975. Two independent causes, both
  fixed: no final norm on the residual stream, and a bf16 output head. Atari
  survived the same code only because 128-dim/8-layer keeps the stream small
  enough; the failure is triggered by scale, not by data.
* **Loss pinned exactly at the data variance means dead, not slow.** Worth
  checking E[x²] before assuming a flat curve is a learning-rate problem.
* **Measure the checkpoint, not the loss curve.** Parameter movement versus a
  fresh init at the same seed separates "learning slowly" from "not learning",
  and costs one restore.
* **Deleting data makes derived analysis silently wrong, not absent.** Re-running
  the camera classifier after purging moving-camera video reported 102 moving
  instead of 406 and overwrote the good artifact. Recovered from a backup;
  `camera_views.json` is now checked into the repo.
* **HF downloads can silently no-op under rate limiting.** `snapshot_download`
  returned without error while fetching nothing for ~600 repos. Verify files
  landed; never trust the absence of an exception.
* **A token matters for *many small files*, not for bandwidth.** Video is
  bandwidth-bound; parquet is ~34k tiny requests and was hard rate-limited.
* **Do not mutate a tree while another job walks it.** Purging during a download
  produced `FileNotFoundError` from `stat()` racing the deletion.
* **`"laptop"` contains `"top"`.** A substring match for top-down cameras
  produced 548 false positives. Tokenize before matching.
* **Fast teleop looks like corruption.** A flat 400 deg/s cap flagged 20% of
  episodes; peak velocity p50 is 267 and p99 708. Real glitches are isolated
  spikes, so the test must be relative to the episode's own p95.
* **Estimates from one sample are wrong.** Download size was projected at 238 GB
  from a single episode's bytes/frame; the real total was 396 GB.
