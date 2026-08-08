# Repo refactor plan

Working plan for restructuring `visionary/` and `scripts/`. Written for whoever
picks this up next — human or agent. Every claim below was verified against the
tree at `91f089f`; file:line references are real.

Read §2 (principles) and §4 (the blocker) before moving anything.

---

## 1. Why

Three things are coming that the current layout does not accommodate:

1. **More model architectures.** Specifically a latent action model (LAM) that
   infers actions from video instead of consuming ground-truth action vectors,
   and possibly architectures that are not transformers at all.
2. **Imagination RL** (Dreamer 4 §policy) — training a policy inside the learned
   world model, no environment in the loop.
3. **Cleanup.** `scripts/` accumulated one-off probes with no separation between
   "load-bearing" and "answered a question in March".

The layout today implicitly assumes one architecture. `visionary/` is a flat
namespace of Dreamer-4 parts (`tokenizer.py`, `dynamics.py`, `transformer.py`,
`sigreg.py`) sitting next to genuinely shared infrastructure (`dataset.py`,
`common/checkpoint.py`). A second architecture has nowhere to go that does not
either collide or invite bad reuse.

## 2. Guiding principles

These are settled. Do not re-litigate them; they were decided explicitly.

- **Self-containment over reuse.** Do not assume every component is shared. If
  code is used by exactly one path, it belongs *in* that path, even at the cost
  of duplication. Duplication is cheaper than a wrong abstraction here.
- **Shared code must not know any model exists.** Models opt in to shared
  utilities; utilities never branch on model identity. Concretely: when the LAM
  lands, do **not** add a third branch to `ActionEmbedding` in
  `visionary/dynamics.py`. It gets its own module.
- **No trainer shims.** Trainers stay as real, standalone, readable scripts in
  `scripts/` — one per model. A thin `train_dynamics.py` delegating into the
  package was proposed and **rejected** as over-abstraction.
- **No generic base `TrainState`.** There isn't one today (all three subclass
  flax's `TrainState` directly; `DynamicsTrainState` is `pass`) and creating one
  was explicitly declined.
- **The Atari path is kept.** It is not legacy. The DQN checkpoints were lost and
  are being retrained to regenerate expert rollouts, and Atari remains the small
  fast dataset for smoke-testing new architectures. The NPZ raw-data path stays
  supported alongside the mp4/ArrayRecord one.

## 3. Current state

`visionary/` — 5,033 lines. `scripts/` — 9,209 lines.

| Path | Role | Verdict |
|---|---|---|
| `visionary/common/{checkpoint,jax,wandb}.py` | genuinely model-agnostic infra | keep at `common/` |
| `visionary/common/{env,buffers,rollout}.py` | Atari-only (gym wrappers, replay buffer) | absorb into the Atari path |
| `visionary/common/train_state.py` | 3 unrelated classes in one file | split; see §5.1 |
| `visionary/dataset.py` | shared video/ArrayRecord pipeline | keep at top level |
| `visionary/{tokenizer,tokenizer_preprocessor,dynamics,transformer,sigreg}.py` | Dreamer-4-specific | move under `models/dreamer4/` |
| `visionary/export/onnx_wrappers.py` (2,546 lines) | shadows the Dreamer-4 architecture | move **inside** `models/dreamer4/` |
| `visionary/lpips/` | perceptual loss, arch-agnostic | keep at top level |
| `visionary/eval/loading.py` | checkpoint/config restore helpers | keep |
| `scripts/{train_tokenizer,train_dynamics}.py` | the real trainers | stay standalone |
| `scripts/dream-arcade/` | DQN train + rollout collection | rename to `scripts/atari/`, self-contain |
| `scripts/data/` | dataset builders, validators | keep |
| `scripts/so101/` | corpus curation, packing | keep |
| `scripts/analysis/` | live probes | keep, consolidate (§5.2) |
| `scripts/deprecated/` | 6 answered probes | keep as-is, do not delete |

`scripts/deprecated/` deliberately lives **under `scripts/`** so the probes'
`sys.path.insert(0, parents[1])` hack still resolves and they stay runnable.

### Already done

- `cloud/` split into `cloud/atari/` (12 yamls) + `cloud/so101/` (5 yamls).
- `cloud/tokenizer_queue.py` (649 lines) deleted.
- `build_raw_index` / `load_train_config` / `restore_params` promoted out of
  `scripts/eval_dynamics_videos.py` into `visionary/eval/loading.py`. The old
  module still imports those names (`eval_dynamics_videos.py:37`), so
  `from eval_dynamics_videos import ...` continues to resolve.
- Six answered probes moved to `scripts/deprecated/`.
- §5.2 probe consolidation landed (`456287f`): `eval_pixel_tf.py` merged into
  `eval_teacher_forced.py` (`--space latent|pixel`), `reconstruct_tokenizer.py`
  into `diagnose_tokenizer.py` (`--mode recon|ablate`).
- §4 resolved by rewriting stored configs (no shims): all 159 checkpoint JSONs
  on GCS (`metadata.json` + `model/<step>/config/metadata`) rewritten to the
  `visionary.models.dreamer4.*` paths, originals kept beside each as
  `*.pre-refactor`. Verified: dynamics + tokenizer train configs instantiate,
  tokenizer export restores end-to-end with weights.
- §5.3 landed: `tokenizer{,_preprocessor}.py`, `dynamics.py`, `transformer.py`,
  `sigreg.py`, `export/onnx_wrappers.py` moved to `visionary/models/dreamer4/`;
  imports and yaml `_target_` strings rewritten repo-wide.
- §5.1 landed: `scripts/dream-arcade` → `scripts/atari`, absorbed
  `common/{env,buffers,rollout}.py` and `TargetTrainState` (now
  `scripts/atari/train_state.py`); `TokenizerTrainState`/`DynamicsTrainState`
  untouched. Verified no live TPU job before moving.
- §6 leftovers fixed: `cloud/README.md` example refs, unused
  `align_actions_to_frames` import. Plan complete except EnvPool (tracked in
  §5.1 notes, needs Linux TPU VM).
- `--config` made required on `diagnose_tokenizer.py` and
  `reconstruct_tokenizer.py` (both previously defaulted to
  `scripts/analysis/config/breakout_tokenizer.yaml`, which does not exist —
  `reconstruct_tokenizer.py` could not run at all).

## 4. The blocker: `_target_` strings live inside checkpoints

**This is the one thing that turns the `models/` reshape from a `git mv` into a
migration.** Understand it before touching `visionary/*.py`.

Hydra `_target_` strings in `scripts/config/*.yaml` name package paths:

```
visionary.tokenizer.Tokenizer                 (tokenizer.yaml:38, so101_tokenizer.yaml:56)
visionary.dynamics.DynamicsModel              (dynamics.yaml:52, so101_dynamics_*.yaml)
visionary.dataset.VideoDataSource             (tokenizer.yaml:24)
visionary.dataset.VideoBytesDataSource        (so101_tokenizer.yaml:33)
visionary.dataset.{RandomVideoCrop,DecodeRandomVideoClip,AugmentVideoClip}
visionary.common.checkpoint.CheckpointManager (every config)
```

The training config is **serialized into orbax checkpoint metadata**.
`visionary/eval/loading.py:load_train_config` reads it back *out of the
checkpoint*, and `instantiate(cfg.dynamics)` resolves the stored string at
restore time.

So moving `visionary/tokenizer.py` → `visionary/models/dreamer4/tokenizer.py`
makes **every existing checkpoint unrestorable**. That breaks not just eval but
**spot-preemption auto-resume**, which is the mechanism the whole TPU fleet
strategy depends on.

Two viable fixes:

- **Compat shims (recommended).** Leave `visionary/tokenizer.py` etc. in place as
  one-line re-exports from the new location. Cheap, reversible, keeps old
  checkpoints loadable forever. Cost: the old names linger.
- **Rewrite stored configs.** Walk every checkpoint's metadata and rewrite the
  `_target_` strings. Complete, but touches artifacts on GCS and is not
  reversible if it goes wrong mid-run.

Whichever is chosen, the acceptance test is the same and must run **before**
anything is pushed to a branch a TPU VM tracks: restore an existing SO-101
dynamics checkpoint end-to-end and confirm `instantiate` resolves.

### Other verified couplings

- `scripts/data/save_dynamics_dataset_from_shards.py` imports 5 helpers
  (`ShardWriter`, `SplitStats`, `build_action_normalizer`, `chunk_starts`,
  `record_bounds`) from `save_dynamics_dataset.py`, and re-implements
  `compute_action_stats` inline — a third copy, after
  `scripts/so101/compute_action_stats.py`.
- `webgpu_app/export/` imports `visionary.{common.checkpoint, dynamics,
  tokenizer, tokenizer_preprocessor, export.onnx_wrappers}`. The demo is 100%
  Breakout-hardcoded.
- `cloud/starter.sh` runs `training.script` as a **path**
  (`uv run python "$TRAIN_SCRIPT"`), and hydra's `config_path="config"` is
  relative to the trainer file. **Trainers and their configs move together or
  not at all.**
- `cloud/starter.sh` resumes with `git pull --ff-only origin $BRANCH`, so any
  force-push to a branch a VM already cloned breaks its restart. If history is
  rewritten, either bring the VMs down first or change this to
  `git reset --hard origin/$BRANCH`.
- `moving_pixels` is copy-pasted across `eval_latent_{denoise,noise,stride}.py`
  (all now in `deprecated/`, so this is no longer worth fixing).
- `cloud/README.md` points at a non-existent `cloud/example_watcher.yaml`.

## 5. Work items

Ordered by dependency. Phases 1 and 2 are independent of the blocker and can
proceed now; phase 3 cannot.

### 5.1 Self-contain the Atari path — *approved, deferred*

Deferred by the user until after the in-flight DQN training run, to avoid moving
files under a running job.

- `git mv scripts/dream-arcade scripts/atari`.
- Absorb `visionary/common/{env,buffers,rollout}.py` into that directory. Sole
  importers are `train_dqn.py` (lines 17, 23, 30) and `collect_rollouts.py`
  (line 16).
- Move `TargetTrainState` in beside them — used only by `train_dqn.py:31` and
  `collect_rollouts.py:17`.
- Leave `TokenizerTrainState` and `DynamicsTrainState` where they are.

**Caution:** do not delete `DynamicsTrainState` as part of "cleanup". It is
imported live at `visionary/eval/loading.py:11` *and* orbax restores into a
target pytree, so removing it can break checkpoint loads even though the class
body is `pass`.

Related, same area: `train_dqn.py` is throughput-bound on
`gym.vector.SyncVectorEnv` (single-threaded — uses ~1 of the TPU host's 180+
vCPUs) and on `ReplayBuffer.sample`, which does numpy fancy-index gathers plus a
`device_put` per call. EnvPool is the intended fix. **Note: EnvPool cannot be
validated on macOS** — `envpool.make_gymnasium("Breakout-v5", ...)` hangs
indefinitely on the arm64 wheel (observed 14h at 99.5% CPU), `envpool.procgen`
fails to link against Qt5, and the bundled gfootball SDL collides with `cv2`'s.
Observation-equivalence must be checked on the Linux TPU VM. Its defaults differ
from our gym stack in four settings — `repeat_action_probability` (0.0 vs our
0.25), `episodic_life`, `reward_clip`, and the episode cap — and it has **no `FireResetEnv` equivalent**.

### 5.2 Consolidate probes

- `eval_pixel_tf.py` + `eval_teacher_forced.py` → one script, `--space latent|pixel`.
- `diagnose_tokenizer.py` + `reconstruct_tokenizer.py` → one script, `--mode recon|ablate`.

### 5.3 Reshape into `visionary/models/` — *blocked on §4*

Target:

```
visionary/
  common/          checkpoint.py  jax.py  wandb.py       # knows about no model
  dataset.py                                             # shared pipeline
  lpips/
  eval/loading.py
  models/
    dreamer4/      tokenizer.py  tokenizer_preprocessor.py
                   dynamics.py  transformer.py  sigreg.py
                   onnx_wrappers.py
    lam/           # future — its own action embedding, no shared branch
scripts/
  train_tokenizer.py  train_dynamics.py  eval_dynamics_videos.py
  atari/  so101/  data/  analysis/  deprecated/
```

Sequence: land compat shims first, verify a real checkpoint restores, *then*
move files.

## 6. Open items

Small, unowned, none blocking:

- `cd81a9d`'s commit message describes only its author's work, but the commit
  also swept up 16 renames from the `cloud/` reorg plus the 649-line
  `tokenizer_queue.py` deletion. Already pushed; left alone deliberately.
- `cloud/README.md` references a non-existent `cloud/example_watcher.yaml`.
- `eval_dynamics_videos.py` has a pre-existing unused `align_actions_to_frames`
  import (line 35).

## 7. Conventions

- New files carry **code only** — no comments. Comments moved verbatim with
  relocated code are fine.
- Commits in this repo omit the `Co-Authored-By` trailer.
- Report the expected cost of every GCP operation and wait for explicit
  confirmation before executing. See `CLAUDE.md`.
