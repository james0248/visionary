**Overview**
`watcher.py` keeps exactly one queued TPU resource alive from an allowlisted TRC machine spec. `starter.sh` now contains the full setup logic as well as the TPU VM startup flow: it clones the repo, installs the right JAX build, mounts the data disk, fetches the W&B secret, and launches training with `checkpoint.resume_step=latest`. The repo-root [setup.sh](/Users/hyeonseok/Documents/Sources/visionary/setup.sh) is only a thin wrapper around `starter.sh --mode setup-only`.

**One-Time Setup**

- Create and populate the training data disk once.
  - For `v6e` / `v5e`, use `hyperdisk-ml`.
  - For `v4`, use `pd-balanced`.
  - Format it once and stage `train/` and `eval/` onto it.
- Create the W&B secret in Secret Manager.
  The config field stores the Secret Manager secret name, not the API key value itself.
- Create two service accounts.
  - Starter service account:
    - `roles/secretmanager.secretAccessor`
    - Cloud Storage object access for checkpoints, typically `roles/storage.objectAdmin` on the checkpoint bucket
    - Cloud Storage bucket metadata read for Orbax-on-GCS, typically `roles/storage.legacyBucketReader` on the checkpoint bucket
  - Watcher caller or impersonated service account:
    - `roles/tpu.admin`
    - permission to use the data disk, typically `roles/compute.storageAdmin`
    - read access to the marker objects, typically `roles/storage.objectViewer` on the marker bucket or prefix
    - `roles/iam.serviceAccountUser` on the starter service account
- Fill in [example_watcher.yaml](/Users/hyeonseok/Documents/Sources/visionary/scripts/cloud/example_watcher.yaml).
  The simplest path is to set a single `machine` block with `family`, `chips`, and `region` or `zone`. The watcher derives `accelerator_type` and the TPU runtime version from the family. Advanced fallback rotation is still available through `candidates`.

**Repeated Use**

- Start the watcher:

```bash
uv run python scripts/cloud/watcher.py --config scripts/cloud/example_watcher.yaml
```

- The watcher will:
  - create one queued resource for the selected `machine`, or the next entry in `candidates`
  - leave it alone while it is queued or active
  - delete and recreate it if the queued resource becomes `FAILED` or `SUSPENDED`
  - rotate to the next allowlisted entry only when you are using `candidates` and queue wait exceeds `job.queue_wait_timeout_seconds`
  - stop when the starter writes the completion marker
- The starter will:
  - clone or update the repo on the TPU VM
  - place it at `$HOME/<repo-name>`
  - run its built-in setup path for `--accelerator tpu`
  - mount the attached data disk
  - fetch the W&B API key from Secret Manager
  - launch the configured training script
  - write either `_training_complete.json` or `_training_failed.json` into the checkpoint directory

**Machine Selection**

- Use `machine.family` as one of `v4`, `v5e`, `v6e`.
- Use `machine.chips` as chip count, not raw `accelerator_type`.
  - `v6e` / `v5e`: valid chip counts are `1, 4, 8, 16, 32, 64`
  - `v4`: valid chip counts are `4, 8, 16, 32`
- Use `machine.region` or `machine.zone`.
  - `v6e` spot: `europe-west4`, `us-east1`
  - `v5e` spot: `europe-west4`, `us-central1`
  - `v4` spot or on-demand: `us-central2`
- The watcher derives the queued-resource runtime version automatically:
  - `v6e` -> `v2-alpha-tpuv6e`
  - `v5e` -> `v2-alpha-tpuv5-lite`
  - `v4` -> `tpu-ubuntu2204-base`
- For `v4`, Google names `accelerator_type` by TensorCore count. The watcher hides that and lets you specify chip count directly.
  - `4 chips` -> `v4-8`
  - `8 chips` -> `v4-16`
  - `16 chips` -> `v4-32`
  - `32 chips` -> `v4-64`

**Notes**

- If you set only one `machine`, watcher will keep retrying that exact TPU spec.
- If you want fallback rotation across multiple specs, use `candidates` instead of `machine`.
- Marker URIs are explicit. The watcher no longer derives them from a separate `training.checkpoint_dir` field.
- The current trainers reconstruct the Grain iterator position from the restored training step instead of saving a separate iterator sidecar. In the current setup this works because the training loaders are deterministic and use `worker_count=0`.
- Exact bitwise determinism is still only realistic when you resume on the same accelerator family and software stack. Restoring a `v4` checkpoint onto `v6e` is supported operationally, but not guaranteed to be numerically identical.
