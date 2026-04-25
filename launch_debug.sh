#!/usr/bin/env bash

BRANCH="refactor-and-perf"
TRAIN_FILE="scripts/dreamer4/train_dynamics.py"
DATA_DISK_SOURCE="projects/visionary-491008/zones/us-east1-d/disks/visionary-data-dynamics-ue1d"
CONFIG_NAME="breakout_dynamics"
EXP_NAME="breakout-dynamics-debug"

gcloud compute tpus queued-resources create "visionary-debug-v6e1" \
  --zone="us-east1-d" \
  --accelerator-type="v6e-1" \
  --runtime-version="v2-alpha-tpuv6e" \
  --node-id="visionary-debug-v6e1" \
  --service-account="visionary-starter@visionary-491008.iam.gserviceaccount.com" \
  --scopes="https://www.googleapis.com/auth/cloud-platform" \
  --data-disk="source=$DATA_DISK_SOURCE,mode=read-only" \
  --spot

echo "ssh:
gcloud compute tpus queued-resources ssh \"visionary-debug-v6e1\" --zone=\"us-east1-d\"

train:
git clone --branch \"$BRANCH\" \"https://github.com/james0248/visionary.git\" ~/visionary && \\
cd ~/visionary && \\
sudo mkdir -p /mnt/data && \\
sudo mount -o ro,defaults /dev/disk/by-id/google-persistent-disk-1 /mnt/data && \\
./setup.sh --accelerator tpu && \\
source ~/.bashrc && \\
uv run python $TRAIN_FILE \\
  --config-name \"$CONFIG_NAME\" \\
  exp_name=\"$EXP_NAME\" \\
  dataset.train_dir=/mnt/data/train \\
  dataset.eval_dir=/mnt/data/eval \\
  wandb.enabled=false"
