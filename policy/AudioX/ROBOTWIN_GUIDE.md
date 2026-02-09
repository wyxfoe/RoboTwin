# AudioX — RoboTwin Fine-tuning & Evaluation Guide

AudioX is an audio-generation Diffusion Transformer (DiT) adapted for robot
trajectory prediction via `stable_audio_tools`.  This document covers the
**complete workflow**: fine-tuning on RoboTwin demonstrations and running
closed-loop evaluation inside RoboTwin environments.

---

## Table of Contents

1. [File Overview](#file-overview)
2. [Prerequisites](#prerequisites)
3. [Model Architecture](#model-architecture)
4. [Fine-tuning](#fine-tuning)
5. [RoboTwin Evaluation](#robotwin-evaluation)
6. [End-to-End Example](#end-to-end-example)
7. [Configuration Reference](#configuration-reference)

---

## File Overview

```
policy/AudioX/
├── configs/
│   ├── robotx_aloha_agilex.json   # ALOHA-Agilex (action_dim=14, default)
│   └── robotx_robotwin.json       # Larger config  (action_dim=16)
├── audiox_model.py                # AudioXRobot wrapper + ActionHead
├── deploy_policy.py               # RoboTwin adapter (get_model / eval / reset_model)
├── deploy_policy.yml              # Evaluation parameters
├── eval.sh                        # Evaluation launch script
├── train_robotx.py                # Fine-tuning script (PyTorch Lightning)
├── inference_robotx.py            # Standalone inference (outside RoboTwin)
└── ROBOTWIN_GUIDE.md              # This document
```

---

## Prerequisites

### AudioX Source

The `stable_audio_tools` package is required (from AudioX source, not pip).

```bash
# Option A: environment variable (recommended)
export AUDIOX_PATH=/path/to/AudioX-

# Option B: place AudioX- as a sibling directory to RoboTwin
#   parent/
#   ├── RoboTwin/
#   └── AudioX-/
#       └── stable_audio_tools/
```

### Python Dependencies

```bash
pip install pytorch-lightning wandb safetensors transformers
```

---

## Model Architecture

### Inputs (3 conditioning streams, all via cross-attention)

| Stream  | Config key | Encoder            | Details                                      |
|---------|------------|--------------------|----------------------------------------------|
| Text    | `prompt`   | T5-base (len=64)   | Natural language task instruction             |
| Vision  | `video`    | CLIP-ViT-B/32      | 3 cameras (head, left, right), CLIP-normalized|
| Proprio | `proprio`  | MLP trajectory      | 14-dim joint state vector                     |

### Output Pipeline (`audiox_model.py: get_action()`)

```
Diffusion sampling  →  (batch, action_dim, chunk_size)    # audio layout
        ↓ .squeeze(0).T
Transpose           →  (chunk_size, action_dim)            # robot layout
        ↓ ActionHead
Norm + MLP + residual                                      # action projection
        ↓ _denormalize_actions
[-1,1] → real joint angles                                 # undo training norm
        ↓ _clamp_gripper
Gripper clamped to [0, 1]                                  # hardware constraint
        ↓ .cpu().numpy()
Output: (50, 14) numpy array                               # 50-step trajectory
```

### Action Dimension Layout (ALOHA-Agilex, dim=14)

```
[left_arm j1..j6,  left_gripper,  right_arm j1..j6,  right_gripper]
 index: 0..5        6              7..12               13
```

---

## Fine-tuning

### 1. Prepare Data

Organize RoboTwin demonstrations as HDF5 episodes:

```
data/<task_name>/demo_clean_500/
├── episode_0001.hdf5
├── episode_0002.hdf5
└── ...
```

### 2. Configure

Edit `configs/robotx_aloha_agilex.json` dataset section:

```json
"dataset": {
    "data_dir": "data/beat_block_hammer/demo_clean_500",
    "task_description": "use the hammer to beat the block",
    "camera_names": ["front_camera", "head_camera", "left_camera", "right_camera"],
    "normalize": true,
    "batch_size": 16
}
```

### 3. Train

```bash
# Single GPU
python policy/AudioX/train_robotx.py \
    --config policy/AudioX/configs/robotx_aloha_agilex.json \
    --data_dir data/beat_block_hammer/demo_clean_500 \
    --save_dir checkpoints/audiox/beat_block_hammer \
    --batch_size 16 --max_steps 50000 --wandb

# Multi-GPU (4x DDP)
python policy/AudioX/train_robotx.py \
    --config policy/AudioX/configs/robotx_aloha_agilex.json \
    --data_dir data/beat_block_hammer/demo_clean_500 \
    --save_dir checkpoints/audiox/beat_block_hammer \
    --batch_size 16 --num_gpus 4 --max_steps 50000 --wandb

# Resume from checkpoint
python policy/AudioX/train_robotx.py \
    --config policy/AudioX/configs/robotx_aloha_agilex.json \
    --ckpt_path checkpoints/audiox/beat_block_hammer/robotx-00020000.ckpt \
    --save_dir checkpoints/audiox/beat_block_hammer
```

### 4. Training Outputs

```
checkpoints/audiox/beat_block_hammer/
├── robotx-00010000.ckpt    # periodic checkpoints
├── robotx_final.pt         # final model weights
├── action_stats.pt         # normalization stats (required for inference)
└── action_head.pt          # ActionHead weights (optional)
```

---

## RoboTwin Evaluation

### 1. Configure `deploy_policy.yml`

```yaml
model_config: configs/robotx_aloha_agilex.json
ckpt_path: checkpoints/audiox/beat_block_hammer/robotx_final.pt
action_stats_path: checkpoints/audiox/beat_block_hammer/action_stats.pt
action_head_path: checkpoints/audiox/beat_block_hammer/action_head.pt
action_dim: 14
action_chunk_size: 50
```

### 2. Run Evaluation

```bash
cd policy/AudioX

# Single task
bash eval.sh beat_block_hammer demo_clean 0 0
#            task_name          task_cfg   seed gpu

# Batch evaluation
for task in beat_block_hammer place_object_scale pick_cube_bottle; do
    bash eval.sh $task demo_clean 0 0
done
```

### 3. Internal Flow

```
eval.sh
  └─ script/eval_policy.py
       ├─ load deploy_policy.yml
       ├─ get_model(usr_args)     → AudioXRobot instance
       └─ loop per episode:
            ├─ obs = TASK_ENV.get_obs()
            ├─ eval(TASK_ENV, model, obs)
            │    ├─ encode_obs()  → 3 RGB images + joint state
            │    ├─ model.get_action() → (50, 14) trajectory
            │    └─ for step in trajectory:
            │         TASK_ENV.take_action(step)
            └─ reset_model()
```

---

## End-to-End Example

```bash
# === Step 0: Environment ===
export AUDIOX_PATH=/path/to/AudioX-
cd /path/to/RoboTwin

# === Step 1: Fine-tune ===
python policy/AudioX/train_robotx.py \
    --config policy/AudioX/configs/robotx_aloha_agilex.json \
    --data_dir data/beat_block_hammer/demo_clean_500 \
    --save_dir checkpoints/audiox/beat_block_hammer \
    --batch_size 16 --max_steps 50000

# === Step 2: Configure inference paths in deploy_policy.yml ===
#   ckpt_path:         checkpoints/audiox/beat_block_hammer/robotx_final.pt
#   action_stats_path: checkpoints/audiox/beat_block_hammer/action_stats.pt

# === Step 3: Evaluate ===
cd policy/AudioX
bash eval.sh beat_block_hammer demo_clean 0 0

# === (Optional) Standalone inference ===
python inference_robotx.py \
    --config configs/robotx_aloha_agilex.json \
    --ckpt_path checkpoints/audiox/beat_block_hammer/robotx_final.pt \
    --action_stats checkpoints/audiox/beat_block_hammer/action_stats.pt \
    --image_path test_obs.png \
    --instruction "use the hammer to beat the block"
```

---

## Configuration Reference

### robotx_aloha_agilex.json (default)

| Parameter            | Value   | Description                        |
|----------------------|---------|------------------------------------|
| action_dim           | 14      | ALOHA: (6 joints + 1 gripper) x 2 |
| action_chunk_size    | 50      | Steps per trajectory               |
| embed_dim            | 512     | DiT hidden dimension               |
| depth                | 8       | Transformer layers                 |
| num_heads            | 8       | Attention heads                    |
| diffusion_objective  | v       | v-prediction                       |
| T5                   | t5-base | Text encoder                       |
| CLIP                 | ViT-B/32| Vision encoder                     |

### robotx_robotwin.json (larger)

| Parameter   | Value | Description       |
|-------------|-------|-------------------|
| action_dim  | 16    | Generic dual-arm  |
| embed_dim   | 768   | Larger DiT        |
| depth       | 12    | Deeper            |
| num_heads   | 12    | More heads        |

### deploy_policy.yml fields

| Field              | Description                                 |
|--------------------|---------------------------------------------|
| model_config       | Path to model config JSON                   |
| ckpt_path          | Fine-tuned model weights                    |
| action_stats_path  | Normalization stats from training           |
| action_head_path   | ActionHead weights (optional, identity if null) |
| action_dim         | Must match JSON config                      |
| action_chunk_size  | Trajectory length per inference             |
| head_camera_type   | Camera hardware type (D435 or L515)         |
| use_half           | fp16 inference mode                         |
