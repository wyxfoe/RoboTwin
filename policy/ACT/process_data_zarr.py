"""
Convert raw RoboTwin HDF5 episode data into zarr format for ACT policy training.

This follows the same zarr structure used by DP (Diffusion Policy):
  zarr_root/
    data/
      cam_high:      (N, 3, H, W) uint8
      cam_left_wrist:  (N, 3, H, W) uint8
      cam_right_wrist: (N, 3, H, W) uint8
      qpos:          (N, D) float32
      action:        (N, D) float32
    meta/
      episode_ends:  (num_episodes,) int64

Usage:
    python process_data_zarr.py <task_name> <task_config> <expert_data_num>
"""

import sys
sys.path.append("./policy/ACT/")

import os
import numpy as np
import cv2
import h5py
import zarr
import shutil
import argparse
import json


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def main():
    parser = argparse.ArgumentParser(description="Convert raw data to zarr format for ACT.")
    parser.add_argument("task_name", type=str, help="Task name (e.g., beat_block_hammer)")
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int, help="Number of episodes to process")
    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    num = args.expert_data_num

    load_dir = os.path.join("../../data/", task_name, task_config, "data")

    save_dir = f"./data/{task_name}-{task_config}-{num}.zarr"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    cam_high_arrays = []
    cam_left_wrist_arrays = []
    cam_right_wrist_arrays = []
    qpos_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    total_count = 0

    for ep_idx in range(num):
        print(f"processing episode: {ep_idx + 1} / {num}", end="\r")

        load_path = os.path.join(load_dir, f"episode{ep_idx}.hdf5")
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = load_hdf5(load_path)

        for j in range(left_gripper_all.shape[0]):
            left_gripper = left_gripper_all[j]
            left_arm = left_arm_all[j]
            right_gripper = right_gripper_all[j]
            right_arm = right_arm_all[j]

            state = np.concatenate(
                (left_arm, [left_gripper], right_arm, [right_gripper]), axis=0
            ).astype(np.float32)

            if j != left_gripper_all.shape[0] - 1:
                # Decode and resize images
                cam_high_bits = image_dict["head_camera"][j]
                cam_high = cv2.imdecode(np.frombuffer(cam_high_bits, np.uint8), cv2.IMREAD_COLOR)
                cam_high = cv2.resize(cam_high, (640, 480))

                cam_left_bits = image_dict["left_camera"][j]
                cam_left = cv2.imdecode(np.frombuffer(cam_left_bits, np.uint8), cv2.IMREAD_COLOR)
                cam_left = cv2.resize(cam_left, (640, 480))

                cam_right_bits = image_dict["right_camera"][j]
                cam_right = cv2.imdecode(np.frombuffer(cam_right_bits, np.uint8), cv2.IMREAD_COLOR)
                cam_right = cv2.resize(cam_right, (640, 480))

                cam_high_arrays.append(cam_high)
                cam_left_wrist_arrays.append(cam_left)
                cam_right_wrist_arrays.append(cam_right)
                qpos_arrays.append(state)

            if j != 0:
                action_arrays.append(state)

        total_count += left_gripper_all.shape[0] - 1
        episode_ends_arrays.append(total_count)

    print()

    # Convert to numpy arrays
    cam_high_arrays = np.array(cam_high_arrays)
    cam_left_wrist_arrays = np.array(cam_left_wrist_arrays)
    cam_right_wrist_arrays = np.array(cam_right_wrist_arrays)
    qpos_arrays = np.array(qpos_arrays)
    action_arrays = np.array(action_arrays)
    episode_ends_arrays = np.array(episode_ends_arrays)

    # NHWC -> NCHW for images
    cam_high_arrays = np.moveaxis(cam_high_arrays, -1, 1)
    cam_left_wrist_arrays = np.moveaxis(cam_left_wrist_arrays, -1, 1)
    cam_right_wrist_arrays = np.moveaxis(cam_right_wrist_arrays, -1, 1)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    zarr_data.create_dataset(
        "cam_high", data=cam_high_arrays,
        chunks=(100, *cam_high_arrays.shape[1:]),
        overwrite=True, compressor=compressor,
    )
    zarr_data.create_dataset(
        "cam_left_wrist", data=cam_left_wrist_arrays,
        chunks=(100, *cam_left_wrist_arrays.shape[1:]),
        overwrite=True, compressor=compressor,
    )
    zarr_data.create_dataset(
        "cam_right_wrist", data=cam_right_wrist_arrays,
        chunks=(100, *cam_right_wrist_arrays.shape[1:]),
        overwrite=True, compressor=compressor,
    )
    zarr_data.create_dataset(
        "qpos", data=qpos_arrays,
        chunks=(100, qpos_arrays.shape[1]),
        dtype="float32", overwrite=True, compressor=compressor,
    )
    zarr_data.create_dataset(
        "action", data=action_arrays,
        chunks=(100, action_arrays.shape[1]),
        dtype="float32", overwrite=True, compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends", data=episode_ends_arrays,
        dtype="int64", overwrite=True, compressor=compressor,
    )

    print(f"Saved zarr dataset to {save_dir}")
    print(f"  cam_high:        {cam_high_arrays.shape}")
    print(f"  cam_left_wrist:  {cam_left_wrist_arrays.shape}")
    print(f"  cam_right_wrist: {cam_right_wrist_arrays.shape}")
    print(f"  qpos:            {qpos_arrays.shape}")
    print(f"  action:          {action_arrays.shape}")
    print(f"  episodes:        {len(episode_ends_arrays)}")


if __name__ == "__main__":
    main()
