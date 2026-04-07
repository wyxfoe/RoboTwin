"""
Convert raw RoboTwin HDF5 episode data into LeRobotDataset format for ACT policy training.

This follows the same conversion pattern used by pi0's convert_aloha_data_to_lerobot_robotwin.py,
creating a LeRobotDataset with:
  - observation.state:              (D,) float32 - joint positions
  - action:                         (D,) float32 - joint actions
  - observation.images.cam_high:    (3, 480, 640) image
  - observation.images.cam_left_wrist:  (3, 480, 640) image
  - observation.images.cam_right_wrist: (3, 480, 640) image

Usage:
    python process_data_lerobot.py <task_name> <task_config> <expert_data_num> [--repo_id <repo_id>]
"""

import sys
sys.path.append("./policy/ACT/")

import os
import numpy as np
import cv2
import h5py
import shutil
import argparse
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
import torch
import tqdm


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


def create_empty_dataset(repo_id, state_dim=14):
    """Create an empty LeRobotDataset with the ACT schema."""
    motors = [
        "left_waist", "left_shoulder", "left_elbow",
        "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate", "left_gripper",
        "right_waist", "right_shoulder", "right_elbow",
        "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate", "right_gripper",
    ]
    # Adjust motor list if state_dim differs
    motors = motors[:state_dim]

    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
    }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    # Remove existing dataset if present
    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type="aloha",
        features=features,
        use_videos=False,
        tolerance_s=0.0001,
        image_writer_processes=4,
        image_writer_threads=4,
    )


def main():
    parser = argparse.ArgumentParser(description="Convert raw data to LeRobot format for ACT.")
    parser.add_argument("task_name", type=str, help="Task name (e.g., beat_block_hammer)")
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int, help="Number of episodes to process")
    parser.add_argument("--repo_id", type=str, default=None,
                        help="LeRobot repo_id (default: act_<task_name>_<config>_<num>)")
    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    num = args.expert_data_num

    repo_id = args.repo_id
    if repo_id is None:
        repo_id = f"act_{task_name}_{task_config}_{num}"

    load_dir = os.path.join("../../data/", task_name, task_config, "data")

    print(f"Creating LeRobot dataset: {repo_id}")
    dataset = create_empty_dataset(repo_id)

    for ep_idx in tqdm.tqdm(range(num), desc="Converting episodes"):
        load_path = os.path.join(load_dir, f"episode{ep_idx}.hdf5")
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = load_hdf5(load_path)

        states = []
        actions = []
        cam_high_imgs = []
        cam_left_imgs = []
        cam_right_imgs = []

        for j in range(left_gripper_all.shape[0]):
            left_gripper = left_gripper_all[j]
            left_arm = left_arm_all[j]
            right_gripper = right_gripper_all[j]
            right_arm = right_arm_all[j]

            state = np.concatenate(
                (left_arm, [left_gripper], right_arm, [right_gripper]), axis=0
            ).astype(np.float32)

            if j != left_gripper_all.shape[0] - 1:
                states.append(state)

                # Decode and resize images
                cam_high = cv2.imdecode(
                    np.frombuffer(image_dict["head_camera"][j], np.uint8), cv2.IMREAD_COLOR
                )
                cam_high = cv2.resize(cam_high, (640, 480))
                cam_high_imgs.append(cam_high)

                cam_left = cv2.imdecode(
                    np.frombuffer(image_dict["left_camera"][j], np.uint8), cv2.IMREAD_COLOR
                )
                cam_left = cv2.resize(cam_left, (640, 480))
                cam_left_imgs.append(cam_left)

                cam_right = cv2.imdecode(
                    np.frombuffer(image_dict["right_camera"][j], np.uint8), cv2.IMREAD_COLOR
                )
                cam_right = cv2.resize(cam_right, (640, 480))
                cam_right_imgs.append(cam_right)

            if j != 0:
                actions.append(state)

        # Add frames to the dataset
        num_frames = len(states)
        for i in range(num_frames):
            frame = {
                "observation.state": torch.from_numpy(states[i]),
                "action": torch.from_numpy(actions[i]),
                "observation.images.cam_high": cam_high_imgs[i],
                "observation.images.cam_left_wrist": cam_left_imgs[i],
                "observation.images.cam_right_wrist": cam_right_imgs[i],
            }
            dataset.add_frame(frame)
        dataset.save_episode()

    print(f"\nLeRobot dataset saved to: {HF_LEROBOT_HOME / repo_id}")
    print(f"  Episodes: {num}")
    print(f"  Repo ID: {repo_id}")


if __name__ == "__main__":
    main()
