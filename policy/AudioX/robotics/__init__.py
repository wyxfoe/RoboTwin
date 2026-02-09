"""AudioX robotics training and inference utilities.

Core modules (always available):
    robot_model     — config adapter + model factory
    action_space    — action normalization / denormalization

Training-only modules (require pytorch_lightning, h5py, cv2):
    robot_training  — Lightning training wrapper
    robotwin_dataset — HDF5 data loader
"""
from .robot_model import create_robot_model_from_config
from .action_space import normalize_actions, denormalize_actions
