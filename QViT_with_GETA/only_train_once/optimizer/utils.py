"""Optimizer utilities"""

import glob
import os

import torch


def save_checkpoint(filepath, obj):
    """Save a checkpoint to a file."""
    print(f"Saving checkpoint to {filepath}")
    torch.save(obj, filepath)
    print("Complete.")


def load_checkpoint(filepath, device):
    """Load a checkpoint from a file."""
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    """Scan the checkpoint directory for the latest checkpoint file."""
    pattern = os.path.join(cp_dir, prefix + "*")
    cp_list = glob.glob(pattern)

    if len(cp_list) > 0:
        last_checkpoint_path = sorted(
            cp_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )[-1]
        print(f"[INFO] Resuming from checkpoint: '{last_checkpoint_path}'")
        return last_checkpoint_path
    return None
