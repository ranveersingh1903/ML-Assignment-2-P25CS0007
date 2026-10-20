"""
Dataset loading and partitioning module for Olivetti Faces.
Roll Number: P25CS0007
Student Name: Ranvir Kumar Singh
Seed: 250007 (derived from all numeric characters of P25CS0007 in order)
"""

import os
import json
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

SEED = 250007

def get_roll_seed(roll_number: str = "P25CS0007") -> int:
    digits = "".join([c for c in roll_number if c.isdigit()])
    return int(digits)

def prepare_dataset(data_dir: str = "data", seed: int = SEED):
    """
    Loads fetch_olivetti_faces and creates stratified split per identity:
    6 train, 2 validation, 2 test images per identity.
    Also creates masked test set with centered 16x16 zero block (rows 24..39, cols 24..39).
    Saves split to split.npz.
    """
    os.makedirs(data_dir, exist_ok=True)
    split_path = os.path.join(data_dir, "split.npz")
    
    faces = fetch_olivetti_faces(shuffle=False)
    images = faces.images  # shape (400, 64, 64)
    data = faces.data      # shape (400, 4096)
    target = faces.target  # shape (400,)
    
    rng = np.random.default_rng(seed)
    
    train_idx = []
    val_idx = []
    test_idx = []
    
    n_identities = 40
    n_per_id = 10
    
    for c in range(n_identities):
        id_indices = np.where(target == c)[0]
        assert len(id_indices) == n_per_id, f"Identity {c} has {len(id_indices)} images, expected {n_per_id}"
        shuffled = rng.permutation(id_indices)
        train_idx.extend(shuffled[:6])
        val_idx.extend(shuffled[6:8])
        test_idx.extend(shuffled[8:10])
        
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    
    X_train = data[train_idx]
    y_train = target[train_idx]
    
    X_val = data[val_idx]
    y_val = target[val_idx]
    
    X_test_clean = data[test_idx]
    y_test = target[test_idx]
    
    # Create masked test set: centred 16x16 block, rows 24..39, cols 24..39 to zero
    test_images_masked = images[test_idx].copy()
    # rows 24 to 39 inclusive is slice 24:40
    test_images_masked[:, 24:40, 24:40] = 0.0
    X_test_masked = test_images_masked.reshape(len(test_idx), -1)
    
    np.savez_compressed(
        split_path,
        seed=seed,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test_clean=X_test_clean,
        X_test_masked=X_test_masked,
        y_test=y_test
    )
    
    metadata = {
        "student_name": "Ranvir Kumar Singh",
        "roll_number": "P25CS0007",
        "seed": seed,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "feature_dim": X_train.shape[1],
        "n_classes": n_identities,
        "mask_region": {"rows": [24, 39], "cols": [24, 39], "size": [16, 16]}
    }
    with open(os.path.join(data_dir, "dataset_summary.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Dataset prepared and saved to {split_path}")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test Clean: {X_test_clean.shape}, Test Masked: {X_test_masked.shape}")
    return split_path

def load_dataset(data_dir: str = "data"):
    split_path = os.path.join(data_dir, "split.npz")
    if not os.path.exists(split_path):
        prepare_dataset(data_dir=data_dir)
    return np.load(split_path)

if __name__ == "__main__":
    prepare_dataset()
