import os
import torch
import rasterio
import numpy as np
from tqdm import tqdm
from dataset import build_paper_splits


def process_and_save(items, split_name, output_base):
    img_out_dir = os.path.join(output_base, split_name, "images")
    lbl_out_dir = os.path.join(output_base, split_name, "labels")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    print(f"Converting {split_name} split ({len(items)} files)...")

    for i, item in enumerate(tqdm(items)):
        img_path = item["img"]
        lbl_path = item["lbl"]

        # 1. Process Image with NaN/Inf handling
        with rasterio.open(img_path) as src:
            img_tensor = src.read().astype(np.float32)  # (C, H, W)
            
            # CRITICAL FIX: Handle NaN and Inf values in Sentinel-1 data
            # Replace NaN, Inf, -Inf with 0 (or a safe value)
            img_tensor = np.nan_to_num(img_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Per-channel min-max normalization
            c_min = img_tensor.min(axis=(1, 2), keepdims=True)  # (C, 1, 1)
            c_max = img_tensor.max(axis=(1, 2), keepdims=True)  # (C, 1, 1)
            diff = c_max - c_min
            
            # Handle case where all values in a channel are identical
            # Use a small epsilon to avoid division by zero
            eps = 1e-8
            diff = np.where(diff < eps, eps, diff)
            
            # Normalize safely
            img_tensor = (img_tensor - c_min) / diff
            
            # Additional safety: clamp to [0, 1] range to handle any remaining numerical issues
            img_tensor = np.clip(img_tensor, 0.0, 1.0)

        # 2. Process Label
        with rasterio.open(lbl_path) as src:
            lbl = src.read(1).astype(np.float32)
            # Map -1 (clouds/shadows) to 0
            lbl = np.where(lbl == -1.0, 0.0, lbl)
            # Clamp labels to [0, 1] range
            lbl = np.clip(lbl, 0.0, 1.0)
            lbl = np.expand_dims(lbl, axis=0)

        # 3. Save as Native PyTorch Tensors with validation
        img_tensor_torch = torch.tensor(img_tensor, dtype=torch.float32)
        lbl_tensor_torch = torch.tensor(lbl, dtype=torch.float32)
        
        # Safety check: verify no NaN/Inf remain
        if torch.isnan(img_tensor_torch).any() or torch.isinf(img_tensor_torch).any():
            print(f"WARNING: NaN/Inf found in image {img_path}, replacing with zeros")
            img_tensor_torch = torch.nan_to_num(img_tensor_torch, nan=0.0, posinf=0.0, neginf=0.0)
        
        if torch.isnan(lbl_tensor_torch).any() or torch.isinf(lbl_tensor_torch).any():
            print(f"WARNING: NaN/Inf found in label {lbl_path}, replacing with zeros")
            lbl_tensor_torch = torch.nan_to_num(lbl_tensor_torch, nan=0.0, posinf=0.0, neginf=0.0)
        
        base_name = f"{split_name}_{i:04d}"
        torch.save(img_tensor_torch, os.path.join(img_out_dir, f"{base_name}.pt"))
        torch.save(lbl_tensor_torch, os.path.join(lbl_out_dir, f"{base_name}.pt"))


if __name__ == "__main__":
    print("--- Starting Dataset Conversion to Native Tensors ---")
    train_items, val_items, test_items = build_paper_splits("sen1floods11/v1.1")
    output_dir = "sen1floods11_fast"

    process_and_save(train_items, "train", output_dir)
    process_and_save(val_items, "val", output_dir)
    process_and_save(test_items, "test", output_dir)

    print("\n[SUCCESS] Conversion complete! Your dataset is now optimized for the RTX 5070.")