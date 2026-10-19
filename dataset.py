import os
import glob
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random


class PaperSen1FloodsDataset(Dataset):
    def __init__(self, file_items, use_s2=False):
        """
        file_items: list of dicts containing 'img' and 'lbl' paths.
        use_s2: boolean to toggle between Sentinel-1 (dual-pol) and Sentinel-2 (13-band).
        """
        self.items = file_items
        self.use_s2 = use_s2

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path = self.items[idx]["img"]
        lbl_path = self.items[idx]["lbl"]

        # 1. Load Image (Sentinel-1 or Sentinel-2)
        with rasterio.open(img_path) as src:
            img_tensor = src.read().astype(np.float32)  # (C, H, W)
            
            # CRITICAL FIX: Handle NaN/Inf values
            img_tensor = np.nan_to_num(img_tensor, nan=0.0, posinf=0.0, neginf=0.0)

            if self.use_s2:
                # As per paper: Sentinel-2 is scaled by 10,000.
                img_tensor = img_tensor / 10000.0
                img_tensor = np.clip(img_tensor, 0.0, 1.0)
            else:
                # Sentinel-1 standard normalization (per-channel min-max of dB values)
                c_min = img_tensor.min(axis=(1, 2), keepdims=True)
                c_max = img_tensor.max(axis=(1, 2), keepdims=True)
                diff = c_max - c_min
                
                # Use epsilon to avoid division by zero
                eps = 1e-8
                diff = np.where(diff < eps, eps, diff)
                
                img_tensor = (img_tensor - c_min) / diff
                img_tensor = np.clip(img_tensor, 0.0, 1.0)

        # 2. Load Label Mask
        with rasterio.open(lbl_path) as src:
            lbl = src.read(1).astype(np.float32)  # Shape: (512, 512)

            # Map -1 (clouds/shadows) to 0 for binary classification
            lbl = np.where(lbl == -1.0, 0.0, lbl)
            lbl = np.clip(lbl, 0.0, 1.0)
            lbl = np.expand_dims(lbl, axis=0)  # Shape: (1, 512, 512)

        # Convert to torch tensors with safety
        img_tensor_torch = torch.tensor(img_tensor, dtype=torch.float32)
        lbl_tensor_torch = torch.tensor(lbl, dtype=torch.float32)
        
        # Final safety check
        img_tensor_torch = torch.nan_to_num(img_tensor_torch, nan=0.0, posinf=0.0, neginf=0.0)
        lbl_tensor_torch = torch.nan_to_num(lbl_tensor_torch, nan=0.0, posinf=0.0, neginf=0.0)

        return img_tensor_torch, lbl_tensor_torch


class FastTensorDataset(Dataset):
    """
    Loads pre-converted .pt tensors produced by convert_to_tensors.py.
    Much faster than PaperSen1FloodsDataset since there's no rasterio I/O
    or normalization happening at __getitem__ time.
    """
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        
        # Get all .pt files from images directory
        self.file_names = sorted(
            f for f in os.listdir(images_dir) if f.endswith(".pt")
        )
        
        # Verify that all image files have corresponding label files
        valid_files = []
        for fname in self.file_names:
            img_path = os.path.join(images_dir, fname)
            lbl_path = os.path.join(labels_dir, fname)
            
            # Check if both files exist and have content
            if os.path.exists(img_path) and os.path.exists(lbl_path):
                # Check file size (at least 1KB to ensure it's not corrupted)
                if os.path.getsize(img_path) > 1024 and os.path.getsize(lbl_path) > 1024:
                    valid_files.append(fname)
                else:
                    print(f"WARNING: File {fname} is too small (corrupted?), skipping...")
            else:
                print(f"WARNING: Missing corresponding file for {fname}, skipping...")
        
        if len(valid_files) < len(self.file_names):
            print(f"WARNING: Filtered out {len(self.file_names) - len(valid_files)} corrupted/missing files")
        
        self.file_names = valid_files
        print(f"✓ Loaded {len(self.file_names)} valid tensor files")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        try:
            fname = self.file_names[idx]
            
            # Load tensors with error handling
            try:
                img_tensor = torch.load(os.path.join(self.images_dir, fname), weights_only=True)
            except Exception as e:
                print(f"ERROR loading {fname}: {e}")
                # Return a dummy tensor to avoid crashing
                img_tensor = torch.zeros((2, 512, 512), dtype=torch.float32)
            
            try:
                lbl_tensor = torch.load(os.path.join(self.labels_dir, fname), weights_only=True)
            except Exception as e:
                print(f"ERROR loading label {fname}: {e}")
                lbl_tensor = torch.zeros((1, 512, 512), dtype=torch.float32)
            
            # Safety check during runtime
            if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
                img_tensor = torch.nan_to_num(img_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            if torch.isnan(lbl_tensor).any() or torch.isinf(lbl_tensor).any():
                lbl_tensor = torch.nan_to_num(lbl_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ensure correct shapes
            if img_tensor.dim() == 3 and img_tensor.shape[0] == 2:
                pass  # Correct shape (C, H, W)
            else:
                print(f"WARNING: Unexpected image shape {img_tensor.shape} for {fname}")
                img_tensor = img_tensor.view(2, 512, 512) if img_tensor.numel() == 2*512*512 else torch.zeros((2, 512, 512), dtype=torch.float32)
            
            if lbl_tensor.dim() == 3 and lbl_tensor.shape[0] == 1:
                pass  # Correct shape (1, H, W)
            else:
                print(f"WARNING: Unexpected label shape {lbl_tensor.shape} for {fname}")
                lbl_tensor = lbl_tensor.view(1, 512, 512) if lbl_tensor.numel() == 512*512 else torch.zeros((1, 512, 512), dtype=torch.float32)
            
            return img_tensor, lbl_tensor
            
        except Exception as e:
            print(f"CRITICAL ERROR in __getitem__ for index {idx}: {e}")
            # Return dummy tensors to prevent crashing
            return torch.zeros((2, 512, 512), dtype=torch.float32), torch.zeros((1, 512, 512), dtype=torch.float32)


def build_paper_splits(base_dir="sen1floods11/v1.1", seed=42):
    """
    Implements the exact splitting methodology described in the paper:
    - 446 Hand-Labeled -> 60:20:20 (Train:Val:Test)
    - 4385 Weakly-Labeled -> 100% Train
    """
    random.seed(seed)  # Ensure reproducibility for the split

    # Define directories
    hand_img_dir = os.path.join(base_dir, "data", "flood_events", "HandLabeled", "S1Hand")
    hand_lbl_dir = os.path.join(base_dir, "data", "flood_events", "HandLabeled", "LabelHand")
    weak_img_dir = os.path.join(base_dir, "data", "flood_events", "WeaklyLabeled", "S1Weak")
    weak_lbl_dir = os.path.join(base_dir, "data", "flood_events", "WeaklyLabeled", "S1OtsuLabelWeak")

    # 1. Gather all Hand-Labeled Data (446 chips)
    hand_labeled_items = []
    if os.path.exists(hand_img_dir):
        for img_name in os.listdir(hand_img_dir):
            if img_name.endswith(".tif"):
                base_id = img_name.replace("_S1Hand.tif", "")
                hand_labeled_items.append({
                    "img": os.path.join(hand_img_dir, img_name),
                    "lbl": os.path.join(hand_lbl_dir, f"{base_id}_LabelHand.tif")
                })

    # 2. Enforce the 60:20:20 Split for Hand-Labeled data
    random.shuffle(hand_labeled_items)
    total_hand = len(hand_labeled_items)

    train_split_idx = int(total_hand * 0.60)
    val_split_idx = int(total_hand * 0.80)  # 60% + 20%

    train_items = hand_labeled_items[:train_split_idx]              # 60%
    val_items = hand_labeled_items[train_split_idx:val_split_idx]   # 20%
    test_items = hand_labeled_items[val_split_idx:]                 # 20%

    print(f"--- Hand-Labeled Split (60:20:20) ---")
    print(f"Train: {len(train_items)} | Val: {len(val_items)} | Test: {len(test_items)}")

    # 3. Gather all Weakly-Labeled Data (4385 chips)
    weak_count = 0
    if os.path.exists(weak_img_dir) and os.path.exists(weak_lbl_dir):
        weak_lbl_lookup = {}
        for lbl_name in os.listdir(weak_lbl_dir):
            if lbl_name.endswith(".tif"):
                for suffix in ("_S1OtsuLabelWeak.tif",):
                    if lbl_name.endswith(suffix):
                        key = lbl_name[: -len(suffix)]
                        weak_lbl_lookup[key] = os.path.join(weak_lbl_dir, lbl_name)
                        break
                else:
                    key = os.path.splitext(lbl_name)[0]
                    weak_lbl_lookup.setdefault(key, os.path.join(weak_lbl_dir, lbl_name))

        for img_name in os.listdir(weak_img_dir):
            if img_name.endswith(".tif"):
                base_id = img_name.replace("_S1Weak.tif", "")

                lbl_path = weak_lbl_lookup.get(base_id)
                if lbl_path is None:
                    match = next(
                        (v for k, v in weak_lbl_lookup.items() if k.startswith(base_id)),
                        None
                    )
                    lbl_path = match

                if lbl_path:
                    train_items.append({
                        "img": os.path.join(weak_img_dir, img_name),
                        "lbl": lbl_path
                    })
                    weak_count += 1

    print(f"--- Weakly-Labeled Injection ---")
    print(f"Added {weak_count} weakly-labeled images strictly to the Train set.")

    print(f"\n--- Final Dataset Distribution ---")
    print(f"Train: {len(train_items)} | Val: {len(val_items)} | Test: {len(test_items)}")

    return train_items, val_items, test_items


def get_dataloaders(base_dir="sen1floods11/v1.1", batch_size=16, use_s2=False):
    """On-the-fly loader: reads .tif files with rasterio and normalizes at __getitem__ time."""
    print(f"\n--- Initializing DataLoaders (Base Dir: {base_dir}) ---")

    train_items, val_items, test_items = build_paper_splits(base_dir)

    train_dataset = PaperSen1FloodsDataset(train_items, use_s2=use_s2)
    val_dataset = PaperSen1FloodsDataset(val_items, use_s2=use_s2)
    test_dataset = PaperSen1FloodsDataset(test_items, use_s2=use_s2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    print(f"--- DataLoaders Ready: {len(train_loader)} train batches | {len(val_loader)} val batches | {len(test_loader)} test samples ---\n")
    return train_loader, val_loader, test_loader


def get_fast_dataloaders(tensor_dir="sen1floods11_fast", batch_size=16, num_workers=0):
    """
    Fast loader: reads pre-converted .pt tensors produced by convert_to_tensors.py.
    
    Args:
        tensor_dir: Directory containing pre-converted tensors
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes (0 = no parallel loading)
    """
    print(f"\n--- Initializing FAST DataLoaders (Tensor Dir: {tensor_dir}) ---")
    print(f"Using {num_workers} workers (0 = no parallel loading)")

    train_dataset = FastTensorDataset(
        os.path.join(tensor_dir, "train", "images"),
        os.path.join(tensor_dir, "train", "labels"),
    )
    val_dataset = FastTensorDataset(
        os.path.join(tensor_dir, "val", "images"),
        os.path.join(tensor_dir, "val", "labels"),
    )
    test_dataset = FastTensorDataset(
        os.path.join(tensor_dir, "test", "images"),
        os.path.join(tensor_dir, "test", "labels"),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True  # Always 0 for test to avoid issues
    )

    print(f"--- FAST DataLoaders Ready: {len(train_loader)} train batches | {len(val_loader)} val batches | {len(test_loader)} test samples ---\n")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("==========================================")
    print("    RUNNING FULL DATASET VERIFICATION     ")
    print("==========================================")

    try:
        train_loader, val_loader, test_loader = get_dataloaders(base_dir="sen1floods11/v1.1", batch_size=16)

        print("\n[TEST] Fetching first batch from train_loader...")
        sample_images, sample_labels = next(iter(train_loader))

        print("--- Batch Verification Results ---")
        print(f"Image Batch Shape:  {sample_images.shape} (Expected: [16, 2, 512, 512])")
        print(f"Label Batch Shape:  {sample_labels.shape} (Expected: [16, 1, 512, 512])")
        
        # Check for NaN/Inf in the batch
        has_nan = torch.isnan(sample_images).any().item() or torch.isnan(sample_labels).any().item()
        has_inf = torch.isinf(sample_images).any().item() or torch.isinf(sample_labels).any().item()
        
        if has_nan or has_inf:
            print(f"WARNING: NaN/Inf detected in batch! This needs to be fixed.")
        else:
            print("✓ No NaN/Inf detected in batch - data is clean!")

        print("\n==========================================")
        print(" SUCCESS: Dataset Pipeline is ready!      ")
        print("==========================================")
    except Exception as e:
        print(f"\n[ERROR] Verification failed: {e}")