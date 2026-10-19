import os
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

class PaperSen1FloodsDataset(Dataset):
    def __init__(self, file_items, use_s2=False):
        self.items = file_items
        self.use_s2 = use_s2

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path = self.items[idx]["img"]
        lbl_path = self.items[idx]["lbl"]

        with rasterio.open(img_path) as src:
            img_tensor = src.read().astype(np.float32) 
            img_tensor = np.nan_to_num(img_tensor, nan=0.0, posinf=0.0, neginf=0.0)

            if self.use_s2:
                img_tensor = img_tensor / 10000.0
                img_tensor = np.clip(img_tensor, 0.0, 1.0)
            else:
                c_min = img_tensor.min(axis=(1, 2), keepdims=True)
                c_max = img_tensor.max(axis=(1, 2), keepdims=True)
                diff = c_max - c_min
                eps = 1e-8
                diff = np.where(diff < eps, eps, diff)
                
                img_tensor = (img_tensor - c_min) / diff
                img_tensor = np.clip(img_tensor, 0.0, 1.0)

        with rasterio.open(lbl_path) as src:
            lbl = src.read(1).astype(np.float32)  
            lbl = np.where(lbl == -1.0, 0.0, lbl)
            lbl = np.clip(lbl, 0.0, 1.0)
            lbl = np.expand_dims(lbl, axis=0)  

        img_tensor_torch = torch.tensor(img_tensor, dtype=torch.float32)
        lbl_tensor_torch = torch.tensor(lbl, dtype=torch.float32)
        
        img_tensor_torch = torch.nan_to_num(img_tensor_torch, nan=0.0, posinf=0.0, neginf=0.0)
        lbl_tensor_torch = torch.nan_to_num(lbl_tensor_torch, nan=0.0, posinf=0.0, neginf=0.0)

        return img_tensor_torch, lbl_tensor_torch


class FastTensorDataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        
        self.file_names = sorted(f for f in os.listdir(images_dir) if f.endswith(".pt"))
        
        valid_files = []
        for fname in self.file_names:
            img_path = os.path.join(images_dir, fname)
            lbl_path = os.path.join(labels_dir, fname)
            
            if os.path.exists(img_path) and os.path.exists(lbl_path):
                if os.path.getsize(img_path) > 1024 and os.path.getsize(lbl_path) > 1024:
                    valid_files.append(fname)
        
        self.file_names = valid_files

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        try:
            fname = self.file_names[idx]
            
            img_tensor = torch.load(os.path.join(self.images_dir, fname), weights_only=True)
            lbl_tensor = torch.load(os.path.join(self.labels_dir, fname), weights_only=True)
            
            if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
                img_tensor = torch.nan_to_num(img_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            if torch.isnan(lbl_tensor).any() or torch.isinf(lbl_tensor).any():
                lbl_tensor = torch.nan_to_num(lbl_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            
            if img_tensor.dim() != 3 or img_tensor.shape[0] != 2:
                img_tensor = img_tensor.view(2, 512, 512) if img_tensor.numel() == 2*512*512 else torch.zeros((2, 512, 512), dtype=torch.float32)
            
            if lbl_tensor.dim() != 3 or lbl_tensor.shape[0] != 1:
                lbl_tensor = lbl_tensor.view(1, 512, 512) if lbl_tensor.numel() == 512*512 else torch.zeros((1, 512, 512), dtype=torch.float32)
            
            return img_tensor, lbl_tensor
            
        except Exception:
            return torch.zeros((2, 512, 512), dtype=torch.float32), torch.zeros((1, 512, 512), dtype=torch.float32)


def build_paper_splits(base_dir="sen1floods11/v1.1", seed=42):
    random.seed(seed) 

    hand_img_dir = os.path.join(base_dir, "data", "flood_events", "HandLabeled", "S1Hand")
    hand_lbl_dir = os.path.join(base_dir, "data", "flood_events", "HandLabeled", "LabelHand")
    weak_img_dir = os.path.join(base_dir, "data", "flood_events", "WeaklyLabeled", "S1Weak")
    weak_lbl_dir = os.path.join(base_dir, "data", "flood_events", "WeaklyLabeled", "S1OtsuLabelWeak")

    hand_labeled_items = []
    if os.path.exists(hand_img_dir):
        for img_name in os.listdir(hand_img_dir):
            if img_name.endswith(".tif"):
                base_id = img_name.replace("_S1Hand.tif", "")
                hand_labeled_items.append({
                    "img": os.path.join(hand_img_dir, img_name),
                    "lbl": os.path.join(hand_lbl_dir, f"{base_id}_LabelHand.tif")
                })

    random.shuffle(hand_labeled_items)
    total_hand = len(hand_labeled_items)
    train_split_idx = int(total_hand * 0.60)
    val_split_idx = int(total_hand * 0.80) 

    train_items = hand_labeled_items[:train_split_idx]             
    val_items = hand_labeled_items[train_split_idx:val_split_idx]   
    test_items = hand_labeled_items[val_split_idx:]                 

    weak_items = []
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
                    match = next((v for k, v in weak_lbl_lookup.items() if k.startswith(base_id)), None)
                    lbl_path = match

                if lbl_path:
                    weak_items.append({
                        "img": os.path.join(weak_img_dir, img_name),
                        "lbl": lbl_path
                    })

    # The 90:10 split to smooth validation plots
    random.shuffle(weak_items)
    weak_split_idx = int(len(weak_items) * 0.90)
    
    train_items.extend(weak_items[:weak_split_idx])
    val_items.extend(weak_items[weak_split_idx:])

    print(f"\n--- Final Dataset Distribution ---")
    print(f"Train: {len(train_items)} | Val: {len(val_items)} | Test: {len(test_items)}")

    return train_items, val_items, test_items


def get_dataloaders(base_dir="sen1floods11/v1.1", batch_size=16, use_s2=False):
    train_items, val_items, test_items = build_paper_splits(base_dir)

    train_dataset = PaperSen1FloodsDataset(train_items, use_s2=use_s2)
    val_dataset = PaperSen1FloodsDataset(val_items, use_s2=use_s2)
    test_dataset = PaperSen1FloodsDataset(test_items, use_s2=use_s2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_fast_dataloaders(tensor_dir="sen1floods11_fast_v2", batch_size=16, num_workers=0):
    train_dataset = FastTensorDataset(os.path.join(tensor_dir, "train", "images"), os.path.join(tensor_dir, "train", "labels"))
    val_dataset = FastTensorDataset(os.path.join(tensor_dir, "val", "images"), os.path.join(tensor_dir, "val", "labels"))
    test_dataset = FastTensorDataset(os.path.join(tensor_dir, "test", "images"), os.path.join(tensor_dir, "test", "labels"))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True) 

    return train_loader, val_loader, test_loader