import os
import torch
from tqdm import tqdm
from dataset_v2 import get_dataloaders
import warnings

warnings.filterwarnings('ignore')

def convert_and_save(loader, split_name, output_base="sen1floods11_fast_v2"):
    img_dir = os.path.join(output_base, split_name, "images")
    lbl_dir = os.path.join(output_base, split_name, "labels")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    
    print(f"\n[*] Converting {split_name.upper()} set...")
    
    pbar = tqdm(loader, desc=f"Saving {split_name}")
    global_idx = 0
    
    for images, labels in pbar:
        for i in range(images.size(0)):
            img = images[i]
            lbl = labels[i]
            
            torch.save(img, os.path.join(img_dir, f"{split_name}_{global_idx:05d}.pt"))
            torch.save(lbl, os.path.join(lbl_dir, f"{split_name}_{global_idx:05d}.pt"))
            
            global_idx += 1

def main():
    print("==========================================")
    print(" SEN1FLOODS11 TENSOR CONVERSION SCRIPT V2 ")
    print("==========================================")
    
    output_dir = "sen1floods11_fast_v2"
    if os.path.exists(output_dir):
        print(f"[!] Warning: '{output_dir}' already exists. Overwriting files...")
    
    train_loader, val_loader, test_loader = get_dataloaders(
        base_dir="sen1floods11/v1.1", 
        batch_size=16, 
        use_s2=False
    )
    
    convert_and_save(train_loader, "train", output_dir)
    convert_and_save(val_loader, "val", output_dir)
    convert_and_save(test_loader, "test", output_dir)
    
    print("\n==========================================")
    print(" SUCCESS: Fast Tensors V2 Generated!      ")
    print("==========================================")

if __name__ == "__main__":
    main()