import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from tqdm import tqdm
from dataset import get_fast_dataloaders  # Original dataset import

warnings.filterwarnings('ignore')

class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, pos_weight_val=5.0):
        super(DiceBCELoss, self).__init__()
        self.bce_weight = bce_weight
        pos_weight = torch.tensor([pos_weight_val]).cuda()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return (self.bce_weight * bce_loss) + ((1 - self.bce_weight) * dice_loss)

def calculate_metrics(preds, labels):
    preds_bool = torch.sigmoid(preds) > 0.5
    labels_bool = labels > 0.5
    tp = (preds_bool & labels_bool).sum().item()
    fp = (preds_bool & ~labels_bool).sum().item()
    fn = (~preds_bool & labels_bool).sum().item()
    tn = (~preds_bool & ~labels_bool).sum().item()
    return tp, fp, fn, tn

def clean_batch(tensor):
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(tensor, 0.0, 1.0)

def gpu_augment(images, labels):
    if torch.rand(1).item() > 0.5:
        images = torch.flip(images, dims=[3]) 
        labels = torch.flip(labels, dims=[3])
    if torch.rand(1).item() > 0.5:
        images = torch.flip(images, dims=[2]) 
        labels = torch.flip(labels, dims=[2])  # Flips both image and mask vertically
    return images, labels

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- FAST LINKNET TRAINING INITIALIZED ON {device.type.upper()} ---")

    optimal_workers = 4 if os.name != 'nt' else 2 
    train_loader, val_loader, test_loader = get_fast_dataloaders(
        tensor_dir="sen1floods11_fast",  # Original tensor directory
        batch_size=16, 
        num_workers=optimal_workers 
    )

    # Linknet Model
    model = smp.Linknet(
        encoder_name="resnet18", 
        encoder_weights="imagenet", 
        in_channels=2, 
        classes=1
    ).to(device)

    criterion = DiceBCELoss(bce_weight=0.5, pos_weight_val=5.0)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    scaler = torch.amp.GradScaler('cuda')
    max_grad_norm = 1.0

    num_epochs = 50 
    patience = 5  
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        tr_tp, tr_fp, tr_fn, tr_tn = 0, 0, 0, 0
        valid_batches = 0
        current_lr = optimizer.param_groups[0]['lr']
        
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{num_epochs}] Train", leave=False)
        
        for images, labels in train_pbar:
            images = clean_batch(images.to(device, non_blocking=True))
            labels = clean_batch(labels.to(device, non_blocking=True))
            
            images, labels = gpu_augment(images, labels)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            if torch.isnan(loss) or torch.isinf(loss): continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            tp, fp, fn, tn = calculate_metrics(outputs, labels)
            tr_tp += tp; tr_fp += fp; tr_fn += fn; tr_tn += tn
            valid_batches += 1
            
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        if valid_batches == 0: continue
        avg_train_loss = train_loss / valid_batches
        train_dice = (2 * tr_tp) / (2 * tr_tp + tr_fp + tr_fn + 1e-8)
        
        model.eval()
        val_loss = 0.0
        v_tp, v_fp, v_fn, v_tn = 0, 0, 0, 0
        valid_val_batches = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1:02d}/{num_epochs}] Val  ", leave=False)
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images = clean_batch(images.to(device, non_blocking=True))
                labels = clean_batch(labels.to(device, non_blocking=True))
                
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    tp, fp, fn, tn = calculate_metrics(outputs, labels)
                    v_tp += tp; v_fp += fp; v_fn += fn; v_tn += tn
                    valid_val_batches += 1
                    val_pbar.set_postfix(loss=f"{loss.item():.4f}")
                
        if valid_val_batches > 0:
            avg_val_loss = val_loss / valid_val_batches
            val_dice = (2 * v_tp) / (2 * v_tp + v_fp + v_fn + 1e-8)
        else:
            avg_val_loss = float('inf')
            val_dice = 0.0
        
        scheduler.step(avg_val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        
        print(f"Epoch [{epoch+1:02d}/{num_epochs}] | LR: {current_lr:.1e} | T.Loss: {avg_train_loss:.4f} | V.Loss: {avg_val_loss:.4f} | T.Dice: {train_dice:.4f} | V.Dice: {val_dice:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "linknet_best_weights.pth")
            print(f"  [*] Best model saved! (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[!] Early stopping triggered! No improvement for {patience} epochs.")
                break

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', color='orange', linewidth=2)
    plt.title('Loss Convergence'); plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_dice'], label='Train Dice', color='blue', linewidth=2)
    plt.plot(history['val_dice'], label='Val Dice', color='orange', linewidth=2)
    plt.title('Dice Score'); plt.xlabel('Epochs'); plt.ylabel('Score')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('linknet_training_plots.png', dpi=300)
    print("\n[+] Training plots saved to 'linknet_training_plots.png'")
    
    print("\n--- Running Final Evaluation on Test Set ---")
    try: 
        model.load_state_dict(torch.load("linknet_best_weights.pth", weights_only=True))
    except Exception as e: 
        print(f"Could not load best weights, evaluating current state: {e}")
    
    model.eval()
    t_tp, t_fp, t_fn, t_tn = 0, 0, 0, 0
    sample_images, sample_preds, sample_labels = [], [], []

    test_pbar = tqdm(test_loader, desc="Evaluating Test Set")

    with torch.no_grad():
        for images, labels in test_pbar:
            images = clean_batch(images.to(device))
            labels = clean_batch(labels.to(device))
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                tp, fp, fn, tn = calculate_metrics(outputs, labels)
                t_tp += tp; t_fp += fp; t_fn += fn; t_tn += tn
                
                if len(sample_images) < 6:
                    sample_images.append(images[0, 0].cpu().numpy()) 
                    sample_labels.append(labels[0, 0].cpu().numpy())
                    sample_preds.append((torch.sigmoid(outputs[0, 0]) > 0.5).cpu().numpy())

    accuracy = (t_tp + t_tn) / (t_tp + t_tn + t_fp + t_fn + 1e-8)
    iou = t_tp / (t_tp + t_fp + t_fn + 1e-8)
    dice = 2 * t_tp / (2 * t_tp + t_fp + t_fn + 1e-8)
    
    print(f"\n==========================================")
    print(f"          FINAL TEST METRICS              ")
    print(f"==========================================")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"IoU:        {iou:.4f}")
    print(f"Dice Coeff: {dice:.4f}")
    print(f"==========================================\n")
    
    if len(sample_images) > 0:
        fig, axes = plt.subplots(len(sample_images), 3, figsize=(12, 4*len(sample_images)))
        if len(sample_images) == 1: axes = [axes]
        
        axes[0, 0].set_title("Input Image (Ch 1)", fontsize=14, fontweight='bold')
        axes[0, 1].set_title("Ground Truth", fontsize=14, fontweight='bold')
        axes[0, 2].set_title("Model Prediction", fontsize=14, fontweight='bold')
        
        for i in range(len(sample_images)):
            axes[i, 0].imshow(sample_images[i], cmap='gray'); axes[i, 0].axis('off')
            axes[i, 1].imshow(sample_labels[i], cmap='Blues'); axes[i, 1].axis('off')
            axes[i, 2].imshow(sample_preds[i], cmap='Reds'); axes[i, 2].axis('off')
            
        plt.tight_layout()
        plt.savefig('linknet_test_samples.png', dpi=300)
        print("[+] Test set visualizations saved to 'linknet_test_samples.png'")

if __name__ == "__main__":
    train()