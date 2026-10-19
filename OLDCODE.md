import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np
from thop import profile
from dataset import get_fast_dataloaders
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(preds, labels):
    preds_bool = torch.sigmoid(preds) > 0.5
    labels_bool = labels > 0.5
    tp = (preds_bool & labels_bool).sum().item()
    fp = (preds_bool & ~labels_bool).sum().item()
    fn = (~preds_bool & labels_bool).sum().item()
    tn = (~preds_bool & ~labels_bool).sum().item()
    return tp, fp, fn, tn

def validate_batch(images, labels, batch_idx):
    """Check for NaN/Inf in batch and clean it up"""
    # Check images
    has_nan_img = torch.isnan(images).any()
    has_inf_img = torch.isinf(images).any()
    
    if has_nan_img or has_inf_img:
        print(f"⚠️  NaN/Inf detected in images batch {batch_idx}, cleaning...")
        images = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check labels
    has_nan_lbl = torch.isnan(labels).any()
    has_inf_lbl = torch.isinf(labels).any()
    
    if has_nan_lbl or has_inf_lbl:
        print(f"⚠️  NaN/Inf detected in labels batch {batch_idx}, cleaning...")
        labels = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check for values outside [0,1] range and clip
    images = torch.clamp(images, 0.0, 1.0)
    labels = torch.clamp(labels, 0.0, 1.0)
    
    return images, labels

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Initializing UNET Model on {device} ---")

    # Use fast dataloaders with pre-converted tensors
    train_loader, val_loader, test_loader = get_fast_dataloaders(
        tensor_dir="sen1floods11_fast", batch_size=16
    )

    # Initialize model
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=2, classes=1).to(device)

    # Profile model
    dummy_input = torch.randn(1, 2, 512, 512).to(device)
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    print(f"[*] Model Parameters: {params / 1e6:.2f} M")
    print(f"[*] Computational Complexity: {macs / 1e9 * 2:.2f} GFLOPs\n")

    # Loss with pos_weight for class imbalance
    pos_weight = torch.tensor([5.0]).to(device) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer with lower learning rate for stability
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    # Gradient clipping to prevent exploding gradients
    max_grad_norm = 1.0

    num_epochs = 50 
    patience = 8  
    best_val_loss = float('inf')
    epochs_no_improve = 0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        tr_tp, tr_fp, tr_fn, tr_tn = 0, 0, 0, 0
        valid_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Validate and clean batch
            images, labels = validate_batch(images, labels, batch_idx)
            
            optimizer.zero_grad()
            
            try:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                # Check if loss is valid
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  Invalid loss detected in batch {batch_idx}, skipping...")
                    continue
                
                # Backward pass with gradient clipping
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                
                # Accumulate metrics
                train_loss += loss.item()
                tp, fp, fn, tn = calculate_metrics(outputs, labels)
                tr_tp += tp
                tr_fp += fp
                tr_fn += fn
                tr_tn += tn
                valid_batches += 1
                
            except Exception as e:
                print(f"⚠️  Error in batch {batch_idx}: {e}, skipping...")
                continue
            
            if batch_idx % 25 == 0:
                print(f"   -> Epoch {epoch+1} Progress: Batch {batch_idx}/{len(train_loader)}...")
        
        # Calculate average loss (only for valid batches)
        if valid_batches > 0:
            avg_train_loss = train_loss / valid_batches
            train_acc = (tr_tp + tr_tn) / (tr_tp + tr_tn + tr_fp + tr_fn + 1e-8)
        else:
            print(f"⚠️  No valid batches in epoch {epoch+1}, skipping...")
            continue
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        v_tp, v_fp, v_fn, v_tn = 0, 0, 0, 0
        valid_val_batches = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Clean validation data
                images, labels = validate_batch(images, labels, -1)
                
                try:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        val_loss += loss.item()
                        tp, fp, fn, tn = calculate_metrics(outputs, labels)
                        v_tp += tp
                        v_fp += fp
                        v_fn += fn
                        v_tn += tn
                        valid_val_batches += 1
                except Exception as e:
                    print(f"⚠️  Error in validation: {e}")
                    continue
                
        if valid_val_batches > 0:
            avg_val_loss = val_loss / valid_val_batches
            val_acc = (v_tp + v_tn) / (v_tp + v_tn + v_fp + v_fn + 1e-8)
        else:
            print(f"⚠️  No valid validation batches in epoch {epoch+1}")
            avg_val_loss = float('inf')
            val_acc = 0.0
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] | T.Loss: {avg_train_loss:.4f} | V.Loss: {avg_val_loss:.4f} | T.Acc: {train_acc:.4f} | V.Acc: {val_acc:.4f}")
        
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "unet_best_weights.pth")
            print(f"[*] Best UNet model saved! (Val Loss: {best_val_loss:.4f})\n")
        else:
            epochs_no_improve += 1
            print(f"[*] No improvement for {epochs_no_improve} epoch(s).\n")
            if epochs_no_improve >= patience:
                print(f"\n[!] Early stopping triggered at epoch {epoch+1}")
                break

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('UNet Loss Convergence')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('UNet Accuracy')
    plt.legend()
    plt.savefig('unet_training_plots.png')
    print("\n✓ Training plots saved to 'unet_training_plots.png'")

    # Evaluation
    print("\n--- Running Evaluation on Test Data ---")
    try:
        model.load_state_dict(torch.load("unet_best_weights.pth"))
    except:
        print("⚠️  No best weights found, using current model state")
    
    model.eval()
    
    test_loss = 0.0
    t_tp, t_fp, t_fn, t_tn = 0, 0, 0, 0
    sample_images, sample_preds, sample_labels = [], [], []
    valid_test_batches = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Clean test data
            images, labels = validate_batch(images, labels, i)
            
            try:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    test_loss += loss.item()
                    tp, fp, fn, tn = calculate_metrics(outputs, labels)
                    t_tp += tp
                    t_fp += fp
                    t_fn += fn
                    t_tn += tn
                    valid_test_batches += 1
                    
                    if len(sample_images) < 6:
                        sample_images.append(images[0, 0].cpu().numpy())
                        sample_labels.append(labels[0, 0].cpu().numpy())
                        sample_preds.append((torch.sigmoid(outputs[0, 0]) > 0.5).cpu().numpy())
            except Exception as e:
                print(f"⚠️  Error in test batch {i}: {e}")
                continue

    if valid_test_batches > 0:
        accuracy = (t_tp + t_tn) / (t_tp + t_tn + t_fp + t_fn + 1e-8)
        precision = t_tp / (t_tp + t_fp + 1e-8)
        recall = t_tp / (t_tp + t_fn + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        iou = t_tp / (t_tp + t_fp + t_fn + 1e-8)
        dice = 2 * t_tp / (2 * t_tp + t_fp + t_fn + 1e-8)
        
        print(f"\n[UNET TEST METRICS]")
        print(f"Test Loss:    {test_loss / valid_test_batches:.4f}")
        print(f"Accuracy:     {accuracy:.4f}")
        print(f"Precision:    {precision:.4f}")
        print(f"Recall:       {recall:.4f}")
        print(f"F1 Score:     {f1_score:.4f}")
        print(f"IoU:          {iou:.4f}")
        print(f"Dice Coeff:   {dice:.4f}")
        
        print("\n[CONFUSION MATRIX]")
        print(f"True Positives:  {t_tp}  | False Positives: {t_fp}")
        print(f"False Negatives: {t_fn}  | True Negatives:  {t_tn}")
        
        # Plot samples
        if len(sample_images) > 0:
            fig, axes = plt.subplots(len(sample_images), 3, figsize=(10, 6*len(sample_images)))
            fig.suptitle('UNet Test Data - Input | Ground Truth | Prediction', fontsize=16)
            
            # Handle case where only one sample
            if len(sample_images) == 1:
                axes = [axes]
                
            for i in range(len(sample_images)):
                axes[i, 0].imshow(sample_images[i], cmap='gray')
                axes[i, 0].axis('off')
                axes[i, 1].imshow(sample_labels[i], cmap='Blues')
                axes[i, 1].axis('off')
                axes[i, 2].imshow(sample_preds[i], cmap='Reds')
                axes[i, 2].axis('off')
            plt.tight_layout()
            plt.savefig('unet_test_samples.png')
            print("\n✓ Test samples saved to 'unet_test_samples.png'")
    else:
        print("⚠️  No valid test batches found!")

if __name__ == "__main__":
    print("="*50)
    print("UNET TRAINING WITH NAN/INF SAFETY")
    print("="*50)
    train()