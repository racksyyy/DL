import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import segmentation_models_pytorch as smp
from tqdm import tqdm
from dataset import get_fast_dataloaders
from fvcore.nn import FlopCountAnalysis  # <-- NEW: Meta's highly accurate FLOP counter

# ==========================================
# 1. METRICS & UTILITIES
# ==========================================
def clean_batch(tensor):
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(tensor, 0.0, 1.0)

def calculate_confusion_elements(preds, labels):
    preds_bool = torch.sigmoid(preds) > 0.5
    labels_bool = labels > 0.5
    tp = (preds_bool & labels_bool).sum().item()
    fp = (preds_bool & ~labels_bool).sum().item()
    fn = (~preds_bool & labels_bool).sum().item()
    tn = (~preds_bool & ~labels_bool).sum().item()
    return tp, fp, fn, tn

def count_parameters(model):
    """Accurately count parameters using native PyTorch"""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def generate_classification_report(tp, fp, fn, tn):
    # Class 1 (Target)
    prec_1 = tp / (tp + fp + 1e-8)
    rec_1 = tp / (tp + fn + 1e-8)
    f1_1 = 2 * (prec_1 * rec_1) / (prec_1 + rec_1 + 1e-8)
    
    # Class 0 (Background)
    prec_0 = tn / (tn + fn + 1e-8)
    rec_0 = tn / (tn + fp + 1e-8)
    f1_0 = 2 * (prec_0 * rec_0) / (prec_0 + rec_0 + 1e-8)
    
    report = (
        f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n"
        f"{'-'*55}\n"
        f"{'0 (Background)':<15} {prec_0:<12.4f} {rec_0:<12.4f} {f1_0:<12.4f}\n"
        f"{'1 (Target)':<15} {prec_1:<12.4f} {rec_1:<12.4f} {f1_1:<12.4f}\n"
    )
    return report

def plot_confusion_matrix(tp, fp, fn, tn, save_path):
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Background', 'Target'], 
                yticklabels=['Background', 'Target'])
    plt.title('Pixel-wise Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==========================================
# 2. CORE EVALUATION FUNCTION
# ==========================================
def evaluate_model(model_name, model, weight_path, test_loader, device):
    print(f"\n{'='*50}\nEvaluating {model_name.upper()}\n{'='*50}")
    
    save_dir = f"eval_results/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[!] Error loading {model_name} weights: {e}")
        return

    # --- 100% ACCURATE PARAMETERS & GFLOPS ---
    total_params = count_parameters(model)
    params_millions = total_params / 1e6

    # Note: Adjust (1, 2, 256, 256) if your test images are a different size!
    dummy_input = torch.randn(1, 2, 256, 256).to(device)
    
    with torch.no_grad():
        flop_analyzer = FlopCountAnalysis(model, dummy_input)
        flop_analyzer.unsupported_ops_warnings(False) # Hides annoying fvcore warnings
        macs = flop_analyzer.total()
        
    gflops = (macs * 2) / 1e9  # 1 MAC = ~2 FLOPs
    # -----------------------------------------

    t_tp, t_fp, t_fn, t_tn = 0, 0, 0, 0
    test_pbar = tqdm(test_loader, desc=f"Testing {model_name}")

    with torch.no_grad():
        for images, labels in test_pbar:
            images = clean_batch(images.to(device))
            labels = clean_batch(labels.to(device))
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                
            tp, fp, fn, tn = calculate_confusion_elements(outputs.float(), labels)
            t_tp += tp; t_fp += fp; t_fn += fn; t_tn += tn

    accuracy = (t_tp + t_tn) / (t_tp + t_tn + t_fp + t_fn + 1e-8)
    iou = t_tp / (t_tp + t_fp + t_fn + 1e-8)
    dice = 2 * t_tp / (2 * t_tp + t_fp + t_fn + 1e-8)
    
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    metrics_path = os.path.join(save_dir, 'evaluation_metrics.txt')
    
    plot_confusion_matrix(t_tp, t_fp, t_fn, t_tn, cm_path)
    class_report = generate_classification_report(t_tp, t_fp, t_fn, t_tn)
    
    metrics_text = (
        f"Model: {model_name.upper()}\n"
        f"----------------------------------------\n"
        f"Total Parameters : {params_millions:.2f} Million\n"
        f"Computational Cost: {gflops:.2f} GFLOPs\n"
        f"----------------------------------------\n"
        f"Overall Accuracy : {accuracy:.4f}\n"
        f"IoU Score        : {iou:.4f}\n"
        f"Dice Coefficient : {dice:.4f}\n\n"
        f"Classification Report:\n"
        f"{class_report}"
    )
    
    with open(metrics_path, 'w') as f:
        f.write(metrics_text)
        
    print(f"\n[+] Results for {model_name}:")
    print(f"    - Parameters: {params_millions:.2f} M | GFLOPs: {gflops:.2f}")
    print(f"    - IoU: {iou:.4f} | Dice: {dice:.4f}")
    print(f"    - Saved Results to: {save_dir}/")

# ==========================================
# 3. MAIN SCRIPT
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluations on: {device.type.upper()}")

    optimal_workers = 4 if os.name != 'nt' else 2 
    _, _, test_loader = get_fast_dataloaders(
        tensor_dir="sen1floods11_fast", 
        batch_size=16, 
        num_workers=optimal_workers 
    )

    models_to_eval = [
        {
            "name": "unet",
            "weight_path": "unet_best_weights.pth",
            "model": smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=2, classes=1)
        },
        {
            "name": "linknet",
            "weight_path": "linknet_best_weights.pth",
            "model": smp.Linknet(encoder_name="resnet18", encoder_weights=None, in_channels=2, classes=1)
        },
        {
            "name": "pspnet",
            "weight_path": "pspnet_best_weights.pth",
            "model": smp.PSPNet(encoder_name="resnet18", encoder_weights=None, in_channels=2, classes=1)
        }
    ]

    for m in models_to_eval:
        if os.path.exists(m["weight_path"]):
            evaluate_model(m["name"], m["model"], m["weight_path"], test_loader, device)
        else:
            print(f"\n[!] Skipping {m['name']} - Weights not found at {m['weight_path']}")

if __name__ == "__main__":
    main()