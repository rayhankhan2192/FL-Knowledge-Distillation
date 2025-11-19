
import os
import logging
import math
import json  # NEW: Import the json library
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import cv2

logger = logging.getLogger("KDTrainEval")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_last_conv_layer(model: torch.nn.Module):
    """Return the last nn.Conv2d layer in the model, or None if not found."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        logger.warning("No Conv2d layer found for Grad-CAM++; XAI will be skipped.")
    return last_conv


def compute_gradcam_pp(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_layer: torch.nn.Module,
    class_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Compute Grad-CAM++ heatmap for a single image batch x (shape [1, C, H, W]).
    Returns a numpy array of shape [H, W] normalized to [0,1].
    """
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        
        logits = model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        
        score = logits[:, class_idx]
        score.backward(retain_graph=False)

        if not activations or not gradients:
            raise RuntimeError("Hooks not triggered - check target_layer")
        
        A = activations[0][0]  # [C, H, W]
        G = gradients[0][0]    # [C, H, W]

        grads2 = G ** 2
        grads3 = G ** 3
        sum_activations = torch.sum(A, dim=(1, 2), keepdim=True)

        eps = 1e-7
        alpha = grads2 / (2 * grads2 + sum_activations * grads3 + eps)
        positive_gradients = F.relu(G)
        weights = torch.sum(alpha * positive_gradients, dim=(1, 2))

        cam = torch.sum(weights.view(-1, 1, 1) * A, dim=0)
        cam = F.relu(cam)

        cam_np = cam.detach().cpu().numpy()
        cam_np = cam_np - cam_np.min()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()
        
        return cam_np
        
    finally:
        handle_f.remove()
        handle_b.remove()
        model.zero_grad(set_to_none=True)


def compute_deletion_auc(
    model: torch.nn.Module,
    x: torch.Tensor,
    cam: np.ndarray,
    class_idx: int,
    device: torch.device,
    steps: int = 10,
) -> float:
    """
    Deletion AUC faithfulness metric.
    Lower values = better faithfulness (rapid confidence drop).
    Expected range: 0.1-0.3 (good), 0.5-0.7 (moderate), >0.7 (poor).
    """
    model.eval()
    
    with torch.no_grad():
        x_mod = x.clone().to(device)
        B, C, H, W = x_mod.shape
        
        if cam.shape != (H, W):
            cam = cv2.resize(cam, (W, H))
        
        cam_flat = cam.reshape(-1)
        indices = np.argsort(-cam_flat)
        
        num_pixels = H * W
        pixels_per_step = max(num_pixels // steps, 1)
        
        mask = torch.ones((1, 1, H, W), device=device, dtype=x_mod.dtype)
        scores: List[float] = []
        
        for step_i in range(steps + 1):
            masked_input = x_mod * mask
            
            logits = model(masked_input)
            probs = F.softmax(logits, dim=1)
            score = float(probs[0, class_idx].item())
            scores.append(score)
            
            if step_i == steps:
                break
            
            start_idx = step_i * pixels_per_step
            end_idx = min((step_i + 1) * pixels_per_step, num_pixels)
            pixels_to_delete = indices[start_idx:end_idx]
            
            for pixel_idx in pixels_to_delete:
                row = int(pixel_idx // W)
                col = int(pixel_idx % W)
                mask[0, 0, row, col] = 0.0
        
        fractions = np.linspace(0.0, 1.0, len(scores))
        auc = float(np.trapz(scores, fractions))
        
        return auc


def save_gradcam_overlay(
    x: torch.Tensor,
    cam: np.ndarray,
    out_dir: str,
    round_num: int,
    idx: int,
    true_label: int,
    pred_label: int,
    auc: Optional[float] = None,
) -> None:
    """Save an overlay of Grad-CAM++ heatmap on top of the input image."""
    _ensure_dir(out_dir)

    img = x[0].detach().cpu()
    
    if img.shape[0] == 1:
        base = img[0].numpy()
        is_grayscale = True
    else:
        base = img.permute(1, 2, 0).numpy()
        is_grayscale = False

    base = base - base.min()
    if base.max() > 0:
        base = base / (base.max() + 1e-8)

    cam_resized = cv2.resize(cam, (base.shape[1], base.shape[0]))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    if is_grayscale:
        axes[0].imshow(base, cmap="gray")
    else:
        axes[0].imshow(base)
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(cam_resized, cmap="jet")
    axes[1].set_title("Grad-CAM++")
    axes[1].axis("off")
    
    if is_grayscale:
        axes[2].imshow(base, cmap="gray")
    else:
        axes[2].imshow(base)
    axes[2].imshow(cam_resized, cmap="jet", alpha=0.5)
    
    match_symbol = "✓" if pred_label == true_label else "✗"
    title = f"Overlay {match_symbol}\nTrue: {true_label}, Pred: {pred_label}"
    if auc is not None:
        title += f"\nDel-AUC: {auc:.4f}"
    axes[2].set_title(title)
    axes[2].axis("off")

    plt.tight_layout()
    
    fname = f"round{round_num:03d}_sample{idx:03d}_true{true_label}_pred{pred_label}"
    if auc is not None:
        fname += f"_auc{auc:.3f}"
    fname += ".png"
    
    save_path = os.path.join(out_dir, fname)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def run_xai_probe_gradcam_pp(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    target_layer: torch.nn.Module,
    num_classes: int,
    save_dir: str,
    epoch: int,
    num_samples: int = 16,
    save_k: int = 4,
) -> Dict[str, float]:
    """
    Run a Grad-CAM++ + Deletion AUC probe on a subset of the validation set.
    Returns:
        {"val_del_auc_mean": ..., "val_del_auc_std": ...}
    """
    if target_layer is None:
        return {"val_del_auc_mean": float("nan"), "val_del_auc_std": float("nan")}

    model.eval()
    del_aucs: List[float] = []
    seen = 0
    saved = 0

    xai_dir = os.path.join(save_dir, "xai")
    _ensure_dir(xai_dir)

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        for i in range(batch_size):
            if seen >= num_samples:
                break

            x = images[i:i+1]
            y_true = int(labels[i].item())

            with torch.no_grad():
                logits = model(x)
                pred_idx = int(torch.argmax(logits, dim=1).item())

            # for deletion we can use the true label if in range, else pred
            class_idx = y_true if 0 <= y_true < num_classes else pred_idx

            # Grad-CAM++ heatmap
            cam = compute_gradcam_pp(model, x, target_layer, class_idx=class_idx)

            # Deletion-AUC
            del_auc = compute_deletion_auc(
                model=model,
                x=x,
                cam=cam,
                class_idx=class_idx,
                device=device,
                steps=10,
            )
            del_aucs.append(del_auc)

            # Save a few overlays for inspection
            if saved < save_k:
                save_gradcam_overlay(
                    x=x,
                    cam=cam,
                    out_dir=xai_dir,
                    round_num=epoch,
                    idx=seen,
                    true_label=y_true,
                    pred_label=pred_idx,
                )
                saved += 1

            seen += 1

        if seen >= num_samples:
            break

    if not del_aucs:
        return {"val_del_auc_mean": float("nan"), "val_del_auc_std": float("nan")}

    arr = np.asarray(del_aucs, dtype=float)
    return {
        "val_del_auc_mean": float(arr.mean()),
        "val_del_auc_std": float(arr.std()),
    }


def train_one_epoch_kd(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> Tuple[float, float]:
    """
    Train student for one epoch using knowledge distillation.
    Returns:
        avg_train_loss, train_accuracy
    """
    student_model.train()
    teacher_model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Training]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            teacher_logits = teacher_model(images)

        student_logits = student_model(images)
        loss = criterion(student_logits, teacher_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # accuracy
        _, preds = torch.max(student_logits, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(train_loader)
    acc = 100.0 * correct / max(total, 1)

    return avg_loss, acc


def evaluate(
    student_model: torch.nn.Module,
    data_loader,
    device: torch.device,
    num_classes: int,
    epoch: int,
    num_epochs: int,
    split_name: str = "Val",
):
    """
    Evaluate student model on a data loader.
    Returns:
        avg_loss, accuracy, auc, y_true, y_pred
    """
    student_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    all_labels: List[torch.Tensor] = []
    all_probs: List[torch.Tensor] = []
    all_preds: List[torch.Tensor] = []

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"Epoch {epoch}/{num_epochs} [{split_name}]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            logits = student_model(images)

            loss = F.cross_entropy(logits, labels, reduction="sum")
            val_loss += loss.item()

            # probabilities for AUC
            probs = F.softmax(logits, dim=1)

            _, preds = torch.max(logits, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())

    avg_loss = val_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)

    all_labels_tensor = torch.cat(all_labels, dim=0)
    all_probs_tensor = torch.cat(all_probs, dim=0)
    all_preds_tensor = torch.cat(all_preds, dim=0)

    # Compute multi-class AUC (macro, one-vs-rest)
    try:
        auc = roc_auc_score(
            all_labels_tensor.numpy(),
            all_probs_tensor.numpy(),
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        # This can happen if some classes are missing in val set for a given epoch
        auc = float("nan")
        logger.warning(
            "AUC could not be computed for this epoch (likely missing classes in this split)."
        )

    return avg_loss, acc, auc, all_labels_tensor, all_preds_tensor

def plot_confusion_matrix(
    cm,
    class_names,
    save_path: str,
    normalize: bool = True,
    title: str = "Confusion Matrix",
):
    """
    Save confusion matrix as a PNG heatmap using a classic white→green colormap.
    Higher values = darker green. Background is near white.
    """
    import numpy as np
    from matplotlib import cm as mpl_cm

    # optional normalization (row-wise)
    if normalize:
        cm = cm.astype("float")
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid div by zero
        cm = cm / row_sums  # values in [0,1]

    green_cmap = mpl_cm.get_cmap("Greens")  # traditional confusion-matrix style

    plt.figure(figsize=(8, 7))
    # vmin/vmax make 0 = light/white, 1 = darkest green when normalized
    plt.imshow(cm, interpolation="nearest", cmap=green_cmap, vmin=0.0, vmax=1.0 if normalize else None)
    plt.title(title)
    plt.colorbar()

    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # annotate cells
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=9,
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix PNG to {save_path}")


def plot_curves(
    history: Dict[str, List[float]],
    save_dir: str,
    run_name: str,
) -> None:
    _ensure_dir(save_dir)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Train vs Val Loss ({run_name})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    loss_path = os.path.join(save_dir, f"{run_name}_loss_curve.png")
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved loss curve to {loss_path}")

    # Accuracy curves
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Train vs Val Accuracy ({run_name})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    acc_path = os.path.join(save_dir, f"{run_name}_acc_curve.png")
    plt.savefig(acc_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved accuracy curve to {acc_path}")

    # AUC curve (validation only)
    if "val_auc" in history and history["val_auc"]:
        plt.figure()
        plt.plot(epochs, history["val_auc"], label="Val AUC (macro)")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.title(f"Validation AUC ({run_name})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        auc_path = os.path.join(save_dir, f"{run_name}_auc_curve.png")
        plt.savefig(auc_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved AUC curve to {auc_path}")

    # Deletion AUC (Grad-CAM++) curve if available
    if "val_del_auc_mean" in history and history["val_del_auc_mean"]:
        plt.figure()
        plt.plot(epochs, history["val_del_auc_mean"], label="Val Deletion AUC (Grad-CAM++)")
        plt.xlabel("Epoch")
        plt.ylabel("Deletion AUC")
        plt.title(f"Validation Deletion AUC ({run_name})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        del_auc_path = os.path.join(save_dir, f"{run_name}_deletion_auc_curve.png")
        plt.savefig(del_auc_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved Deletion AUC curve to {del_auc_path}")


def train_kd(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
    num_epochs: int,
    save_dir: str,
    student_model_name: str,
    teacher_name: str,
    num_classes: int,
):
    """
    Full KD training loop with metrics, curves, confusion matrix, AUC, and Grad-CAM++ XAI.
    Returns:
        history dict, best_model_path, confusion_matrix (for best epoch)
    """
    _ensure_dir(save_dir)

    run_name = f"student_{student_model_name}_from_teacher_{teacher_name}"

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_auc": [],
        "val_del_auc_mean": [],
        "val_del_auc_std": [],
    }

    best_val_loss = float("inf")
    best_model_path = None
    best_cm = None

    # Prepare Grad-CAM++ target layer for the student
    target_layer = find_last_conv_layer(student_model)

    # MOVED: Define metrics directory and JSON path *before* the loop
    metrics_dir = os.path.join(save_dir, "metrics")
    _ensure_dir(metrics_dir)
    # NEW: Define the path for the JSON log
    metrics_json_path = os.path.join(metrics_dir, f"{run_name}_metrics_history.json")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch_kd(
            student_model,
            teacher_model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            num_epochs,
        )

        val_loss, val_acc, val_auc, y_true, y_pred = evaluate(
            student_model,
            val_loader,
            device,
            num_classes=num_classes,
            epoch=epoch,
            num_epochs=num_epochs,
            split_name="Val",
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        # Confusion matrix for this epoch
        cm = confusion_matrix(
            y_true.numpy(), y_pred.numpy(), labels=list(range(num_classes))
        )

        logger.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Val AUC: {val_auc:.4f}"
        )
        logger.info(f"Validation Confusion Matrix (epoch {epoch}):\n{cm}")

        # XAI probe: Grad-CAM++ + Deletion AUC on validation subset
        # Note: metrics_dir is already defined above the loop
        xai_metrics = run_xai_probe_gradcam_pp(
            model=student_model,
            val_loader=val_loader,
            device=device,
            target_layer=target_layer,
            num_classes=num_classes,
            save_dir=metrics_dir, # Use pre-defined metrics_dir
            epoch=epoch,
            num_samples=16,  # how many val images to probe
            save_k=4,        # how many Grad-CAM++ overlays to save
        )
        history["val_del_auc_mean"].append(xai_metrics["val_del_auc_mean"])
        history["val_del_auc_std"].append(xai_metrics["val_del_auc_std"])
        logger.info(
            f"XAI (Grad-CAM++) - Deletion AUC mean: {xai_metrics['val_del_auc_mean']:.4f}, "
            f"std: {xai_metrics['val_del_auc_std']:.4f}"
        )

        # NEW: Save metrics to JSON file after each epoch
        try:
            with open(metrics_json_path, "w") as f:
                json.dump(history, f, indent=4)
            logger.info(f"Metrics history saved to {metrics_json_path}")
        except Exception as e:
            logger.warning(f"Failed to save metrics JSON on epoch {epoch}: {e}")

        # Scheduler (ReduceLROnPlateau expects a metric to minimize, here val_loss)
        if scheduler is not None:
            scheduler.step(val_loss)

        # Save best model by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, f"{run_name}.pth")
            torch.save(student_model.state_dict(), best_model_path)
            best_cm = cm
            logger.info(f"New best student model saved to {best_model_path}")

    # After all epochs, plot curves and save confusion matrix
    # metrics_dir is already defined and created
    plot_curves(history, metrics_dir, run_name)

    if best_cm is not None:
        # 1) save raw confusion matrix as text
        cm_txt_path = os.path.join(metrics_dir, f"{run_name}_confusion_matrix.txt")
        with open(cm_txt_path, "w") as f:
            f.write(str(best_cm))
        logger.info(f"Saved best confusion matrix to {cm_txt_path}")

        # 2) save normalized confusion matrix as PNG (white→green)
        class_names = ["Glioma_tumor", "Healthy", "Meningioma_tumor", "Pituitary_tumor"]
        cm_png_path = os.path.join(metrics_dir, f"{run_name}_confusion_matrix.png")
        plot_confusion_matrix(
            best_cm,
            class_names=class_names,
            save_path=cm_png_path,
            normalize=True,
            title=f"Confusion Matrix ({run_name})",
        )

    return history, best_model_path, best_cm