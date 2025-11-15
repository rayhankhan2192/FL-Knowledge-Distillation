import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# gRPC stability / payloads
os.environ.setdefault("GRPC_KEEPALIVE_TIME_MS", "30000")
os.environ.setdefault("GRPC_KEEPALIVE_TIMEOUT_MS", "10000")
os.environ.setdefault("GRPC_HTTP2_MAX_PINGS_WITHOUT_DATA", "0")
os.environ.setdefault("GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS", "1")
os.environ.setdefault("GRPC_MAX_RECEIVE_MESSAGE_LENGTH", str(200 * 1024 * 1024))
os.environ.setdefault("GRPC_MAX_SEND_MESSAGE_LENGTH",    str(200 * 1024 * 1024))

import logging, warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

import multiprocessing as mp
import time
import argparse
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # NEW IMPORT
import flwr as fl
import grpc
import cv2
import matplotlib.pyplot as plt  # NEW IMPORT

from utils.label_focalLoss import FocalLoss, LabelSmoothingLoss
from utils.dataloder import create_data_loaders, get_class_weights  # (spelling kept)
from utils.evaluation import ModelTrainer, ModelMetrics
from models.modelEngine import get_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

RESULTS_BASE_DIR = os.path.abspath(os.path.join("Result", "clientresult"))
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

def _set_runtime_knobs(num_threads: int = 4) -> None:
    """
    Configure CPU threads to reduce contention on CPU-only boxes.
    """
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    try:
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


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
    class_idx: int = None,
) -> np.ndarray:
    """
    Compute Grad-CAM++ heatmap for a single image batch x (shape [1, C, H, W]).
    Returns a numpy array of shape [H, W] normalized to [0,1].
    """
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    logits = model(x)
    if class_idx is None:
        class_idx = int(torch.argmax(logits, dim=1).item())
    score = logits[:, class_idx]
    
    # Use enable_grad context for backward pass
    with torch.enable_grad():
        score.backward(retain_graph=False)

    A = activations[0]  # [B, C, H, W]
    G = gradients[0]    # [B, C, H, W]

    A = A[0]  # [C, H, W]
    G = G[0]  # [C, H, W]

    grads2 = G ** 2
    grads3 = G ** 3
    sum_activations = torch.sum(A, dim=(1, 2), keepdim=True)  # [C, 1, 1]

    eps = 1e-7
    alpha = grads2 / (2 * grads2 + sum_activations * grads3 + eps)  # [C, H, W]
    positive_gradients = F.relu(G)  # [C, H, W]
    weights = torch.sum(alpha * positive_gradients, dim=(1, 2))  # [C]

    cam = torch.sum(weights.view(-1, 1, 1) * A, dim=0)  # [H, W]
    cam = F.relu(cam)

    cam_np = cam.detach().cpu().numpy()
    cam_np = cam_np - cam_np.min()
    if cam_np.max() > 0:
        cam_np = cam_np / cam_np.max()

    handle_f.remove()
    handle_b.remove()
    return cam_np


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
    Iteratively removes the most important pixels according to cam and tracks
    the predicted probability for class_idx. Returns the AUC over fraction deleted.
    """
    model.eval()
    with torch.no_grad():
        x_mod = x.clone().to(device)  # [1, C, H, W]
        _, _, H, W = x_mod.shape

        cam_flat = cam.reshape(-1)
        indices = np.argsort(-cam_flat)  # high to low
        num_pixels = H * W
        pixels_per_step = max(num_pixels // steps, 1)

        scores: List[float] = []
        mask = torch.ones((1, 1, H, W), device=device) # Assumes 1-channel mask

        for i in range(steps + 1):
            # Apply mask. Ensure mask works for multi-channel images if needed.
            # If x_mod is [1, C, H, W], mask should be broadcastable.
            if x_mod.size(1) == 3: # Handle 3-channel input
                 masked_input = x_mod * mask.expand(-1, 3, -1, -1)
            else: # Handle 1-channel input
                 masked_input = x_mod * mask

            logits = model(masked_input)
            probs = F.softmax(logits, dim=1)
            scores.append(float(probs[0, class_idx].item()))

            if i == steps:
                break

            start = i * pixels_per_step
            end = (i + 1) * pixels_per_step
            end = min(end, num_pixels)
            idx = indices[start:end]

            flat_mask = mask.view(-1)
            # 0-out the selected pixels
            rows = idx // W
            cols = idx % W
            flat_mask[idx] = 0.0
            mask = flat_mask.view(1, 1, H, W)


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
) -> None:
    """
    Save an overlay of Grad-CAM++ heatmap on top of the input image.
    x: [1, C, H, W] tensor on device.
    """
    _ensure_dir(out_dir)

    img = x[0].detach().cpu()
    # Handle both 1-channel (C=1) and 3-channel (C=3)
    if img.shape[0] == 1:
        base = img[0].numpy() # [H, W]
    else:
        base = img.permute(1, 2, 0).numpy() # [H, W, C]

    # Normalize base image to [0,1] for display
    base = base - base.min()
    if base.max() > 0:
        base = base / (base.max() + 1e-8)

    # Resize CAM to match image dimensions
    cam_resized = cv2.resize(cam, (base.shape[1], base.shape[0]))
    
    cam_norm = cam_resized
    cam_norm = cam_norm - cam_norm.min()
    if cam_norm.max() > 0:
        cam_norm = cam_norm / (cam_norm.max() + 1e-8)

    plt.figure(figsize=(4, 4))
    if base.ndim == 2: # Grayscale
        plt.imshow(base, cmap="gray")
    else: # RGB
        plt.imshow(base)
    plt.imshow(cam_norm, cmap="jet", alpha=0.5)
    plt.axis("off")

    fname = f"round{round_num}_idx{idx}_true{true_label}_pred{pred_label}.png"
    save_path = os.path.join(out_dir, fname)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()

# =============================================================================
# END OF NEW XAI FUNCTIONS
# =============================================================================


# Flower Client
class MedicalFLClient(fl.client.NumPyClient):
    """
    Federated Learning client for medical image classification (PyTorch).
    """
    def __init__(
        self,
        client_id: int,
        data_dir: str,
        device: torch.device,
        model_name: str = "customcnn",
        num_classes: int = 4,
        batch_size: int = 16,
        local_epochs: int = 8,
        num_workers: int = 4,
        results_base_dir: str = RESULTS_BASE_DIR,
    ):
        self.client_id = client_id
        self.data_dir = data_dir
        self.device = device
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.num_workers = num_workers
        self.current_round = 0 # Track current FL round

        # Per-client folders
        self.results_base_dir = results_base_dir
        os.makedirs(self.results_base_dir, exist_ok=True)
        self.client_root = os.path.join(self.results_base_dir, f"client_{client_id}")
        self.ckpt_dir = os.path.join(self.client_root, "checkpoints")
        self.log_dir = os.path.join(self.client_root, "logs")
        self.xai_dir = os.path.join(self.client_root, "xai")
        self.pred_dir = os.path.join(self.client_root, "predictions")
        self.metrics_dir = os.path.join(self.client_root, "metrics")
        for d in [self.client_root, self.ckpt_dir, self.log_dir, self.xai_dir, self.pred_dir, self.metrics_dir]:
            os.makedirs(d, exist_ok=True)

        # Model
        self.model = get_model(model_name, num_classes, pretrained=True)
        self.model.to(device)

        # *** MODIFIED: Use new function ***
        self.target_layer = find_last_conv_layer(self.model)
        if self.target_layer is None:
            logger.warning("No Conv2d layer found for Grad-CAM. XAI probe will be skipped.")

        logger.info(f"Client {client_id}: Loading data from {data_dir}")
        ctx = mp.get_context("spawn")
        self.train_loader, self.val_loader, self.test_loader = self._create_loaders_spawn_safe(
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=(224, 224),
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            num_workers=num_workers,
            multiprocessing_context=ctx,
        )

        # Class weights
        self.class_weights = get_class_weights(self.train_loader)
        try:
            cw_log = self.class_weights.tolist()
        except Exception:
            cw_log = self.class_weights
        logger.info(f"Client {client_id}: Class weights: {cw_log}")

        # Trainer
        self.trainer = ModelTrainer(self.model, device, self.ckpt_dir, self.log_dir)
        self.learning_rate = 0.001
        self.weight_decay = 1e-4

        logger.info(f"Client {client_id} initialized successfully")
        logger.info(f"  - Training samples:   {len(self.train_loader.dataset)}")
        logger.info(f"  - Validation samples: {len(self.val_loader.dataset)}")
        logger.info(f"  - Test samples:       {len(self.test_loader.dataset)}")

    @staticmethod
    def _create_loaders_spawn_safe(
        data_dir: str,
        batch_size: int,
        image_size: Tuple[int, int],
        train_split: float,
        val_split: float,
        test_split: float,
        num_workers: int,
        multiprocessing_context,
    ):
        """
        Try to pass 'multiprocessing_context' to your create_data_loaders if supported.
        If not, fall back without it (still fine if mp.set_start_method('spawn') is set).
        """
        try:
            loaders = create_data_loaders(
                data_dir=data_dir,
                batch_size=batch_size,
                train_split=train_split,
                val_split=val_split,
                test_split=test_split,
                image_size=image_size,
                num_workers=num_workers,
                multiprocessing_context=multiprocessing_context, 
            )
            return loaders
        except TypeError:
            logger.warning("utils.dataloder.create_data_loaders does not accept 'multiprocessing_context'; "
                           "falling back without it. Make sure mp.set_start_method('spawn') is set in main().")
            loaders = create_data_loaders(
                data_dir=data_dir,
                batch_size=batch_size,
                train_split=train_split,
                val_split=val_split,
                test_split=test_split,
                image_size=image_size,
                num_workers=num_workers,
            )
            return loaders

    # Flower NumPyClient API ----
    def get_parameters(self, config: Dict = None) -> List[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        own_state = self.model.state_dict()
        keys = list(own_state.keys())
        incoming = OrderedDict()
        for k, v in zip(keys, parameters):
            t = torch.tensor(v, dtype=own_state[k].dtype)
            incoming[k] = t
        merged = OrderedDict((k, incoming.get(k, own_state[k])) for k in own_state.keys())
        self.model.load_state_dict(merged, strict=False)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        logger.info(f"Client {self.client_id}: Starting local training round")
        self.set_parameters(parameters)

        # Server-configurable knobs
        self.current_round = int(config.get("server_round", 0)) # Store current round
        local_epochs   = int(config.get("local_epochs", self.local_epochs))
        learning_rate  = float(config.get("learning_rate", self.learning_rate))
        weight_decay   = float(config.get("weight_decay", self.weight_decay))
        loss_function  = str(config.get("loss_function", "crossentropy")).lower()
        optimizer_name = str(config.get("optimizer", "adamw")).lower()
        scheduler_name = str(config.get("scheduler", "plateau")).lower()
        use_scheduler  = bool(config.get("use_scheduler", True))

        # Loss
        if loss_function == "focal":
            criterion = FocalLoss(alpha=1.0, gamma=2.0)
        elif loss_function == "label_smoothing":
            criterion = LabelSmoothingLoss(num_classes=self.num_classes, smoothing=0.1)
        else:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))

        # Train
        train_history = self.trainer.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=local_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            class_weights=self.class_weights,
            use_scheduler=use_scheduler,
            patience=10,
            criterion=criterion,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
        )

        # Evaluate on test
        test_metrics = self.trainer.evaluate(self.test_loader)

        # XAI probe (optional)
        xai_metrics = self._xai_probe(
            loader=self.val_loader, 
            num_samples=16, 
            save_k=3, 
            epoch=self.current_round
        )

        # Save checkpoint
        best_model_path = os.path.join(self.ckpt_dir, f"client_{self.client_id}_best_model.pth")
        torch.save(self.model.state_dict(), best_model_path)
        logger.info(f"Client {self.client_id}: Best model saved to {best_model_path}")

        # Scalar metrics only (keep payload small)
        metrics = {
            "train_loss": float(train_history["train_loss"][-1]),
            "train_accuracy": float(train_history["train_accuracy"][-1]),
            "val_loss": float(train_history["val_loss"][-1]),
            "val_accuracy": float(train_history["val_accuracy"][-1]),
            "val_f1": float(train_history["val_f1_macro"][-1]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_f1": float(test_metrics["f1_macro"]),
            "num_examples": int(len(self.train_loader.dataset)),
            **xai_metrics, # Add val_del_auc_mean and val_del_auc_std
        }

        logger.info(f"Client {self.client_id}: Local training completed")
        return self.get_parameters(), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        logger.info(f"Client {self.client_id}: Starting evaluation")
        self.set_parameters(parameters)
        self.current_round = int(config.get("server_round", 0))

        test_metrics = self.trainer.evaluate(self.test_loader)
        add_xai = self._xai_probe(
            loader=self.val_loader, 
            num_samples=12, 
            save_k=0,
            epoch=self.current_round
        )
        test_metrics.update(add_xai)

        logger.info(f"Client {self.client_id}: Evaluation completed")
        logger.info(f"  - Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  - Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
        logger.info(f"  - XAI Deletion AUC: {add_xai['val_del_auc_mean']:.4f}")

        return (
            float(test_metrics.get("loss", 0.0)),
            int(len(self.test_loader.dataset)),
            {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in test_metrics.items()},
        )

    # *** MODIFIED: Rewritten _xai_probe function ***
    def _xai_probe(self, loader, num_samples: int = 16, save_k: int = 3, epoch: int = 0) -> Dict[str, float]:
        """
        Run a Grad-CAM++ + Deletion AUC probe on a subset of the validation set.
        Returns:
            {"val_del_auc_mean": ..., "val_del_auc_std": ...}
        """
        if self.target_layer is None:
            logger.warning("No target_layer found, skipping XAI probe.")
            return {"val_del_auc_mean": float("nan"), "val_del_auc_std": float("nan")}

        self.model.eval()
        del_aucs: List[float] = []
        seen = 0
        saved = 0

        # Use the xai_dir defined in __init__
        _ensure_dir(self.xai_dir) 

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)

            for i in range(batch_size):
                if seen >= num_samples:
                    break

                x = images[i:i+1] # [1, C, H, W]
                y_true = int(labels[i].item())

                with torch.no_grad():
                    logits = self.model(x)
                    pred_idx = int(torch.argmax(logits, dim=1).item())

                # Use the true label for the test
                class_idx = y_true

                # 1. Grad-CAM++ heatmap
                # We need to enable grad for CAM calculation
                x.requires_grad_(True)
                cam = compute_gradcam_pp(self.model, x, self.target_layer, class_idx=class_idx)
                x.requires_grad_(False) # Detach

                # 2. Deletion-AUC
                del_auc = compute_deletion_auc(
                    model=self.model,
                    x=x.detach(), # Use detached tensor
                    cam=cam,
                    class_idx=class_idx,
                    device=self.device,
                    steps=10,
                )
                del_aucs.append(del_auc)

                # 3. Save overlays
                if saved < save_k:
                    save_gradcam_overlay(
                        x=x.detach(),
                        cam=cam,
                        out_dir=self.xai_dir,
                        round_num=epoch, # Use round_num as epoch
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
        metrics = {
            "val_del_auc_mean": float(arr.mean()),
            "val_del_auc_std": float(arr.std()),
        }
        logger.info(f"XAI Probe (Round {epoch}): Deletion AUC Mean={metrics['val_del_auc_mean']:.4f} Std={metrics['val_del_auc_std']:.4f}")
        return metrics


def create_client(client_id: int, data_dir: str, model_name: str = "customcnn",
                  batch_size: int = 16, local_epochs: int = 50, num_workers: int = 1) -> MedicalFLClient:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    client = MedicalFLClient(
        client_id=client_id,
        data_dir=data_dir,
        device=device,
        model_name=model_name,
        num_classes=4,
        batch_size=batch_size,
        local_epochs=local_epochs,
        num_workers=num_workers,
    )
    return client


def run_flower(server_address: str, client: MedicalFLClient) -> None:
    """Start Flower with simple auto-reconnect on transient UNAVAILABLE."""
    while True:
        try:
            fl.client.start_client(
                server_address=server_address,
                client=client.to_client()
            )
            break  # finished cleanly
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                logger.warning("Server UNAVAILABLE (gRPC 14). Reconnecting in 5s...")
                time.sleep(5)
                continue
            else:
                raise


def main():
    # Use spawn to avoid forking a multi-threaded process (Flower/gRPC)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set in this process
        pass

    _set_runtime_knobs(num_threads=4)

    parser = argparse.ArgumentParser(description="Federated Learning Client for Medical Imaging")
    parser.add_argument("--client-id", type=int, default=1, help="Client ID")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to client data directory")
    parser.add_argument("--server-address", type=str, default="localhost:8080", help="FL server address")
    parser.add.argument("--model", type=str, default="customcnn",
                        choices=["mobilenetv3", "hybridmodel", "resnet50", "cnn", "hybridswin", "densenet121"],
                        help="Model architecture")
    parser.add_argument("--train-local", action="store_true", help="Run local training only (no FL server)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (CPU-only: 32 for ResNet/DenseNet, 64 for small CNN/EfficientNetB0)")
    parser.add_argument("--local-epochs", type=int, default=50, help="Local epochs per round")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (set 0 if problems)")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory not found: {args.data_dir}")

    client = create_client(
        client_id=args.client_id,
        data_dir=args.data_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        num_workers=args.num_workers,
    )

    if args.train_local:
        logger.info("Running standalone local training (no FL server)")
        updated_params, num_examples, train_metrics = client.fit(client.get_parameters(), config={"server_round": 0})
        test_loss, test_examples, test_metrics = client.evaluate(updated_params, config={"server_round": 0})
        logger.info("Local training and evaluation completed:")
        logger.info(f"  - Final train metrics: {train_metrics}")
        logger.info(f"  - Final test metrics:  {test_metrics}")
        return

    logger.info(f"Starting FL client {args.client_id} connecting to {args.server_address}")
    run_flower(args.server_address, client)


if __name__ == "__main__":
    main()