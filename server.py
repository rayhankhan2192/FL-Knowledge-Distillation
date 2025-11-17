import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

import argparse
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict
import time
import json
from datetime import datetime
import threading
import sys
import subprocess #For running the distillation script

import numpy as np
import torch
import flwr as fl
from flwr.server.client_manager import SimpleClientManager, ClientProxy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.modelEngine import get_model

RESULTS_BASE_DIR = os.path.abspath("Result/FLResult")
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
logger = logging.getLogger("FL-Server")
GRPC_MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024

def get_init_parameters(model_name: str, num_classes: int) -> fl.common.Parameters:
    try:
        model = get_model(model_name, num_classes, pretrained=False)
        with torch.no_grad():
            parameters = [v.cpu().numpy() for _, v in model.state_dict().items()]
        logger.info(f"Initial model parameters loaded for: {model_name} (Classes: {num_classes})")
        return fl.common.ndarrays_to_parameters(parameters)
    except Exception as e:
        logger.error(f"Failed to get initial parameters: {e}", exc_info=True)
        return None

def fit_config(server_round: int, local_epochs: int) -> Dict[str, fl.common.Scalar]:
    """Per-round training config broadcast to clients."""
    config = {
        "server_round": server_round,
        "local_epochs": local_epochs,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "loss_function": "cross_entropy",
        "optimizer": "adamw",
        "scheduler": "plateau",
        "use_scheduler": True,
        "batch_size": 16,
        "xai_probe": True,
        "xai_samples": 16,
        "xai_save_k": 0,
    }
    logger.info(
        f"Round {server_round} training config: "
        f"epochs={config['local_epochs']}, lr={config['learning_rate']}, loss={config['loss_function']}"
    )
    return config

def evaluate_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    return {"server_round": server_round}

def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    if not metrics: return {}
    total_samples = sum(num_samples for num_samples, _ in metrics)
    if total_samples == 0: return {}
    aggregated_metrics: Dict[str, float] = {}
    for num_samples, client_metrics in metrics:
        weight = num_samples / total_samples
        for key, value in client_metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                aggregated_metrics[key] = aggregated_metrics.get(key, 0.0) + weight * float(value)
    return aggregated_metrics

class MedicalFLStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg strategy extended with:
      - history tracking
      - best/last global checkpoint saving
      - detailed round logging & plots
      - optional XAI metrics aggregation
      
    """
    def __init__(
        self,
        *,
        model_name: str,
        num_classes: int,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = True,
        initial_parameters: Optional[fl.common.Parameters] = None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        results_base_dir: str = RESULTS_BASE_DIR,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.model_name = model_name
        self.num_classes = num_classes

        self.history = {
            "round": [],
            "train_loss": [],
            "train_accuracy": [],
            "train_f1": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "test_loss": [],
            "test_accuracy": [],
            "test_f1": [],
            "num_clients": [],
            "client_data_sizes": [],
            "aggregation_time": [],
            # XAI history
            "xai_del_auc_mean": [], "xai_del_auc_std": [],
            "xai_heat_in_mask_mean": [], "xai_heat_in_mask_std": [],
            # NEW: AUC history
            "train_auc_roc_macro": [],
            "val_auc_roc_macro": [],
        }
        self.best_accuracy = 0.0
        self.best_f1 = 0.0
        self.best_round = 0
        self.best_parameters: Optional[fl.common.Parameters] = None
        self.last_parameters: Optional[fl.common.Parameters] = None

        self.connected_clients = set()
        self.client_metrics_history = {}

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_base_dir = os.path.join(results_base_dir, f"fl_results_{ts}")
        os.makedirs(self.results_base_dir, exist_ok=True)
        self._save_strategy_config()

        logger.info("FL Strategy initialized")
        logger.info(f"   → results_base_dir: {self.results_base_dir}")
        logger.info(f"   → model={self.model_name}, num_classes={self.num_classes}")

    def _save_strategy_config(self):
        config = {
            "strategy": "MedicalFLStrategy",
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "fraction_fit": self.fraction_fit,
            "fraction_evaluate": self.fraction_evaluate,
            "min_fit_clients": self.min_fit_clients,
            "min_evaluate_clients": self.min_evaluate_clients,
            "min_available_clients": self.min_available_clients,
            "accept_failures": self.accept_failures,
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(self.results_base_dir, "strategy_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def configure_fit(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        logger.info(f"Round {server_round}: configuring clients for training...")
        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        sample_size, min_num = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)
        self.connected_clients.update({c.cid for c in clients})

        logger.info(f"Selected {len(clients)} clients: {sorted([c.cid for c in clients])}")
        fit_ins = fl.common.FitIns(parameters, config)
        return [(c, fit_ins) for c in clients]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        if self.fraction_evaluate == 0.0:
            return []
        logger.info(f"Round {server_round}: configuring clients for evaluation...")
        config = self.on_evaluate_config_fn(server_round) if self.on_evaluate_config_fn else {}
        sample_size, min_num = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)
        logger.info(f"Selected {len(clients)} clients for evaluation")

        eval_ins = fl.common.EvaluateIns(parameters, config)
        return [(c, eval_ins) for c in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        t0 = time.time()
        logger.info(
            f"Round {server_round}: aggregating fit results (success={len(results)}, failures={len(failures)})"
        )

        if len(results) < self.min_fit_clients:
            logger.warning(
                f"Not enough results to aggregate. Expected {self.min_fit_clients}, got {len(results)}"
            )
            return None, {}

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is None:
            return None, aggregated_metrics

        self.last_parameters = aggregated_parameters

        # Summaries from client metrics
        summary = self._calculate_fit_metrics(results)
        self.history["round"].append(server_round)
        self.history["train_loss"].append(summary["train_loss_avg"])
        self.history["train_accuracy"].append(summary["train_accuracy_avg"])
        self.history["train_f1"].append(summary["train_f1_avg"])
        self.history["val_loss"].append(summary["val_loss_avg"])
        self.history["val_accuracy"].append(summary["val_accuracy_avg"])
        self.history["val_f1"].append(summary["val_f1_avg"])
        self.history["num_clients"].append(len(results))
        self.history["client_data_sizes"].append(summary["client_data_sizes"])
        self.history["aggregation_time"].append(time.time() - t0)

        # AUC history (handles None safely)
        self.history["train_auc_roc_macro"].append(
            summary.get("train_auc_roc_macro_avg", np.nan)
        )
        self.history["val_auc_roc_macro"].append(
            summary.get("val_auc_roc_macro_avg", np.nan)
        )

        # XAI history (only if keys exist in summary)
        self.history["xai_del_auc_mean"].append(summary.get("xai_del_auc_mean_avg", np.nan))
        self.history["xai_del_auc_std"].append(summary.get("xai_del_auc_std_avg", np.nan))
        self.history["xai_heat_in_mask_mean"].append(summary.get("xai_heat_in_mask_mean_avg", np.nan))
        self.history["xai_heat_in_mask_std"].append(summary.get("xai_heat_in_mask_std_avg", np.nan))

        # Track best by validation F1
        if summary["val_f1_avg"] > self.best_f1:
            self.best_f1 = summary["val_f1_avg"]
            self.best_accuracy = summary["val_accuracy_avg"]
            self.best_round = server_round
            self.best_parameters = aggregated_parameters
            self.save_best_model()
            logger.info(
                f"🏆 New best model: round={self.best_round}, val_f1={self.best_f1:.4f}, val_acc={self.best_accuracy:.4f}"
            )

        aggregated_metrics.update({k: v for k, v in summary.items() if k.startswith("xai_")})
        aggregated_metrics["aggregation_time"] = self.history["aggregation_time"][-1]
        self._log_round_summary(server_round, summary, len(results))

        # Periodic snapshot
        if server_round % 10 == 0:
            self.save_intermediate_results(server_round)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        logger.info(
            f"Round {server_round}: aggregating evaluation results (success={len(results)} fail={len(failures)})"
        )
        if not results:
            return None, {}
        test = self._calculate_eval_metrics(results)
        if len(self.history["test_loss"]) < len(self.history["round"]):
            self.history["test_loss"].append(test["test_loss_avg"])
            self.history["test_accuracy"].append(test["test_accuracy_avg"])
            self.history["test_f1"].append(test["test_f1_avg"])
        logger.info(
            f"   Test: loss={test['test_loss_avg']:.4f} acc={test['test_accuracy_avg']:.4f} f1={test['test_f1_avg']:.4f}"
        )
        return test["test_loss_avg"], test

    def _calculate_fit_metrics(self, results):
        total_examples = sum(fit_res.num_examples for _, fit_res in results) or 1
        metric_keys = [
            "train_loss", "train_accuracy", "train_f1",
            "val_loss", "val_accuracy", "val_f1",
            # NEW: per-client AUC metrics
            "train_auc_roc_macro", "val_auc_roc_macro",
            "xai_del_auc_mean", "xai_del_auc_std",
            "xai_heat_in_mask_mean", "xai_heat_in_mask_std",
        ]

        weighted_sums = {key: 0.0 for key in metric_keys}
        present = {key: False for key in metric_keys}
        client_data_sizes, client_metrics_list = [], []

        for client_proxy, fit_res in results:
            weight = fit_res.num_examples / total_examples
            client_data_sizes.append(fit_res.num_examples)
            metrics_for_client = {}
            for key in metric_keys:
                val = fit_res.metrics.get(key, None) if fit_res.metrics else None
                if isinstance(val, (int, float, np.integer, np.floating)):
                    weighted_sums[key] += float(val) * weight
                    present[key] = True
                    metrics_for_client[key] = float(val)
            client_metrics_list.append({
                "client_id": client_proxy.cid,
                "num_examples": fit_res.num_examples,
                "metrics": metrics_for_client,
            })

        current_round = len(self.history["round"]) + 1
        self.client_metrics_history[current_round] = client_metrics_list

        out = {
            "train_loss_avg": weighted_sums["train_loss"],
            "train_accuracy_avg": weighted_sums["train_accuracy"],
            "train_f1_avg": weighted_sums["train_f1"],
            "val_loss_avg": weighted_sums["val_loss"],
            "val_accuracy_avg": weighted_sums["val_accuracy"],
            "val_f1_avg": weighted_sums["val_f1"],
            # NEW: aggregated AUC (or None if not present)
            "train_auc_roc_macro_avg": weighted_sums["train_auc_roc_macro"] if present["train_auc_roc_macro"] else None,
            "val_auc_roc_macro_avg": weighted_sums["val_auc_roc_macro"] if present["val_auc_roc_macro"] else None,

            "total_examples": total_examples,
            "client_data_sizes": client_data_sizes,
            "num_participating_clients": len(results),
        }
        if present["xai_del_auc_mean"]:
            out["xai_del_auc_mean_avg"] = weighted_sums["xai_del_auc_mean"]
            out["xai_del_auc_std_avg"]  = weighted_sums["xai_del_auc_std"]
        if present["xai_heat_in_mask_mean"]:
            out["xai_heat_in_mask_mean_avg"] = weighted_sums["xai_heat_in_mask_mean"]
            out["xai_heat_in_mask_std_avg"]  = weighted_sums["xai_heat_in_mask_std"]
        return out

    def _calculate_eval_metrics(
        self, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]]
    ) -> Dict:
        total_examples = sum(eval_res.num_examples for _, eval_res in results) or 1
        weighted_loss = 0.0
        weighted_accuracy = 0.0
        weighted_f1 = 0.0

        for _, eval_res in results:
            weight = eval_res.num_examples / total_examples
            weighted_loss += float(eval_res.loss or 0.0) * weight
            if eval_res.metrics:
                weighted_accuracy += float(eval_res.metrics.get("accuracy", 0.0)) * weight
                weighted_f1 += float(eval_res.metrics.get("f1_macro", 0.0)) * weight

        return {
            "test_loss_avg": weighted_loss,
            "test_accuracy_avg": weighted_accuracy,
            "test_f1_avg": weighted_f1,
            "total_test_examples": total_examples,
            "num_eval_clients": len(results),
        }

    def _log_round_summary(self, round_num: int, summary: Dict, num_clients: int):
        logger.info("=" * 80)
        logger.info(f"ROUND {round_num} SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Clients: {num_clients} | examples: {summary['total_examples']:,}")
        logger.info(
            f"Train: loss={summary['train_loss_avg']:.4f} "
            f"acc={summary['train_accuracy_avg']:.4f} f1={summary['train_f1_avg']:.4f}"
        )
        logger.info(
            f"Val  : loss={summary['val_loss_avg']:.4f} "
            f"acc={summary['val_accuracy_avg']:.4f} f1={summary['val_f1_avg']:.4f}"
        )
        logger.info(f"Best : round={self.best_round} val_f1={self.best_f1:.4f} val_acc={self.best_accuracy:.4f}")
        val_auc = summary.get("val_auc_roc_macro_avg", None)
        if val_auc is not None:
            logger.info(
                f"Round {round_num}: val_auc_roc_macro={val_auc:.4f}"
            )

        xai_mean = summary.get("xai_del_auc_mean_avg", None)
        if xai_mean is not None:
            logger.info(
                f"XAI  : delAUC={xai_mean:.4f} "
                f"mask_in={summary.get('xai_heat_in_mask_mean_avg','NA')}"
            )

        logger.info("=" * 80)

    def save_best_model(self) -> None:
        if self.best_parameters is None:
            logger.warning("No best parameters available, skipping model save.")
            return
        try:
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            if models_dir not in sys.path:
                sys.path.insert(0, models_dir)

            model = get_model(self.model_name, num_classes=self.num_classes, pretrained=False)

            best_state_dict = OrderedDict()
            for (name, param), arr in zip(
                model.state_dict().items(),
                fl.common.parameters_to_ndarrays(self.best_parameters),
            ):
                best_state_dict[name] = torch.as_tensor(arr, dtype=param.dtype)

            checkpoint = {
                "round": self.best_round,
                "model_state_dict": best_state_dict,
                "best_f1": self.best_f1,
                "best_accuracy": self.best_accuracy,
                "model_name": self.model_name,
                "num_classes": self.num_classes,
                "timestamp": datetime.now().isoformat(),
            }

            save_path = os.path.join(self.results_base_dir, f"best_model_round_{self.best_round}.pth")
            torch.save(checkpoint, save_path)
            logger.info(f"Best model saved successfully → {save_path}")

        except Exception as exc:
            logger.error(f"Failed to save best model: {exc}", exc_info=True)

    def save_last_model(self) -> None:
        if self.last_parameters is None:
            logger.warning("No last parameters available, skipping model save.")
            return
        try:
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            if models_dir not in sys.path:
                sys.path.insert(0, models_dir)

            model = get_model(self.model_name, num_classes=self.num_classes, pretrained=False)

            last_state_dict = OrderedDict()
            for (name, param), arr in zip(
                model.state_dict().items(),
                fl.common.parameters_to_ndarrays(self.last_parameters),
            ):
                last_state_dict[name] = torch.as_tensor(arr, dtype=param.dtype)

            checkpoint = {
                "model_state_dict": last_state_dict,
                "model_name": self.model_name,
                "num_classes": self.num_classes,
                "timestamp": datetime.now().isoformat(),
            }

            save_path = os.path.join(self.results_base_dir, "last_global_model.pth")
            torch.save(checkpoint, save_path)
            logger.info(f"Last global model saved successfully → {save_path}")

        except Exception as exc:
            logger.error(f"Failed to save last model: {exc}", exc_info=True)

    def save_intermediate_results(self, rnd: int):
        try:
            with open(os.path.join(self.results_base_dir, f"history_round_{rnd}.json"), "w") as f:
                json.dump(self.history, f, indent=2)
            with open(os.path.join(self.results_base_dir, f"client_metrics_round_{rnd}.json"), "w") as f:
                json.dump(self.client_metrics_history, f, indent=2)
            self.plot_training_curves(save_suffix=f"_round_{rnd}")
            logger.info(f"Intermediate results saved for round {rnd}")
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}", exc_info=True)

    def save_final_results(self):
        try:
            with open(os.path.join(self.results_base_dir, "final_training_history.json"), "w") as f:
                json.dump(self.history, f, indent=2)
            with open(os.path.join(self.results_base_dir, "final_client_metrics.json"), "w") as f:
                json.dump(self.client_metrics_history, f, indent=2)
            self.plot_training_curves(save_suffix="_final")
            logger.info(f"Final results saved in {self.results_base_dir}")
        except Exception as e:
            logger.error(f"Failed to save final results: {e}", exc_info=True)
    def plot_training_curves(self, save_suffix: str = ""):
        if not self.history["round"]:
            return
        rounds = self.history["round"]
        
        # Use 'ggplot' style for a different look
        plt.style.use('ggplot')
        plt.rcParams.update({
            'font.size': 11,
            'axes.facecolor': '#F3F3F3',
            'figure.facecolor': 'white',
            'axes.edgecolor': 'white',
            'axes.grid': True,
            'grid.color': 'white',
            'grid.linestyle': '--',
            'grid.linewidth': 1.5
        })

        # PLOT 1: METRICS (Loss, Accuracy, F1, Aggregation Time)
        fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
        
        # Loss Plot
        ax = axes1[0, 0]
        ax.plot(rounds, self.history["train_loss"], label="Train Loss", color='tab:blue', linestyle='-', marker="o", markersize=5)
        ax.plot(rounds, self.history["val_loss"], label="Validation Loss", color='tab:orange', linestyle='--', marker="s", markersize=5)
        if self.history["test_loss"]:
            ax.plot(rounds, self.history["test_loss"], label="Test Loss", color='tab:green', linestyle=':', marker="^", markersize=5)
        ax.set_title("Loss Across Rounds", fontsize=15)
        ax.set_xlabel("Federated Round", fontsize=12)
        ax.set_ylabel("Loss Value", fontsize=12)
        ax.legend(loc="upper right", fontsize=10)

        # Accuracy Plot
        ax = axes1[0, 1]
        ax.plot(rounds, self.history["train_accuracy"], label="Train Accuracy", color='tab:blue', linestyle='-', marker="o", markersize=5)
        ax.plot(rounds, self.history["val_accuracy"], label="Validation Accuracy", color='tab:orange', linestyle='--', marker="s", markersize=5)
        if self.history["test_accuracy"]:
            ax.plot(rounds, self.history["test_accuracy"], label="Test Accuracy", color='tab:green', linestyle=':', marker="^", markersize=5)
        ax.set_title("Accuracy Across Rounds", fontsize=15)
        ax.set_xlabel("Federated Round", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right", fontsize=10)

        # F1-Score Plot 
        ax = axes1[1, 0]
        ax.plot(rounds, self.history["train_f1"], label="Train F1-Score", color='tab:blue', linestyle='-', marker="o", markersize=5)
        ax.plot(rounds, self.history["val_f1"], label="Validation F1-Score", color='tab:orange', linestyle='--', marker="s", markersize=5)
        if self.history["test_f1"]:
            ax.plot(rounds, self.history["test_f1"], label="Test F1-Score", color='tab:green', linestyle=':', marker="^", markersize=5)
        ax.set_title("F1-Score Across Rounds", fontsize=15)
        ax.set_xlabel("Federated Round", fontsize=12)
        ax.set_ylabel("F1-Score", fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right", fontsize=10)

        # After F1 plot
        ax = axes1[1, 1]  # use the remaining subplot
        if self.history["val_auc_roc_macro"]:
            ax.plot(rounds, self.history["val_auc_roc_macro"], label="Val AUC (macro)", marker="o")
        if self.history["train_auc_roc_macro"]:
            ax.plot(rounds, self.history["train_auc_roc_macro"], label="Train AUC (macro)", linestyle="--", marker="s")
        ax.set_title("AUC Across Rounds", fontsize=15)
        ax.set_xlabel("Federated Round", fontsize=12)
        ax.set_ylabel("AUC (macro-averaged)", fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right", fontsize=10)


        # Aggregation Time Plot
        ax = axes1[1, 1]
        if self.history["aggregation_time"]:
            ax.plot(rounds, self.history["aggregation_time"], color='tab:purple', linewidth=2, marker="d", markersize=5)
            ax.set_title("Aggregation Time (seconds)", fontsize=15)
            ax.set_xlabel("Federated Round", fontsize=12)
            ax.set_ylabel("Time (s)", fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No Aggregation Time Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, c='gray')
            ax.set_title("Aggregation Time (seconds)", fontsize=15)
        
        # Final Touches for Plot 1
        fig1.suptitle(f"Federated Learning Training Metrics ({self.model_name})", fontsize=20, y=1.03)
        fig1.tight_layout(rect=[0, 0, 1, 0.98])
        out1 = os.path.join(self.results_base_dir, f"training_metrics{save_suffix}.png")
        fig1.savefig(out1, dpi=300, bbox_inches="tight")
        plt.close(fig1)
        logger.info(f"Saved plot 1 → {out1}")


        # PLOT 2: CLIENT & DATA (Clients per Round, Data Distribution)
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 7))

        # Clients per Round (Bar Chart)
        ax = axes2[0]
        if self.history["num_clients"]:
            ax.bar(rounds, self.history["num_clients"], alpha=0.8, color='tab:cyan')
            ax.set_title("Number of Clients per Round", fontsize=15)
            ax.set_xlabel("Federated Round", fontsize=12)
            ax.set_ylabel("Number of Clients", fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No Client Count Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, c='gray')
            ax.set_title("Number of Clients per Round", fontsize=15)
            

        # Data Distribution (Pie Chart)
        ax = axes2[1]
        if self.history["client_data_sizes"]:
            latest_data_sizes = self.history["client_data_sizes"][-1]
            labels = [f"Client {i+1}" for i in range(len(latest_data_sizes))]
            
            # Use a different colormap for the pie
            pie_colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))
            
            ax.pie(latest_data_sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=pie_colors,
                   wedgeprops={'edgecolor': 'white', 'linewidth': 1})
            ax.set_title("Data Distribution (Latest Round)", fontsize=15)
            ax.axis('equal')
        else:
            ax.text(0.5, 0.5, 'No Data Distribution Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, c='gray')
            ax.set_title("Data Distribution (Latest Round)", fontsize=15)
            ax.axis('equal')

        # Final Touches for Plot 2
        fig2.suptitle(f"FL Client & Data Overview ({self.model_name})", fontsize=20, y=1.04)
        fig2.tight_layout(rect=[0, 0, 1, 0.98])
        out2 = os.path.join(self.results_base_dir, f"client_data_distribution{save_suffix}.png")
        fig2.savefig(out2, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        logger.info(f"Saved plot 2 → {out2}")

        # Reset style to default if other plots are made elsewhere
        plt.style.use('default')


# LoggingClientManager, start_waiting_heartbeat, create_server_strategy
class LoggingClientManager(SimpleClientManager):
    def __init__(self, expected_clients: int):
        super().__init__()
        self.expected_clients = expected_clients

    def register(self, client: ClientProxy) -> bool:
        ok = super().register(client)
        n = self.num_available()
        remaining = max(self.expected_clients - n, 0)
        logger.info(f"Client connected: {client.cid} | connected={n} | waiting={remaining}")
        if n >= self.expected_clients:
            logger.info("Required clients connected. Starting rounds as soon as strategy is ready.")
        return ok

    def unregister(self, client: ClientProxy) -> None:
        super().unregister(client)
        n = self.num_available()
        remaining = max(self.expected_clients - n, 0)
        logger.info(f"Client disconnected: {client.cid} | connected={n} | waiting={remaining}")

def start_waiting_heartbeat(cm: SimpleClientManager, target: int, interval_sec: float = 2.0):
    stop_evt = threading.Event()

    def _loop():
        while not stop_evt.is_set():
            connected = cm.num_available()
            remaining = max(target - connected, 0)
            if remaining <= 0:
                stop_evt.set()
                break
            logger.info(f"⏳ Waiting for clients… connected={connected} | waiting={remaining}")
            time.sleep(interval_sec)

    thr = threading.Thread(target=_loop, daemon=True)
    thr.start()
    return stop_evt

def create_server_strategy(*, min_clients: int, fraction_fit: float, fraction_evaluate: float,
                           model_name: str, num_classes: int, local_epochs: int) -> MedicalFLStrategy:
    initial_parameters = get_init_parameters(model_name, num_classes)
    if initial_parameters is None:
        raise RuntimeError("Failed to initialize model parameters")
    return MedicalFLStrategy(
        model_name=model_name, num_classes=num_classes,
        fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_clients, min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        on_fit_config_fn=lambda r: fit_config(r, local_epochs),
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )


def main():
    parser = argparse.ArgumentParser("Federated Learning Server")
    # Standard FL Args 
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8080, help="Port")
    parser.add_argument("--rounds", type=int, default=3, help="FL rounds")
    parser.add_argument("--min-clients", type=int, default=1, help="Minimum clients per round")
    parser.add_argument("--fraction-fit", type=float, default=1.0)
    parser.add_argument("--fraction-evaluate", type=float, default=1.0)
    parser.add_argument("--model", type=str, default="densenet121", 
                        choices=["mobilenetv3", "hybridmodel", 
                                 "resnet50", "cnn", "HSwinDNMLP", "densenet121", "effnetb3", "effnetb4"], help="Teacher model architecture")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of classes")
    parser.add_argument("--local-epochs", type=int, default=6)
    
    # Distillation Pipeline Args 
    parser.add_argument("--run-distillation", action="store_true", 
                        help="Automatically run distillation after FL training completes")
    parser.add_argument("--student-model", type=str, default="mobilenetv3", 
                        help="Student model architecture for distillation")
    parser.add_argument("--distill-data-dir", type=str, default="Dataset", 
                        help="Path to the FULL dataset for distillation training")
    parser.add_argument("--distill-save-dir", type=str, default="Result/Distillation", 
                        help="Where to save final student models")
    parser.add_argument("--distill-epochs", type=int, default=50, help="Epochs for distillation training")
    parser.add_argument("--distill-batch-size", type=int, default=32, help="Batch size for distillation")
    parser.add_argument("--expected-clients", type=int, default=None,
                        help="How many clients you expect to connect (for logs). Defaults to --min-clients.")
    args = parser.parse_args()

    if args.expected_clients is None:
        args.expected_clients = args.min_clients

    # Logging setup
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", force=True)
    logger.info("Starting Federated Learning Server (Teacher Training)")
    logger.info(f"Config: {vars(args)}")

    strategy = None # Define strategy in outer scope
    try:
        strategy = create_server_strategy(
            min_clients=args.min_clients,
            fraction_fit=args.fraction_fit,
            fraction_evaluate=args.fraction_evaluate,
            model_name=args.model,
            num_classes=args.num_classes,
            local_epochs=args.local_epochs,
        )
        client_manager = LoggingClientManager(expected_clients=args.expected_clients)
        server_cfg = fl.server.ServerConfig(num_rounds=args.rounds)
        server_addr = f"{args.host}:{args.port}"

        logger.info(f"🌐 Flower gRPC server → {server_addr}")
        logger.info(f"🎯 Expecting {args.expected_clients} clients to connect")

        hb_stop = start_waiting_heartbeat(client_manager, args.expected_clients, interval_sec=2.0)
        
        fl.server.start_server(
            server_address=server_addr,
            config=server_cfg,
            strategy=strategy,
            client_manager=client_manager,
            grpc_max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        )
        try:
            hb_stop.set()
        except Exception:
            pass

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"FL Server failed: {e}", exc_info=True)
    finally:
        # Dynamic Distillation Trigger
        if strategy is None or not strategy.history["round"]:
             logger.warning("FL training did not complete. No results to save or distill.")
             return # Exit if FL failed

        try:
            # Save all FL results first
            strategy.save_final_results()
            strategy.save_last_model()
            logger.info("\nFL training complete. Final results saved.")
            logger.info(f"Results: {strategy.results_base_dir}")
            logger.info(f"Best round (Teacher): {strategy.best_round} | Best Val F1: {strategy.best_f1:.4f}")

            # Check if distillation is requested
            if args.run_distillation:
                logger.info("=" * 80)
                logger.info("STARTING DYNAMIC KNOWLEDGE DISTILLATION")
                logger.info("=" * 80)

                # Find the best saved teacher model
                teacher_path = os.path.join(
                    strategy.results_base_dir,
                    f"best_model_round_{strategy.best_round}.pth"
                )
                if not os.path.exists(teacher_path) or strategy.best_round == 0:
                    # Fallback to last model if best wasn't saved
                    teacher_path = os.path.join(strategy.results_base_dir, "last_global_model.pth")

                if not os.path.exists(teacher_path):
                    logger.error("Could not find a trained teacher model to distill from. Skipping.")
                    return # Exit

                logger.info(f"Using Teacher Model: {args.model} from {teacher_path}")
                logger.info(f"Training Student Model: {args.student_model}")

                # Prepare the command to call distillation/distill.py
                # Using sys.executable ensures we use the same Python (e.g., from venv)
                cmd = [
                    sys.executable,
                    "distillation/distill.py",
                    "--teacher-model", args.model,
                    "--teacher-path", teacher_path,
                    "--student-model", args.student_model,
                    "--data-dir", args.distill_data_dir,
                    "--save-dir", args.distill_save_dir,
                    "--num-classes", str(args.num_classes),
                    "--epochs", str(args.distill_epochs),
                    "--batch-size", str(args.distill_batch_size),
                ]
                
                logger.info(f"Running command: {' '.join(cmd)}")

                # Run the distillation script as a subprocess
                # We stream the output directly to the console instead of capturing
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
                
                # Log output line by line as it comes
                for line in iter(process.stdout.readline, ''):
                    logger.info(f"[Distill] {line.strip()}")
                
                process.wait() # Wait for it to finish

                if process.returncode == 0:
                    logger.info("Knowledge Distillation completed successfully.")
                else:
                    logger.error(f"Knowledge Distillation FAILED with return code {process.returncode}.")

        except Exception as e:
            logger.error(f"Error during shutdown/distillation: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()    