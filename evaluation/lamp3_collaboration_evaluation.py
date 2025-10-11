"""Evaluation script for the collaborative model on the LaMP3 dataset.

This module evaluates the performance of the collaborative inference
pipeline (large language model + small language model + fusion/weight
network) on the LaMP3 rating prediction task.

The implementation mirrors the evaluation helpers that live under the
``evaluation/`` directory for other LaMP tasks so that it fits neatly
into the existing experimentation workflow.
"""
from __future__ import annotations

import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.dataset import RatingDataset
from models.collaborative_inference import CollaborativeInference
from models.model import LargeModelLoader, TinyModelLoader
from models.tokenizer import Tokenizer
from models.weight_network import WeightNetwork
from utils.config_loader import load_config


@dataclass
class EvaluationBatch:
    """Container used to keep a batch of tokenised samples together."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    raw_labels: List[str]


class Lamp3CollaborativeEvaluator:
    """Evaluate the collaborative model on the LaMP3 rating task."""

    def __init__(self, batch_size: int | None = None) -> None:
        self.config = load_config()
        self.tokenizer = Tokenizer.load_tokenizer()

        os.environ["CUDA_VISIBLE_DEVICES"] = self.config["base"].get("device_id", "0")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = batch_size or self.config["combModel_training"].get("batch_size", 4)

        self._load_models()
        self._load_weight_network()

        dataset_split = RatingDataset.format_data_combModel()
        self.test_dataset = dataset_split["test"]

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        self.collaborative_inference = CollaborativeInference(
            self.large_model,
            self.tiny_model,
            self.weight_network,
            self.tokenizer,
            self.device,
        )

    # ------------------------------------------------------------------
    # Loading utilities
    # ------------------------------------------------------------------
    def _load_models(self) -> None:
        """Load the fine-tuned large and tiny models for inference."""

        self.large_model = LargeModelLoader.load_finetuned_model()
        self.large_model.eval()

        self.tiny_model = TinyModelLoader.load_finetuned_model()
        self.tiny_model.eval()

    def _load_weight_network(self) -> None:
        """Load the trained weight network from the configured checkpoint."""

        checkpoint_path = self.config["base"].get("fusion_network_path")
        if not checkpoint_path:
            raise ValueError(
                "`fusion_network_path` is not set in config/config.yaml. "
                "Please provide the path to the trained weight network checkpoint."
            )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Could not find fusion network checkpoint at: {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        tiny_model_id = self.config["base"].get("tiny_model_id", "")
        ctx_dim = 1024 if tiny_model_id == "Qwen/Qwen1.5-0.5B-Chat" else 2048

        self.weight_network = WeightNetwork(
            vocab_size=len(self.tokenizer),
            hidden_dims=[512, 512],
            ctx_dim=ctx_dim,
        )

        if "model_state" not in checkpoint:
            raise KeyError(
                "The checkpoint file does not contain `model_state`. "
                "Ensure you are providing a checkpoint produced by the trainer."
            )

        self.weight_network.load_state_dict(checkpoint["model_state"])
        self.weight_network.to(self.device)
        self.weight_network.eval()

    # ------------------------------------------------------------------
    # Data utilities
    # ------------------------------------------------------------------
    def collate_fn(self, batch: List[Dict[str, object]]) -> EvaluationBatch:
        """Tokenise batch samples into tensors ready for the models."""

        texts = [item["text"] for item in batch]
        titles = [item["title"] for item in batch]

        text_encodings = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt",
            add_special_tokens=False,
            padding_side="left",
        )

        title_encodings = self.tokenizer(
            titles,
            padding="max_length",
            truncation=True,
            max_length=self.config["base"].get("max_length", 35),
            return_tensors="pt",
        )

        labels = title_encodings["input_ids"].clone()
        eos_token_id = self.tokenizer.eos_token_id
        for i in range(labels.size(0)):
            if labels[i, -1] != eos_token_id:
                labels[i, -1] = eos_token_id

        return EvaluationBatch(
            input_ids=text_encodings["input_ids"].to(self.device),
            attention_mask=text_encodings["attention_mask"].to(self.device),
            labels=labels.to(self.device),
            raw_labels=titles,
        )

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def calculate_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the cross-entropy loss ignoring padding tokens."""

        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)

        return torch.nn.functional.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=self.tokenizer.pad_token_id,
        )

    @staticmethod
    def _normalise_rating(text: str) -> str:
        """Extract a single rating token (1-5) from the decoded text."""

        cleaned = text.strip()
        match = re.search(r"[1-5]", cleaned)
        if match:
            return match.group(0)
        return "invalid"

    def _compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, object]:
        """Compute accuracy and macro-F1 along with per-class metrics."""

        metrics: Dict[str, float] = {}
        correct = sum(pred == ref for pred, ref in zip(predictions, references))
        total = len(references)
        metrics["accuracy"] = correct / total if total > 0 else 0.0

        per_class_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

        for pred, ref in zip(predictions, references):
            if pred == ref:
                per_class_stats[ref]["tp"] += 1
            else:
                per_class_stats[pred]["fp"] += 1
                per_class_stats[ref]["fn"] += 1

        per_class_metrics: Dict[str, Dict[str, float]] = {}
        f1_sum = 0.0
        class_count = 0

        for label in sorted(set(references)):
            stats = per_class_stats[label]
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            per_class_metrics[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

            f1_sum += f1
            class_count += 1

        metrics["macro_f1"] = f1_sum / class_count if class_count > 0 else 0.0
        metrics["per_class"] = per_class_metrics
        metrics["invalid_rate"] = sum(pred == "invalid" for pred in predictions) / total if total > 0 else 0.0

        return metrics

    def evaluate(self) -> Dict[str, object]:
        """Run collaborative inference on the test split and compute metrics."""

        self.large_model.eval()
        self.tiny_model.eval()
        self.weight_network.eval()

        total_loss = 0.0
        batch_count = 0

        all_predictions: List[str] = []
        all_references: List[str] = []

        with torch.no_grad():
            for batch in self.test_loader:
                if isinstance(batch, EvaluationBatch):
                    eval_batch = batch
                else:
                    eval_batch = EvaluationBatch(**batch)

                outputs = self.collaborative_inference.forward(
                    eval_batch.input_ids,
                    eval_batch.attention_mask,
                    use_past=False,
                )

                logits = outputs["combined_logits"]
                loss = self.calculate_loss(logits, eval_batch.labels)
                total_loss += loss.item()
                batch_count += 1

                generated = outputs["generated_tokens"]
                decoded_predictions = self.tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                )

                normalised_predictions = [self._normalise_rating(text) for text in decoded_predictions]
                normalised_references = [self._normalise_rating(label) for label in eval_batch.raw_labels]

                all_predictions.extend(normalised_predictions)
                all_references.extend(normalised_references)

        metrics = self._compute_metrics(all_predictions, all_references)
        metrics["loss"] = total_loss / batch_count if batch_count > 0 else 0.0

        self._pretty_print(metrics)

        return metrics

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def _pretty_print(self, metrics: Dict[str, object]) -> None:
        """Nicely format the evaluation metrics for console output."""

        print("\n===== LaMP3 Collaborative Evaluation =====")
        print(f"Test Loss     : {metrics['loss']:.4f}")
        print(f"Accuracy      : {metrics['accuracy']:.4f}")
        print(f"Macro F1      : {metrics['macro_f1']:.4f}")
        print(f"Invalid Rate  : {metrics['invalid_rate']:.4f}")
        print("-----------------------------------------")
        print("Per-class metrics:")
        for label, scores in metrics["per_class"].items():
            print(
                f"  Rating {label}: precision={scores['precision']:.4f}, "
                f"recall={scores['recall']:.4f}, f1={scores['f1']:.4f}"
            )
        print("=========================================\n")


# ----------------------------------------------------------------------
# Entrypoint utilities
# ----------------------------------------------------------------------

def set_random_seed(seed: int = 1057) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible evaluation."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    set_random_seed()
    evaluator = Lamp3CollaborativeEvaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
