import os
import re

import numpy as np
import pandas as pd
import torch
from imblearn.metrics import sensitivity_score, specificity_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_avg_metrics(ground_truth, activations):
    ground_truth = _to_numpy(ground_truth)
    activations = _to_numpy(activations)
    predictions = np.argmax(activations, -1)
    mean_acc = accuracy_score(y_true=ground_truth, y_pred=predictions)
    f1_macro = f1_score(y_true=ground_truth, y_pred=predictions, average="macro")
    try:
        auc = roc_auc_score(y_true=ground_truth, y_score=activations, multi_class="ovr")
    except ValueError:
        auc = 0.0
    bac = balanced_accuracy_score(y_true=ground_truth, y_pred=predictions)
    sens_macro = sensitivity_score(y_true=ground_truth, y_pred=predictions, average="macro")
    spec_macro = specificity_score(y_true=ground_truth, y_pred=predictions, average="macro")
    return mean_acc, f1_macro, auc, bac, sens_macro, spec_macro


def compute_per_class_metrics(ground_truth, activations, num_classes=None):
    ground_truth = _to_numpy(ground_truth)
    activations = _to_numpy(activations)
    predictions = np.argmax(activations, -1)

    if num_classes is None:
        num_classes = int(activations.shape[1])
    labels = list(range(int(num_classes)))
    cm = confusion_matrix(y_true=ground_truth, y_pred=predictions, labels=labels)
    total = float(cm.sum())

    per_class = []
    for c in labels:
        tp = float(cm[c, c])
        fn = float(cm[c, :].sum() - tp)
        fp = float(cm[:, c].sum() - tp)
        tn = float(total - tp - fn - fp)

        support = int(tp + fn)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        # Use recall-only definition to keep all long-tail metrics on the same axis.
        bac = recall
        bacc = 0.5 * (recall + spec)

        y_true_bin = (ground_truth == c).astype(np.int32)
        y_score = activations[:, c]
        if y_true_bin.min() == y_true_bin.max():
            auc = float("nan")
        else:
            try:
                auc = float(roc_auc_score(y_true=y_true_bin, y_score=y_score))
            except ValueError:
                auc = float("nan")

        per_class.append(
            {
                "class_id": int(c),
                "support": support,
                "acc": float(recall),
                "recall": float(recall),
                "f1": float(f1),
                "auc": float(auc),
                "bac": float(bac),
                "bacc": float(bacc),
                "sens": float(recall),
                "spec": float(spec),
                "precision": float(precision),
            }
        )
    return per_class


def compute_macro_metric(per_class_metrics, metric_key):
    if per_class_metrics is None:
        return float("nan")
    values = []
    for row in per_class_metrics:
        val = float(row.get(metric_key, float("nan")))
        if not np.isnan(val):
            values.append(val)
    if not values:
        return float("nan")
    return float(np.mean(values))


def build_group_specs(class_names, class_counts):
    class_names = list(class_names)
    class_counts = np.asarray(class_counts, dtype=np.int64)
    order = np.argsort(-class_counts)
    group_names = ("head", "medium", "tail")
    groups = []
    for group_name, idxs in zip(group_names, np.array_split(order, len(group_names))):
        idxs = np.asarray(idxs, dtype=np.int64)
        groups.append(
            {
                "name": group_name,
                "indices": [int(i) for i in idxs.tolist()],
                "members": [class_names[int(i)] for i in idxs],
            }
        )
    return groups


def build_group_lookup(groups):
    lookup = {}
    if groups is None:
        return lookup
    for group in groups:
        for class_id in group["indices"]:
            lookup[int(class_id)] = group["name"]
    return lookup


def compute_group_metric(per_class_metrics, groups, metric_key="acc", weighted=False):
    if per_class_metrics is None or groups is None:
        return None
    metric_map = {int(row["class_id"]): float(row.get(metric_key, float("nan"))) for row in per_class_metrics}
    support_map = {int(row["class_id"]): int(row.get("support", 0)) for row in per_class_metrics}
    summary = {}
    for group in groups:
        vals = []
        weights = []
        for class_id in group["indices"]:
            val = metric_map.get(class_id, float("nan"))
            support = support_map.get(class_id, 0)
            if np.isnan(val):
                continue
            vals.append(val)
            weights.append(max(1, support))
        if not vals:
            summary[group["name"]] = float("nan")
            continue
        if weighted:
            vals_np = np.asarray(vals, dtype=np.float64)
            weights_np = np.asarray(weights, dtype=np.float64)
            summary[group["name"]] = float((vals_np * weights_np).sum() / weights_np.sum())
        else:
            summary[group["name"]] = float(np.mean(vals))
    return summary


def append_per_class_records(records_path, epoch, split_name, per_class_metrics, class_names):
    rows = []
    for row in per_class_metrics:
        item = dict(row)
        item["epoch"] = int(epoch)
        item["split"] = split_name
        item["class_name"] = str(class_names[int(row["class_id"])])
        rows.append(item)
    frame = pd.DataFrame(rows)
    header = not os.path.exists(records_path)
    frame.to_csv(records_path, mode="a", header=header, index=False)


def format_tail_lines(per_class_metrics, class_names, tail_classes, split_name):
    metric_map = {int(row["class_id"]): row for row in per_class_metrics}
    lines = []
    for class_id in tail_classes:
        row = metric_map.get(int(class_id))
        if row is None:
            continue
        lines.append(
            (
                f"{split_name}_tail {class_names[int(class_id)]}: "
                f"acc={row['acc']:.4f}, recall={row['recall']:.4f}, f1={row['f1']:.4f}, "
                f"bac={row['bac']:.4f}, bacc={row['bacc']:.4f}, n={row['support']}"
            )
        )
    return lines


def format_group_class_lines(per_class_metrics, class_names, groups, split_name):
    metric_map = {int(row["class_id"]): row for row in per_class_metrics}
    lines = []
    for group in groups:
        lines.append(f"{split_name}_{group['name']}_classes:")
        for class_id in group["indices"]:
            row = metric_map.get(int(class_id))
            if row is None:
                continue
            lines.append(
                (
                    f"  {class_names[int(class_id)]}: "
                    f"acc={row['acc']:.4f}, recall={row['recall']:.4f}, f1={row['f1']:.4f}, "
                    f"bac={row['bac']:.4f}, bacc={row['bacc']:.4f}, n={row['support']}"
                )
            )
    return lines


def update_curve_history(history, epoch, split_name, per_class_metrics, groups):
    history[split_name]["epoch"].append(int(epoch))
    history[split_name]["acc"].append(compute_group_metric(per_class_metrics, groups, metric_key="acc"))
    history[split_name]["bac"].append(compute_group_metric(per_class_metrics, groups, metric_key="bac"))


def plot_group_curves(history, out_path, split_name):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = history[split_name]["epoch"]
    if not epochs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=140)
    for ax, metric_key, title in zip(axes, ("acc", "bac"), ("Accuracy", "Balanced Accuracy")):
        for group_name in ("head", "medium", "tail"):
            values = [item.get(group_name, float("nan")) for item in history[split_name][metric_key]]
            ax.plot(epochs, values, label=group_name)
        ax.set_title(f"{split_name.upper()} {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_key.upper())
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def update_loss_history(history, epoch, loss_value):
    history["train"]["epoch"].append(int(epoch))
    history["train"]["loss"].append(float(loss_value))


def plot_loss_curve(history, out_path, title="Train Loss"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = history["train"]["epoch"]
    losses = history["train"]["loss"]
    if not epochs or not losses:
        return

    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    ax.plot(epochs, losses, color="#d62728", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_loss_curve_from_log(log_path, out_path, title="Train Loss"):
    if not os.path.isfile(log_path):
        return False

    pattern = re.compile(r"Epoch\s+(\d+)(?:/\d+)?\s+loss=([0-9eE+\-.]+)")
    history = {"train": {"epoch": [], "loss": []}}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match is None:
                continue
            history["train"]["epoch"].append(int(match.group(1)))
            history["train"]["loss"].append(float(match.group(2)))

    if not history["train"]["epoch"]:
        return False
    plot_loss_curve(history, out_path, title=title)
    return True
