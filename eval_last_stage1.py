import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.dataset import ISICDataset
from data.transforms import Transforms
from models import ResNetBackbone
from utils.checkpoint_utils import load_state_dict_flexible
from utils.metrics import (
    build_group_lookup,
    build_group_specs,
    compute_avg_metrics,
    compute_macro_metric,
    compute_group_metric,
    compute_per_class_metrics,
    format_group_class_lines,
    format_tail_lines,
    plot_loss_curve_from_log,
)
from visualize_embeddings import save_tsne_plot


class SingleTransformDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        return self.transform(img), label


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_class_counts(csv_file):
    frame = pd.read_csv(csv_file)
    labels = frame.iloc[:, 1:].replace(
        {
            True: 1,
            False: 0,
            "True": 1,
            "False": 0,
            "true": 1,
            "false": 0,
        }
    ).astype(int)
    return labels.sum(axis=0).to_numpy(dtype=np.int64), list(frame.columns[1:])


def evaluate(encoder, classifier, loader, device, num_classes):
    encoder.eval()
    classifier.eval()
    ground_truth = []
    activations = []
    features = []
    with torch.no_grad():
        for image, label in loader:
            image = image.to(device)
            label = label.to(device)
            feat = encoder(image)
            logits = classifier(feat)
            probs = torch.softmax(logits, dim=1)
            features.append(feat.detach().cpu())
            ground_truth.append(label)
            activations.append(probs)
    ground_truth = torch.cat(ground_truth, dim=0)
    activations = torch.cat(activations, dim=0)
    features = torch.cat(features, dim=0)
    avg_metrics = compute_avg_metrics(ground_truth, activations)
    per_class_metrics = compute_per_class_metrics(ground_truth, activations, num_classes=num_classes)
    return avg_metrics, per_class_metrics, features, ground_truth.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--csv_file_train", required=True)
    parser.add_argument("--csv_file_val", required=True)
    parser.add_argument("--csv_file_test", required=True)
    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--classifier_ckpt", required=True)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    class_counts, class_names = build_class_counts(args.csv_file_train)
    groups = build_group_specs(class_names, class_counts)
    group_lookup = build_group_lookup(groups)
    _, _, tail_classes = np.array_split(np.argsort(-class_counts), 3)
    tail_classes = [int(x) for x in tail_classes.tolist()]

    transforms = Transforms(args.image_size)
    val_base = ISICDataset(args.data_path, args.csv_file_val, transform=None)
    test_base = ISICDataset(args.data_path, args.csv_file_test, transform=None)

    val_loader = DataLoader(
        SingleTransformDataset(val_base, transforms.test_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        SingleTransformDataset(test_base, transforms.test_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    encoder = ResNetBackbone(args.backbone, pretrained=False).to(device)
    classifier = nn.Linear(encoder.feat_dim, len(class_names)).to(device)
    load_state_dict_flexible(encoder, args.encoder_ckpt)
    load_state_dict_flexible(classifier, args.classifier_ckpt)

    val_metrics, val_per_class, val_features, val_labels = evaluate(encoder, classifier, val_loader, device, len(class_names))
    test_metrics, test_per_class, test_features, test_labels = evaluate(
        encoder, classifier, test_loader, device, len(class_names)
    )

    val_group_acc = compute_group_metric(val_per_class, groups, metric_key="acc")
    val_group_bac = compute_group_metric(val_per_class, groups, metric_key="bac")
    val_group_bacc = compute_group_metric(val_per_class, groups, metric_key="bacc")
    test_group_acc = compute_group_metric(test_per_class, groups, metric_key="acc")
    test_group_bac = compute_group_metric(test_per_class, groups, metric_key="bac")
    test_group_bacc = compute_group_metric(test_per_class, groups, metric_key="bacc")
    val_bacc = compute_macro_metric(val_per_class, metric_key="bacc")
    test_bacc = compute_macro_metric(test_per_class, metric_key="bacc")

    output_dir = args.output_dir or os.path.join(os.path.dirname(args.encoder_ckpt), "offline_eval_stage1")
    os.makedirs(output_dir, exist_ok=True)
    run_dir = os.path.dirname(args.encoder_ckpt)
    run_name = os.path.basename(run_dir)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    inferred_log_path = os.path.join(repo_root, "log", "tpcsd", f"{run_name}.log")

    rows = []
    for split_name, metrics in (("val", val_per_class), ("test", test_per_class)):
        for row in metrics:
            item = dict(row)
            item["split"] = split_name
            item["class_name"] = class_names[int(row["class_id"])]
            item["group_name"] = group_lookup.get(int(row["class_id"]), "unknown")
            rows.append(item)
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "per_class_metrics.csv"), index=False)

    summary = {
        "dataset": args.dataset,
        "encoder_ckpt": args.encoder_ckpt,
        "classifier_ckpt": args.classifier_ckpt,
        "val": {
            "acc": float(val_metrics[0]),
            "f1": float(val_metrics[1]),
            "auc": float(val_metrics[2]),
            "bac": float(val_metrics[3]),
            "bacc": float(val_bacc),
            "sens": float(val_metrics[4]),
            "spec": float(val_metrics[5]),
            "group_acc": val_group_acc,
            "group_bac": val_group_bac,
            "group_bacc": val_group_bacc,
        },
        "test": {
            "acc": float(test_metrics[0]),
            "f1": float(test_metrics[1]),
            "auc": float(test_metrics[2]),
            "bac": float(test_metrics[3]),
            "bacc": float(test_bacc),
            "sens": float(test_metrics[4]),
            "spec": float(test_metrics[5]),
            "group_acc": test_group_acc,
            "group_bac": test_group_bac,
            "group_bacc": test_group_bacc,
        },
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    torch.save({"features": val_features, "labels": val_labels}, os.path.join(output_dir, "val_embeddings.pt"))
    torch.save({"features": test_features, "labels": test_labels}, os.path.join(output_dir, "test_embeddings.pt"))
    save_tsne_plot(
        features=val_features,
        labels=val_labels,
        class_names=class_names,
        out_path=os.path.join(output_dir, "val_tsne.png"),
        title=f"{args.dataset} Stage1 VAL t-SNE",
        max_points=4000,
        seed=args.seed,
    )
    save_tsne_plot(
        features=test_features,
        labels=test_labels,
        class_names=class_names,
        out_path=os.path.join(output_dir, "test_tsne.png"),
        title=f"{args.dataset} Stage1 TEST t-SNE",
        max_points=4000,
        seed=args.seed,
    )
    plot_loss_curve_from_log(
        inferred_log_path,
        os.path.join(output_dir, "train_loss_curve.png"),
        title="Stage1 Train Loss",
    )

    print(
        "val: "
        f"acc={val_metrics[0]:.6f}, f1={val_metrics[1]:.6f}, auc={val_metrics[2]:.6f}, "
        f"bac={val_metrics[3]:.6f}, bacc={val_bacc:.6f}"
    )
    print(
        "test: "
        f"acc={test_metrics[0]:.6f}, f1={test_metrics[1]:.6f}, auc={test_metrics[2]:.6f}, "
        f"bac={test_metrics[3]:.6f}, bacc={test_bacc:.6f}"
    )
    print(f"val_group_acc: {val_group_acc}")
    print(f"val_group_bac: {val_group_bac}")
    print(f"val_group_bacc: {val_group_bacc}")
    print(f"test_group_acc: {test_group_acc}")
    print(f"test_group_bac: {test_group_bac}")
    print(f"test_group_bacc: {test_group_bacc}")
    for line in format_group_class_lines(val_per_class, class_names, groups, "val"):
        print(line)
    for line in format_group_class_lines(test_per_class, class_names, groups, "test"):
        print(line)
    for line in format_tail_lines(val_per_class, class_names, tail_classes, "val"):
        print(line)
    for line in format_tail_lines(test_per_class, class_names, tail_classes, "test"):
        print(line)
    if os.path.isfile(inferred_log_path):
        print(f"train_loss_curve: {os.path.join(output_dir, 'train_loss_curve.png')}")
    print(f"val_tsne: {os.path.join(output_dir, 'val_tsne.png')}")
    print(f"test_tsne: {os.path.join(output_dir, 'test_tsne.png')}")
    print(f"saved to: {output_dir}")


if __name__ == "__main__":
    main()
