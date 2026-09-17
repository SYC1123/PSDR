import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.dataset import ISICDataset
from data.transforms import Transforms
from models import Projector, ResNetBackbone
from utils.checkpoint_utils import load_state_dict_flexible
from utils.losses import active_prototype_mask, ensure_3d_prototypes, prototype_uniformity_loss, reduce_prototypes_mean
from utils.metrics import build_group_specs
from visualize_embeddings import save_named_point_tsne, save_prototype_mean_tsne, save_similarity_heatmap


class SingleTransformDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        return self.transform(image), label


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_projector_dims(projector_ckpt, feat_dim, default_proj_dim, default_hidden_dim):
    if not projector_ckpt or not os.path.isfile(projector_ckpt):
        return False, feat_dim, 0
    state = torch.load(projector_ckpt, map_location="cpu")
    if "net.weight" in state:
        return True, int(state["net.weight"].shape[0]), 0
    if "net.0.weight" in state and "net.2.weight" in state:
        hidden_dim = int(state["net.0.weight"].shape[0])
        proj_dim = int(state["net.2.weight"].shape[0])
        return True, proj_dim, hidden_dim
    return True, default_proj_dim, default_hidden_dim


def extract_features(encoder, projector, loader, device):
    encoder.eval()
    if projector is not None:
        projector.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for image, label in loader:
            image = image.to(device)
            feat = encoder(image)
            if projector is not None:
                feat = projector(feat)
            feats.append(feat.detach().cpu())
            labels.append(label.detach().cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def compute_class_stats(feats, labels, num_classes):
    means = torch.zeros(num_classes, feats.shape[1], dtype=feats.dtype)
    stds = torch.zeros(num_classes, feats.shape[1], dtype=feats.dtype)
    for class_id in range(num_classes):
        idx = labels == class_id
        if idx.sum() == 0:
            continue
        class_feats = feats[idx]
        means[class_id] = class_feats.mean(dim=0)
        stds[class_id] = class_feats.std(dim=0, unbiased=False)
    return means, stds


def safe_group_mean(rows, indices, key):
    vals = [float(rows[int(i)][key]) for i in indices if not np.isnan(rows[int(i)][key])]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def plot_alignment(rows, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    class_names = [row["class_name"] for row in rows]
    cosine_vals = [row["proto_mean_cosine"] for row in rows]
    l2_vals = [row["proto_mean_l2"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=140)
    axes[0].bar(class_names, cosine_vals)
    axes[0].set_title("Prototype vs Class Mean Cosine")
    axes[0].set_ylabel("Cosine")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(class_names, l2_vals)
    axes[1].set_title("Prototype vs Class Mean L2")
    axes[1].set_ylabel("L2 Distance")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def compute_prototype_cosine_matrix(prototypes):
    prototypes_n = F.normalize(prototypes, dim=1)
    return torch.matmul(prototypes_n, prototypes_n.t())


def summarize_off_diagonal_similarity(cosine_matrix, class_names, threshold=0.8):
    matrix = cosine_matrix.detach().cpu().numpy()
    num_classes = matrix.shape[0]
    off_diag = matrix[~np.eye(num_classes, dtype=bool)]
    rows = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            rows.append(
                {
                    "class_i": class_names[i],
                    "class_j": class_names[j],
                    "cosine_similarity": float(matrix[i, j]),
                    "above_threshold": bool(matrix[i, j] > threshold),
                }
            )
    rows = sorted(rows, key=lambda x: x["cosine_similarity"], reverse=True)
    return {
        "mean_off_diagonal_cosine": float(off_diag.mean()) if off_diag.size > 0 else float("nan"),
        "max_off_diagonal_cosine": float(off_diag.max()) if off_diag.size > 0 else float("nan"),
        "min_off_diagonal_cosine": float(off_diag.min()) if off_diag.size > 0 else float("nan"),
        "high_similarity_pairs": [row for row in rows if row["above_threshold"]],
        "top_pairs": rows[: min(10, len(rows))],
        "all_pairs": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--csv_file_train", required=True)
    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--prototype_ckpt", required=True)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--projector_ckpt", default="")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--proj_hidden_dim", type=int, default=0)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    transforms = Transforms(args.image_size)
    train_base = ISICDataset(args.data_path, args.csv_file_train, transform=None)
    train_loader = DataLoader(
        SingleTransformDataset(train_base, transforms.test_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    encoder = ResNetBackbone(args.backbone, pretrained=False).to(device)
    load_state_dict_flexible(encoder, args.encoder_ckpt)

    use_projector, inferred_proj_dim, inferred_hidden_dim = infer_projector_dims(
        args.projector_ckpt, encoder.feat_dim, args.proj_dim, args.proj_hidden_dim
    )
    projector = None
    feature_dim = encoder.feat_dim
    if use_projector:
        projector = Projector(encoder.feat_dim, proj_dim=inferred_proj_dim, hidden_dim=inferred_hidden_dim).to(device)
        load_state_dict_flexible(projector, args.projector_ckpt)
        feature_dim = inferred_proj_dim

    prototypes = ensure_3d_prototypes(torch.load(args.prototype_ckpt, map_location="cpu").float())
    if prototypes.shape[-1] != feature_dim:
        raise ValueError(
            f"prototype dim mismatch: checkpoint={prototypes.shape[-1]}, expected={feature_dim}"
        )
    proto_active_mask = active_prototype_mask(prototypes)
    prototype_mean = reduce_prototypes_mean(prototypes, proto_active_mask)

    feats, labels = extract_features(encoder, projector, train_loader, device)
    means, stds = compute_class_stats(feats, labels, train_base.n_class)

    prototypes_n = F.normalize(prototype_mean, dim=1)
    means_n = F.normalize(means, dim=1)
    prototype_cosine_matrix = compute_prototype_cosine_matrix(prototype_mean)
    prototype_cosine_summary = summarize_off_diagonal_similarity(
        prototype_cosine_matrix, list(train_base.class_names), threshold=0.8
    )
    flat_active_proto = prototypes[proto_active_mask]
    prototype_uniformity = float(prototype_uniformity_loss(flat_active_proto, t=2.0).item()) if flat_active_proto.numel() > 0 else 0.0

    rows = []
    for class_id, class_name in enumerate(train_base.class_names):
        class_mask = labels == class_id
        class_feats = feats[class_mask]
        proto = prototype_mean[class_id]
        mean = means[class_id]
        proto_n = prototypes_n[class_id]
        mean_n = means_n[class_id]

        proto_mean_cosine = float(torch.sum(proto_n * mean_n).item())
        proto_mean_l2 = float(torch.norm(proto - mean, p=2).item())

        if class_feats.shape[0] > 0:
            class_feats_n = F.normalize(class_feats, dim=1)
            real_to_proto_cos = torch.matmul(class_feats_n, proto_n.unsqueeze(1)).squeeze(1)
            real_to_mean_cos = torch.matmul(class_feats_n, mean_n.unsqueeze(1)).squeeze(1)
            real_to_proto_l2 = torch.norm(class_feats - proto.unsqueeze(0), dim=1, p=2)
            real_to_mean_l2 = torch.norm(class_feats - mean.unsqueeze(0), dim=1, p=2)
            feat_var_mean = float(class_feats.std(dim=0, unbiased=False).mean().item())
        else:
            real_to_proto_cos = torch.tensor([])
            real_to_mean_cos = torch.tensor([])
            real_to_proto_l2 = torch.tensor([])
            real_to_mean_l2 = torch.tensor([])
            feat_var_mean = float("nan")

        rows.append(
            {
                "class_id": int(class_id),
                "class_name": str(class_name),
                "support": int(class_mask.sum().item()),
                "active_proto_count": int(proto_active_mask[class_id].sum().item()),
                "proto_mean_cosine": proto_mean_cosine,
                "proto_mean_l2": proto_mean_l2,
                "real_to_proto_cos_mean": float(real_to_proto_cos.mean().item()) if real_to_proto_cos.numel() > 0 else float("nan"),
                "real_to_mean_cos_mean": float(real_to_mean_cos.mean().item()) if real_to_mean_cos.numel() > 0 else float("nan"),
                "real_to_proto_l2_mean": float(real_to_proto_l2.mean().item()) if real_to_proto_l2.numel() > 0 else float("nan"),
                "real_to_mean_l2_mean": float(real_to_mean_l2.mean().item()) if real_to_mean_l2.numel() > 0 else float("nan"),
                "feat_var_mean": feat_var_mean,
                "class_std_mean": float(stds[class_id].mean().item()),
            }
        )

    class_counts = np.asarray([row["support"] for row in rows], dtype=np.int64)
    groups = build_group_specs(train_base.class_names, class_counts)
    group_summary = {}
    for group in groups:
        group_summary[group["name"]] = {
            "proto_mean_cosine": safe_group_mean(rows, group["indices"], "proto_mean_cosine"),
            "proto_mean_l2": safe_group_mean(rows, group["indices"], "proto_mean_l2"),
            "real_to_proto_cos_mean": safe_group_mean(rows, group["indices"], "real_to_proto_cos_mean"),
            "real_to_mean_cos_mean": safe_group_mean(rows, group["indices"], "real_to_mean_cos_mean"),
            "real_to_proto_l2_mean": safe_group_mean(rows, group["indices"], "real_to_proto_l2_mean"),
            "real_to_mean_l2_mean": safe_group_mean(rows, group["indices"], "real_to_mean_l2_mean"),
        }

    output_dir = args.output_dir or os.path.join(os.path.dirname(args.prototype_ckpt), "offline_prototype_alignment")
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "prototype_alignment.csv"), index=False)
    pd.DataFrame(
        prototype_cosine_matrix.detach().cpu().numpy(),
        index=list(train_base.class_names),
        columns=list(train_base.class_names),
    ).to_csv(os.path.join(output_dir, "prototype_cosine_similarity.csv"))
    pd.DataFrame(prototype_cosine_summary["all_pairs"]).to_csv(
        os.path.join(output_dir, "prototype_similarity_pairs.csv"), index=False
    )
    torch.save(
        {
            "features": feats,
            "labels": labels,
            "class_means": means,
            "class_stds": stds,
            "prototypes": prototypes,
            "prototype_mean": prototype_mean,
            "prototype_cosine_matrix": prototype_cosine_matrix,
        },
        os.path.join(output_dir, "feature_stats.pt"),
    )
    plot_alignment(rows, os.path.join(output_dir, "prototype_alignment.png"))
    save_similarity_heatmap(
        prototype_cosine_matrix,
        train_base.class_names,
        os.path.join(output_dir, "prototype_cosine_similarity_heatmap.png"),
        title=f"{args.dataset} Prototype Cosine Similarity",
    )
    save_named_point_tsne(
        features=prototype_mean,
        class_names=train_base.class_names,
        out_path=os.path.join(output_dir, "prototype_tsne.png"),
        title=f"{args.dataset} Prototype t-SNE",
        seed=args.seed,
    )
    save_prototype_mean_tsne(
        prototypes=prototype_mean,
        means=means,
        class_names=train_base.class_names,
        out_path=os.path.join(output_dir, "prototype_mean_tsne.png"),
        title=f"{args.dataset} Prototype vs Class Mean t-SNE",
        seed=args.seed,
    )

    summary = {
        "dataset": args.dataset,
        "encoder_ckpt": args.encoder_ckpt,
        "projector_ckpt": args.projector_ckpt,
        "prototype_ckpt": args.prototype_ckpt,
        "mean_proto_mean_cosine": float(np.mean([row["proto_mean_cosine"] for row in rows])),
        "min_proto_mean_cosine": float(np.min([row["proto_mean_cosine"] for row in rows])),
        "mean_proto_mean_l2": float(np.mean([row["proto_mean_l2"] for row in rows])),
        "prototype_uniformity_loss_t2": prototype_uniformity,
        "prototype_cosine_summary": {
            "mean_off_diagonal_cosine": prototype_cosine_summary["mean_off_diagonal_cosine"],
            "max_off_diagonal_cosine": prototype_cosine_summary["max_off_diagonal_cosine"],
            "min_off_diagonal_cosine": prototype_cosine_summary["min_off_diagonal_cosine"],
            "high_similarity_pairs": prototype_cosine_summary["high_similarity_pairs"],
            "top_pairs": prototype_cosine_summary["top_pairs"],
        },
        "group_summary": group_summary,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"mean proto-mean cosine: {summary['mean_proto_mean_cosine']:.6f}")
    print(f"min proto-mean cosine: {summary['min_proto_mean_cosine']:.6f}")
    print(f"mean proto-mean l2: {summary['mean_proto_mean_l2']:.6f}")
    print(f"prototype uniformity loss (t=2.0): {summary['prototype_uniformity_loss_t2']:.6f}")
    print(
        "prototype off-diagonal cosine: "
        f"mean={prototype_cosine_summary['mean_off_diagonal_cosine']:.6f}, "
        f"max={prototype_cosine_summary['max_off_diagonal_cosine']:.6f}, "
        f"min={prototype_cosine_summary['min_off_diagonal_cosine']:.6f}"
    )
    if prototype_cosine_summary["high_similarity_pairs"]:
        print("high-similarity prototype pairs (>0.8):")
        for item in prototype_cosine_summary["high_similarity_pairs"]:
            print(f"  {item['class_i']} vs {item['class_j']}: {item['cosine_similarity']:.6f}")
    else:
        print("high-similarity prototype pairs (>0.8): none")
    for group_name, metrics in group_summary.items():
        print(
            f"{group_name}: "
            f"cos={metrics['proto_mean_cosine']:.6f}, "
            f"l2={metrics['proto_mean_l2']:.6f}, "
            f"real->proto cos={metrics['real_to_proto_cos_mean']:.6f}, "
            f"real->mean cos={metrics['real_to_mean_cos_mean']:.6f}"
        )
    print(f"prototype_tsne: {os.path.join(output_dir, 'prototype_tsne.png')}")
    print(f"prototype_mean_tsne: {os.path.join(output_dir, 'prototype_mean_tsne.png')}")
    print(f"prototype_cosine_similarity_heatmap: {os.path.join(output_dir, 'prototype_cosine_similarity_heatmap.png')}")
    print(f"saved to: {output_dir}")


if __name__ == "__main__":
    main()
