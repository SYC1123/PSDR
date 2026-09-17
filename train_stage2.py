import argparse
import csv
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data.dataset import ISICDataset
from data.transforms import Transforms
from models import CosineClassifier, Projector, ResNetBackbone
from utils.csv_utils import label_frame_to_int
from utils.losses import active_prototype_mask, ensure_3d_prototypes, reduce_prototypes_mean
from utils.metrics import (
    append_per_class_records,
    build_group_specs,
    compute_avg_metrics,
    compute_macro_metric,
    compute_per_class_metrics,
    format_tail_lines,
    plot_loss_curve,
    plot_group_curves,
    update_curve_history,
    update_loss_history,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def load_state_dict_compat(model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {ckpt_path}")

    normalized_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            normalized_state_dict[key[len("module."):]] = value
        else:
            normalized_state_dict[key] = value
    model.load_state_dict(normalized_state_dict)


class SingleTransformDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        img = self.transform(img)
        return img, label


def build_class_weights(train_csv, device):
    file = pd.read_csv(train_csv)
    labels = label_frame_to_int(file.iloc[:, 1:])
    counts = labels.sum(axis=0).to_numpy(dtype=np.float32)
    counts[counts == 0] = 1.0
    weights = counts.max() / counts
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights, counts


def extract_backbone_features(encoder, loader, device):
    encoder.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for img, label in loader:
            img = img.to(device)
            feat = encoder(img)
            feats.append(feat.detach().cpu())
            labels.append(label.detach().cpu())
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels


def project_feature_tensor(projector, feats, device, batch_size):
    projector.eval()
    projected = []
    with torch.no_grad():
        for start in range(0, feats.shape[0], batch_size):
            end = min(start + batch_size, feats.shape[0])
            batch = feats[start:end].to(device)
            projected.append(projector(batch).detach().cpu())
    return torch.cat(projected, dim=0)


def compute_class_stats(feats, labels, num_classes):
    mu = torch.zeros(num_classes, feats.shape[1], dtype=feats.dtype)
    sigma = torch.zeros(num_classes, feats.shape[1], dtype=feats.dtype)
    for c in range(num_classes):
        idx = labels == c
        if idx.sum() == 0:
            continue
        f = feats[idx]
        mu[c] = f.mean(dim=0)
        sigma[c] = f.std(dim=0, unbiased=False)
    return mu, sigma


def compute_shared_covariance(feats, labels, mu, jitter=1e-4):
    if feats.numel() == 0:
        return torch.eye(mu.shape[1], dtype=mu.dtype)

    centered = feats - mu.index_select(0, labels.long())
    denom = max(int(centered.shape[0]) - 1, 1)
    cov = centered.t().matmul(centered) / float(denom)
    cov = 0.5 * (cov + cov.t())
    cov = cov + torch.eye(cov.shape[0], dtype=cov.dtype) * float(jitter)
    return cov


def compute_classifier_per_class_acc(classifier, feats, labels, num_classes, device):
    classifier.eval()
    with torch.no_grad():
        x = feats.to(device)
        y = labels.to(device)
        logits = classifier(x)
        pred = logits.argmax(dim=1)
    acc = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        idx = y == c
        if idx.sum() == 0:
            acc[c] = 0.0
        else:
            acc[c] = (pred[idx] == c).float().mean().item()
    return acc


def compute_prototype_per_class_acc(feats, labels, prototypes, num_classes, device):
    with torch.no_grad():
        x = F.normalize(feats.to(device), p=2, dim=1)
        proto = F.normalize(ensure_3d_prototypes(prototypes).to(device), p=2, dim=2)
        scores = torch.einsum("bd,ckd->bck", x, proto).max(dim=2).values
        pred = scores.argmax(dim=1).cpu()
    acc = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        idx = labels == c
        if idx.sum() == 0:
            acc[c] = 0.0
        else:
            acc[c] = (pred[idx] == c).float().mean().item()
    return acc


def select_hardest_classes(per_class_acc, hardest_k, hardest_fraction):
    num_classes = int(per_class_acc.shape[0])
    if hardest_k > 0:
        k = min(num_classes, int(hardest_k))
    else:
        hardest_fraction = float(hardest_fraction)
        k = int(np.ceil(num_classes * hardest_fraction))
        k = max(1, min(num_classes, k))
    order = torch.argsort(per_class_acc, descending=False)
    return order[:k]


def allocate_hardest_virtual_counts(per_class_acc, hardest_indices, alpha, total):
    counts = torch.zeros_like(per_class_acc, dtype=torch.long)
    weights = torch.zeros_like(per_class_acc, dtype=torch.float32)
    total = int(total)
    if total <= 0 or hardest_indices.numel() == 0:
        return counts, weights

    hardest_acc = per_class_acc[hardest_indices]
    scores = torch.exp(float(alpha) * (1.0 - hardest_acc))
    hardest_weights = scores / scores.sum()
    hardest_counts = torch.floor(hardest_weights * float(total)).to(torch.long)
    diff = int(total - hardest_counts.sum().item())
    if diff > 0:
        order = torch.argsort(hardest_weights, descending=True)
        for i in range(diff):
            hardest_counts[order[i % len(order)]] += 1

    counts[hardest_indices] = hardest_counts
    weights[hardest_indices] = hardest_weights
    return counts, weights


def sample_virtual_features(mu, shared_cov, counts, delta=0.01, cov_scale_factor=1.0, active_mask=None):
    device = mu.device
    feats = []
    labels = []
    centers = ensure_3d_prototypes(mu)
    num_classes, num_proto, _ = centers.shape
    if active_mask is None:
        active_mask = active_prototype_mask(centers)
    base_cov = shared_cov * float(cov_scale_factor)
    base_cov = base_cov + torch.eye(shared_cov.shape[0], device=device, dtype=shared_cov.dtype) * (float(delta) ** 2)
    for c in range(num_classes):
        k = int(counts[c].item())
        if k <= 0:
            continue
        class_active = torch.flatnonzero(active_mask[c], as_tuple=False).flatten()
        if class_active.numel() == 0:
            class_active = torch.arange(num_proto, device=device)
        per_proto = torch.full((class_active.numel(),), k // max(1, class_active.numel()), device=device, dtype=torch.long)
        remainder = int(k - per_proto.sum().item())
        for idx in range(remainder):
            per_proto[idx % class_active.numel()] += 1
        for slot_idx, slot in enumerate(class_active.tolist()):
            slot_count = int(per_proto[slot_idx].item())
            if slot_count <= 0:
                continue
            mvn = torch.distributions.MultivariateNormal(loc=centers[c, slot], covariance_matrix=base_cov)
            z = mvn.sample((slot_count,))
            feats.append(z)
            labels.append(torch.full((slot_count,), c, device=device, dtype=torch.long))
    if not feats:
        return torch.empty(0, centers.shape[-1], device=device), torch.empty(0, dtype=torch.long, device=device)
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels


def filter_virtual_features(
    virtual_feats,
    virtual_labels,
    classifier,
    class_centers,
    device,
    conf_thresh,
    center_cos_thresh,
    batch_size,
):
    total = int(virtual_labels.shape[0])
    if total == 0:
        return (
            torch.empty(0, virtual_feats.shape[1], dtype=virtual_feats.dtype),
            torch.empty(0, dtype=torch.long),
            total,
            0,
        )

    classifier.eval()
    kept_feats = []
    kept_labels = []
    norm_centers = F.normalize(class_centers.to(device), p=2, dim=1)

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            z = virtual_feats[start:end].to(device)
            y = virtual_labels[start:end].to(device)
            logits = classifier(z)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            center_cos = F.cosine_similarity(
                F.normalize(z, p=2, dim=1),
                norm_centers.index_select(0, y),
                dim=1,
            )
            keep = pred.eq(y) & (conf >= float(conf_thresh)) & (center_cos >= float(center_cos_thresh))
            if torch.any(keep):
                kept_feats.append(z[keep].detach().cpu())
                kept_labels.append(y[keep].detach().cpu())

    if not kept_feats:
        return (
            torch.empty(0, virtual_feats.shape[1], dtype=virtual_feats.dtype),
            torch.empty(0, dtype=torch.long),
            total,
            0,
        )
    feats = torch.cat(kept_feats, dim=0)
    labels = torch.cat(kept_labels, dim=0)
    return feats, labels, total, int(labels.shape[0])


def summarize_gaussian_stats(mu, shared_cov):
    mu_norms = torch.norm(mu, p=2, dim=1)
    mu_norm_mean = float(mu_norms.mean().item())
    mu_norm_std = float(mu_norms.std(unbiased=False).item())

    cov_diag = torch.diag(shared_cov)
    cov_trace = float(cov_diag.sum().item())
    cov_diag_mean = float(cov_diag.mean().item())
    cov_fro = float(torch.norm(shared_cov, p="fro").item())

    eigvals = torch.linalg.eigvalsh(shared_cov)
    eig_min = float(eigvals.min().item())
    eig_max = float(eigvals.max().item())

    cond = float("inf")
    if eig_min > 1e-12:
        cond = eig_max / eig_min

    return {
        "mu_norm_mean": mu_norm_mean,
        "mu_norm_std": mu_norm_std,
        "cov_trace": cov_trace,
        "cov_diag_mean": cov_diag_mean,
        "cov_fro": cov_fro,
        "cov_eig_min": eig_min,
        "cov_eig_max": eig_max,
        "cov_cond": cond,
    }


def gaussian_stats_delta(curr, prev):
    if prev is None:
        return None
    out = {}
    for k, v in curr.items():
        out[k] = float(v - prev[k])
    return out


def projector_anchor_loss(projected, projected_anchor, eps=1e-12):
    projected = F.normalize(projected, p=2, dim=1, eps=eps)
    projected_anchor = F.normalize(projected_anchor, p=2, dim=1, eps=eps)
    return (1.0 - F.cosine_similarity(projected, projected_anchor, dim=1)).mean()


def evaluate_classifier(classifier, feats, labels, device, num_classes):
    classifier.eval()
    with torch.no_grad():
        logits = classifier(feats.to(device))
        probs = torch.softmax(logits, dim=1).cpu()
    acc, f1, auc, bac, sens, spec = compute_avg_metrics(labels, probs)
    per_class = compute_per_class_metrics(labels, probs, num_classes=num_classes)
    bacc = compute_macro_metric(per_class, metric_key="bacc")
    return (acc, f1, auc, bac, bacc, sens, spec), per_class


def next_batch(loader, iterator):
    if loader is None:
        return None, iterator
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ISIC2019LT")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--csv_file_train", required=True)
    parser.add_argument("--csv_file_val", required=True)
    parser.add_argument("--csv_file_test", required=True)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--checkpoints", default="./checkpoints")
    parser.add_argument("--log_dir", default="./log/tpcsd")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--stage2_batch_size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--use_projector", action="store_true")
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--proj_hidden_dim", type=int, default=0)
    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--projector_ckpt", required=True)
    parser.add_argument("--prototype_ckpt", required=True)
    parser.add_argument("--lambda_mu", type=float, default=0.5)
    parser.add_argument("--delta_noise", type=float, default=0.01)
    parser.add_argument("--cov_scale_factor", type=float, default=1.0)
    parser.add_argument("--aas_alpha", type=float, default=2.0)
    parser.add_argument("--virtual_ratio", type=float, default=1.0)
    parser.add_argument("--merge_real", action="store_true")
    parser.add_argument("--train_noise_std", type=float, default=0.0)
    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument("--cosine_scale", type=float, default=16.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--projector_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hardest_k", type=int, default=3)
    parser.add_argument("--hardest_fraction", type=float, default=0.5)
    parser.add_argument("--virtual_conf_thresh", type=float, default=0.6)
    parser.add_argument("--virtual_center_cos_thresh", type=float, default=0.2)
    parser.add_argument("--anchor_weight", type=float, default=0.05)
    parser.add_argument("--virtual_loss_weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.merge_real:
        raise ValueError("Stage2 projector fine-tuning expects --merge_real so real features remain in training.")
    if args.hardest_k < 0:
        raise ValueError("hardest_k must be >= 0")
    if args.hardest_k == 0 and not (0.0 < float(args.hardest_fraction) <= 1.0):
        raise ValueError("hardest_fraction must be in (0, 1] when hardest_k is 0")

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    transforms = Transforms(args.image_size)
    train_base = ISICDataset(args.data_path, args.csv_file_train, transform=None)
    val_base = ISICDataset(args.data_path, args.csv_file_val, transform=None)
    test_base = ISICDataset(args.data_path, args.csv_file_test, transform=None)

    train_ds = SingleTransformDataset(train_base, transforms.test_transform)
    val_ds = SingleTransformDataset(val_base, transforms.test_transform)
    test_ds = SingleTransformDataset(test_base, transforms.test_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    num_classes = train_base.n_class
    class_weights, counts = build_class_weights(args.csv_file_train, device)
    if not args.use_class_weight:
        class_weights = None
    group_specs = build_group_specs(train_base.class_names, counts)
    _, _, tail_classes = np.array_split(np.argsort(-counts), 3)
    tail_classes = [int(x) for x in tail_classes.tolist()]

    if not args.use_projector:
        raise ValueError("Stage2 requires --use_projector because prototypes live in the projected feature space.")
    if not os.path.isfile(args.projector_ckpt):
        raise FileNotFoundError(f"projector_ckpt not found: {args.projector_ckpt}")
    if not os.path.isfile(args.prototype_ckpt):
        raise FileNotFoundError(f"prototype_ckpt not found: {args.prototype_ckpt}")

    encoder = ResNetBackbone(args.backbone, pretrained=False)
    feat_dim = encoder.feat_dim
    if not os.path.isfile(args.encoder_ckpt):
        raise FileNotFoundError(f"encoder_ckpt not found: {args.encoder_ckpt}")
    load_state_dict_compat(encoder, args.encoder_ckpt)
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)

    train_backbone_feats, train_labels = extract_backbone_features(encoder, train_loader, device)
    val_backbone_feats, val_labels = extract_backbone_features(encoder, val_loader, device)
    test_backbone_feats, test_labels = extract_backbone_features(encoder, test_loader, device)

    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    projector = Projector(feat_dim, proj_dim=args.proj_dim, hidden_dim=args.proj_hidden_dim)
    load_state_dict_compat(projector, args.projector_ckpt)
    projector = projector.to(device)
    projector.eval()
    for p in projector.parameters():
        p.requires_grad = False

    prototypes = ensure_3d_prototypes(torch.load(args.prototype_ckpt, map_location="cpu").float())
    if prototypes.shape[-1] != args.proj_dim:
        raise ValueError("prototype dimension mismatch with projector output dimension")
    proto_active_mask = active_prototype_mask(prototypes)
    prototype_mean = reduce_prototypes_mean(prototypes, proto_active_mask)

    classifier = CosineClassifier(args.proj_dim, num_classes, scale=args.cosine_scale).to(device)
    with torch.no_grad():
        classifier.weight.copy_(prototype_mean.to(device))

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        projector = nn.DataParallel(projector)
        classifier = nn.DataParallel(classifier)

    optimizer = optim.SGD(
        [
            {"params": classifier.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
        momentum=0.9,
    )

    ckpt_root = os.path.join(args.checkpoints, args.run_name or f"run_tpcsd_stage2_{int(time.time())}")
    os.makedirs(ckpt_root, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{os.path.basename(ckpt_root)}.log")
    per_class_csv = os.path.join(ckpt_root, "per_class_metrics.csv")
    gaussian_stats_csv = os.path.join(ckpt_root, "gaussian_stats_history.csv")
    stage2_mode_line = (
        "Stage2 mode: projector is frozen; projector_lr and anchor_weight are retained "
        "for CLI compatibility but ignored during optimization."
    )
    with open(gaussian_stats_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "mu_norm_mean", "mu_norm_std",
            "cov_trace", "cov_diag_mean", "cov_fro",
            "cov_eig_min", "cov_eig_max", "cov_cond",
            "d_mu_norm_mean", "d_mu_norm_std",
            "d_cov_trace", "d_cov_diag_mean", "d_cov_fro",
            "d_cov_eig_min", "d_cov_eig_max", "d_cov_cond",
        ])
    curve_history = {
        "train": {"epoch": [], "loss": []},
        "val": {"epoch": [], "acc": [], "bac": []},
        "test": {"epoch": [], "acc": [], "bac": []},
    }

    best_val_acc = -1.0
    prev_stats = None
    proto = prototypes.to(device)
    proto_active_mask = proto_active_mask.to(device)
    train_proj_feats = project_feature_tensor(projector, train_backbone_feats, device, args.stage2_batch_size)
    val_proj_feats = project_feature_tensor(projector, val_backbone_feats, device, args.stage2_batch_size)
    test_proj_feats = project_feature_tensor(projector, test_backbone_feats, device, args.stage2_batch_size)

    print(stage2_mode_line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(stage2_mode_line + "\n")

    for epoch in range(1, args.epochs + 1):
        mu_real, sigma_real = compute_class_stats(train_proj_feats, train_labels, num_classes)
        shared_cov_real = compute_shared_covariance(train_proj_feats, train_labels, mu_real)
        mu = mu_real.to(device)
        sigma = sigma_real.to(device)
        shared_cov = shared_cov_real.to(device)

        if epoch == 1:
            per_class_acc = compute_prototype_per_class_acc(train_proj_feats, train_labels, proto, num_classes, device)
        else:
            per_class_acc = compute_classifier_per_class_acc(classifier, train_proj_feats, train_labels, num_classes, device)

        hardest_indices = select_hardest_classes(per_class_acc, args.hardest_k, args.hardest_fraction)
        total_virtual = int(len(train_labels) * float(args.virtual_ratio))
        alloc, weights = allocate_hardest_virtual_counts(per_class_acc, hardest_indices, args.aas_alpha, total_virtual)

        mu_tilde_real = args.lambda_mu * mu.unsqueeze(1) + (1.0 - args.lambda_mu) * proto
        class_centers_real = reduce_prototypes_mean(mu_tilde_real, proto_active_mask)
        seed_virtual_feats, seed_virtual_labels = sample_virtual_features(
            mu_tilde_real,
            shared_cov,
            alloc,
            delta=args.delta_noise,
            cov_scale_factor=args.cov_scale_factor,
            active_mask=proto_active_mask,
        )
        accepted_seed_feats, accepted_seed_labels, seed_total, seed_kept = filter_virtual_features(
            seed_virtual_feats,
            seed_virtual_labels,
            classifier,
            class_centers_real,
            device,
            args.virtual_conf_thresh,
            args.virtual_center_cos_thresh,
            args.stage2_batch_size,
        )

        if accepted_seed_feats.numel() > 0:
            mix_feats = torch.cat([train_proj_feats, accepted_seed_feats], dim=0)
            mix_labels = torch.cat([train_labels, accepted_seed_labels], dim=0)
            mu_mix, sigma_mix = compute_class_stats(mix_feats, mix_labels, num_classes)
            shared_cov_mix = compute_shared_covariance(mix_feats, mix_labels, mu_mix)
            mu = mu_mix.to(device)
            sigma = sigma_mix.to(device)
            shared_cov = shared_cov_mix.to(device)
        else:
            mu = mu_real.to(device)
            sigma = sigma_real.to(device)
            shared_cov = shared_cov_real.to(device)

        mu_tilde = args.lambda_mu * mu.unsqueeze(1) + (1.0 - args.lambda_mu) * proto
        class_centers = reduce_prototypes_mean(mu_tilde, proto_active_mask)
        final_virtual_feats, final_virtual_labels = sample_virtual_features(
            mu_tilde,
            shared_cov,
            alloc,
            delta=args.delta_noise,
            cov_scale_factor=args.cov_scale_factor,
            active_mask=proto_active_mask,
        )
        accepted_virtual_feats, accepted_virtual_labels, final_total, final_kept = filter_virtual_features(
            final_virtual_feats,
            final_virtual_labels,
            classifier,
            class_centers,
            device,
            args.virtual_conf_thresh,
            args.virtual_center_cos_thresh,
            args.stage2_batch_size,
        )

        stats = summarize_gaussian_stats(mu, shared_cov)
        delta_stats = gaussian_stats_delta(stats, prev_stats)

        real_dataset = TensorDataset(train_proj_feats, train_labels)
        real_loader = DataLoader(
            real_dataset,
            batch_size=args.stage2_batch_size,
            shuffle=True,
            drop_last=False,
        )
        if accepted_virtual_feats.numel() > 0:
            virtual_dataset = TensorDataset(accepted_virtual_feats, accepted_virtual_labels)
            virtual_loader = DataLoader(
                virtual_dataset,
                batch_size=args.stage2_batch_size,
                shuffle=True,
                drop_last=False,
            )
        else:
            virtual_loader = None

        classifier.train()
        loss_sum = 0.0
        real_cls_sum = 0.0
        virtual_cls_sum = 0.0
        step_count = 0
        real_iter = iter(real_loader)
        virtual_iter = iter(virtual_loader) if virtual_loader is not None else None
        num_steps = max(1, max(len(real_loader), len(virtual_loader) if virtual_loader is not None else 0))

        for _ in range(num_steps):
            real_batch, real_iter = next_batch(real_loader, real_iter)
            virt_batch, virtual_iter = next_batch(virtual_loader, virtual_iter)

            real_x, real_y = real_batch
            real_x = real_x.to(device)
            real_y = real_y.to(device)

            real_proj_for_cls = real_x
            if args.train_noise_std > 0:
                real_proj_for_cls = real_proj_for_cls + args.train_noise_std * torch.randn_like(real_proj_for_cls)
            real_logits = classifier(real_proj_for_cls)
            real_cls_loss = nn.functional.cross_entropy(real_logits, real_y, weight=class_weights)

            loss = real_cls_loss
            virt_cls_loss = real_cls_loss.new_tensor(0.0)

            if virt_batch is not None:
                virt_x, virt_y = virt_batch
                virt_x = virt_x.to(device)
                virt_y = virt_y.to(device)
                if args.train_noise_std > 0:
                    virt_x = virt_x + args.train_noise_std * torch.randn_like(virt_x)
                virt_logits = classifier(virt_x)
                virt_cls_loss = nn.functional.cross_entropy(virt_logits, virt_y, weight=class_weights)
                loss = loss + args.virtual_loss_weight * virt_cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            real_cls_sum += real_cls_loss.item()
            virtual_cls_sum += virt_cls_loss.item()
            step_count += 1

        val_metrics, val_per_class = evaluate_classifier(classifier, val_proj_feats, val_labels, device, num_classes)
        test_metrics, test_per_class = evaluate_classifier(classifier, test_proj_feats, test_labels, device, num_classes)

        train_loss = loss_sum / max(1, step_count)
        train_real_loss = real_cls_sum / max(1, step_count)
        train_virtual_loss = virtual_cls_sum / max(1, step_count)
        train_anchor_loss = 0.0
        hardest_names = [train_base.class_names[int(idx)] for idx in hardest_indices.tolist()]

        append_per_class_records(per_class_csv, epoch, "val", val_per_class, train_base.class_names)
        append_per_class_records(per_class_csv, epoch, "test", test_per_class, train_base.class_names)
        update_loss_history(curve_history, epoch, train_loss)
        update_curve_history(curve_history, epoch, "val", val_per_class, group_specs)
        update_curve_history(curve_history, epoch, "test", test_per_class, group_specs)
        plot_loss_curve(curve_history, os.path.join(ckpt_root, "train_loss_curve.png"), title="Stage2 Train Loss")
        plot_group_curves(curve_history, os.path.join(ckpt_root, "val_group_curves.png"), "val")
        plot_group_curves(curve_history, os.path.join(ckpt_root, "test_group_curves.png"), "test")

        log_line = (
            f"Epoch {epoch}/{args.epochs} "
            f"loss={train_loss:.6f} "
            f"loss_real={train_real_loss:.6f} "
            f"loss_virtual={train_virtual_loss:.6f} "
            f"loss_anchor={train_anchor_loss:.6f} "
            f"hardest_k={len(hardest_names)} "
            f"seed_virtual={seed_kept}/{seed_total} "
            f"final_virtual={final_kept}/{final_total} "
            f"cov_scale={args.cov_scale_factor:.4f} "
            f"cos_scale={args.cosine_scale:.4f} "
            f"val_acc={val_metrics[0]:.6f} val_bac={val_metrics[3]:.6f} val_bacc={val_metrics[4]:.6f} "
            f"test_acc={test_metrics[0]:.6f} test_bac={test_metrics[3]:.6f} test_bacc={test_metrics[4]:.6f}"
        )
        hardest_line = "Hardest classes: " + ", ".join(hardest_names)

        gaussian_line = (
            "Gaussian stats: "
            f"mu_norm_mean={stats['mu_norm_mean']:.6f} "
            f"mu_norm_std={stats['mu_norm_std']:.6f} "
            f"cov_trace={stats['cov_trace']:.6f} "
            f"cov_diag_mean={stats['cov_diag_mean']:.6f} "
            f"cov_fro={stats['cov_fro']:.6f} "
            f"cov_eig_min={stats['cov_eig_min']:.6e} "
            f"cov_eig_max={stats['cov_eig_max']:.6e} "
            f"cov_cond={stats['cov_cond']:.6e}"
        )

        if delta_stats is None:
            gaussian_delta_line = "Gaussian delta: init"
        else:
            gaussian_delta_line = (
                "Gaussian delta: "
                f"d_mu_norm_mean={delta_stats['mu_norm_mean']:+.6e} "
                f"d_mu_norm_std={delta_stats['mu_norm_std']:+.6e} "
                f"d_cov_trace={delta_stats['cov_trace']:+.6e} "
                f"d_cov_diag_mean={delta_stats['cov_diag_mean']:+.6e} "
                f"d_cov_fro={delta_stats['cov_fro']:+.6e} "
                f"d_cov_eig_min={delta_stats['cov_eig_min']:+.6e} "
                f"d_cov_eig_max={delta_stats['cov_eig_max']:+.6e} "
                f"d_cov_cond={delta_stats['cov_cond']:+.6e}"
            )

        print(log_line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
            print(hardest_line)
            f.write(hardest_line + "\n")
            for line in format_tail_lines(val_per_class, train_base.class_names, tail_classes, "val"):
                print(line)
                f.write(line + "\n")
            print(gaussian_line)
            f.write(gaussian_line + "\n")
            print(gaussian_delta_line)
            f.write(gaussian_delta_line + "\n")
            for line in format_tail_lines(test_per_class, train_base.class_names, tail_classes, "test"):
                print(line)
                f.write(line + "\n")

        prev_stats = stats

        with open(gaussian_stats_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                stats["mu_norm_mean"], stats["mu_norm_std"],
                stats["cov_trace"], stats["cov_diag_mean"], stats["cov_fro"],
                stats["cov_eig_min"], stats["cov_eig_max"], stats["cov_cond"],
                (float("nan") if delta_stats is None else delta_stats["mu_norm_mean"]),
                (float("nan") if delta_stats is None else delta_stats["mu_norm_std"]),
                (float("nan") if delta_stats is None else delta_stats["cov_trace"]),
                (float("nan") if delta_stats is None else delta_stats["cov_diag_mean"]),
                (float("nan") if delta_stats is None else delta_stats["cov_fro"]),
                (float("nan") if delta_stats is None else delta_stats["cov_eig_min"]),
                (float("nan") if delta_stats is None else delta_stats["cov_eig_max"]),
                (float("nan") if delta_stats is None else delta_stats["cov_cond"]),
            ])

        if val_metrics[0] > best_val_acc:
            best_val_acc = val_metrics[0]
            torch.save(unwrap_model(classifier).state_dict(), os.path.join(ckpt_root, "classifier_best.pth"))
            torch.save(unwrap_model(projector).state_dict(), os.path.join(ckpt_root, "projector_best.pth"))
            torch.save(mu.cpu(), os.path.join(ckpt_root, "gaussian_mu_best.pth"))
            torch.save(sigma.cpu(), os.path.join(ckpt_root, "gaussian_sigma_best.pth"))
            torch.save(shared_cov.cpu(), os.path.join(ckpt_root, "shared_cov_best.pth"))

        torch.save(unwrap_model(classifier).state_dict(), os.path.join(ckpt_root, "classifier_latest.pth"))
        torch.save(unwrap_model(projector).state_dict(), os.path.join(ckpt_root, "projector_latest.pth"))
        torch.save(mu.cpu(), os.path.join(ckpt_root, "gaussian_mu_latest.pth"))
        torch.save(sigma.cpu(), os.path.join(ckpt_root, "gaussian_sigma_latest.pth"))
        torch.save(shared_cov.cpu(), os.path.join(ckpt_root, "shared_cov_latest.pth"))


if __name__ == "__main__":
    main()
