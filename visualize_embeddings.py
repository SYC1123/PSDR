import argparse
import os

import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _balanced_subsample(features, labels, max_points=4000, seed=42):
    features = _to_numpy(features)
    labels = _to_numpy(labels).astype(np.int64)

    if features.shape[0] <= max_points:
        return features, labels

    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    per_class_cap = max(1, int(np.ceil(max_points / max(1, len(unique_labels)))))

    keep_indices = []
    for class_id in unique_labels:
        idx = np.flatnonzero(labels == class_id)
        if idx.size <= per_class_cap:
            keep_indices.extend(idx.tolist())
        else:
            chosen = rng.choice(idx, size=per_class_cap, replace=False)
            keep_indices.extend(chosen.tolist())

    keep_indices = np.asarray(sorted(keep_indices), dtype=np.int64)
    if keep_indices.size > max_points:
        keep_indices = rng.choice(keep_indices, size=max_points, replace=False)
        keep_indices = np.asarray(sorted(keep_indices), dtype=np.int64)
    return features[keep_indices], labels[keep_indices]


def _compute_tsne_embedding(features, seed=42, small_n_perplexity=5):
    from sklearn.manifold import TSNE

    features = _to_numpy(features)
    if features.shape[0] < 2:
        raise ValueError("Need at least two points to compute a 2D embedding.")
    if features.shape[0] == 2:
        return np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    perplexity = max(2, min(small_n_perplexity, features.shape[0] - 1))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(features)


def save_tsne_plot(features, labels, class_names, out_path, title="", max_points=4000, seed=42):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    features, labels = _balanced_subsample(features, labels, max_points=max_points, seed=seed)
    if features.shape[0] < 2:
        return False

    embedding = _compute_tsne_embedding(features, seed=seed, small_n_perplexity=30)

    class_names = list(class_names)
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", max(20, len(class_names)))

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    for color_idx, class_id in enumerate(unique_labels):
        idx = labels == class_id
        class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else str(class_id)
        ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            s=11,
            alpha=0.75,
            color=cmap(color_idx),
            label=class_name,
            linewidths=0,
        )
    ax.set_title(title or "t-SNE")
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.grid(True, alpha=0.15)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def save_named_point_tsne(features, class_names, out_path, title="Prototype t-SNE", seed=42):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    features = _to_numpy(features)
    if features.shape[0] < 2:
        return False

    embedding = _compute_tsne_embedding(features, seed=seed, small_n_perplexity=5)
    cmap = plt.cm.get_cmap("tab20", max(20, len(class_names)))

    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    for class_id, class_name in enumerate(class_names):
        ax.scatter(
            embedding[class_id, 0],
            embedding[class_id, 1],
            s=120,
            alpha=0.9,
            color=cmap(class_id),
            edgecolors="black",
            linewidths=0.5,
        )
        ax.text(
            embedding[class_id, 0],
            embedding[class_id, 1],
            f" {class_name}",
            fontsize=9,
            va="center",
            ha="left",
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def save_prototype_mean_tsne(prototypes, means, class_names, out_path, title="Prototype vs Class Mean t-SNE", seed=42):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    prototypes = _to_numpy(prototypes)
    means = _to_numpy(means)
    if prototypes.shape != means.shape or prototypes.shape[0] < 2:
        return False

    all_points = np.concatenate([means, prototypes], axis=0)
    embedding = _compute_tsne_embedding(all_points, seed=seed, small_n_perplexity=8)
    mean_points = embedding[: len(class_names)]
    proto_points = embedding[len(class_names) :]
    cmap = plt.cm.get_cmap("tab20", max(20, len(class_names)))

    fig, ax = plt.subplots(figsize=(9, 8), dpi=160)
    for class_id, class_name in enumerate(class_names):
        color = cmap(class_id)
        ax.plot(
            [mean_points[class_id, 0], proto_points[class_id, 0]],
            [mean_points[class_id, 1], proto_points[class_id, 1]],
            color=color,
            alpha=0.5,
            linewidth=1.2,
        )
        ax.scatter(
            mean_points[class_id, 0],
            mean_points[class_id, 1],
            s=80,
            alpha=0.85,
            color=color,
            marker="o",
            edgecolors="black",
            linewidths=0.4,
        )
        ax.scatter(
            proto_points[class_id, 0],
            proto_points[class_id, 1],
            s=95,
            alpha=0.95,
            color=color,
            marker="^",
            edgecolors="black",
            linewidths=0.4,
        )
        ax.text(
            proto_points[class_id, 0],
            proto_points[class_id, 1],
            f" {class_name}",
            fontsize=8.5,
            va="center",
            ha="left",
        )

    ax.scatter([], [], color="gray", marker="o", label="class mean")
    ax.scatter([], [], color="gray", marker="^", label="prototype")
    ax.set_title(title)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def save_similarity_heatmap(matrix, class_names, out_path, title="Cosine Similarity Heatmap", vmin=-1.0, vmax=1.0):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    matrix = _to_numpy(matrix)
    class_names = list(class_names)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False

    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    im = ax.imshow(matrix, cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def save_real_virtual_tsne(
    real_features,
    real_labels,
    virtual_features,
    virtual_labels,
    class_names,
    out_path,
    title="Real vs Virtual t-SNE",
    max_real_points=2500,
    max_virtual_points=2500,
    seed=42,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    real_features, real_labels = _balanced_subsample(real_features, real_labels, max_points=max_real_points, seed=seed)
    virtual_features, virtual_labels = _balanced_subsample(
        virtual_features, virtual_labels, max_points=max_virtual_points, seed=seed + 1
    )
    if real_features.shape[0] + virtual_features.shape[0] < 2:
        return False

    features = np.concatenate([real_features, virtual_features], axis=0)
    labels = np.concatenate([real_labels, virtual_labels], axis=0)
    sources = np.concatenate(
        [
            np.zeros(real_features.shape[0], dtype=np.int64),
            np.ones(virtual_features.shape[0], dtype=np.int64),
        ],
        axis=0,
    )

    embedding = _compute_tsne_embedding(features, seed=seed, small_n_perplexity=30)

    class_names = list(class_names)
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", max(20, len(class_names)))

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    for color_idx, class_id in enumerate(unique_labels):
        class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else str(class_id)
        real_idx = (labels == class_id) & (sources == 0)
        virt_idx = (labels == class_id) & (sources == 1)
        if np.any(real_idx):
            ax.scatter(
                embedding[real_idx, 0],
                embedding[real_idx, 1],
                s=10,
                alpha=0.55,
                color=cmap(color_idx),
                marker="o",
                linewidths=0,
            )
        if np.any(virt_idx):
            ax.scatter(
                embedding[virt_idx, 0],
                embedding[virt_idx, 1],
                s=18,
                alpha=0.85,
                color=cmap(color_idx),
                marker="^",
                linewidths=0,
                label=class_name,
            )
        elif np.any(real_idx):
            ax.scatter([], [], color=cmap(color_idx), marker="o", label=class_name)

    ax.scatter([], [], color="black", marker="o", label="real")
    ax.scatter([], [], color="black", marker="^", label="virtual")
    ax.set_title(title)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.grid(True, alpha=0.15)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--class_names", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="t-SNE")
    parser.add_argument("--max_points", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features = torch.load(args.features, map_location="cpu")
    labels = torch.load(args.labels, map_location="cpu")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ok = save_tsne_plot(
        features=features,
        labels=labels,
        class_names=args.class_names,
        out_path=args.output,
        title=args.title,
        max_points=args.max_points,
        seed=args.seed,
    )
    if ok:
        print(f"saved to: {args.output}")
    else:
        print("t-SNE generation skipped")


if __name__ == "__main__":
    main()
