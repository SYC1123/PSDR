import math

import torch
import torch.nn.functional as F


def ensure_3d_prototypes(prototypes):
    if prototypes.dim() == 2:
        return prototypes.unsqueeze(1)
    if prototypes.dim() == 3:
        return prototypes
    raise ValueError(f"Unsupported prototype shape: {tuple(prototypes.shape)}")


def active_prototype_mask(prototypes, eps=1e-12):
    prototypes_3d = ensure_3d_prototypes(prototypes)
    return torch.norm(prototypes_3d, p=2, dim=2) > float(eps)


def reduce_prototypes_mean(prototypes, active_mask=None, eps=1e-12):
    prototypes_3d = ensure_3d_prototypes(prototypes)
    if active_mask is None:
        active_mask = active_prototype_mask(prototypes_3d, eps=eps)
    active_mask = active_mask.to(device=prototypes_3d.device, dtype=prototypes_3d.dtype)
    denom = torch.clamp(active_mask.sum(dim=1, keepdim=True), min=1.0)
    return (prototypes_3d * active_mask.unsqueeze(-1)).sum(dim=1) / denom


def max_cosine_scores_to_class_prototypes(features, prototypes):
    if features.numel() == 0:
        num_classes = ensure_3d_prototypes(prototypes).shape[0]
        return features.new_zeros((0, num_classes))

    prototypes_3d = ensure_3d_prototypes(prototypes)
    feat_n = F.normalize(features, p=2, dim=1)
    proto_n = F.normalize(prototypes_3d, p=2, dim=2)
    scores = torch.einsum("bd,ckd->bck", feat_n, proto_n)
    return scores.max(dim=2).values


def _farthest_point_init(features, k):
    n = int(features.shape[0])
    if n == 0 or k <= 0:
        return features.new_zeros((0, features.shape[1]))
    if k >= n:
        return features[:k].clone()

    feat_n = F.normalize(features, p=2, dim=1)
    mean_feat = F.normalize(features.mean(dim=0, keepdim=True), p=2, dim=1)
    first_idx = torch.argmin(torch.matmul(feat_n, mean_feat.t()).squeeze(1))
    selected = [int(first_idx.item())]

    while len(selected) < k:
        selected_feats = feat_n[selected]
        sim = torch.matmul(feat_n, selected_feats.t())
        min_dist = (1.0 - sim).min(dim=1).values
        min_dist[selected] = -1.0
        next_idx = torch.argmax(min_dist)
        selected.append(int(next_idx.item()))
    return features[selected].clone()


def _mini_batch_kmeans(features, k, iters=3):
    n = int(features.shape[0])
    if n == 0:
        return features.new_zeros((0, features.shape[1]))
    k = max(1, min(int(k), n))
    if k == 1:
        return features.mean(dim=0, keepdim=True)

    centers = _farthest_point_init(features, k)
    feat_n = F.normalize(features, p=2, dim=1)
    for _ in range(max(1, int(iters))):
        center_n = F.normalize(centers, p=2, dim=1)
        assign = torch.matmul(feat_n, center_n.t()).argmax(dim=1)
        new_centers = []
        for idx in range(k):
            mask = assign == idx
            if torch.any(mask):
                new_centers.append(features[mask].mean(dim=0))
            else:
                new_centers.append(centers[idx])
        centers = torch.stack(new_centers, dim=0)
    return centers


def _match_centroids_to_slots(centroids, class_prototypes, class_seen):
    num_centroids = int(centroids.shape[0])
    num_slots = int(class_prototypes.shape[0])
    mapping = centroids.new_full((num_centroids,), -1, dtype=torch.long)
    available_slots = list(range(num_slots))
    seen_slots = [idx for idx in available_slots if bool(class_seen[idx].item())]

    if seen_slots:
        centroid_n = F.normalize(centroids, p=2, dim=1)
        proto_n = F.normalize(class_prototypes[seen_slots], p=2, dim=1)
        sim = torch.matmul(centroid_n, proto_n.t())
        while sim.numel() > 0:
            flat_idx = int(torch.argmax(sim).item())
            c_idx = flat_idx // sim.shape[1]
            s_idx = flat_idx % sim.shape[1]
            if mapping[c_idx] >= 0:
                sim[c_idx, :] = -2.0
                continue
            slot = seen_slots[s_idx]
            mapping[c_idx] = slot
            sim[c_idx, :] = -2.0
            sim[:, s_idx] = -2.0
            if slot in available_slots:
                available_slots.remove(slot)

    unseen_slots = [idx for idx in available_slots if not bool(class_seen[idx].item())]
    remaining = [idx for idx in range(num_centroids) if mapping[idx] < 0]
    for centroid_idx, slot in zip(remaining, unseen_slots):
        mapping[centroid_idx] = slot
        if slot in available_slots:
            available_slots.remove(slot)

    remaining = [idx for idx in range(num_centroids) if mapping[idx] < 0]
    for centroid_idx, slot in zip(remaining, available_slots):
        mapping[centroid_idx] = slot

    return mapping


def balanced_softmax_loss(logits, labels, class_counts):
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    counts = torch.as_tensor(class_counts, dtype=logits.dtype, device=logits.device)
    counts = torch.clamp(counts, min=1.0)
    balanced_logits = logits + counts.log().unsqueeze(0)
    return F.cross_entropy(balanced_logits, labels)


def deferred_balanced_softmax_loss(logits, labels, class_counts, epoch, warmup_epochs):
    if int(epoch) < int(warmup_epochs):
        return F.cross_entropy(logits, labels)
    return balanced_softmax_loss(logits, labels, class_counts)


def enqueue_feature_queue(feature_queue, features, label_queue=None, labels=None):
    if feature_queue.numel() == 0 or features.numel() == 0:
        return

    capacity = int(feature_queue.shape[0])
    num_new = min(capacity, int(features.shape[0]))
    if num_new <= 0:
        return

    features = features.detach()[-num_new:]
    if label_queue is not None:
        if labels is None:
            raise ValueError("labels must be provided when label_queue is used")
        labels = labels.detach()[-num_new:]

    if num_new < capacity:
        feature_queue[:-num_new] = feature_queue[num_new:].clone()
        if label_queue is not None:
            label_queue[:-num_new] = label_queue[num_new:].clone()
    feature_queue[-num_new:] = features
    if label_queue is not None:
        label_queue[-num_new:] = labels


def sp_kd_loss(student_features, teacher_features, student_memory=None, teacher_memory=None, eps=1e-12):
    if student_features.numel() == 0:
        return student_features.new_tensor(0.0)

    if student_memory is not None and student_memory.numel() > 0:
        student_all = torch.cat([student_features, student_memory], dim=0)
    else:
        student_all = student_features

    if teacher_memory is not None and teacher_memory.numel() > 0:
        teacher_all = torch.cat([teacher_features, teacher_memory], dim=0)
    else:
        teacher_all = teacher_features

    gram_s = torch.matmul(student_features, student_all.t())
    gram_t = torch.matmul(teacher_features, teacher_all.t())
    gram_s = gram_s / torch.clamp(torch.norm(gram_s, p="fro"), min=eps)
    gram_t = gram_t / torch.clamp(torch.norm(gram_t, p="fro"), min=eps)
    return torch.sum((gram_s - gram_t) ** 2)


def prototype_uniformity_loss(prototypes, t=2.0, active_mask=None):
    if prototypes.numel() == 0:
        return prototypes.new_tensor(0.0)

    if active_mask is not None:
        active_mask = active_mask.to(device=prototypes.device, dtype=torch.bool)
        prototypes = prototypes[active_mask]

    if prototypes.numel() == 0 or prototypes.shape[0] < 2:
        return prototypes.new_tensor(0.0)

    prototypes = F.normalize(prototypes, p=2, dim=1)
    sim_matrix = torch.matmul(prototypes, prototypes.t())
    off_diag_mask = ~torch.eye(sim_matrix.shape[0], device=sim_matrix.device, dtype=torch.bool)
    off_diag = sim_matrix[off_diag_mask]

    if off_diag.numel() == 0:
        return sim_matrix.new_tensor(0.0)

    return torch.logsumexp(float(t) * off_diag, dim=0) - math.log(off_diag.numel())


def prototype_nearest_neighbor_separation_loss(prototypes, margin=0.5, active_mask=None):
    if prototypes.numel() == 0:
        return prototypes.new_tensor(0.0)

    if active_mask is not None:
        active_mask = active_mask.to(device=prototypes.device, dtype=torch.bool)
        prototypes = prototypes[active_mask]

    if prototypes.numel() == 0 or prototypes.shape[0] < 2:
        return prototypes.new_tensor(0.0)

    prototypes = F.normalize(prototypes, p=2, dim=1)
    sim_matrix = torch.matmul(prototypes, prototypes.t())
    sim_matrix.fill_diagonal_(-1.0)
    nearest_neighbor_sim, _ = sim_matrix.max(dim=1)
    return torch.clamp(nearest_neighbor_sim - float(margin), min=0.0).mean()


def pcd_loss(z, labels, prototypes, temperature=0.07, sample_weights=None, pcd_margin=0.85):
    if z.numel() == 0:
        return z.new_tensor(0.0)
    z = F.normalize(z, p=2, dim=1)
    prototypes_3d = ensure_3d_prototypes(prototypes)
    p = F.normalize(prototypes_3d, p=2, dim=2)
    target_proto = p[labels]
    proto_scores = torch.einsum("bkd,bd->bk", target_proto, z)
    best_idx = proto_scores.argmax(dim=1)
    target_proto = target_proto[torch.arange(target_proto.shape[0], device=z.device), best_idx]
    cos_sim = F.cosine_similarity(z, target_proto, dim=1)
    loss = torch.clamp(float(pcd_margin) - cos_sim, min=0.0)
    if sample_weights is not None:
        loss = loss * sample_weights
    return loss.mean()


def compute_batch_class_means(z, labels, num_classes):
    if z.numel() == 0:
        return z.new_zeros(num_classes, 0), torch.zeros(num_classes, dtype=torch.bool, device=z.device)

    batch_class_mean = z.new_zeros(num_classes, z.shape[1])
    valid_mask = torch.zeros(num_classes, dtype=torch.bool, device=z.device)
    for c in range(num_classes):
        idx = labels == c
        if idx.sum() == 0:
            continue
        batch_class_mean[c] = z[idx].mean(dim=0)
        valid_mask[c] = True
    return batch_class_mean, valid_mask


def compute_batch_multi_prototype_means(z, labels, prototypes, prototype_seen=None, kmeans_iters=3):
    prototypes_3d = ensure_3d_prototypes(prototypes)
    num_classes, num_proto, feat_dim = prototypes_3d.shape
    if z.numel() == 0:
        return (
            z.new_zeros(num_classes, num_proto, feat_dim),
            torch.zeros(num_classes, num_proto, dtype=torch.bool, device=z.device),
        )

    if prototype_seen is None:
        prototype_seen = active_prototype_mask(prototypes_3d)
    prototype_seen = prototype_seen.to(device=z.device, dtype=torch.bool)

    batch_proto_mean = z.new_zeros(num_classes, num_proto, feat_dim)
    valid_mask = torch.zeros(num_classes, num_proto, dtype=torch.bool, device=z.device)

    for class_id in range(num_classes):
        idx = labels == class_id
        if idx.sum() == 0:
            continue
        class_feats = z[idx]
        k = min(num_proto, int(class_feats.shape[0]))
        centroids = _mini_batch_kmeans(class_feats, k, iters=kmeans_iters)
        slot_map = _match_centroids_to_slots(centroids, prototypes_3d[class_id], prototype_seen[class_id])
        for centroid_idx in range(int(centroids.shape[0])):
            slot = int(slot_map[centroid_idx].item())
            if slot < 0:
                continue
            batch_proto_mean[class_id, slot] = centroids[centroid_idx]
            valid_mask[class_id, slot] = True
    return batch_proto_mean, valid_mask


def ema_update_prototypes(prototypes, batch_class_mean, valid_mask, momentum=0.96):
    if batch_class_mean.numel() == 0 or not torch.any(valid_mask):
        return prototypes
    if prototypes.dim() == 2:
        prototypes[valid_mask] = (
            momentum * prototypes[valid_mask] + (1.0 - momentum) * batch_class_mean[valid_mask]
        )
        return prototypes

    prototypes_3d = ensure_3d_prototypes(prototypes)
    batch_3d = ensure_3d_prototypes(batch_class_mean)
    mask = valid_mask.to(device=prototypes_3d.device, dtype=torch.bool)
    prototypes_3d[mask] = momentum * prototypes_3d[mask] + (1.0 - momentum) * batch_3d[mask]
    return prototypes_3d


def recalibrate_prototypes(prototypes, batch_class_mean, valid_mask, tail_mask=None, alpha=0.15, tail_factor=1.5):
    if batch_class_mean.numel() == 0 or not torch.any(valid_mask):
        return prototypes

    if prototypes.dim() == 2:
        recal_factor = torch.ones(prototypes.shape[0], dtype=prototypes.dtype, device=prototypes.device)
        if tail_mask is not None:
            recal_factor = recal_factor + (float(tail_factor) - 1.0) * tail_mask.to(prototypes.dtype)
        alpha_vec = float(alpha) * recal_factor

        prototypes[valid_mask] = (
            (1.0 - alpha_vec[valid_mask]).unsqueeze(1) * prototypes[valid_mask]
            + alpha_vec[valid_mask].unsqueeze(1) * batch_class_mean[valid_mask]
        )
        return prototypes

    prototypes_3d = ensure_3d_prototypes(prototypes)
    batch_3d = ensure_3d_prototypes(batch_class_mean)
    class_factor = torch.ones(prototypes_3d.shape[0], dtype=prototypes_3d.dtype, device=prototypes_3d.device)
    if tail_mask is not None:
        class_factor = class_factor + (float(tail_factor) - 1.0) * tail_mask.to(prototypes_3d.dtype)
    alpha_mat = float(alpha) * class_factor.unsqueeze(1).expand(-1, prototypes_3d.shape[1])
    mask = valid_mask.to(device=prototypes_3d.device, dtype=torch.bool)
    prototypes_3d[mask] = (
        (1.0 - alpha_mat[mask]).unsqueeze(1) * prototypes_3d[mask]
        + alpha_mat[mask].unsqueeze(1) * batch_3d[mask]
    )
    return prototypes_3d
