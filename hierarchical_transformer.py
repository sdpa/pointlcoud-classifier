import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sampling & Grouping ────────────────────────────────────────────────────────

def farthest_point_sample(xyz: torch.Tensor, n_centroids: int) -> torch.Tensor:
    """
    Iteratively select the point farthest from the already-chosen set.
    Guarantees good spatial coverage across the point cloud.

    Args:
        xyz:         (B, N, 3)
        n_centroids: number of centroids M to select

    Returns:
        centroid_idx: (B, M) indices into xyz
    """
    B, N, _ = xyz.shape
    device = xyz.device

    centroid_idx = torch.zeros(B, n_centroids, dtype=torch.long, device=device)
    # Distance from each point to the closest already-selected centroid
    min_dist = torch.full((B, N), 1e10, device=device)
    # Start from a random point in each batch element
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(n_centroids):
        centroid_idx[:, i] = farthest
        # (B, 1, 3) centroid coords
        c = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)
        dist = torch.sum((xyz - c) ** 2, dim=-1)       # (B, N)
        min_dist = torch.minimum(min_dist, dist)
        farthest = torch.argmax(min_dist, dim=-1)       # pick the farthest

    return centroid_idx


def knn_group(xyz: torch.Tensor, centroid_idx: torch.Tensor, k: int):
    """
    For each centroid, gather its k nearest neighbours and return their
    coordinates expressed relative to the centroid.

    Args:
        xyz:         (B, N, 3) — full point cloud
        centroid_idx:(B, M)    — FPS centroid indices
        k:           neighbourhood size

    Returns:
        grouped_xyz: (B, M, k, 3)  relative coords
        nn_idx:      (B, M, k)     absolute indices (for feature gathering)
    """
    B, N, _ = xyz.shape
    M = centroid_idx.shape[1]
    device = xyz.device

    centroid_xyz = xyz[torch.arange(B, device=device).unsqueeze(1), centroid_idx]  # (B, M, 3)

    # Pairwise squared distances from every centroid to every point: (B, M, N)
    diff = centroid_xyz.unsqueeze(2) - xyz.unsqueeze(1)   # (B, M, N, 3)
    dist = torch.sum(diff ** 2, dim=-1)                    # (B, M, N)

    _, nn_idx = torch.topk(dist, k, dim=-1, largest=False)  # (B, M, k)

    # Gather absolute coords then make them relative to their centroid
    nn_idx_exp  = nn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    xyz_exp     = xyz.unsqueeze(1).expand(-1, M, -1, -1)    # (B, M, N, 3)
    grouped_xyz = torch.gather(xyz_exp, 2, nn_idx_exp)       # (B, M, k, 3)
    grouped_xyz = grouped_xyz - centroid_xyz.unsqueeze(2)    # relative

    return grouped_xyz, nn_idx


# ── Local Attention Block ──────────────────────────────────────────────────────

class LocalAttentionBlock(nn.Module):
    """
    Encode a single k-NN neighbourhood with one Transformer layer, then
    max-pool to produce a single feature vector per centroid.

    Input:  (B*M, k, 3) relative neighbourhood coords
    Output: (B*M, embed_dim)
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()

        # Lift raw 3-D coords into embedding space
        self.embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, grouped_xyz: torch.Tensor) -> torch.Tensor:
        # grouped_xyz: (B*M, k, 3)
        x = self.embed(grouped_xyz)   # (B*M, k, embed_dim)
        x = self.attn(x)              # (B*M, k, embed_dim)
        x = torch.max(x, dim=1)[0]   # (B*M, embed_dim)  — max pool over k
        return x


# ── Hierarchical Transformer ───────────────────────────────────────────────────

class HierarchicalTransformerClassifier(nn.Module):
    """
    Local-to-Global Hierarchical Transformer for point cloud classification.

    Pipeline
    --------
    1. Farthest Point Sampling  — downsample N → M centroid points
    2. k-NN Grouping            — build M local neighbourhoods of size k
    3. Local Self-Attention      — encode each neighbourhood → 1 feature vector
    4. Global Self-Attention     — M feature vectors attend to each other
    5. Max Pool + MLP Head       — classify

    Complexity comparison vs. flat transformer
    ------------------------------------------
    Flat:        O(N²)          N = 1024
    Hierarchical:O(M·k + M²)   M = 256, k = 16  →  ~10× cheaper
    """

    def __init__(
        self,
        num_classes:  int = 40,
        num_centroids: int = 256,
        k:            int = 16,
        embed_dim:    int = 128,
        num_heads:    int = 4,
        num_layers:   int = 3,
    ):
        super().__init__()
        self.num_centroids = num_centroids
        self.k = k

        # Stage 1 — local neighbourhood encoder
        self.local_attn = LocalAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)

        # Stage 2 — global transformer across centroids
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.global_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, N)  — channel-first layout expected by train.py
        B, _, N = x.shape
        xyz = x.permute(0, 2, 1)   # (B, N, 3)

        # ── 1. Farthest Point Sampling ─────────────────────────────────────
        centroid_idx = farthest_point_sample(xyz, self.num_centroids)  # (B, M)

        # ── 2. k-NN Grouping ───────────────────────────────────────────────
        grouped_xyz, _ = knn_group(xyz, centroid_idx, self.k)  # (B, M, k, 3)

        # ── 3. Local Self-Attention ────────────────────────────────────────
        # Flatten batch & centroid dims so every neighbourhood is one sequence
        grouped_flat = grouped_xyz.view(B * self.num_centroids, self.k, 3)
        local_feats  = self.local_attn(grouped_flat)                     # (B*M, embed_dim)
        local_feats  = local_feats.view(B, self.num_centroids, -1)       # (B, M, embed_dim)

        # ── 4. Global Self-Attention ───────────────────────────────────────
        global_feats = self.global_transformer(local_feats)   # (B, M, embed_dim)

        # ── 5. Global Max Pool ─────────────────────────────────────────────
        global_feat = torch.max(global_feats, dim=1)[0]       # (B, embed_dim)

        # ── 6. Classify ────────────────────────────────────────────────────
        logits = self.classifier(global_feat)
        return F.log_softmax(logits, dim=1)
