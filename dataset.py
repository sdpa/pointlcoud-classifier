"""
dataset.py  —  ModelNet40 HDF5 loader with pluggable Stage-2 subsampling

Stage-2 subsampling: 2048 stored points → num_points used per forward pass
---------------------------------------------------------------------------
  random  (default)
    np.random.choice selects num_points indices with no spatial bias.
    Fast, but the drawn subset may cluster or leave gaps purely by chance.

  fps  (Farthest Point Sampling)
    Greedily picks the next point that is farthest from all already-selected
    points.  Guarantees maximal spread across the cloud — the worst-case
    nearest-neighbour distance across all selected points is maximised.
    O(num_points × N_stored) per sample; noticeably slower than random but
    produces a structurally representative subset even when num_points << N.
"""

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# ── HDF5 loader ───────────────────────────────────────────────────────────────

def load_h5_data(data_dir: str, split: str):
    all_data, all_labels = [], []
    h5_files = sorted(glob.glob(os.path.join(data_dir, f'ply_data_{split}*.h5')))
    if not h5_files:
        raise ValueError(f"No h5 files found for split '{split}' in {data_dir}")
    for path in h5_files:
        with h5py.File(path, 'r') as f:
            all_data.append(f['data'][:].astype('float32'))
            all_labels.append(f['label'][:].astype('int64'))
    data   = np.concatenate(all_data,   axis=0)
    labels = np.concatenate(all_labels, axis=0).squeeze(-1)
    return data, labels


# ── Augmentation helpers ───────────────────────────────────────────────────────

def translate_pointcloud(pointcloud: np.ndarray) -> np.ndarray:
    scale = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
    shift = np.random.uniform(low=-0.2,       high=0.2,        size=[3])
    return np.add(np.multiply(pointcloud, scale), shift).astype('float32')


def jitter_pointcloud(pointcloud: np.ndarray,
                      sigma: float = 0.01, clip: float = 0.05) -> np.ndarray:
    N, C = pointcloud.shape
    noise = np.clip(sigma * np.random.randn(N, C), -clip, clip)
    return (pointcloud + noise).astype('float32')


def rotate_pointcloud(pointcloud: np.ndarray) -> np.ndarray:
    theta = np.pi * 2 * np.random.uniform()
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0,              0,             1]])
    return pointcloud.dot(R).astype('float32')


# ── Farthest Point Sampling (numpy, CPU) ───────────────────────────────────────

def fps_subsample(points: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Greedy farthest-point subsampling.

    Algorithm
    ---------
    1. Pick a random starting point.
    2. Maintain min_dist[i] = distance from point i to the closest
       already-selected point.
    3. At each step, add the point with the largest min_dist value.
    4. Update min_dist for all remaining points.

    This maximises the minimum pairwise distance in the selected set,
    giving the best possible spatial coverage for a fixed budget of
    n_samples points.

    Complexity: O(n_samples × N)  where N = len(points).
    """
    N = len(points)
    selected = np.empty(n_samples, dtype=np.int64)
    # Initialise: every point is infinitely far from the (empty) selected set
    min_dists = np.full(N, np.inf)

    cur = np.random.randint(0, N)
    for i in range(n_samples):
        selected[i] = cur
        # Squared Euclidean distance from cur to every point
        diff  = points - points[cur]          # (N, 3)
        dists = np.einsum('ij,ij->i', diff, diff)   # (N,)
        # Each point's distance to its nearest selected point
        min_dists = np.minimum(min_dists, dists)
        # Next centroid: farthest from the current selected set
        cur = int(np.argmax(min_dists))

    return selected


# ── Dataset ────────────────────────────────────────────────────────────────────

class ModelNet40Dataset(Dataset):
    """
    Parameters
    ----------
    data_dir    : path to directory containing ply_data_{split}*.h5 files
    split       : 'train' or 'test'
    num_points  : number of points to feed to the model per sample
    augment     : apply random rotation / jitter / translation at load time
    subsample   : 'random' — uniform random subset (fast, default)
                  'fps'    — farthest-point subset (slower, better coverage)
    """

    def __init__(self, data_dir: str, split: str = 'train',
                 num_points: int = 1024, augment: bool = False,
                 subsample: str = 'random'):
        assert subsample in ('random', 'fps'), \
            f"subsample must be 'random' or 'fps', got '{subsample}'"
        self.num_points = num_points
        self.augment    = augment
        self.split      = split
        self.subsample  = subsample

        self.data, self.labels = load_h5_data(data_dir, split)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        cloud = self.data[idx]   # (2048, 3) or whatever was stored

        # ── Stage-2 subsampling ────────────────────────────────────────────
        if self.subsample == 'fps':
            pt_idx = fps_subsample(cloud, self.num_points)
        else:
            pt_idx = np.random.choice(cloud.shape[0], self.num_points,
                                      replace=False)
        pointcloud = cloud[pt_idx]   # (num_points, 3)

        # ── Augmentation (train only) ──────────────────────────────────────
        if self.augment and self.split == 'train':
            pointcloud = rotate_pointcloud(pointcloud)
            pointcloud = jitter_pointcloud(pointcloud)
            pointcloud = translate_pointcloud(pointcloud)

        label = self.labels[idx]
        return torch.from_numpy(pointcloud), torch.tensor(label, dtype=torch.long)
