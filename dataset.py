import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

def load_h5_data(data_dir, split):
    all_data = []
    all_labels = []
    h5_files = glob.glob(os.path.join(data_dir, f'ply_data_{split}*.h5'))
    for h5_name in h5_files:
        f = h5py.File(h5_name, 'r')
        data = f['data'][:].astype('float32')
        labels = f['label'][:].astype('int64')
        all_data.append(data)
        all_labels.append(labels)
        f.close()
    if len(all_data) == 0:
        raise ValueError(f"No h5 files found for split {split} in {data_dir}")
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0).squeeze(-1)
    return all_data, all_labels

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_pointcloud = (pointcloud + jittered_data).astype('float32')
    return jittered_pointcloud

def rotate_pointcloud(pointcloud):
    theta = np.pi * 2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    rotated_pointcloud = pointcloud.dot(rotation_matrix).astype('float32')
    return rotated_pointcloud

class ModelNet40Dataset(Dataset):
    def __init__(self, data_dir, split='train', num_points=1024, augment=False):
        self.data_dir = data_dir
        self.split = split
        self.num_points = num_points
        self.augment = augment
        
        # Load all data into memory
        self.data, self.labels = load_h5_data(data_dir, split)
        
    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, idx):
        pt_idx = np.random.choice(self.data.shape[1], self.num_points, replace=False)
        pointcloud = self.data[idx, pt_idx, :]
        label = self.labels[idx]
        
        if self.augment and self.split == 'train':
            pointcloud = rotate_pointcloud(pointcloud)
            pointcloud = jitter_pointcloud(pointcloud)
            pointcloud = translate_pointcloud(pointcloud)
            
        # Switch to channel first for Conv1d in PointNet
        # Note: We will handle the permutation in the model or collate function.
        # It's standard for pointnet to accept (B, 3, N)
        return torch.from_numpy(pointcloud), torch.tensor(label, dtype=torch.long)
