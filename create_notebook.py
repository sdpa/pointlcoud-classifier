import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # 1. Introduction Markdown
    md_intro = """# 3D Point Cloud Classifier: PointNet vs Transformer
**Objective**: This notebook implements a lightweight point cloud classifier on the ModelNet40 dataset. We compare a PointNet-style baseline against a Transformer-based architecture to study the benefits of self-attention in 3D data.
"""
    
    # 2. Imports Setup
    code_imports = """import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Import our custom modules
from dataset import ModelNet40Dataset
from pointnet import PointNetClassifier
from transformer import TransformerClassifier
from train import run_experiment

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")
"""
    
    # 3. Dataset Setup
    md_data = """## Data Preparation Tools
We use the `ModelNet40Dataset` to load HDF5 representations containing 2048 sampled points per shape. During training, we randomly subsample this to `N=1024` and apply geometric augmentations including jittering, translation, and rotation along the vertical axis.
"""
    code_data = """train_dataset = ModelNet40Dataset(data_dir='data/modelnet40_ply_hdf5_2048', split='train', num_points=1024, augment=True)
test_dataset = ModelNet40Dataset(data_dir='data/modelnet40_ply_hdf5_2048', split='test', num_points=1024, augment=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train Points: {len(train_dataset)}, Test Points: {len(test_dataset)}")
"""

    code_vis = """# Visualize a point cloud
data, label = train_dataset[0]
points = data.numpy()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5)
ax.set_title(f'Label: {label.item()}')
plt.show()
"""
    
    # 4. PointNet
    md_pointnet = """## Week 1 Baseline: PointNet
PointNet utilizes shared MLPs (implemented efficiently as Conv1D) to map individual 3D points into a higher dimensional latent space, followed by a symmetric aggregation function (Global Max Pooling) creating a permutation-invariant holistic representation.
"""
    code_pointnet = """pointnet_model = PointNetClassifier(k=40)
print(f"PointNet Parameters: {sum(p.numel() for p in pointnet_model.parameters() if p.requires_grad)}")

# Run 10 Epochs for evaluation (increase to ~250 for >88% accuracy)
print("Starting PointNet Training...")
history_pointnet = run_experiment(pointnet_model, train_loader, test_loader, epochs=10, lr=1e-3, device=device)
"""

    code_plot_pointnet = """# Plot PointNet metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_pointnet['train_loss'], label='Train')
plt.plot(history_pointnet['test_loss'], label='Test')
plt.title('PointNet Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_pointnet['train_acc'], label='Train')
plt.plot(history_pointnet['test_acc'], label='Test')
plt.title('PointNet Accuracy')
plt.legend()
plt.show()
"""

    # 5. Transformer
    md_transformer = """## Week 2: Lightweight Transformer Classifier
This model maps raw XYZ coordinates to a dense point embedding sequentially, then employs standard PyTorch Multi-Head Self-Attention layers without strict positional encoding (since the geometric coords act as features/positions already), followed by global max pooling.
"""
    code_transformer = """transformer_model = TransformerClassifier(num_classes=40, embed_dim=128, num_heads=4, num_layers=2)
print(f"Transformer Parameters: {sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)}")

print("Starting Transformer Training...")
history_transformer = run_experiment(transformer_model, train_loader, test_loader, epochs=10, lr=1e-3, device=device)
"""

    code_plot_transformer = """# Plot Transformer metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_transformer['train_loss'], label='Train')
plt.plot(history_transformer['test_loss'], label='Test')
plt.title('Transformer Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_transformer['train_acc'], label='Train')
plt.plot(history_transformer['test_acc'], label='Test')
plt.title('Transformer Accuracy')
plt.legend()
plt.show()
"""

    md_comparison = """## Final Comparison
These charts give you an idea of memory vs accuracy tradeoffs for both architectures across identically matched geometric augmentations.
"""

    nb.cells = [
        nbf.v4.new_markdown_cell(md_intro),
        nbf.v4.new_code_cell(code_imports),
        nbf.v4.new_markdown_cell(md_data),
        nbf.v4.new_code_cell(code_data),
        nbf.v4.new_code_cell(code_vis),
        nbf.v4.new_markdown_cell(md_pointnet),
        nbf.v4.new_code_cell(code_pointnet),
        nbf.v4.new_code_cell(code_plot_pointnet),
        nbf.v4.new_markdown_cell(md_transformer),
        nbf.v4.new_code_cell(code_transformer),
        nbf.v4.new_code_cell(code_plot_transformer),
        nbf.v4.new_markdown_cell(md_comparison)
    ]
    
    with open('pointcloud_classifier.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Notebook pointcloud_classifier.ipynb created successfully!")

if __name__ == "__main__":
    create_notebook()
