import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import argparse
from dataset import ModelNet40Dataset
from pointnet import PointNetClassifier
from transformer import TransformerClassifier
from torch.utils.data import DataLoader, Subset

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in tqdm(loader, desc="Training", leave=False):
        data, target = data.to(device), target.to(device)
        # Convert to B, C, N layout for Conv1D compatibility
        data = data.permute(0, 2, 1)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.size(0)
        
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in tqdm(loader, desc="Evaluating", leave=False):
        data, target = data.to(device), target.to(device)
        data = data.permute(0, 2, 1)
        
        output = model(data)
        loss = F.nll_loss(output, target)
        
        total_loss += loss.item() * data.size(0)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.size(0)
        
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def run_experiment(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu', save_dir='./checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0.0
    
    print(f"Training on device: {device}")
    
    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = eval_epoch(model, test_loader, device)
        scheduler.step()
        end_time = time.time()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Time: {end_time-start_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
              
        # Save best model checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"  -> Saved new best model with Test Acc: {best_acc:.4f}")
            
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    print(f"Final model saved to {save_dir}/final_model.pth")
    
    # Save History as CSV
    csv_path = os.path.join(save_dir, 'training_history.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
        for i in range(epochs):
            writer.writerow([i+1, history['train_loss'][i], history['train_acc'][i], history['test_loss'][i], history['test_acc'][i]])
    print(f"Metrics saved to {csv_path}")
            
    # Save Graphs as PNG
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['test_acc'], label='Test Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    graph_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(graph_path)
    plt.close()
    print(f"Plots saved to {graph_path}")
              
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D Point Cloud Classifier")
    parser.add_argument('--model', type=str, default='pointnet', choices=['pointnet', 'transformer'], help='Model architecture to run')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--debug', action='store_true', help='Use a minimal subset of data for debugging/testing')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    try:
        full_train_dataset = ModelNet40Dataset(data_dir='data/modelnet40_ply_hdf5_2048', split='train', num_points=1024, augment=True)
        full_test_dataset = ModelNet40Dataset(data_dir='data/modelnet40_ply_hdf5_2048', split='test', num_points=1024, augment=False)
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}. Please ensure data is downloaded.")
        exit(1)
    
    if args.debug:
        print("Running in DEBUG mode with a tiny data subset...")
        subset_size = 128
        train_dataset = Subset(full_train_dataset, list(range(min(subset_size, len(full_train_dataset)))))
        test_dataset = Subset(full_test_dataset, list(range(min(subset_size, len(full_test_dataset)))))
    else:
        train_dataset = full_train_dataset
        test_dataset = full_test_dataset
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train Points: {len(train_dataset)}, Test Points: {len(test_dataset)}")
    
    if args.model == 'pointnet':
        print("Initializing PointNet Baseline...")
        model = PointNetClassifier(k=40)
    elif args.model == 'transformer':
        print("Initializing Lightweight Transformer...")
        model = TransformerClassifier(num_classes=40)
    
    save_dir = f'./checkpoints_{args.model}'
    
    print(f"Starting Training for {args.model}...")
    run_experiment(model, train_loader, test_loader, epochs=args.epochs, lr=1e-3, device=device, save_dir=save_dir)
    print("Training loop completed successfully.")
