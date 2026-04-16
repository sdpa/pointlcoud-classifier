# Point Cloud Classifier

A PyTorch-based 3D point cloud classification project trained on the **ModelNet40** dataset. Three model architectures are implemented and compared:

- **PointNet** — A lightweight baseline using shared MLP layers and global max pooling.
- **Transformer** — A lightweight flat self-attention encoder that models global interactions between all 1024 point embeddings.
- **Hierarchical Transformer** — A local-to-global architecture that first applies Farthest Point Sampling + k-NN grouping to build local neighbourhoods, encodes each neighbourhood with a small self-attention block, then runs a global transformer over the 256 resulting centroid features. Bridges the gap between PointNet's local structure sensitivity and the Transformer's global reasoning, while reducing attention complexity from O(N²) to O(M·k + M²).

---

## Project Structure

```
pointcloud_classifier/
├── data/                        # Dataset directory (created after download)
│   └── modelnet40_ply_hdf5_2048/
├── checkpoints_pointnet/        # Saved checkpoints for PointNet
├── checkpoints_transformer/     # Saved checkpoints for Transformer
├── checkpoints_hierarchical/    # Saved checkpoints for Hierarchical Transformer
├── dataset.py                   # ModelNet40 PyTorch Dataset with augmentation
├── pointnet.py                  # PointNet model architecture
├── transformer.py               # Flat Transformer model architecture
├── hierarchical_transformer.py  # Hierarchical (Local-to-Global) Transformer
├── train.py                     # Training script (main entry point)
├── download_data.py             # Downloads the ModelNet40 HDF5 dataset
├── preprocess_data.py           # Preprocesses raw .OFF mesh files into HDF5
├── generate_synthetic_data.py   # Generates a small synthetic dataset for testing
├── eda.py                       # Exploratory data analysis on raw mesh files
├── pointcloud_classifier.ipynb  # Jupyter notebook walkthrough
├── requirements.txt             # Python dependencies
└── report.tex                   # LaTeX report
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Data

### Option A — Download the pre-built HDF5 dataset (recommended)

This downloads and extracts the ModelNet40 dataset (~435 MB) pre-sampled at 2048 points per shape into `data/modelnet40_ply_hdf5_2048/`.

```bash
python download_data.py
```

### Option B — Use a synthetic dataset (for quick testing only)

If you just want to verify the training pipeline without downloading real data, generate a small random dataset:

```bash
python generate_synthetic_data.py
```

> **Note:** Models trained on synthetic data will not produce meaningful accuracy results. Use this only for smoke-testing the code.

### Option C — Preprocess raw ModelNet40 meshes

If you have the raw `.OFF` mesh files from ModelNet40, place them under `data/raw_modelnet40/` (with the standard class/split folder structure) and run:

```bash
python preprocess_data.py
```

This samples 2048 points per mesh, normalizes them to a unit sphere, and saves them as HDF5 files in `data/modelnet40_ply_hdf5_2048/`.

---

## Training

Run `train.py` with your chosen model architecture.

### Train PointNet (default)

```bash
python train.py --model pointnet --epochs 100 --batch_size 64
```

### Train Transformer

```bash
python train.py --model transformer --epochs 100 --batch_size 64
```

### Train Hierarchical Transformer (Local-to-Global)

```bash
python train.py --model hierarchical --epochs 100 --batch_size 32
```

> **Note:** The hierarchical model runs Farthest Point Sampling on the CPU each forward pass. Use `--batch_size 32` if you encounter memory pressure; the FPS loop is the main bottleneck on large batches.

### All CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `pointnet` | Model to train: `pointnet` or `transformer` |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `64` | Batch size |
| `--debug` | `False` | Use a tiny 128-sample subset for fast iteration |

### Debug / quick smoke-test

```bash
python train.py --model pointnet --debug
```

This runs only 128 training and 128 test samples, allowing you to verify the full pipeline in seconds.

---

## Outputs

After training, the following files are saved to `checkpoints_<model>/`:

| File | Description |
|---|---|
| `best_model.pth` | Model weights with the highest test accuracy |
| `final_model.pth` | Model weights at the end of training |
| `training_history.csv` | Per-epoch loss and accuracy metrics |
| `training_curves.png` | Loss and accuracy plots |

---

## Exploratory Data Analysis

To analyze point count distributions across the raw ModelNet40 `.OFF` mesh files (requires raw data in `data/raw_modelnet40/`):

```bash
python eda.py
```

This prints summary statistics and saves a histogram to `points_distribution.png`.

---

## Jupyter Notebook

A notebook walkthrough is included for interactive exploration:

```bash
jupyter notebook pointcloud_classifier.ipynb
```

---

## Device Support

Training automatically uses the best available device:

- **CUDA** (NVIDIA GPU) — if available
- **MPS** (Apple Silicon GPU) — if available
- **CPU** — fallback

---

## Running on a SLURM Cluster

### Step 1 — Copy the project to scratch (do this once)

```bash
cp -r pointcloud_classifier $SCRATCH/pointcloud_classifier
cd $SCRATCH/pointcloud_classifier
```

### Step 2 — Set up the environment (do this once, on the login node)

```bash
bash setup_env.sh
```

This creates `venv/`, installs all dependencies, and downloads the ModelNet40 dataset into `data/`.

### Step 3 — Submit a job

```bash
# Train with default settings (PointNet, 100 epochs, batch 64)
sbatch job.slurm

# Choose a different model
sbatch --export=MODEL=transformer job.slurm
sbatch --export=MODEL=hierarchical job.slurm

# Override any hyperparameter
sbatch --export=MODEL=hierarchical,EPOCHS=50,BATCH=32 job.slurm

# Quick smoke-test (128 samples, completes in ~1 min)
sbatch --export=MODEL=pointnet,DEBUG=1 job.slurm
```

### Step 4 — Monitor the job

```bash
squeue -u $USER                     # check job status
cat logs/slurm_<job_id>_pointcloud.out   # live stdout
```

> **Before submitting:** open `job.slurm` and update `#SBATCH --partition=gpu` to match your cluster's GPU partition name. You can find available partitions with `sinfo`.

---

## Expected Results

On the full ModelNet40 dataset with default hyperparameters (100 epochs):

| Model | Test Accuracy | Notes |
|---|---|---|
| PointNet | ~85–87% | Fastest to train |
| Transformer (flat) | ~85–88% | O(N²) on all 1024 points |
| Hierarchical Transformer | ~88–91% | O(M·k + M²), M=256, k=16 |

Actual results may vary depending on hardware, random seed, and number of epochs.
