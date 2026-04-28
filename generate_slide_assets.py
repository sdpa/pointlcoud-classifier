"""
generate_slide_assets.py
Generates all figures needed for the presentation slides.
Run: python generate_slide_assets.py
Output: slide_assets/ directory
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (side-effect import)

os.makedirs('slide_assets', exist_ok=True)

RESULTS_BASE = 'results'

PALETTE = {
    'blue':       '#2563EB',
    'green':      '#16A34A',
    'orange':     '#EA580C',
    'purple':     '#7C3AED',
    'red':        '#DC2626',
    'gray':       '#6B7280',
    'light_blue': '#93C5FD',
    'light_green':'#86EFAC',
    'bg':         '#F8FAFC',
    'text':       '#1E293B',
}


# ── helpers ────────────────────────────────────────────────────────────────────

def styled_fig(w=12, h=6):
    fig = plt.figure(figsize=(w, h), facecolor=PALETTE['bg'])
    return fig


def styled_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('#FFFFFF')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CBD5E1')
    ax.spines['bottom'].set_color('#CBD5E1')
    ax.tick_params(colors=PALETTE['text'], labelsize=10)
    ax.xaxis.label.set_color(PALETTE['text'])
    ax.yaxis.label.set_color(PALETTE['text'])
    if title:  ax.set_title(title,  color=PALETTE['text'], fontsize=13, fontweight='bold', pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=PALETTE['text'], fontsize=11)
    if ylabel: ax.set_ylabel(ylabel, color=PALETTE['text'], fontsize=11)


def load_history(checkpoint_dir):
    path = os.path.join(RESULTS_BASE, checkpoint_dir, 'training_history.csv')
    rows = list(csv.DictReader(open(path)))
    epochs     = [int(r['epoch'])       for r in rows]
    train_acc  = [float(r['train_acc']) for r in rows]
    test_acc   = [float(r['test_acc'])  for r in rows]
    train_loss = [float(r['train_loss'])for r in rows]
    test_loss  = [float(r['test_loss']) for r in rows]
    return epochs, train_acc, test_acc, train_loss, test_loss


def best_test(checkpoint_dir):
    _, _, test_acc, _, _ = load_history(checkpoint_dir)
    return max(test_acc) * 100


# ══════════════════════════════════════════════════════════════════════════════
# FIG 0 — Raw ModelNet40 objects (4 distinct categories from .off meshes)
#   Visualised as surface-sampled point clouds to show what the raw data
#   looks like before any preprocessing pipeline is applied.
# ══════════════════════════════════════════════════════════════════════════════

import h5py
import trimesh

def sample_raw_off(path, n=2048, seed=0):
    """Load a .off mesh and return a normalised surface-sampled point cloud."""
    np.random.seed(seed)
    mesh = trimesh.load(path, force='mesh')
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    pts = pts - pts.mean(axis=0)
    pts = pts / np.max(np.linalg.norm(pts, axis=1))
    return pts.astype(np.float32)


RAW_OBJECTS = [
    ('airplane', 'data/raw_modelnet40/airplane/train/airplane_0001.off', (20, -60),  PALETTE['blue']),
    ('chair',    'data/raw_modelnet40/chair/train/chair_0001.off',       (20, -45),  PALETTE['green']),
    ('sofa',     'data/raw_modelnet40/sofa/train/sofa_0001.off',          (25, -45),  PALETTE['orange']),
    ('lamp',     'data/raw_modelnet40/lamp/train/lamp_0001.off',         (15, -60),  PALETTE['purple']),
]

print("Generating raw object visualisations ...")
for name, off_path, (elev, azim), color in RAW_OBJECTS:
    pts = sample_raw_off(off_path)

    fig = plt.figure(figsize=(5, 5), facecolor=PALETTE['bg'])
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3, c=color, alpha=0.80)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(name.capitalize(), color=PALETTE['text'],
                 fontsize=13, fontweight='bold', pad=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_facecolor(PALETTE['bg'])
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')

    fname = f'slide_assets/raw_{name}.png'
    plt.savefig(fname, dpi=180, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ saved  {fname}")

# Combined 4-panel version
print("Generating fig_raw_objects.png (combined) ...")
fig = styled_fig(18, 5.2)
fig.suptitle('ModelNet40 Raw Data — 4 Object Categories  (surface-sampled, unit-sphere normalised)',
             color=PALETTE['text'], fontsize=13, fontweight='bold', y=1.02)

for i, (name, off_path, (elev, azim), color) in enumerate(RAW_OBJECTS):
    pts = sample_raw_off(off_path)
    ax  = fig.add_subplot(1, 4, i + 1, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3, c=color, alpha=0.80)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(name.capitalize(), color=PALETTE['text'],
                 fontsize=12, fontweight='bold', pad=6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_facecolor(PALETTE['bg'])
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')

plt.tight_layout()
plt.savefig('slide_assets/fig_raw_objects.png', dpi=150, bbox_inches='tight',
            facecolor=PALETTE['bg'])
plt.close()
print("  ✓ saved  slide_assets/fig_raw_objects.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Sampling method comparison
#   Loads the SAME airplane (index 0, label=0) from each real HDF5 dataset.
#   All four files processed the same source files in sorted order, so
#   index 0 is airplane_0001.off in every dataset — only the sampling differs.
# ══════════════════════════════════════════════════════════════════════════════

def fps_np(pts, k):
    N = len(pts)
    sel = np.zeros(k, dtype=int)
    min_d = np.full(N, np.inf)
    cur = 0
    for i in range(k):
        sel[i] = cur
        d = np.sum((pts - pts[cur])**2, axis=1)
        min_d = np.minimum(min_d, d)
        cur = int(np.argmax(min_d))
    return sel


def load_cloud(method, obj_idx=0, split='train'):
    """Load a single 2048-pt cloud from the preprocessed HDF5 file."""
    path = f'data/hdf5_{method}/ply_data_{split}_0.h5'
    with h5py.File(path, 'r') as f:
        pts = f['data'][obj_idx].astype(np.float32)   # (2048, 3)
    return pts


print("Generating individual sampling method images ...")
np.random.seed(42)

# airplane_0001 → index 0 (label 0) in all four sorted HDF5 datasets
AIRPLANE_IDX = 0

METHOD_LABELS = [
    ('uniform',   'Uniform',             'Area-weighted Random',   PALETTE['blue']),
    ('poisson',   'Poisson Disk',        'Blue-noise',             PALETTE['green']),
    ('curvature', 'Curvature-Weighted',  'Detail-aware',           PALETTE['orange']),
    ('vertex',    'Vertex Sampling',     'Topology-driven',        PALETTE['purple']),
]

# consistent viewing angle so the airplane silhouette is clear across all images
ELEV, AZIM = 20, -60

for method, title, subtitle, color in METHOD_LABELS:
    pts = load_cloud(method, AIRPLANE_IDX)

    fig = plt.figure(figsize=(6, 5), facecolor=PALETTE['bg'])
    ax  = fig.add_subplot(111, projection='3d')

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, c=color, alpha=0.80)
    ax.view_init(elev=ELEV, azim=AZIM)

    ax.set_title(f'{title}\n({subtitle})',
                 color=PALETTE['text'], fontsize=12, fontweight='bold', pad=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_facecolor(PALETTE['bg'])
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')

    fname = f'slide_assets/airplane_{method}.png'
    plt.savefig(fname, dpi=180, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ saved  {fname}")

# also keep the combined 4-panel version for reference
print("Generating fig_sampling_comparison.png (combined) ...")
fig = styled_fig(16, 5.0)
fig.suptitle(
    'Stage-1 Sampling Methods — Same Airplane (airplane_0001), 2048 pts each',
    color=PALETTE['text'], fontsize=13, fontweight='bold', y=1.02)

for idx, (method, title, subtitle, color) in enumerate(METHOD_LABELS):
    pts = load_cloud(method, AIRPLANE_IDX)
    ax  = fig.add_subplot(1, 4, idx + 1, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3, c=color, alpha=0.75)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_title(f'{title}\n({subtitle})',
                 color=PALETTE['text'], fontsize=10, fontweight='bold', pad=6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_facecolor(PALETTE['bg'])
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')

plt.tight_layout()
plt.savefig('slide_assets/fig_sampling_comparison.png', dpi=150, bbox_inches='tight',
            facecolor=PALETTE['bg'])
plt.close()
print("  ✓ saved  slide_assets/fig_sampling_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — FPS vs Random subsampling  (same airplane, uniform HDF5)
#   Grey = full 2048-pt stored cloud.  Colour = 512 selected points.
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_fps_vs_random.png ...")
np.random.seed(7)

# Use the uniform HDF5 cloud — this is what Stage-2 operates on at runtime
dense2 = load_cloud('uniform', AIRPLANE_IDX)
N_sub  = 512

rand_idx = np.random.choice(len(dense2), N_sub, replace=False)
fps_idx  = fps_np(dense2, N_sub)

fig = styled_fig(12, 5.2)
fig.suptitle('Stage-2 Subsampling: 2048 → 512 pts  (uniform airplane cloud)',
             color=PALETTE['text'], fontsize=13, fontweight='bold', y=1.01)

titles  = ['Random Subsample', 'Farthest Point Sampling (FPS)']
indices = [rand_idx, fps_idx]
cols    = [PALETTE['orange'], PALETTE['blue']]

for i, (t, idx, c) in enumerate(zip(titles, indices, cols)):
    ax = fig.add_subplot(1, 2, i+1, projection='3d')
    ax.scatter(dense2[:, 0], dense2[:, 1], dense2[:, 2], s=1, c='#CBD5E1', alpha=0.25)
    sel_pts = dense2[idx]
    ax.scatter(sel_pts[:, 0], sel_pts[:, 1], sel_pts[:, 2], s=10, c=c, alpha=0.9)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_title(t, color=PALETTE['text'], fontsize=12, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_facecolor(PALETTE['bg'])

plt.tight_layout()
plt.savefig('slide_assets/fig_fps_vs_random.png', dpi=150, bbox_inches='tight',
            facecolor=PALETTE['bg'])
plt.close()
print("  ✓ saved")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Results horizontal bar chart
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_results_bar.png ...")

experiments = [
    ('Poisson + FPS',              'checkpoints_pointnet_hdf5_poisson_fps',    PALETTE['blue'],   True),
    ('Poisson + Random',           'checkpoints_pointnet_hdf5_poisson_random',  PALETTE['blue'],   False),
    ('Uniform + FPS',              'checkpoints_pointnet_hdf5_uniform_fps',     PALETTE['green'],  True),
    ('Uniform + Random',           'checkpoints_pointnet_hdf5_uniform_random',  PALETTE['green'],  False),
    ('Curvature + FPS',            'checkpoints_pointnet_hdf5_curvature_fps',   PALETTE['orange'], True),
    ('Curvature + Random',         'checkpoints_pointnet_hdf5_curvature_random',PALETTE['orange'], False),
    ('Vertex + FPS',               'checkpoints_pointnet_hdf5_vertex_fps',      PALETTE['purple'], True),
    ('Vertex + Random',            'checkpoints_pointnet_hdf5_vertex_random',   PALETTE['purple'], False),
    ('PointNet Baseline',          'checkpoints_pointnet',                       PALETTE['gray'],   False),
    ('Transformer (comparison)',   'checkpoints_transformer 2',                  PALETTE['red'],    False),
]

labels  = [e[0] for e in experiments]
accs    = [best_test(e[1]) for e in experiments]
colors  = [e[2] for e in experiments]
alphas  = [1.0 if e[3] else 0.55 for e in experiments]

fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=PALETTE['bg'])
styled_ax(ax, title='Best Test Accuracy by Experiment  (PointNet backbone)',
          xlabel='Best Test Accuracy (%)')

y_pos = np.arange(len(labels))
bars = ax.barh(y_pos, accs, color=colors, alpha=1.0, height=0.65,
               edgecolor='white', linewidth=0.8)

for bar, al in zip(bars, alphas):
    bar.set_alpha(al)

# value labels
for bar, acc in zip(bars, accs):
    ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height()/2,
            f'{acc:.2f}%', va='center', ha='left', fontsize=10,
            color=PALETTE['text'], fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlim(70, 92)

# Baseline = PointNet original
baseline_acc = best_test('checkpoints_pointnet')
ax.axvline(x=baseline_acc, color=PALETTE['gray'], linestyle='--', linewidth=1.5,
           label=f'PointNet baseline ({baseline_acc:.2f}%)')
ax.axvline(x=86.99, color=PALETTE['blue'], linestyle='--', linewidth=1.5,
           label='Best result — Poisson + FPS (86.99%)')

# legend for FPS vs Random
fps_patch   = mpatches.Patch(color=PALETTE['gray'], alpha=1.0, label='FPS subsample')
rand_patch  = mpatches.Patch(color=PALETTE['gray'], alpha=0.5, label='Random subsample')
ax.legend(handles=[fps_patch, rand_patch], loc='lower right', fontsize=10,
          framealpha=0.9, edgecolor='#CBD5E1')

ax.invert_yaxis()
plt.tight_layout()
plt.savefig('slide_assets/fig_results_bar.png', dpi=150, bbox_inches='tight',
            facecolor=PALETTE['bg'])
plt.close()
print("  ✓ saved")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Training curves comparison (top 4 + transformer)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_training_curves_comparison.png ...")

KEY_RUNS = [
    ('PointNet Baseline  (uniform+random)', 'checkpoints_pointnet',                       PALETTE['gray'],  '-'),
    ('Poisson + FPS  (best)',               'checkpoints_pointnet_hdf5_poisson_fps',       PALETTE['blue'],  '-'),
    ('Poisson + Random',                    'checkpoints_pointnet_hdf5_poisson_random',    '#60A5FA',        '--'),
    ('Uniform + FPS',                       'checkpoints_pointnet_hdf5_uniform_fps',       PALETTE['green'], '--'),
    ('Transformer  (comparison)',           'checkpoints_transformer 2',                   PALETTE['red'],   ':'),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=PALETTE['bg'])
fig.suptitle('Training Convergence — Key Experiments  (100 epochs)',
             color=PALETTE['text'], fontsize=13, fontweight='bold')

ax_loss, ax_acc = axes
styled_ax(ax_loss, title='Test Loss', xlabel='Epoch', ylabel='Loss')
styled_ax(ax_acc,  title='Test Accuracy', xlabel='Epoch', ylabel='Accuracy (%)')

for label, ckpt, color, ls in KEY_RUNS:
    try:
        epochs, train_acc, test_acc, train_loss, test_loss = load_history(ckpt)
        test_acc_pct = [a*100 for a in test_acc]
        ax_loss.plot(epochs, test_loss, color=color, linestyle=ls, linewidth=2, label=label)
        ax_acc.plot(epochs,  test_acc_pct, color=color, linestyle=ls, linewidth=2, label=label)
    except Exception as e:
        print(f"  [skip] {ckpt}: {e}")

ax_acc.axhline(y=86.99, color=PALETTE['blue'], linestyle=':', linewidth=1,
               alpha=0.5, label='Best (86.99%)')
ax_acc.legend(fontsize=9, framealpha=0.9, edgecolor='#CBD5E1', loc='lower right')
ax_loss.legend(fontsize=9, framealpha=0.9, edgecolor='#CBD5E1')

plt.tight_layout()
plt.savefig('slide_assets/fig_training_curves_comparison.png', dpi=150, bbox_inches='tight',
            facecolor=PALETTE['bg'])
plt.close()
print("  ✓ saved")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Results summary table
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_results_table.png ...")

table_data = [
    ['Experiment',                  'Stage-1',   'Stage-2', 'Best Test Acc', 'Last Train Acc'],
    ['Poisson + FPS',               'Poisson',   'FPS',     '86.99%',        '85.81%'],
    ['Poisson + Random',            'Poisson',   'Random',  '86.30%',        '85.43%'],
    ['Uniform + FPS',               'Uniform',   'FPS',     '86.30%',        '85.63%'],
    ['Curvature + FPS',             'Curvature', 'FPS',     '85.66%',        '84.87%'],
    ['Curvature + Random',          'Curvature', 'Random',  '85.29%',        '84.42%'],
    ['Vertex + FPS',                'Vertex',    'FPS',     '82.94%',        '81.33%'],
    ['Vertex + Random',             'Vertex',    'Random',  '81.89%',        '80.87%'],
    ['PointNet Baseline ★',         'Uniform',   'Random',  '85.21%',        '85.48%'],
    ['Transformer (comparison)',    'Uniform',   'Random',  '76.99%',        '85.68%'],
]

fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=PALETTE['bg'])
ax.axis('off')

col_widths = [0.32, 0.14, 0.12, 0.18, 0.18]
col_positions = [0.02, 0.34, 0.49, 0.62, 0.81]
row_h = 0.082
header_h = 0.88

ROW_COLORS = ['#EFF6FF', '#F0FDF4', '#FFF7ED', '#FAF5FF', '#FEF2F2']

for r_idx, row in enumerate(table_data):
        y = header_h - r_idx * row_h
        is_header   = r_idx == 0
        is_best     = r_idx == 1   # Poisson+FPS
        is_baseline = r_idx == 8   # PointNet Baseline

        if is_header:
            rect = mpatches.FancyBboxPatch((0.01, y - row_h*0.5), 0.98, row_h*0.92,
                                            boxstyle='round,pad=0.005',
                                            facecolor=PALETTE['blue'], edgecolor='none',
                                            transform=ax.transAxes, zorder=1)
        elif is_best:
            rect = mpatches.FancyBboxPatch((0.01, y - row_h*0.5), 0.98, row_h*0.92,
                                            boxstyle='round,pad=0.005',
                                            facecolor='#DBEAFE', edgecolor=PALETTE['blue'],
                                            linewidth=1.5, transform=ax.transAxes, zorder=1)
        elif is_baseline:
            rect = mpatches.FancyBboxPatch((0.01, y - row_h*0.5), 0.98, row_h*0.92,
                                            boxstyle='round,pad=0.005',
                                            facecolor='#F1F5F9', edgecolor=PALETTE['gray'],
                                            linewidth=1.5, transform=ax.transAxes, zorder=1)
        else:
            bg = '#FFFFFF' if r_idx % 2 == 0 else '#F8FAFC'
            rect = mpatches.FancyBboxPatch((0.01, y - row_h*0.5), 0.98, row_h*0.92,
                                            boxstyle='round,pad=0.005',
                                            facecolor=bg, edgecolor='none',
                                            transform=ax.transAxes, zorder=1)
        ax.add_patch(rect)

        for c_idx, (cell, xpos) in enumerate(zip(row, col_positions)):
            color = 'white' if is_header else PALETTE['text']
            weight = 'bold' if is_header or (c_idx == 3) else 'normal'
            size = 10.5 if is_header else (11 if c_idx == 3 else 10)
            ax.text(xpos, y, cell, transform=ax.transAxes,
                    fontsize=size, color=color, fontweight=weight,
                    va='center', ha='left', zorder=2)

# annotations
ax.text(0.99, header_h - 1*row_h, '▲ Best', transform=ax.transAxes,
        fontsize=10, color=PALETTE['blue'], fontweight='bold',
        va='center', ha='right', zorder=3)
ax.text(0.99, header_h - 8*row_h, '← Baseline', transform=ax.transAxes,
        fontsize=9, color=PALETTE['gray'], fontweight='bold',
        va='center', ha='right', zorder=3)

ax.set_title('Summary of All Experiments — PointNet on ModelNet40',
             color=PALETTE['text'], fontsize=13, fontweight='bold',
             pad=10, loc='left')

plt.tight_layout()
plt.savefig('slide_assets/fig_results_table.png', dpi=150, bbox_inches='tight',
            facecolor=PALETTE['bg'])
plt.close()
print("  ✓ saved")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Architecture comparison diagram
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_architecture_comparison.png ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 6.5), facecolor=PALETTE['bg'])
fig.suptitle('Model Architecture Comparison', color=PALETTE['text'],
             fontsize=14, fontweight='bold', y=1.02)

ARCH_SPECS = [
    {
        'title': 'PointNet\n(0.8M params)',
        'color': PALETTE['green'],
        'blocks': [
            ('Input\n(B, 3, N)', '#F1F5F9'),
            ('Conv1D 3→64\n+ BN + ReLU', '#D1FAE5'),
            ('Conv1D 64→128\n+ BN + ReLU', '#D1FAE5'),
            ('Conv1D 128→1024\n+ BN', '#D1FAE5'),
            ('Global Max-Pool\n→ (B, 1024)', '#FEF9C3'),
            ('FC 1024→512\n+ BN + ReLU', '#D1FAE5'),
            ('FC 512→256\n+ Dropout(0.4)', '#D1FAE5'),
            ('FC 256→40\nLog-Softmax', '#E0F2FE'),
        ],
        'note': 'O(N) — point-wise ops\n+ single max-pool',
    },
    {
        'title': 'Lightweight Transformer\n(0.4M params)',
        'color': PALETTE['blue'],
        'blocks': [
            ('Input\n(B, 3, N)', '#F1F5F9'),
            ('Point Embed\n3→64→128', '#DBEAFE'),
            ('TransformerEncoder\n3 layers, 4 heads\nd=128, FFN=256', '#BFDBFE'),
            ('Global Max-Pool\n→ (B, 128)', '#FEF9C3'),
            ('MLP Classifier\n128→256→128→40', '#DBEAFE'),
            ('Log-Softmax', '#E0F2FE'),
        ],
        'note': 'O(N²) self-attention\nfull pairwise context',
    },
    {
        'title': 'Hierarchical Transformer\n(0.5M params)',
        'color': PALETTE['purple'],
        'blocks': [
            ('Input\n(B, 3, N=1024)', '#F1F5F9'),
            ('FPS\nN=1024 → M=256', '#EDE9FE'),
            ('k-NN Grouping\nk=16 neighbours\nper centroid', '#EDE9FE'),
            ('Local Attention\nper neighbourhood\n→ (B, M, 128)', '#DDD6FE'),
            ('Global Attention\nM=256 centroid tokens', '#DDD6FE'),
            ('Max-Pool + MLP\n→ 40 classes', '#E0F2FE'),
        ],
        'note': 'O(M·k + M²)\n~10× cheaper than flat',
    },
]

for ax, spec in zip(axes, ARCH_SPECS):
    ax.set_facecolor(PALETTE['bg'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(spec['title'], color=spec['color'], fontsize=12, fontweight='bold', pad=8)

    n = len(spec['blocks'])
    block_h = 0.09
    gap = (0.88 - n * block_h) / max(n - 1, 1)
    x0, w = 0.08, 0.84

    for i, (label, bg) in enumerate(spec['blocks']):
        y = 0.94 - i * (block_h + gap)
        rect = mpatches.FancyBboxPatch((x0, y - block_h), w, block_h,
                                        boxstyle='round,pad=0.01',
                                        facecolor=bg, edgecolor=spec['color'],
                                        linewidth=1.2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.5, y - block_h/2, label, transform=ax.transAxes,
                ha='center', va='center', fontsize=8.5, color=PALETTE['text'])
        if i < n - 1:
            ax.annotate('', xy=(0.5, y - block_h - gap),
                        xytext=(0.5, y - block_h),
                        xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color=spec['color'],
                                        lw=1.5))

    ax.text(0.5, 0.02, spec['note'], transform=ax.transAxes,
            ha='center', va='bottom', fontsize=8.5, color='#64748B',
            style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F1F5F9', edgecolor='none'))

plt.tight_layout()
plt.savefig('slide_assets/fig_architecture_comparison.png', dpi=150, bbox_inches='tight',
            facecolor=PALETTE['bg'])
plt.close()
print("  ✓ saved")

# Individual architecture images — PointNet and Lightweight Transformer only
INDIVIDUAL_ARCHS = [
    (ARCH_SPECS[0], 'arch_pointnet.png'),
    (ARCH_SPECS[1], 'arch_transformer.png'),
]

for spec, fname in INDIVIDUAL_ARCHS:
    fig, ax = plt.subplots(figsize=(4.5, 7.5), facecolor=PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # larger title above the diagram
    fig.text(0.5, 0.97, spec['title'], ha='center', va='top',
             color=spec['color'], fontsize=14, fontweight='bold')

    n = len(spec['blocks'])
    block_h = 0.085
    total_arrow_space = 0.025 * (n - 1)
    total_block_space = block_h * n
    top_margin  = 0.10   # space below title
    bot_margin  = 0.10   # space above note
    available   = 1.0 - top_margin - bot_margin
    gap = (available - total_block_space) / max(n - 1, 1)
    x0, w = 0.06, 0.88

    for i, (label, bg) in enumerate(spec['blocks']):
        y = (1.0 - top_margin) - i * (block_h + gap)
        rect = mpatches.FancyBboxPatch((x0, y - block_h), w, block_h,
                                        boxstyle='round,pad=0.012',
                                        facecolor=bg, edgecolor=spec['color'],
                                        linewidth=1.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.5, y - block_h / 2, label, transform=ax.transAxes,
                ha='center', va='center', fontsize=9.5, color=PALETTE['text'])
        if i < n - 1:
            ax.annotate('', xy=(0.5, y - block_h - gap),
                        xytext=(0.5, y - block_h),
                        xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color=spec['color'], lw=1.8))

    ax.text(0.5, 0.03, spec['note'], transform=ax.transAxes,
            ha='center', va='bottom', fontsize=9, color='#64748B', style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F1F5F9', edgecolor='none'))

    plt.savefig(f'slide_assets/{fname}', dpi=180, bbox_inches='tight',
                facecolor=PALETTE['bg'])
    plt.close()
    print(f"  ✓ saved  slide_assets/{fname}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Pipeline overview
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig_pipeline_overview.png ...")

fig, ax = plt.subplots(figsize=(14, 3.5), facecolor=PALETTE['bg'])
ax.axis('off')
ax.set_xlim(0, 14); ax.set_ylim(0, 3.5)

STEPS = [
    ('Raw .off\nMesh',        0.9,  '#F1F5F9', PALETTE['gray']),
    ('Stage-1\nSampling\n(2048 pts)', 3.2,  '#D1FAE5', PALETTE['green']),
    ('HDF5\nStorage',         5.5,  '#DBEAFE', PALETTE['blue']),
    ('Stage-2\nSubsample\n(1024 pts)', 7.8,  '#FEF9C3', '#B45309'),
    ('Neural\nNetwork',       10.1, '#EDE9FE', PALETTE['purple']),
    ('40-class\nPrediction',  12.4, '#FEE2E2', PALETTE['red']),
]

bw, bh, by = 1.6, 1.2, 1.15
for label, cx, bg, ec in STEPS:
    rect = mpatches.FancyBboxPatch((cx - bw/2, by), bw, bh,
                                    boxstyle='round,pad=0.1',
                                    facecolor=bg, edgecolor=ec, linewidth=2)
    ax.add_patch(rect)
    ax.text(cx, by + bh/2, label, ha='center', va='center',
            fontsize=10, fontweight='bold', color=PALETTE['text'])

# arrows
for i in range(len(STEPS) - 1):
    cx_from = STEPS[i][1] + bw/2
    cx_to   = STEPS[i+1][1] - bw/2
    cy = by + bh/2
    ax.annotate('', xy=(cx_to, cy), xytext=(cx_from, cy),
                arrowprops=dict(arrowstyle='->', color='#94A3B8', lw=2))

# stage labels below
for label, cx, _, _ in [('OFFLINE\n(once)', 4.0, None, None),
                          ('ONLINE\n(each epoch)', 7.8, None, None)]:
    ax.text(cx, by - 0.45, label, ha='center', va='top', fontsize=9,
            color='#64748B', style='italic')

# bracket for stage 1
ax.annotate('', xy=(5.3, by - 0.15), xytext=(0.0, by - 0.15),
            arrowprops=dict(arrowstyle='-', color='#16A34A', lw=2))
ax.annotate('', xy=(5.3, by - 0.15), xytext=(5.3, by - 0.05),
            arrowprops=dict(arrowstyle='-', color='#16A34A', lw=2))
ax.annotate('', xy=(0.0, by - 0.15), xytext=(0.0, by - 0.05),
            arrowprops=dict(arrowstyle='-', color='#16A34A', lw=2))

ax.text(7, 0.5, '4 Stage-1 methods  ×  2 Stage-2 methods  =  8 combinations tested',
        ha='center', va='center', fontsize=10, color=PALETTE['text'],
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8FAFC',
                  edgecolor='#CBD5E1', linewidth=1))

ax.set_title('Two-Stage Sampling Pipeline', color=PALETTE['text'],
             fontsize=13, fontweight='bold', pad=8, loc='left')

plt.tight_layout()
plt.savefig('slide_assets/fig_pipeline_overview.png', dpi=150, bbox_inches='tight',
            facecolor=PALETTE['bg'])
plt.close()
print("  ✓ saved")


print("\nAll assets saved to slide_assets/")
print("\nFiles generated:")
for f in sorted(os.listdir('slide_assets')):
    size = os.path.getsize(f'slide_assets/{f}') // 1024
    print(f"  {f}  ({size} KB)")
