"""
preprocess_data.py  —  Mesh → HDF5 point cloud converter for ModelNet40

Four sampling strategies are available via --method:

  uniform   (default)
    Area-weighted triangle selection, then a uniformly-random barycentric
    point inside that triangle.  This is what trimesh.sample.sample_surface
    does internally.  Produces a distribution proportional to surface area,
    so large flat patches get the most points regardless of local detail.

  poisson
    Poisson-disk (blue-noise) sampling via trimesh.sample.sample_surface_even.
    Enforces a minimum inter-point distance, guaranteeing no two sampled
    points are closer than r = sqrt(area / (pi * N)).  The result is
    spatially uniform in 3-D — no clustering, no large voids — unlike the
    random clustering that pure uniform sampling exhibits.  If the library
    returns fewer than num_points, the gap is filled with fallback uniform
    samples.

  curvature
    Blends area-proportional weight (50 %) with curvature-proportional
    weight (50 %) so that high-curvature regions (edges, corners, sharp
    features) receive a higher point density.  Per-face curvature is
    estimated by averaging the angle-deficit (discrete Gaussian curvature)
    of each face's three vertices.  The final sample is drawn via the same
    random-barycentric scheme as uniform but with the modified face
    probability vector.  This is motivated by the observation that flat
    surfaces are geometrically redundant — a classifier needs only a few
    points to identify a flat plane, but many more to resolve a detailed
    handle or leg joint.

  vertex
    Directly subsamples the raw mesh vertices without any surface
    interpolation.  This preserves the exact keypoints the mesh designer
    placed, but density is topology-dependent (a tesselated sphere has poles
    with more vertices than equatorial regions).  Useful as a baseline to
    compare sampled-surface methods against raw vertex structure.

Usage
-----
    python preprocess_data.py --method uniform   --source data/raw_modelnet40 --dest data/hdf5_uniform
    python preprocess_data.py --method poisson   --source data/raw_modelnet40 --dest data/hdf5_poisson
    python preprocess_data.py --method curvature --source data/raw_modelnet40 --dest data/hdf5_curvature
    python preprocess_data.py --method vertex    --source data/raw_modelnet40 --dest data/hdf5_vertex
"""

import os
import glob
import argparse
import h5py
import numpy as np
import trimesh
from tqdm import tqdm


# ── Sampling strategies ────────────────────────────────────────────────────────

def sample_uniform(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    """Area-weighted uniform surface sampling (original method)."""
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points.astype(np.float32)


def sample_poisson(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    """
    Poisson-disk (blue-noise) sampling.

    trimesh.sample.sample_surface_even guarantees no two points are closer
    than a radius derived from total surface area / N.  The returned count
    may differ slightly from num_points; we fix this with a uniform fallback
    or a random trim.
    """
    points, _ = trimesh.sample.sample_surface_even(mesh, num_points)
    n = len(points)

    if n >= num_points:
        # Trim to exact count randomly
        idx = np.random.choice(n, num_points, replace=False)
        points = points[idx]
    else:
        # Fill shortage with ordinary uniform samples
        extra, _ = trimesh.sample.sample_surface(mesh, num_points - n)
        points = np.vstack([points, extra])

    return points.astype(np.float32)


def sample_curvature(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    """
    Curvature-weighted surface sampling.

    Face sampling probability = 0.5 * (normalised area)
                               + 0.5 * (normalised mean curvature of face)

    Per-face curvature is the mean of its three vertices' discrete Gaussian
    curvature estimates (angle deficit).  A flat face has defect ≈ 0 and is
    sampled only in proportion to its area; a highly curved face is boosted.
    """
    areas = mesh.area_faces                          # (F,)

    try:
        vertex_curvature = np.abs(mesh.vertex_defects)   # (V,)
    except Exception:
        # Fallback: all curvatures equal — degenerates to uniform
        vertex_curvature = np.ones(len(mesh.vertices), dtype=np.float32)

    # Replace NaN / inf that can appear on degenerate geometry
    vertex_curvature = np.nan_to_num(vertex_curvature, nan=0.0, posinf=0.0)

    face_curvature = vertex_curvature[mesh.faces].mean(axis=1)   # (F,)

    # Normalise each term to a proper probability distribution
    area_w = areas / (areas.sum() + 1e-10)
    curv_w = face_curvature / (face_curvature.sum() + 1e-10)

    weights = 0.5 * area_w + 0.5 * curv_w
    weights /= weights.sum()

    # Sample face indices according to blended weights
    face_idx = np.random.choice(len(mesh.faces), size=num_points, p=weights)

    # Random barycentric coordinate inside each chosen triangle
    # Using the square-root trick to get uniform distribution within triangle
    r1 = np.sqrt(np.random.uniform(0.0, 1.0, num_points))
    r2 = np.random.uniform(0.0, 1.0, num_points)
    a = 1.0 - r1
    b = r1 * (1.0 - r2)
    c = r1 * r2

    verts = mesh.vertices[mesh.faces[face_idx]]   # (N, 3, 3)
    points = (a[:, None] * verts[:, 0]
            + b[:, None] * verts[:, 1]
            + c[:, None] * verts[:, 2])
    return points.astype(np.float32)


def sample_vertex(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    """
    Subsample directly from the mesh's existing vertices.

    No interpolation occurs — every returned point is an original vertex
    coordinate.  When the mesh has fewer vertices than num_points, we
    oversample with replacement (which introduces duplicates but keeps the
    output shape fixed).
    """
    vertices = mesh.vertices
    V = len(vertices)
    replace = V < num_points
    idx = np.random.choice(V, num_points, replace=replace)
    return vertices[idx].astype(np.float32)


SAMPLING_METHODS = {
    'uniform':   sample_uniform,
    'poisson':   sample_poisson,
    'curvature': sample_curvature,
    'vertex':    sample_vertex,
}


# ── Normalisation (shared) ─────────────────────────────────────────────────────

def normalize_to_unit_sphere(points: np.ndarray) -> np.ndarray:
    """
    Translate centroid to origin, then scale so the farthest point lies on
    the unit sphere.  This makes all shapes scale- and translation-invariant
    and is standard practice for ModelNet40.
    """
    centroid = points.mean(axis=0)
    points = points - centroid
    radius = np.max(np.linalg.norm(points, axis=1))
    points = points / (radius + 1e-10)
    return points


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_modelnet40(source_dir: str, dest_dir: str,
                       num_points: int = 2048, method: str = 'uniform'):
    sample_fn = SAMPLING_METHODS[method]
    os.makedirs(dest_dir, exist_ok=True)

    classes = sorted([d for d in os.listdir(source_dir)
                      if os.path.isdir(os.path.join(source_dir, d))])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    if not classes:
        print(f"No classes found in {source_dir}")
        return

    print(f"Method: {method}  |  Points per cloud: {num_points}")
    print(f"Found {len(classes)} classes.")

    for split in ['train', 'test']:
        print(f"\nProcessing '{split}' split ...")
        filepaths = []
        for cls_name in classes:
            cls_dir = os.path.join(source_dir, cls_name, split)
            if not os.path.exists(cls_dir):
                continue
            for f in glob.glob(os.path.join(cls_dir, '*.off')):
                filepaths.append((f, class_to_idx[cls_name]))

        all_data, all_labels = [], []
        for filepath, label in tqdm(filepaths, desc=f"  Sampling [{method}] {split}"):
            try:
                mesh = trimesh.load(filepath, force='mesh')
                points = sample_fn(mesh, num_points)
                points = normalize_to_unit_sphere(points)
                all_data.append(points)
                all_labels.append([label])
            except Exception as e:
                print(f"\n  [WARN] Skipped {filepath}: {e}")

        if not all_data:
            print(f"  No data processed for {split}.")
            continue

        data_np   = np.array(all_data,   dtype=np.float32)
        labels_np = np.array(all_labels, dtype=np.int64)

        out_file = os.path.join(dest_dir, f'ply_data_{split}_0.h5')
        print(f"  Saving {len(data_np)} clouds → {out_file}")
        with h5py.File(out_file, 'w') as hf:
            hf.create_dataset('data',  data=data_np)
            hf.create_dataset('label', data=labels_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess ModelNet40 meshes into HDF5 point clouds.')
    parser.add_argument('--method', type=str, default='uniform',
                        choices=list(SAMPLING_METHODS.keys()),
                        help='Point sampling strategy (default: uniform)')
    parser.add_argument('--source', type=str, default='data/raw_modelnet40',
                        help='Root directory of raw ModelNet40 .off files')
    parser.add_argument('--dest', type=str, default=None,
                        help='Output HDF5 directory. Defaults to '
                             'data/modelnet40_hdf5_<method>')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Points to sample per cloud (default: 2048)')
    args = parser.parse_args()

    dest = args.dest or f'data/modelnet40_hdf5_{args.method}'
    process_modelnet40(args.source, dest,
                       num_points=args.num_points, method=args.method)
    print("\nPreprocessing complete!")
