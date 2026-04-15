import os
import glob
import h5py
import numpy as np
import trimesh
from tqdm import tqdm

def process_modelnet40(source_dir, dest_dir, num_points=2048):
    os.makedirs(dest_dir, exist_ok=True)
    
    classes = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    if len(classes) == 0:
        print("No classes found in", source_dir)
        return

    for split in ['train', 'test']:
        print(f"Processing {split} split...")
        all_data = []
        all_labels = []
        
        # Gather all files first for tqdm
        filepaths = []
        for cls_name in classes:
            cls_dir = os.path.join(source_dir, cls_name, split)
            if not os.path.exists(cls_dir): continue
            
            off_files = glob.glob(os.path.join(cls_dir, '*.off'))
            for f in off_files:
                filepaths.append((f, class_to_idx[cls_name]))
                
        # Parse and sample
        for filepath, label in tqdm(filepaths, desc=f"Sampling MVP {split}"):
            try:
                mesh = trimesh.load(filepath, force='mesh')
                # Sample points on the surface
                points, _ = trimesh.sample.sample_surface(mesh, num_points)
                # Normalize points into a unit sphere (standard practice)
                centroid = np.mean(points, axis=0)
                points = points - centroid
                m = np.max(np.sqrt(np.sum(points**2, axis=1)))
                points = points / m
                
                all_data.append(points.astype(np.float32))
                all_labels.append([label])
            except Exception as e:
                # Some off files might be corrupted or empty
                print(f"Error processing {filepath}: {e}")
                
        if len(all_data) == 0:
            print(f"No data processed for {split}.")
            continue
            
        data_np = np.array(all_data)
        labels_np = np.array(all_labels).astype(np.int64)
        
        out_file = os.path.join(dest_dir, f'ply_data_{split}_0.h5')
        print(f"Saving {len(data_np)} examples to {out_file}...")
        with h5py.File(out_file, 'w') as f:
            f.create_dataset('data', data=data_np)
            f.create_dataset('label', data=labels_np)

if __name__ == '__main__':
    source = "data/raw_modelnet40"
    dest = "data/modelnet40_ply_hdf5_2048"
    process_modelnet40(source, dest)
    print("Preprocessing complete!")
