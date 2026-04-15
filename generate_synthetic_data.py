import os
import h5py
import numpy as np

def create_synthetic_modelnet40():
    data_dir = "data/modelnet40_ply_hdf5_2048"
    os.makedirs(data_dir, exist_ok=True)

    # We will generate a small subset for demonstration purposes
    # 2 batches of 100 samples each for train, 1 batch of 50 samples for test
    num_points = 2048

    print(f"Generating synthetic ModelNet40 dataset in {data_dir}...")

    # Train files
    for i in range(2):
        print(f"Generating ply_data_train{i}.h5")
        num_samples = 100
        data = np.random.randn(num_samples, num_points, 3).astype('float32') # Random point cloud
        label = np.random.randint(0, 40, size=(num_samples, 1)).astype('int64')
        
        with h5py.File(os.path.join(data_dir, f'ply_data_train{i}.h5'), 'w') as f:
            f.create_dataset('data', data=data)
            f.create_dataset('label', data=label)

    # Test file
    print("Generating ply_data_test0.h5")
    num_samples = 50
    data = np.random.randn(num_samples, num_points, 3).astype('float32')
    label = np.random.randint(0, 40, size=(num_samples, 1)).astype('int64')
    
    with h5py.File(os.path.join(data_dir, 'ply_data_test0.h5'), 'w') as f:
        f.create_dataset('data', data=data)
        f.create_dataset('label', data=label)

    print("Synthetic dataset generation complete. You can now use the Notebook.")

if __name__ == "__main__":
    create_synthetic_modelnet40()
