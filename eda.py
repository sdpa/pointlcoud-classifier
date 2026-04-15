import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_num_vertices(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line == 'OFF':
                # Normal OFF format
                second_line = f.readline().strip()
                parts = second_line.split()
                if len(parts) >= 1:
                    return int(parts[0])
            elif first_line.startswith('OFF'):
                # Handle cases where OFF is joined with the numbers
                parts = first_line[3:].strip().split()
                if len(parts) >= 1:
                    return int(parts[0])
    except Exception as e:
        pass
    return -1

def main():
    source_dir = "data/raw_modelnet40"
    all_off_files = glob.glob(os.path.join(source_dir, "**", "*.off"), recursive=True)
    
    print(f"Found {len(all_off_files)} .off files. Analyzing point distributions...")
    
    vertex_counts = []
    
    for filepath in tqdm(all_off_files, desc="Parsing OFF headers"):
        count = get_num_vertices(filepath)
        if count > 0:
            vertex_counts.append(count)
            
    if not vertex_counts:
        print("No valid vertex counts found.")
        return
        
    counts = np.array(vertex_counts)
    
    print(f"\n--- Point Distribution Data ---")
    print(f"Total shapes parsed: {len(counts)}")
    print(f"Minimum Points: {np.min(counts):,}")
    print(f"Maximum Points: {np.max(counts):,}")
    print(f"Mean Points:    {np.mean(counts):,.1f}")
    print(f"Median Points:  {np.median(counts):,}")
    print(f"5th Percentile: {np.percentile(counts, 5):,.0f}")
    print(f"95th Percentile:{np.percentile(counts, 95):,.0f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # We will log-scale the x-axis or use a standard distribution 
    # since max points could be very large and skew the graph
    plt.hist(counts, bins=100, range=(0, np.percentile(counts, 98)), color='skyblue', edgecolor='black')
    
    plt.title('Distribution of Point Counts (Vertices) in ModelNet40 RAW Meshes')
    plt.xlabel('Number of Points/Vertices (clipped at 98th percentile)')
    plt.ylabel('Frequency (Number of Meshes)')
    plt.grid(axis='y', alpha=0.75)
    
    plt.axvline(np.median(counts), color='red', linestyle='dashed', linewidth=2, label=f'Median: {int(np.median(counts))}')
    plt.legend()
    
    out_path = 'points_distribution.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {out_path}")

if __name__ == "__main__":
    main()
