import os
import zipfile
import urllib.request
from tqdm import tqdm

DATA_DIR = "data"
URL = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
ZIP_FILE = os.path.join(DATA_DIR, "modelnet40_ply_hdf5_2048.zip")

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_and_extract():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    extracted_folder = os.path.join(DATA_DIR, "modelnet40_ply_hdf5_2048")
    if os.path.exists(extracted_folder):
        print(f"Dataset already exists at {extracted_folder}. Skipping download.")
        return
        
    if not os.path.exists(ZIP_FILE):
        print(f"Downloading {URL}...")
        download_url(URL, ZIP_FILE)
    
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    # Optional: cleanup zip
    try:
        os.remove(ZIP_FILE)
        print("Cleaned up zip file.")
    except Exception as e:
        pass
    print("Data ready!")

if __name__ == "__main__":
    download_and_extract()
