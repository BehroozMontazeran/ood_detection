import zipfile
from tqdm import tqdm
import os

def extract_zip(zip_path: str, extract_to: str):
    """Extracts a zip file to a specified directory with a progress bar."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of all files in the zip
        total_files = len(zip_ref.infolist())
        
        # Iterate over each file in the zip archive
        with tqdm(total=total_files, desc="Extracting", unit="file") as pbar:
            for file in zip_ref.infolist():
                zip_ref.extract(file, extract_to)
                pbar.update(1)

if __name__ == "__main__":
    zip_path = 'data/CELEBA/archive.zip'
    extract_to = 'data/CELEBA'
    
    # Ensure the output directory exists
    os.makedirs(extract_to, exist_ok=True)
    
    extract_zip(zip_path, extract_to)
