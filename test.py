import zipfile
# from tqdm import tqdm
import os

def extract_zip(zip_path: str, extract_to: str):
    """Extracts a zip file to a specified directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

extract_zip('data/CELEBA/archive.zip', 'data/CELEBA')

if __name__ == "__main__":
    zip_path = 'data/CELEBA/archive.zip'
    extract_to = 'data/CELEBA'

    # Ensure the output directory exists
    os.makedirs(extract_to, exist_ok=True)

    extract_zip(zip_path, extract_to)
