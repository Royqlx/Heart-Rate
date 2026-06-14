import os
import sys
import gdown

DATA_DIR = os.path.join("data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

FOLDER_URL = "https://drive.google.com/drive/folders/1U3fzIOESmaBAyikGF0cKI2wW3YK8JqCK"

print("Downloading MR-NIRP-D dataset...")
print(f"Target directory: {os.path.abspath(DATA_DIR)}")

gdown.download_folder(
    url=FOLDER_URL,
    output=DATA_DIR,
    quiet=False,
    use_cookies=False
)

print("\nDownload completed successfully.")