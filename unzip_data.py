import zipfile
import os

# ✅ Fix path using raw string or forward slashes
zip_path = r"C:\Users\yasar beg\traffic_sign_recognition\data\archive (2).zip"
extract_dir = r"C:\Users\yasar beg\traffic_sign_recognition\data"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("✅ Dataset extracted successfully to:", extract_dir)

