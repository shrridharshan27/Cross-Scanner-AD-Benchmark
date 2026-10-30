"""
AIBL Scan Organizer & Cross-Scanner Pipeline Utility
Extracts and organizes AIBL T1w/MPRAGE scans downloaded from LONI IDA (Project: AIBL).
"""

import os
import sys
import glob
import shutil
import zipfile
import tarfile
from pathlib import Path
import pandas as pd

OUTPUT_DIR = "./data/aibl"

def unpack_archives(source_dir, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    archive_patterns = ["*.zip", "*.tar", "*.tar.gz", "*.tgz"]
    archives = []
    for pat in archive_patterns:
        archives.extend(glob.glob(os.path.join(source_dir, pat)))
    
    if not archives:
        return
        
    print(f"Found {len(archives)} archive(s) in {source_dir}. Extracting...")
    for arc in archives:
        print(f"Extracting {os.path.basename(arc)}...")
        try:
            if zipfile.is_zipfile(arc):
                with zipfile.ZipFile(arc, 'r') as zf:
                    zf.extractall(extract_to)
            elif tarfile.is_tarfile(arc):
                with tarfile.open(arc, 'r:*') as tf:
                    tf.extractall(extract_to)
        except Exception as e:
            print(f"Error extracting {arc}: {e}")

def organize_aibl(raw_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    scan_files = []
    for ext in ("*.nii", "*.nii.gz", "*.dcm"):
        scan_files.extend(Path(raw_dir).rglob(ext))

    if not scan_files:
        print(f"No scan files found in {raw_dir}.")
        print("Once your AIBL IDA package is downloaded, place it in ./data/aibl_raw or Downloads.")
        return

    print(f"Found {len(scan_files)} scans. Organizing into {target_dir}...")
    manifest = []
    for sf in scan_files:
        # AIBL format typical path or subject pattern
        dest_file = os.path.join(target_dir, sf.name)
        if not os.path.exists(dest_file):
            shutil.copy2(sf, dest_file)
        manifest.append({'filename': sf.name, 'path': os.path.relpath(dest_file, start=target_dir)})

    df = pd.DataFrame(manifest)
    manifest_csv = os.path.join(target_dir, "aibl_manifest.csv")
    df.to_csv(manifest_csv, index=False)
    print(f"Organized {len(df)} files into {target_dir}. Manifest saved to {manifest_csv}.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Organize AIBL T1w Scans")
    parser.add_argument("--source-dir", default="./data/aibl_raw")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    unpack_archives(args.source_dir, args.source_dir)
    organize_aibl(args.source_dir, args.output_dir)

if __name__ == "__main__":
    main()
