"""
ADNI Scan Organizer & Cross-Scanner Pipeline Utility
Matches downloaded LONI IDA image collections against ADNI_MPRAGE_metadata.csv,
organizes scans by Subject ID, Diagnosis (AD/MCI/CN), and Field Strength (1.5T vs 3.0T).
"""

import os
import sys
import glob
import shutil
import zipfile
import tarfile
import pandas as pd
from pathlib import Path

METADATA_PATH = "./ADNI_MPRAGE_metadata.csv"
OUTPUT_DIR = "./data/adni"

def load_metadata(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: Metadata file {csv_path} not found.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    print(f"Loaded {len(df)} metadata records for {df['Subject ID'].nunique()} unique subjects.")
    return df

def unpack_archives(source_dir, extract_to):
    """Unpack any .zip or .tar files from source_dir into extract_to."""
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

def organize_scans(raw_dir, metadata_df, target_dir):
    """
    Search for NIfTI (.nii, .nii.gz) or DICOM (.dcm) files in raw_dir,
    match them to metadata by Subject ID (e.g. 002_S_0295), and organize.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Find all scan files (.nii, .nii.gz, or DICOM dirs)
    scan_files = []
    for ext in ("*.nii", "*.nii.gz", "*.dcm"):
        scan_files.extend(Path(raw_dir).rglob(ext))

    if not scan_files:
        print(f"No scan files (.nii, .nii.gz, .dcm) found in {raw_dir}.")
        print("Please place the downloaded LONI IDA package in this directory or Downloads.")
        return

    print(f"Found {len(scan_files)} scan files. Organizing...")
    
    manifest_rows = []
    
    for file_path in scan_files:
        path_str = str(file_path)
        
        # Match subject ID (Pattern: \d{3}_S_\d{4})
        matched_subj = None
        for subj in metadata_df['Subject ID'].unique():
            if subj in path_str:
                matched_subj = subj
                break
                
        if not matched_subj:
            continue

        subj_meta = metadata_df[metadata_df['Subject ID'] == matched_subj].iloc[0]
        group = str(subj_meta.get('Research Group', 'Unknown')).strip()
        protocol = str(subj_meta.get('Imaging Protocol', 'Unknown')).strip().replace('Field Strength=', '') + 'T'
        sex = str(subj_meta.get('Sex', 'NA')).strip()
        age = str(subj_meta.get('Age', 'NA')).strip()

        # Destination structure: ./data/adni/{FieldStrength}/{Group}/{SubjectID}/filename
        dest_folder = os.path.join(target_dir, protocol, group, matched_subj)
        os.makedirs(dest_folder, exist_ok=True)
        
        dest_file = os.path.join(dest_folder, file_path.name)
        if not os.path.exists(dest_file):
            shutil.copy2(file_path, dest_file)
            
        manifest_rows.append({
            'Subject ID': matched_subj,
            'Research Group': group,
            'Field Strength': protocol,
            'Age': age,
            'Sex': sex,
            'Filename': file_path.name,
            'Relative Path': os.path.relpath(dest_file, start=target_dir)
        })

    manifest_df = pd.DataFrame(manifest_rows).drop_duplicates()
    manifest_csv = os.path.join(target_dir, "adni_organized_manifest.csv")
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"\nSuccessfully organized {len(manifest_df)} scans into {target_dir}!")
    print(f"Manifest written to: {manifest_csv}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Organize ADNI MPRAGE Scans")
    parser.add_argument("--source-dir", default="./data/adni_raw", help="Folder containing raw downloaded zip/nii files")
    parser.add_argument("--metadata-csv", default=METADATA_PATH, help="Path to ADNI metadata CSV")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Destination directory for organized scans")
    args = parser.parse_args()

    meta_df = load_metadata(args.metadata_csv)
    
    # Check if user has downloads in default Downloads folder
    user_downloads = os.path.expanduser(r"~\Downloads")
    if not os.path.exists(args.source_dir) or not os.listdir(args.source_dir):
        print(f"Checking {user_downloads} for downloaded IDA zip files...")
        ida_zips = glob.glob(os.path.join(user_downloads, "*ADNI*.zip")) + glob.glob(os.path.join(user_downloads, "*ida*.zip"))
        if ida_zips:
            print(f"Found {len(ida_zips)} ADNI archive(s) in {user_downloads}!")
            unpack_archives(user_downloads, args.source_dir)
            
    if os.path.exists(args.source_dir):
        unpack_archives(args.source_dir, args.source_dir)
        organize_scans(args.source_dir, meta_df, args.output_dir)
    else:
        print(f"Directory {args.source_dir} is ready. Place your downloaded LONI packages there and run this script.")

if __name__ == "__main__":
    main()
