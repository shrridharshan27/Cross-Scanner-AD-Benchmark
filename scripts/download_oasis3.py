"""
OASIS-3 T1w MRI Scan Downloader (NITRC-IR REST API)
Cross-platform Python implementation for downloading T1w scans from OASIS-3.

Requirements:
- Valid NITRC account approved for OASIS-3 on NITRC-IR (https://www.nitrc.org/ir)
- requests, tqdm
"""

import os
import sys
import glob
import shutil
import zipfile
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

NITRC_BASE_URL = "https://www.nitrc.org/ir"

def start_session(username, password):
    """Authenticate with NITRC-IR and obtain a JSESSIONID cookie."""
    session = requests.Session()
    session.auth = (username, password)
    login_url = f"{NITRC_BASE_URL}/data/JSESSION"
    
    print(f"Authenticating '{username}' on NITRC-IR ({login_url})...")
    resp = session.post(login_url)
    
    if resp.status_code not in (200, 201):
        print(f"\n[!] Authentication failed: HTTP {resp.status_code}")
        print("Please check your NITRC username and password, and ensure your OASIS-3 DUA access is active.")
        sys.exit(1)
        
    print("Authentication successful! Active Session ID acquired.")
    return session

def get_oasis3_mr_experiments(session):
    """Fetch list of OASIS-3 MR experiments from NITRC-IR, excluding data file sessions."""
    url = f"{NITRC_BASE_URL}/data/projects/OASIS3/experiments?format=json"
    print(f"Fetching experiment list from OASIS-3 project...")
    resp = session.get(url)
    if resp.status_code != 200:
        print(f"Failed to fetch experiments list: HTTP {resp.status_code}")
        return []
    
    data = resp.json()
    results = data.get('ResultSet', {}).get('Result', [])
    mr_experiments = [
        r for r in results 
        if r.get('xsiType') == 'xnat:mrSessionData' and '0AS' not in r.get('label', '')
    ]
    print(f"Found {len(mr_experiments)} MR imaging experiments in OASIS-3.")
    return mr_experiments

def download_metadata_spreadsheets(session, output_dir):
    """Attempt to download OASIS-3 clinical and MR spreadsheets from 0AS_data_files."""
    meta_dir = os.path.join(output_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    url = f"{NITRC_BASE_URL}/data/projects/OASIS3/subjects/0AS_data_files/experiments?format=json"
    try:
        resp = session.get(url)
        if resp.status_code == 200:
            exp_list = resp.json().get('ResultSet', {}).get('Result', [])
            for exp in exp_list:
                exp_id = exp.get('ID')
                label = exp.get('label')
                files_url = f"{NITRC_BASE_URL}/data/experiments/{exp_id}/resources/DATA/files?format=zip"
                zpath = os.path.join(meta_dir, f"{label}.zip")
                r = session.get(files_url, stream=True)
                if r.status_code == 200 and int(r.headers.get('content-length', 1)) > 500:
                    with open(zpath, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=16384):
                            f.write(chunk)
                    if zipfile.is_zipfile(zpath):
                        with zipfile.ZipFile(zpath, 'r') as zf:
                            zf.extractall(meta_dir)
                        os.remove(zpath)
                        print(f"Downloaded OASIS-3 metadata spreadsheets to {meta_dir}")
    except Exception as e:
        print(f"Note: Could not automatically fetch 0AS_data_files spreadsheets: {e}")

def organize_unpacked_scan(exp_dir):
    """
    Rearrange nested scans folder structure so files are cleanly placed:
    exp_dir/anat1/file.nii.gz, etc.
    """
    scans_dir = os.path.join(exp_dir, "scans")
    if not os.path.exists(scans_dir):
        return
        
    for scan_folder in Path(scans_dir).iterdir():
        if scan_folder.is_dir():
            scan_name = scan_folder.name.split('-')[0]
            target_subfolder = os.path.join(exp_dir, scan_name)
            os.makedirs(target_subfolder, exist_ok=True)
            
            # Find all files inside resources/*/files
            for file_path in scan_folder.rglob('*'):
                if file_path.is_file():
                    dest_file = os.path.join(target_subfolder, file_path.name)
                    if not os.path.exists(dest_file):
                        shutil.move(str(file_path), dest_file)
                        
    # Clean up empty scans directory
    shutil.rmtree(scans_dir, ignore_errors=True)

def download_scan(session, project_id, subject_id, experiment_id, scan_type, output_dir):
    """Download and extract a scan for a specific experiment."""
    exp_dir = os.path.join(output_dir, experiment_id)
    if os.path.exists(exp_dir) and any(os.scandir(exp_dir)):
        print(f"[{experiment_id}] Already exists in {output_dir}, skipping.")
        return True

    download_url = (
        f"{NITRC_BASE_URL}/data/archive/projects/{project_id}/"
        f"subjects/{subject_id}/experiments/{experiment_id}/scans/{scan_type}/files?format=zip"
    )
    zip_path = os.path.join(output_dir, f"{experiment_id}.zip")

    try:
        resp = session.get(download_url, stream=True)
        if resp.status_code != 200:
            print(f"[{experiment_id}] No scan found or HTTP {resp.status_code} for scan type '{scan_type}'.")
            return False

        total_size = int(resp.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f, tqdm(
            desc=experiment_id,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            leave=False
        ) as bar:
            for chunk in resp.iter_content(chunk_size=65536):
                size = f.write(chunk)
                bar.update(size)

        if zipfile.is_zipfile(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(zip_path)
            
            # Organize nested subfolders
            organize_unpacked_scan(exp_dir)
            print(f"[{experiment_id}] Downloaded & organized successfully.")
            return True
        else:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            print(f"[{experiment_id}] Archive was empty or invalid (no '{scan_type}' scan present).")
            return False

    except Exception as e:
        print(f"[{experiment_id}] Error: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False

def main():
    parser = argparse.ArgumentParser(description="Download OASIS-3 scans from NITRC-IR")
    parser.add_argument("--username", help="NITRC-IR username")
    parser.add_argument("--password", help="NITRC-IR password (or set NITRC_PASSWORD env var)")
    parser.add_argument("--input-csv", help="Optional CSV file containing experiment IDs (e.g. OAS30001_MR_d0129)")
    parser.add_argument("--scan-type", default="T1w", help="Scan type filter (default: T1w)")
    parser.add_argument("--output-dir", default="./data/oasis3", help="Output directory (default: ./data/oasis3)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of experiments to download")
    parser.add_argument("--download-metadata", action="store_true", help="Download OASIS-3 clinical & session spreadsheets")

    args = parser.parse_args()

    username = args.username or os.environ.get("NITRC_USERNAME")
    if not username:
        if sys.stdin.isatty():
            username = input("Enter NITRC-IR Username: ").strip()
        else:
            print("[!] Error: NITRC username is required. Pass --username <username>.")
            sys.exit(1)

    password = args.password or os.environ.get("NITRC_PASSWORD")
    if not password:
        if sys.stdin.isatty():
            import getpass
            password = getpass.getpass("Enter NITRC-IR Password: ")
        else:
            print("\n" + "="*70)
            print("[!] AUTHENTICATION REQUIRED")
            print("NITRC-IR requires your account password to authenticate session tokens.")
            print("\nOptions to run:")
            print("1. Set environment variable:  $env:NITRC_PASSWORD = 'your_password'")
            print("   Then run:                  python scripts/download_oasis3.py --username shrridharshan --scan-type T1w --output-dir ./data/oasis3")
            print("2. Pass --password:           python scripts/download_oasis3.py --username shrridharshan --password 'your_password' --scan-type T1w --output-dir ./data/oasis3")
            print("3. Run in your terminal where it will prompt interactively for your password.")
            print("="*70 + "\n")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    session = start_session(username, password)

    if args.download_metadata:
        download_metadata_spreadsheets(session, args.output_dir)

    experiments = []
    if args.input_csv:
        import pandas as pd
        df = pd.read_csv(args.input_csv)
        col = df.columns[0]
        experiments = df[col].dropna().astype(str).tolist()
        print(f"Loaded {len(experiments)} experiment IDs from {args.input_csv}")
    else:
        exp_records = get_oasis3_mr_experiments(session)
        experiments = [r.get('label') or r.get('ID') for r in exp_records if r.get('label') or r.get('ID')]

    if args.limit:
        experiments = experiments[:args.limit]

    print(f"\nStarting download of {len(experiments)} experiments (Scan Type: {args.scan_type}) to {args.output_dir}...")
    success_count = 0
    for exp_id in experiments:
        sub_id = exp_id.split('_')[0]
        proj_id = "OASIS3"
        ok = download_scan(session, proj_id, sub_id, exp_id, args.scan_type, args.output_dir)
        if ok:
            success_count += 1

    print(f"\nFinished. Successfully processed {success_count}/{len(experiments)} scans.")

if __name__ == "__main__":
    main()
