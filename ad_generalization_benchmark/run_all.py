"""
Convenience runner. Execute from C:\\Project\\ML Project\\
Usage: python ad_generalization_benchmark/run_all.py --step audit
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

STEPS = {
    'audit_oasis': ROOT / 'data/audit/audit_oasis1.py',
    'audit_spanish': ROOT / 'data/audit/audit_spanish.py',
    'check_leakage': ROOT / 'data/audit/check_leakage.py',
    'splits': ROOT / 'splits/generate_splits.py',
    'extract_spanish': ROOT / 'features/extract_spanish.py',
    'audit_units': ROOT / 'data/audit/audit_feature_units.py',
    'train_baseline': ROOT / 'training/train_baseline.py',
    'evaluate_internal': ROOT / 'evaluation/evaluate.py',
    'evaluate_external': ROOT / 'evaluation/evaluate_external_trackA.py',
    'combat': ROOT / 'training/train_combat.py',
    'dann': ROOT / 'training/train_dann.py',
    'final_comparison': ROOT / 'evaluation/final_comparison.py',
    'test': None,  # handled separately
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=list(STEPS.keys()) + ['test', 'all'],
                        required=True)
    args = parser.parse_args()

    if args.step == 'test':
        subprocess.run([sys.executable, '-m', 'pytest',
                        str(ROOT / 'tests'), '-v'], check=True)
        return

    if args.step == 'all':
        for name, script in STEPS.items():
            if script is None:
                continue
            print(f"\n{'='*50}")
            print(f"Running: {name}")
            print('='*50)
            subprocess.run([sys.executable, str(script)], check=True)
        return

    script = STEPS[args.step]
    subprocess.run([sys.executable, str(script)], check=True)

if __name__ == '__main__':
    main()
