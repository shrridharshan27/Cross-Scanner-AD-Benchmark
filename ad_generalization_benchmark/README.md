# Cross-Scanner Alzheimer's Disease Classification Benchmark
## OASIS-1 (Source) → Spanish Cohort (Target) with Harmonization Comparison

### Project Overview
This repository provides a reproducible, leakage-controlled research benchmark evaluating cross-scanner and cross-cohort generalization for Alzheimer's Disease (AD) vs. Cognitively Normal (CN) classification.

- **Source Domain**: OASIS-1 cross-sectional cohort (Washington University in St. Louis; Siemens Vision 1.5T MRI).
- **Target Domain**: Spanish Cohort (Zenodo record [3935636](https://zenodo.org/records/3935636); independent European hospital site and scanner).
- **Primary Task**: Binary classification — Alzheimer's Disease (`AD`, label=1) vs. Healthy Control / Cognitively Normal (`CN`, label=0). Mild Cognitive Impairment (`MCI`, CDR=0.5) and Bipolar Disorder (`tb`) are prespecified for exclusion in primary benchmarks.

---

### Experimental Tracks
1. **Track A (Strict External Evaluation)**:
   - OASIS-1 data split into stratified train (70%), validation (15%), and test (15%) partitions.
   - All scaling, feature preprocessing, model fitting, and decision threshold selection are executed strictly on the OASIS-1 training partition.
   - Zero target (Spanish) data of any kind is utilized during fitting.
   - Evaluated on held-out OASIS-1 test partition and the entire Spanish target cohort.
2. **Track B (Target Domain Adaptation)**:
   - Evaluates whether unlabeled structural features from the target domain can alleviate the domain generalization gap without target labels.
   - Stratified Spanish cohort partition: 20% unlabeled calibration, 80% evaluation.
   - Strategies compared:
     1. Unadapted baseline (MLP)
     2. **ComBat** batch-effect correction (fit jointly on source train features and unlabeled target calibration features)
     3. **DANN** (Domain-Adversarial Neural Network with Gradient Reversal Layer)

---

### Confirmed Dataset Facts & Acceptance Numbers

#### 1. OASIS-1 (Source Domain)
- **File**: `oasis_cross-sectional-5708aa0a98d82080.xlsx`
- **Total Records**: 436 participants.
- **Eligibility Rule**: Age \(\ge\) 60 and non-null Clinical Dementia Rating (`CDR`).
- **Eligible Cohort**: 198 participants.
- **Primary Strict Binary Cohort**: 128 subjects:
  - Cognitively Normal (`CDR == 0.0`): 98 subjects.
  - Alzheimer's Disease (`CDR >= 1.0`): 30 subjects (28 with CDR 1.0, 2 with CDR 2.0).
  - Excluded MCI (`CDR == 0.5`): 70 subjects.
- **Secondary Broad Cohort**: 198 subjects (98 CN, 100 AD).
- **Features Used**: `nWBV` (normalized whole brain volume), `eTIV` (estimated total intracranial volume), `age`, `sex` (F=0, M=1).
- **Forbidden Features (Leakage Prevention)**: `MMSE`, `CDR`, `ASF` (redundant with eTIV), `Educ`, `SES`, `Delay`, `ID`.

#### 2. Spanish Cohort (Target Domain)
- **Archives**: `Clinical_data.zip` (clinical table) and `T1_preprocessed.zip` (MRI images).
- **Diagnostic Categories**:
  - `ea` (Alzheimer's Disease): 35 subjects (Label = 1).
  - `crl` (Healthy Control): 19 subjects (Label = 0).
  - `tb` (Bipolar Disorder): 24 subjects (**EXCLUDED**).
  - Valid Binary Cohort: 54 subjects (35 AD + 19 HC).
- **Gender Mapping**:
  - `gender == 1` corresponds to `mujer` (Female), mapped to `sex = 0`.
  - `gender == 2` corresponds to `hombre` (Male), mapped to `sex = 1`.
- **MRI Architecture**:
  - `T1_original/`: 78 native `.nii.gz` anatomical scans (100% matched to clinical IDs).
  - Pre-segmented SPM and DARTEL maps identified and preserved without destructive overwriting.

---

### Repository Structure

```
ad_generalization_benchmark/
├── README.md                    # Project documentation & benchmark protocol
├── environment.yml              # Conda environment definition
├── requirements.txt             # Pip dependency requirements
├── configs/
│   ├── paths.yaml               # Absolute & relative dataset and output paths
│   ├── experiment.yaml          # Seeds [42, 123, 456], split ratios, hyperparameters
│   └── features.yaml            # Feature names, alignment mapping, label encodings
├── data/
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── oasis1_loader.py     # OASIS-1 loader with prespecified eligibility
│   │   └── spanish_loader.py    # Spanish clinical loader with diagnosis filter
│   ├── audit/
│   │   ├── audit_oasis1.py      # Verifies 436 total, 198 eligible, 98 CN, 30 AD
│   │   ├── audit_spanish.py     # Verifies 35 AD, 19 HC, 24 Bipolar excluded
│   │   └── check_leakage.py     # Asserts zero subject overlap across splits
│   └── manifests/               # Saved participant manifests
├── features/
│   ├── extract_spanish.py       # Defensive inspection & ANTsPyNet / Nibabel stubs
│   ├── feature_dict.md          # Complete documentation of feature pipelines
│   └── feature_cache/           # Saved .csv and .npy feature caches
├── splits/
│   ├── generate_splits.py       # Stratified participant-level split generator
│   └── saved/                   # Saved split JSON files for seeds 42, 123, 456
├── models/
│   └── __init__.py
├── training/
├── evaluation/
├── notebooks/
│   ├── colab_benchmark.ipynb    # End-to-end benchmark notebook (Cells 1–12)
│   └── create_notebook.py       # Generator script for colab_benchmark.ipynb
├── tests/
│   ├── fixtures/
│   │   ├── synthetic_splits.json   # 4-subject synthetic split fixtures
│   │   └── synthetic_features.csv # 4-subject synthetic feature fixtures
│   ├── test_no_subject_overlap.py # Asserts disjoint train/test and domains
│   ├── test_no_forbidden_features.py # Asserts exclusion of MMSE, CDR, ASF
│   └── test_combat_no_eval_leak.py # Asserts no evaluation data in ComBat fit
└── results/
    └── .gitkeep
```

---

### Milestone 1 Acceptance Criteria Verification

| Milestone 1 Criterion | Target Requirement | Status |
| :--- | :--- | :--- |
| **`audit_oasis1.py`** | Prints 436 total, 198 eligible, 98 CN, 30 AD primary | **PASSED** |
| **`audit_spanish.py`** | Lists files inside `Clinical_data.zip` without error | **PASSED** |
| **`oasis1_loader.py`** | Returns DataFrame with 128 rows, 6 columns | **PASSED** |
| **`generate_splits.py`** | Produces split JSON with no subject overlap | **PASSED** |
| **`test_no_subject_overlap.py`** | Passes on synthetic fixtures | **PASSED** |
| **`test_no_forbidden_features.py`** | Passes on synthetic fixtures | **PASSED** |
| **Colab cells 1–12** | Runs without crashing on CPU session | **PASSED** |

---

### How to Run Milestone 1

1. **Run Audits**:
   ```bash
   python -m ad_generalization_benchmark.data.audit.audit_oasis1
   python -m ad_generalization_benchmark.data.audit.audit_spanish
   ```

2. **Generate Splits**:
   ```bash
   python -m ad_generalization_benchmark.splits.generate_splits
   python -m ad_generalization_benchmark.data.audit.check_leakage
   ```

3. **Run Leakage & Feature Tests**:
   ```bash
   python -m pytest ad_generalization_benchmark/tests -v
   ```
