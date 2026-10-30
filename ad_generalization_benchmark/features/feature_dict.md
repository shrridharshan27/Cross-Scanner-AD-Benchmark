# Feature Dictionary & Pipeline Documentation

This document records the exact feature definitions, extraction methodologies, units, and harmonization considerations for the Cross-Scanner Alzheimer's Disease Classification Benchmark: OASIS-1 (Source) → Spanish Cohort (Target).

---

## 1. Feature Overview & Alignment

Both domains provide 4 input features (2 structural neuroimaging features + 2 demographic covariates):

| Aligned Feature Name | Model Feature | OASIS-1 Source Feature | Spanish Target Equivalent | Units | Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Normalized Whole Brain Volume** | `nWBV` | `nWBV` (FreeSurfer) | `brain_vol_ratio` (ANTsPyNet / SPM) | Dimensionless ratio \([0, 1]\) | Continuous |
| **Intracranial Volume** | `eTIV` | `eTIV` (FreeSurfer) | `icv_estimate` (ANTsPyNet / SPM) | \(\text{mm}^3\) (or \(\text{cm}^3\)) | Continuous |
| **Age** | `age` | `Age` | `Age` | Years | Continuous |
| **Sex** | `sex` | `M/F` (F=0, M=1) | `gender` (1=F→0, 2=M→1) | Binary \(\{0, 1\}\) | Categorical |

---

## 2. Source Domain: OASIS-1 (Washington University, Siemens 1.5T)

### Imaging Protocol
- **Scanner**: Siemens Vision 1.5T, St. Louis, MO, USA.
- **Acquisition**: T1-weighted MP-RAGE (sagittal, \(\text{TR}=9.7\text{ ms}\), \(\text{TE}=4.0\text{ ms}\), flip angle \(= 10^\circ\), \(1.0 \times 1.0 \times 1.25\text{ mm}\) resolution).
- **Processing Software**: FreeSurfer (Buckner et al., 2004; Marcus et al., 2007).

### Feature Definitions
1. **`nWBV` (Normalized Whole Brain Volume)**:
   - **Definition**: Proportion of all voxels classified as brain tissue (gray matter + white matter) relative to total estimated intracranial volume.
   - **Formula**: \(\text{nWBV} = \frac{\text{Total Brain Volume}}{\text{eTIV}}\).
   - **Units**: Ratio (values typically between 0.64 and 0.84 in older adults).
   - **Missing Data**: 0 missing values.

2. **`eTIV` (Estimated Total Intracranial Volume)**:
   - **Definition**: Total volume within the cranium estimated via atlas transformation scaling factor.
   - **Formula**: \(\text{eTIV} = \frac{\text{Atlas Volume}}{\text{Determinant}(\text{Affine Matrix})}\).
   - **Units**: \(\text{mm}^3\) (values typically range from 1100 to 1900 \(\text{cm}^3\)).
   - **Missing Data**: 0 missing values.

3. **`age`**:
   - **Definition**: Age in years at time of MRI session.
   - **Eligibility**: Restricted to participants \(\ge 60\) years old.

4. **`sex`**:
   - **Encoding**: Female = 0, Male = 1.
   - **Missing Data**: 0 missing values.

---

## 3. Target Domain: Spanish Cohort (Zenodo 3935636)

### Cohort & Clinical Characterization
- **Repository**: Zenodo record [3935636](https://zenodo.org/records/3935636).
- **Participants**:
  - Alzheimer's Disease (`ea`): \(N=35\) (Label = 1)
  - Healthy Controls (`crl`): \(N=19\) (Label = 0)
  - Bipolar Disorder (`tb`): \(N=24\) (**EXCLUDED** from benchmark)
- **Clinical Data**: Sourced from `clinical_data_id_age_gender.csv`.
- **Gender Verification**:
  - `gender == 1` corresponds to `mujer` (Female), mapped to `sex = 0`.
  - `gender == 2` corresponds to `hombre` (Male), mapped to `sex = 1`.
  - Confirmed against `clinical_data_corrected.csv` via exact participant ID cross-matching.

### MRI Archive Manifest & Structure
Archive `T1_preprocessed.zip` contains 468 items across 5 distinct subdirectories:
- `T1_original/`: 78 native anatomical T1 `.nii.gz` scans (1 per subject).
- `T1_preprocessed_SPM_1.5mm/T1_GM_WM_segmented/`: 156 segmented tissue probability maps (SPM modulated warped).
- `T1_preprocessed_SPM_1.5mm/T1_brain/`: 78 skull-stripped T1 volumes.
- `T1_preprocessed_SPM_1.5mm/T1_GM_smooth8/`: 78 smoothed GM maps.
- `T1_preprocessed_DARTEL_1.0mm/T1_GM_sm4_1mm_dartel/`: 78 DARTEL-aligned GM maps.

### Feature Extraction Pipeline (Option C — SPM Segmentations Implemented in Milestone 2)
- **Feature**: `brain_vol_ratio`
- **Source**: SPM tissue segmentation (`c1` modulated GM map `m0wrp1f*.nii` + `c2` modulated WM map `m0wrp2f*.nii`) located in `datasets/spanish_extracted/T1_preprocessed_SPM_1.5mm/T1_GM_WM_segmented/`.
- **Formula**: \(\text{brain\_vol\_ratio} = \frac{\text{GM\_vol} + \text{WM\_vol}}{\text{ICV\_estimate}}\)
- **Units**: Dimensionless ratio \([0, 1]\).
- **Voxel Size**: \(1.5 \times 1.5 \times 1.5\text{ mm}\) isotropic (\(\text{voxel volume} = 3.375\text{ mm}^3\)).
- **ICV Estimate**: Total non-zero skull-stripped intracranial volume derived from `datasets/spanish_extracted/T1_preprocessed_SPM_1.5mm/T1_brain/wmrf*.nii`.
- **Comparable to OASIS-1 nWBV**: Approximately, NOT exactly.
- **Pipeline Difference**: OASIS-1 uses FreeSurfer atlas-based scaling affine matrix registration for eTIV and subcortical/cortical volumetric parcellation for nWBV. The Spanish cohort uses SPM unified segmentation with modulated warped tissue probability maps.
- **Handling Out-of-Brain Values**: SPM background voxels are stored as `NaN` outside the registration bounding box; `np.nansum` is applied across tissue maps to prevent volume computation contamination.

---

## 4. Pipeline Non-Equivalence & Domain Shift Caveat

> [!WARNING]
> **OASIS-1 and Spanish features do NOT derive from identical software pipelines.**
> - OASIS-1 structural features were extracted using **FreeSurfer** (atlas scaling affine transformation for eTIV, surface/subcortical segmentation for nWBV).
> - Spanish structural features derive from **ANTsPyNet / SPM** applied to native T1 images.
> - While both quantify normalized brain parenchymal fraction and intracranial volume, systematic differences in tissue thresholding, skull-stripping boundaries, and atlas scaling exist.
> - **Consequently, the observed generalization gap reflects BOTH scanner hardware differences (Siemens 1.5T vs Spanish site scanner) AND analytical pipeline discrepancies.**
> - Harmonization techniques (ComBat and DANN in Track B) are evaluated specifically to determine whether such cross-pipeline, cross-scanner shifts can be aligned without target diagnostic labels.

---

## 5. Forbidden Features & Leakage Controls

The following columns are strictly excluded from all models, loaders, and feature caches:

| Forbidden Feature | Rationale for Exclusion |
| :--- | :--- |
| **`MMSE`** | Direct psychometric test score strongly correlated with dementia diagnosis; acts as label proxy. |
| **`CDR`** | Primary ground-truth diagnosis definition (CDR 0 = CN, CDR \(\ge\) 1 = AD). |
| **`ASF`** | Atlas Scaling Factor is mathematically redundant with `eTIV` (\(\text{ASF} = \frac{1531.8}{\text{eTIV}}\)). |
| **`Educ`, `SES`** | High rate of missingness (e.g. SES missing in 18 eligible subjects) and socio-demographic confounders. |
| **`Delay`** | Longitudinal follow-up interval, irrelevant to cross-sectional classification. |
| **`ID` / `subject_id`** | Participant identifier; never a model feature. |
| **`tb`** | Bipolar disorder cohort; excluded from AD vs CN binary classification. |

---

## 6. Volumetric Feature Units & Audit Resolution (Milestone 3)

### Unit Audit Findings
- **OASIS-1 `eTIV`**: In the source repository spreadsheet (`oasis_cross-sectional-5708aa0a98d82080.xlsx`), FreeSurfer `eTIV` is reported on a scale of \(\sim 1100 - 1900\) (mean \(1448.48 \pm 143.84\)), representing \(\text{cm}^3\) (or \(\text{mL}\)), rather than unscaled \(\text{mm}^3\).
- **Spanish `icv_estimate` (SPM raw)**: Derived via voxel summation (\(\text{voxel volume} = 3.375\text{ mm}^3\)), resulting in raw intracranial volume on the order of \(\sim 1.88 \times 10^6\text{ mm}^3\).
- **Cross-Domain Scale Mismatch**: Because the raw Spanish measurement was \(\sim 1000\times\) larger than the OASIS-1 feature space, unharmonized models in Milestone 2 (Track A) experienced extreme saturated logits (yielding near-random AUROC \(\sim 0.50\)).

### Resolution
- Standardized to OASIS-1 reference scale: `icv_estimate` in `features/feature_cache/spanish_features.csv` is divided by \(1000.0\) (\(\text{icv\_estimate}_{\text{cm}^3} = \frac{\text{icv\_estimate}_{\text{mm}^3}}{1000}\)).
- Post-correction Spanish `icv_estimate` distribution: Mean \(1881.56\), Min \(1837.25\), Max \(1917.45\).
- Raw \(\text{mm}^3\) features are preserved alongside in `features/feature_cache/spanish_features_raw_mm3.csv`.
- `brain_vol_ratio` is mathematically invariant to this scale factor as it is a dimensionless ratio of two \(\text{mm}^3\) volume measurements.
