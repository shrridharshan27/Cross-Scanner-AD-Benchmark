# Research Presentation Dashboard

A standalone, zero-dependency interactive dashboard for presenting the Cross-Scanner Alzheimer's Disease Classification Benchmark.

## Quick Start

### 1. Generate Live Data
```bash
python generate_dashboard_data.py
```
This reads empirical results from `../ad_generalization_benchmark/results/` and exports `dashboard_data.js`.

### 2. View in Browser
Open `index.html` directly in any web browser (Chrome, Edge, Firefox, Safari). No web server, Node.js, or build step required.

### 3. Features
- **Live Empirical Data**: Dynamic metrics and results table loaded directly from `dashboard_data.js`.
- **Interactive Anatomical Brain (SVG)**:
  - Top-down view with 4 clickable regions: **Hippocampus**, **Frontal Lobe**, **Temporal Lobe**, and **Parietal Lobe**.
  - **Healthy Brain ↔ Alzheimer's Brain Toggle**: Interactive animated atrophy transition (Hippocampus shrinks to 60% and turns red, frontal turns amber, temporal turns orange).
  - **Scanner Domain Shift Simulator**: Compare OASIS-1 (Siemens 1.5T) vs Spanish hospital scanner measurements.
- **Scroll-Triggered Visualizations**: Animated horizontal bars for generalization gap comparisons.
- **Publication-Ready Styling**: Deep Navy theme (`#0A1628`) with Medical Teal accents (`#00B4D8`).
