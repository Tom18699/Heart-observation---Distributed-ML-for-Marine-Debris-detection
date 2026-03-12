# Marine Debris Detection — Distributed ML on MARIDA
<img width="2257" height="824" alt="image" src="https://github.com/user-attachments/assets/3b26a379-6460-43b2-befc-be983872c93e" />
<img width="591" height="591" alt="image" src="https://github.com/user-attachments/assets/0a6d00d7-4652-400f-9ef7-afb6607237ec" />

**Authors:** Lorenzo Albani, Luigi Ascione, Tommaso Maitino

---

## Overview

Distributed machine learning pipeline built with **Apache Spark / PySpark** for detecting marine plastic pollution from **Sentinel-2 multispectral satellite imagery**, using the [MARIDA](https://github.com/marine-debris/marine-debris.github.io) benchmark dataset (~837K annotated pixels, 15 thematic classes).

The dataset combines three feature groups: raw **spectral bands** (11 Sentinel-2 channels), **spectral indices** (NDVI, FDI, NDWI, NRD, NDMI, BSI), and **GLCM texture features** (Contrast, Dissimilarity, Homogeneity, Energy, ASM). The target class — Marine Debris — represents only **0.41%** of samples, making class imbalance the central challenge of the project.

---

## Pipeline

### Preprocessing
- PCA on spectral bands (PC1 captures 92.78% of variance)
- Redundant feature removal (SI, HOMO, ASM, FAI)
- Z-score outlier removal (threshold = 3.5) → final dataset: **795,955 samples**

### Clustering (Unsupervised)
Distributed **K-Means** via MLlib, evaluated with elbow method + silhouette score. Three phases:
1. Baseline (k=15) — spectral structure does not replicate the 15-class taxonomy
2. Optimal global config (k=6) — interpretable macro-environments (water, sediment, floating matter, clouds)
3. **Sub-clustering on Marine Debris only (k=4)** — 4 physical macro-states identified:
   - Dense/Pure debris (highest annotator confidence: 64.9%)
   - Sparse/Submerged debris (maximum uncertainty: 29.4% low confidence)
   - Debris mixed with vegetation (NDVI peak: +1.12)
   - Coastal/Sediment-mixed debris (BSI peak: +0.65)

### Classification (Supervised)
- **XGBoost** (multiclass + binary) with 5-fold CV random search
- **Random Forest** multiclass
- Imbalanced learning:
  - **ADASYN** — hybrid distributed (sklearn KNN + PySpark reinjection)
  - **Cluster Centroids** — fully distributed via MLlib KMeans
  - Target threshold: 10,000 samples per class

### Domain Shift Analysis
Train on **Central America** (88% of data), test on geographically unseen regions (SE Asia, Africa, Europe).

### Explainable AI
SHAP values via `TreeExplainer` distributed with `pandas_udf`. Top features for Marine Debris: **FDI** (1.71), **NDWI** (1.50), **NDMI** (1.14), **CON** (1.13).

---

## Results

| Model | Marine Debris F1 |
|---|---|
| MARIDA benchmark (best RF) | 0.80 |
| XGBoost Multiclass (baseline) | 0.88 |
| **XGBoost + ADASYN Multiclass** | **0.91** ✅ |
| XGBoost Binary | 0.90 |
| Random Forest Multiclass | 0.79 |
| Domain Shift (Binary Unweighted) | 0.71 |

> Best model (**XGBoost + ADASYN**) surpasses the MARIDA benchmark by ~11 percentage points on the harder 15-class schema.

<img width="900" height="650" alt="prec_rec_tradeoff" src="https://github.com/user-attachments/assets/7afb5645-050f-43f1-b022-40c2ff73a3e4" />



---

## Tech Stack

| Tool | Purpose |
|---|---|
| **PySpark / MLlib** | Distributed pipeline, K-Means, PCA |
| **SparkXGBClassifier** | Primary classifier |
| **scikit-learn** | KNN (ADASYN) |
| **SHAP** | Explainability via TreeExplainer + pandas_udf |
| **HDF5 / h5py** | Dataset loading |

---

## Dataset Setup

1. Download `dataset.h5` from the [MARIDA repository](https://github.com/marine-debris/marine-debris.github.io)
2. Generate `dataset_si.h5` and `dataset_glcm.h5` using the MARIDA data engineering scripts
3. Place all files in `data/`

---

## References

- Kikaki et al. (2022) — *MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data*
- [ADASYN for Scala](https://github.com/fsleeman/spark-class-balancing)
- [Cluster Centroids for Scala](https://github.com/ElsevierSoftwareX/SOFTX_2019_253)
