# uci-secom-analysis
Defect prediction on imbalanced, high-dimensional semiconductor manufacturing data (UCI SECOM) — from preprocessing to constraint-based model selection.

# SECOM Semiconductor Defect Prediction

An end-to-end machine learning pipeline for predicting wafer defects on the
**UCI SECOM** semiconductor manufacturing dataset. The project addresses three
challenges that occur simultaneously in real fab data — high dimensionality,
heavy missingness, and severe class imbalance — and builds a classifier that
remains usable under a realistic operating constraint.

---

## Overview

Semiconductor manufacturing produces hundreds of sensor measurements per wafer,
and final yield depends on subtle interactions among those signals. The goal here
is to flag defective wafers before they pass through later (expensive) process
steps. The dataset is heavily imbalanced — only **6.6%** of wafers are defective —
so accuracy is meaningless (a trivial "always pass" classifier already scores
93.4%). The pipeline therefore optimizes for **precision under a recall floor**
rather than a single aggregate metric.

The central finding: the model with the best PR-AUC is **not** the model that is
actually deployable. After applying a domain constraint (recall ≥ 0.70), only two
of the top-10 candidates survive, and the final choice ranked 10th on PR-AUC.

---

## Dataset

- **Source:** [UCI SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom)
  (M. McCann & A. Johnston, 2008)
- **Samples:** 1,567 wafers
- **Features:** 590 anonymized continuous sensor / process signals
- **Target:** binary Pass (0) / Fail (1); defect rate ≈ 6.6% (~1:14 imbalance)
- **Note:** original labels (−1 / 1) are remapped to (0 / 1).

---

## Pipeline

The analysis is organized as a staged pipeline across three notebooks.

### 1. Preprocessing (`secom_preprocessing.ipynb`)
Reduces 590 raw features to 244 while preventing data leakage (all statistics are
estimated on the training split only):

| Step | Action | Rationale |
| --- | --- | --- |
| 1 | Stratified 8:2 Train/Test split | Preserve class ratio, isolate the test set |
| 2 | Drop high-missing features (> 50%) | KNN imputation unreliable above this rate |
| 3 | KNN imputation (k = 5) | Uses local feature structure, not just the mean |
| 4 | Remove zero-variance features | No discriminative information |
| 5 | Robust scaling | `x_scaled = (x − median(x)) / IQR(x)`; resistant to outliers |
| 6 | Smart-drop multicollinearity (\|ρ\| > 0.90) | Keep the less-redundant / lower-missing feature |

Output: **`secom_preprocessed.csv`** (244 features, referred to as `secom_base`).

### 2. Feature engineering — first attempt (`secom_feature_1_.ipynb`)
Tests whether *adding* features helps:
- IsolationForest anomaly score as an extra feature (`secom_FE1`)
- Targeted interaction features from top point-biserial-correlated variables (`secom_FE2`)

**Result:** neither addition produced a meaningful PR-AUC gain — consistent with
the known result that on noisy high-dimensional data, added signal rarely exceeds
added noise. This motivates a switch from *adding* features to *selecting* them.

### 3. Feature selection + modeling (`secom_feature_2_.ipynb`)
- **Mutual-information** feature selection, producing four candidate datasets:
  `secom_fs_mi30`, `fs_mi50`, `fs_mi80`, `fs_mi160` (top-k features by MI).
- **Stage 2 — combination grid:** 5 datasets × 4 models
  (Logistic Regression, Random Forest, XGBoost, LightGBM) × 2 imbalance flows
  (Flow A: SMOTEENN, Flow B: cost-sensitive) = **40 combinations**, each scored by
  PR-AUC under 5-fold × 3-repeat `RepeatedStratifiedKFold`.
- **Stage 3 — threshold search:** for the top-10 PR-AUC combinations, search for a
  threshold satisfying **recall ≥ 0.70**, then maximize precision among feasible ones.
- **Stage 4 — hyperparameter tuning:** custom scorer = "precision at the threshold
  meeting recall ≥ 0.70"; 50 Random Forest candidates evaluated.
- **Stage 5 — final test evaluation:** single, leak-free evaluation on the held-out test set.

---

## Key Results

| Property | Value |
| --- | --- |
| Final model | `fs_mi50` + Random Forest + Flow B (cost-sensitive) |
| Hyperparameters | `n_estimators=200`, `max_depth=5`, `min_samples_leaf=4`, `max_features='sqrt'` |
| Operating threshold | 0.27 |
| Test precision | 0.102 |
| Test recall | 0.762 (meets the 0.70 domain floor) |
| Test F1 / F2 | 0.180 / 0.332 |
| Test PR-AUC | 0.151 |
| Confusion matrix | TN=152, FP=141, FN=5, TP=16 |

Notable observations:
- **Cost-sensitive beats SMOTEENN:** 9 of the top-10 PR-AUC combinations use Flow B.
  Synthetic minority samples on noisy high-dimensional data behave like
  boundary noise rather than useful signal.
- **Boosting wins PR-AUC but loses on deployability:** XGBoost / LightGBM dominate
  PR-AUC, yet their probability outputs are compressed too low to reach recall 0.70
  at any threshold. Random Forest's flatter vote-based probabilities allow a usable
  operating point.
- **OOF ↔ Test consistency:** cross-validated and test scores agree within standard
  error, indicating no overfitting and no leakage.

---

## Repository Structure

```
.
├── secom_preprocessing.ipynb     # Stage 1: preprocessing (590 → 244 features)
├── secom_feature_1_.ipynb        # First attempt: feature engineering (rejected)
├── secom_feature_2_.ipynb        # MI feature selection + 40-combo experiment + tuning + final eval
├── ucisecom.csv                  # Raw UCI SECOM data (1567 × 592)
├── secom_preprocessed.csv        # Preprocessed base dataset (244 features + split)
├── secom_fs_mi30.csv             # MI top-30 feature subset
├── secom_fs_mi50.csv             # MI top-50 feature subset (final model input)
├── secom_fs_mi80.csv             # MI top-80 feature subset
└── secom_fs_mi160.csv            # MI top-160 feature subset
```

Each `secom_fs_mi*.csv` includes the selected features plus the `Pass/Fail` label
and a `split` column marking Train/Test membership.

---

## Requirements

```
python >= 3.9
numpy
pandas
scikit-learn
imbalanced-learn
xgboost
lightgbm
matplotlib
tqdm
```

Install with:

```bash
pip install numpy pandas scikit-learn imbalanced-learn xgboost lightgbm matplotlib tqdm
```

---

## How to Run

Run the notebooks in order:

1. `secom_preprocessing.ipynb` — produces `secom_preprocessed.csv`
2. `secom_feature_2_.ipynb` — produces the `secom_fs_mi*.csv` subsets, runs the
   40-combination experiment, tunes the final model, and reports test results.

`secom_feature_1_.ipynb` documents the rejected feature-engineering attempt and is
included for completeness; it is not required to reproduce the final model.

A fixed random seed is used throughout, so the pipeline is reproducible from the
same inputs.

---

## Methodological Takeaways

1. On noisy, high-dimensional data, **feature selection outperforms feature addition**.
2. **PR-AUC is a filter, not a final decision metric** — operating constraints can
   reorder the candidate ranking entirely.
3. Encoding the **domain cost structure** (a recall floor) directly into the scorer
   makes the selected model operationally meaningful, not just statistically strong.

---

## References

Key references for the methods used (full IEEE-style list is in the report):

- Chawla et al., *SMOTE*, JAIR 2002
- Batista et al., *SMOTEENN behavior study*, SIGKDD Explorations 2004
- Elkan, *Foundations of cost-sensitive learning*, IJCAI 2001
- Breiman, *Random Forests*, Machine Learning 2001
- Chen & Guestrin, *XGBoost*, KDD 2016
- Ke et al., *LightGBM*, NeurIPS 2017
- Saito & Rehmsmeier, *PR plot vs. ROC on imbalanced data*, PLoS ONE 2015
- Guyon & Elisseeff, *Variable and feature selection*, JMLR 2003
