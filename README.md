
**Taha** | Data Analytics Intern, ItSimpleRa Solutions
Independent AI/ML Research Challenge | Submitted 12 August 2026

---

## 1. Overview

This repository contains the implementation of the Week 7–8 capstone proposal: a credit card fraud detection system built on the ULB benchmark dataset, evaluated under a strictly leakage-free protocol.

Fraud detection is a rare-event classification problem. In this dataset frauds are 0.172% of all transactions, which means a model that predicts "not fraud" for every single row scores 99.83% accuracy while catching nothing. Every design decision in this project follows from that fact: accuracy is never reported, class-specific metrics are used throughout, and the evaluation protocol is built to avoid the specific leakage failure that inflates published results in this space.

**Deliverable:** `Taha_Fraud_Detection_Capstone.ipynb` — a single Google Colab notebook, top-to-bottom runnable.

---

## 2. Dataset

**Source:** [Credit Card Fraud Detection Dataset — ULB Machine Learning Group (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
|---|---|
| Transactions | 284,807 |
| Confirmed frauds | 492 (0.172%) |
| Features | 30 (V1–V28 PCA components, Time, Amount) |
| Target | `Class` (0 = genuine, 1 = fraud) |
| Origin | European cardholders, two days in September 2013 |
| File | `creditcard.csv`, ~144 MB |

V1–V28 are anonymised PCA components — the original features were transformed for confidentiality. `Time` and `Amount` are the only two raw features.

### Why this dataset

It is the standard benchmark for fraud detection research, so results are directly comparable against a large body of published work. Its extreme class imbalance is precisely the difficulty this project sets out to address. It has not been used in any previous week of this internship.

### Getting the data

The dataset requires a free Kaggle account and cannot be fetched anonymously. Two options, both implemented in Section 0 of the notebook:

**Option A — manual upload**
1. Open the dataset link, click **Download** → you get `archive.zip`
2. Unzip on your machine → `creditcard.csv`
3. Run the upload cell in the notebook and select the file

**Option B — Kaggle API** (faster, avoids a 144 MB browser upload)
1. Kaggle → profile picture → **Settings** → **API** → **Create New Token** → downloads `kaggle.json`
2. Run the Kaggle API cell and upload `kaggle.json` when prompted
3. The notebook downloads and unzips the dataset directly

Run only one of the two cells.

---

## 3. Running the notebook

1. Go to [Google Colab](https://colab.research.google.com) → **File → Upload notebook** → select `Taha_Fraud_Detection_Capstone.ipynb`
2. **Runtime → Change runtime type → T4 GPU** (optional — speeds up the autoencoder; CPU works fine)
3. Run Section 0 to load the dataset
4. **Runtime → Run all**, or step through cell by cell

**Expected runtime:** 15–20 minutes end to end. The 5-fold cross-validation cell (Section 7) is the slowest at roughly 5–10 minutes, since it retrains an autoencoder inside every fold.

### Dependencies

Installed by the notebook itself; listed here for reference.

```
numpy, pandas, matplotlib, seaborn
scikit-learn
xgboost
imbalanced-learn
tensorflow / keras
joblib
```

`RANDOM_STATE = 42` is set for NumPy, TensorFlow, and every scikit-learn/XGBoost estimator, so runs are reproducible. Minor variation on GPU is still possible from non-deterministic cuDNN kernels.

---

## 4. Notebook structure

| Section | Contents |
|---|---|
| 0 | Dataset acquisition (two options) |
| 1 | Imports, seeds, plotting setup |
| 2 | Load and inspect data, exploratory plots |
| 3 | Leakage-free train/test split |
| 4 | Evaluation helper functions |
| 5 | Baseline models + leakage demonstration |
| 6 | Proposed improvement — autoencoder feature |
| 7 | 5-fold cross-validated comparison |
| 8 | Threshold tuning and final results |
| 9 | Business-cost view |
| 10 | Save artefacts |
| 11 | Notes for the write-up |

---

## 5. Methodology

### 5.1 Preprocessing

`Time` is dropped. It encodes seconds elapsed since the first transaction in a two-day window — an absolute timestamp with no meaning outside this particular sample, and keeping it invites the model to memorise a period rather than learn fraud structure. `Amount` is retained and scaled; V1–V28 are already PCA outputs and are used as-is for the tree models, scaled only where the estimator requires it.

### 5.2 The leakage-free protocol

This is the methodological core of the project, following Kabane (2024).

The common error in this literature is applying SMOTE or another resampling method to the full dataset **before** splitting into train and test. When that happens, synthetic minority samples interpolated from test-set rows end up in the training data. The model has effectively seen the test set, and reported performance is inflated — sometimes dramatically.

This project enforces the correct ordering:

1. Stratified train/test split happens **first** (80/20)
2. Every transformation that learns from data — `StandardScaler`, SMOTE, the autoencoder — is fit on training data only
3. The test set is transformed using those already-fitted objects and never contributes to fitting
4. Inside cross-validation, all of the above is repeated per fold, including retraining the autoencoder from scratch

Section 5.4 runs the wrong version deliberately, as a controlled demonstration, and prints the leaky AUPRC alongside the correct one. The gap between them is leakage, not skill. That inflated number is never reported as a result.

### 5.3 Baseline models

**Logistic Regression** (`class_weight='balanced'`) — a linear baseline establishing the performance floor. Its coefficients are directly readable, and the notebook plots the largest ones. Interpretability is limited by the PCA anonymisation, but relative magnitudes still indicate which components carry signal.

**Random Forest** (`class_weight='balanced_subsample'`, 300 trees) — a bagged ensemble capturing non-linear interactions among the PCA components. Requires no feature scaling and handles the imbalance reasonably once class-weighted.

**XGBoost** (`scale_pos_weight` set to the negative/positive ratio, `eval_metric='aucpr'`) — gradient-boosted trees, the model that performs strongest on this benchmark in prior work. This is the baseline the proposed improvement has to beat.

### 5.4 Proposed improvement

The reviewed literature shows a trade-off. One-class and graph-based methods (Zaffar et al., 2023; Duan et al., 2024) improve minority-class recall but add architectural complexity and cost interpretability. Simple supervised baselines deploy easily but under-detect fraud unless resampling is handled correctly — and Kabane (2024) shows it frequently is not.

The approach here targets both problems at once:

1. Train an **autoencoder on genuine transactions only**, drawn from the training split. Architecture: 29 → 24 → 16 → 8 → 16 → 24 → 29, ReLU activations, linear output, Adam optimiser, MSE loss, early stopping on a held-out genuine validation slice.
2. The autoencoder learns to reconstruct normal transaction structure. Fraudulent rows, unseen during training, reconstruct poorly.
3. Compute per-row **mean squared reconstruction error** for both training and test rows.
4. Append that single column as an engineered feature and retrain XGBoost on the augmented matrix.

The intent is to capture the anomaly sensitivity of one-class methods while keeping the deployed model a plain tree ensemble — one column of extra feature engineering rather than a new architecture.

The autoencoder never sees a labelled fraud example and never sees a test row during training, so the feature introduces no leakage.

---

## 6. Evaluation

Accuracy is not reported anywhere. At 0.172% prevalence it carries no information.

| Metric | Why |
|---|---|
| **Precision, Recall, F1** (fraud class) | Class-specific performance on the minority class that actually matters |
| **AUPRC** | More informative than ROC-AUC under severe imbalance; not inflated by the enormous true-negative count |
| **MCC** | Used in Randhawa et al. (2018); accounts for all four confusion-matrix cells in one balanced score, enabling direct comparison with that work |
| **Confusion matrix** | The real-world trade-off — missed fraud vs. false alarms — which is where the business cost sits |

### Threshold selection

A 0.5 cutoff is arbitrary for a fraud system. The notebook selects the operating threshold by maximising fraud-class F1 on a **validation split carved out of the training data**, then applies that threshold unchanged to the test set. Tuning a threshold on test scores is itself a form of leakage and is avoided.

### Cross-validation

A single split leaves roughly 98 fraud cases in the test set — few enough that one lucky split can move F1 by several points. Section 7 runs 5-fold stratified CV comparing baseline XGBoost against the augmented version, rebuilding the autoencoder inside each fold, and reports mean and standard deviation per metric with boxplots.

**Read the standard deviation before the mean.** If the gap between the two models is smaller than the fold-to-fold spread, there is no demonstrated improvement, regardless of what the single-split numbers showed.

### Business-cost view

Section 9 converts the confusion matrix into money using two editable assumptions — cost per missed fraud and cost per false alarm — and sweeps the threshold to find the cost-minimising operating point. These figures are illustrative placeholders, not measured from the data, and should be labelled as assumptions in the write-up. The purpose is to show the threshold decision can be argued in business terms rather than metric terms.

---

## 7. Interpreting your results

Fill the write-up from actual output rather than expectation.

**Did the autoencoder feature help?** Compare CV means in Section 7 against their standard deviations. On this dataset the gain is often small or absent, and there is a good reason: V1–V28 are already PCA components, so much of the structure an autoencoder would learn is present in the inputs, and a well-tuned gradient-boosted ensemble extracts most of it directly.

If the boxplots overlap, the honest conclusion is that the feature produced no measurable gain under a leakage-free protocol. **That is a legitimate and valuable capstone finding.** A clean negative result, reported alongside a working leakage demonstration, is stronger work than a marginal improvement that turns out to be fold noise. Overclaiming a gain that the variance does not support is the one outcome to avoid.

**Where `recon_error` ranked** in feature importance (Section 6.1) tells you whether the model found the signal useful at all, independent of the headline metric. A high rank with no metric gain is itself interesting — it means the signal is real but redundant with what the PCA components already encode.

**The leakage cell** gives a concrete number for Kabane (2024)'s argument. Quote the inflated and correct AUPRC side by side.

---

## 8. Limitations

State these explicitly in the report:

- **Temporal scope.** Two days of transactions from September 2013. Fraud patterns drift; these numbers would not hold on live traffic without retraining.
- **Geographic scope.** European cardholders only. No claim of generalisation to other markets.
- **Anonymised features.** V1–V28 are PCA outputs, so feature-level interpretation is structural at best — you cannot say *which* real-world behaviour drives a prediction.
- **No temporal validation.** Splits are random and stratified, not chronological. A production system should validate on future transactions relative to training, which is a stricter and more realistic test.
- **Assumed costs.** The Section 9 cost figures are illustrative, not derived from the data.
- **Single dataset.** All conclusions are conditional on this one benchmark.

---

## 9. Outputs

Section 10 writes to `artifacts/`:

```
artifacts/
├── xgboost_with_ae_feature.joblib   # trained augmented classifier
├── ae_scaler.joblib                 # fitted StandardScaler (training data only)
├── autoencoder.keras                # trained autoencoder
├── results_summary.csv              # final metrics table
└── cv_results.csv                   # per-fold CV results
```

A commented cell zips and downloads these from Colab. Note that Colab sessions are ephemeral — download anything you need before the runtime disconnects.

---

## 10. References

1. Randhawa, K., Loo, C.K., Seera, M., Lim, C.P., & Nandi, A.K. (2018). Credit Card Fraud Detection Using AdaBoost and Majority Voting. *IEEE Access*, 6, 14277–14284.
2. Zaffar, Z., Sohrab, F., Kanniainen, J., & Gabbouj, M. (2023). Credit Card Fraud Detection with Subspace Learning-based One-Class Classification. arXiv:2309.14880.
3. Duan, Y., Zhang, G., Wang, S., Peng, X., Wang, Z., Mao, J., Wu, H., Jiang, X., & Wang, K. (2024). CaT-GNN: Enhancing Credit Card Fraud Detection via Causal Temporal Graph Neural Networks. arXiv:2402.14708.
4. Lu, Y., & Zhan, F. (2024). Kolmogorov Arnold Networks in Fraud Detection: Bridging the Gap Between Theory and Practice. arXiv:2408.10263.
5. Kabane, S. (2024). Impact of Sampling Techniques and Data Leakage on XGBoost Performance in Credit Card Fraud Detection. arXiv:2412.07437.

**Dataset:** Dal Pozzolo, A., Caelen, O., Johnson, R.A., & Bontempi, G. Calibrating Probability with Undersampling for Unbalanced Classification. Machine Learning Group, Université Libre de Bruxelles (ULB).

---

## 11. Troubleshooting

**`FileNotFoundError: creditcard.csv`** — Section 0 did not complete. Check the file landed in the Colab working directory: `!ls -la *.csv`

**Kaggle API returns 403** — `kaggle.json` is missing or misplaced. Confirm with `!ls -la ~/.kaggle/`, and make sure the token was created under your own account.

**Out-of-memory during cross-validation** — reduce `n_estimators` in `cv_compare`, or drop to 3 folds. On the free Colab tier the full run is close to the RAM ceiling.

**Autoencoder loss goes to NaN** — usually a scaling problem. Confirm the `StandardScaler` cell ran before the autoencoder cell; the network expects standardised inputs.

**Session disconnects mid-run** — Colab free tier times out on idle. Keep the tab active, or run Sections 1–6 and Section 7 in separate sittings.
