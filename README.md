# Survival Classification in Rituximab‑treated WES Cohort

End‑to‑end R workflow to clean clinical data, define an overall survival category, select features, and train multiple classifiers (RF, XGBoost, penalized GLMs, logistic regression). This README documents the logic behind your script and provides reproducible, polished code snippets.

---

## TL;DR

* **Goal:** Predict short OS (≤3 years) vs longer OS (>3 years) in **Rituximab‑positive** patients from the **WES** cohort.
* **Label:** `OS_Category = 1` if `OS.YEARS ≤ 3`, else `0`.
* **Models:** Random Forest, XGBoost, Ridge/Lasso/Elastic Net (glmnet), full & stepwise Logistic Regression, RFE.
* **Feature Selection:** Boruta (with TentativeRoughFix), optional manual 5-feature LR.
* **Evaluation:** Cross-validated training + held‑out test; report Accuracy + ROC‑AUC (recommended).

---

## Data

* **Input:** `lesions_rituximab.csv`
* **Required columns (at minimum):**

  * `OS.YEARS` (numeric), `OS.Censor` (0/1), `Rituximab` (0/1), `Cohort` (e.g., `WES`), plus candidate features (SNVs/CNVs/genes/etc.)
* **Filtering used in the analysis:**

  1. Drop rows where `OS.YEARS` or `OS.Censor` are missing
  2. Drop rows where `OS.YEARS == 0`
  3. Keep only `Rituximab == 1`
  4. Keep only `Cohort == "WES"`


---

## Environment

```r
# R >= 4.2 recommended
install.packages(c(
  "tidyverse", "data.table", "caret", "ranger", "xgboost",
  "glmnet", "Boruta", "MASS", "pROC"
))

---

## How to run
Rscript scripts/analysis.R data/lesions_rituximab.csv
# or, if your CSV lives elsewhere:
# Rscript scripts/analysis.R /full/path/to/lesions_rituximab.csv


