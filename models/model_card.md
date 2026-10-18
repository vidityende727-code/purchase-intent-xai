# Model Card — E-Commerce Purchase Intent Prediction

## 1. Model Overview

This project uses **CatBoost Classifier** as the final model for predicting purchase intent (binary classification: whether an e-commerce session results in a purchase — `Converted` = 1/0).

During model selection, six algorithms were trained and compared: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, and CatBoost. CatBoost was chosen as the final model: its **PR-AUC (0.855)** is effectively tied with LightGBM (0.8550 vs 0.8552, a gap far below noise), and both clearly outperform the other four models. PR-AUC is the primary evaluation metric given the dataset's class imbalance (only 15.72% of sessions convert), and it is more reliable than accuracy or ROC-AUC in imbalanced settings because it focuses on how well the model identifies the minority (buyer) class.

CatBoost's strength likely comes from its native handling of categorical features — the dataset has 7 categorical columns (e.g., Month, VisitorType, TrafficType), and CatBoost is specifically designed for datasets with many categorical variables. LightGBM performed almost identically on this dataset, so the choice reflects CatBoost's native categorical handling and the fact that the full pipeline (API, SHAP, LIME) is built around it, rather than a clear performance win. 

## 2. Training Data

The dataset consists of **12,000 e-commerce sessions** with 17 original features (18 including the target), sourced from `ecommerce_sessions.csv`. The target variable is `Converted`, indicating whether the session resulted in a purchase.

- **Class distribution:** 15.72% positive (purchase), 84.28% negative (no purchase) — a significantly imbalanced dataset.
- **Data quality:** No missing values and no duplicate rows across all 18 columns — clean dataset requiring no imputation.
- **Data gap:** No sessions exist for January or April, meaning the model must be able to handle unseen `Month` categories at prediction time (handled via `OneHotEncoder(handle_unknown="ignore")`).
- **Train/test split:** 80/20 stratified split (9,600 training rows, 2,400 test rows), preserving the 15.7% positive rate in both sets.
- **Class balancing:** SMOTE (Synthetic Minority Oversampling) applied to the training set only, balancing it to a 50/50 class distribution (9,600 → 16,182 rows) before model training. The test set was left untouched to reflect real-world imbalance during evaluation.  

## 2.5 Feature Engineering

Beyond the original 17 features, 4 new behavioral features were engineered from session data to test whether aggregated engagement signals could improve prediction:

- **TotalPagesViewed** = Administrative + Informational + ProductRelated (overall session engagement)
- **TotalDuration** = sum of all three duration columns (total time spent in session)
- **AvgTimePerPage** = TotalDuration / TotalPagesViewed (engagement quality — time spent per page)
- **ProductEngagementRatio** = ProductRelated / TotalPagesViewed (proportion of session focused on product pages)

**Feature selection — 2-lens evaluation:** Each engineered feature was evaluated using two independent importance lenses before deciding whether to keep it, per best-practice feature selection methodology:

1. **Model-based (CatBoost's built-in `feature_importances_`)** — measures how much each feature contributed to reducing loss during tree splits.
2. **SHAP-based (mean absolute SHAP value)** — measures each feature's average impact on individual predictions.

**Results:**
- **TotalPagesViewed** ranked highly in both lenses (SHAP rank #3, model-based rank #11) — **kept**.
- **TotalDuration** and **AvgTimePerPage** showed moderate but consistent signal in both lenses — **kept**.
- **ProductEngagementRatio** ranked at the bottom of both lenses. This makes sense: it is mathematically derived from `ProductRelated` and `TotalPagesViewed`, both of which are already present in the model individually, making it largely redundant. It was **dropped**.

**Impact on performance:** After retraining with the final 3 engineered features (removing `ProductEngagementRatio`), CatBoost's PR-AUC remained effectively unchanged (0.856 → 0.855), confirming that the original 17 features already captured most of the predictive signal — but the exercise validated that `PageValues` remains overwhelmingly the dominant predictor, consistent with earlier EDA and SHAP findings.

## 3. Performance Metrics

All 6 models were evaluated on the held-out test set (2,400 sessions, natural imbalanced distribution). Comparison summary:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.903 | 0.630 | 0.931 | 0.752 | 0.970 | 0.836 |
| Decision Tree | 0.898 | 0.653 | 0.743 | 0.695 | 0.835 | 0.525 |
| Random Forest | 0.920 | 0.724 | 0.793 | 0.757 | 0.965 | 0.834 |
| XGBoost | 0.926 | 0.765 | 0.761 | 0.763 | 0.968 | 0.837 |
| LightGBM | 0.933 | 0.764 | 0.825 | 0.793 | 0.972 | 0.855 |
| **CatBoost (final)** | 0.931 | 0.762 | 0.817 | 0.789 | 0.970 | **0.855** |

*(Note: CatBoost metrics above are at default threshold 0.5. At the tuned threshold of 0.40 used in production, CatBoost achieves Precision 0.721, Recall 0.849, F1 0.780 — prioritizing recall to reduce missed buyers.)*

**Final model (CatBoost) — selected for PR-AUC (tied best, 0.855):**
- PR-AUC: 0.855 (tied with LightGBM; ahead of the other 4 models)
- At tuned threshold (0.40): Precision 0.721, Recall 0.849, F1 0.780

**Cross-validation:** 5-fold StratifiedKFold on SMOTE-resampled training data gave a mean PR-AUC of 0.9932 (std 0.0007), confirming training stability. This is higher than the test-set PR-AUC (0.855) because CV ran on balanced (50/50) resampled data — the test-set score remains the realistic, imbalanced-data estimate.

## 4. Threshold Decision

By default, classification models use a 0.5 probability threshold to decide the predicted class. However, in this business context, **missing a buyer (false negative)** is more costly than a **false alarm (false positive)** — a missed buyer means a lost sale opportunity, while a false alarm just means an extra (low-cost) marketing nudge to a non-buyer.

To account for this, threshold values from 0.2 to 0.5 were tested on the CatBoost model to compare the precision, recall and F1 trade-off at each level. The results showed:

- **Default threshold (0.5):** F1 = 0.789, Precision = 0.762, Recall = 0.817
- **Tuned threshold (0.40):** F1 = 0.780, Precision = 0.721, Recall = 0.849

The tuned threshold of **0.40** was selected as the final decision threshold because it improves recall (0.849 vs 0.817, about 3 percentage points more of the actual buyers caught) at a modest cost in precision (0.721 vs 0.762). The default 0.5 threshold gives a marginally higher F1 (0.789 vs 0.780), so 0.40 is a deliberate recall-first choice rather than an F1-optimal one. This means the model is intentionally biased toward flagging more sessions as "likely to purchase," reducing missed sales opportunities at the cost of a few more false positives. 

## 5. Key Explainability Findings (SHAP)

SHAP (SHapley Additive exPlanations) was used to interpret the CatBoost model's predictions, using `shap.TreeExplainer` since CatBoost is a tree-based model.

**Global Feature Importance:**
- **PageValues** is by far the most influential feature, with a mean absolute SHAP importance of 4.09 — roughly **10x higher** than the second-most important feature, `ProductRelated` (0.41); the engineered feature `TotalPagesViewed` ranks third (0.26).
- This finding mathematically confirms the earlier EDA observation that buyers had roughly 10x higher PageValues than non-buyers, making it the strongest behavioral signal of purchase intent.
- The summary plot showed clear, logical directionality: high **PageValues** and **ProductRelated** push predictions toward "buyer," while high **ExitRates** and **BounceRates** push predictions toward "non-buyer" — consistent with intuitive user behavior (engaged users convert more; users who bounce/exit quickly don't).

**Local (Per-Prediction) Explainability:**
- A sample buyer session was explained individually:  the model predicted a **77.82% purchase probability**, led by a positive contribution from PageValues (+1.57 SHAP value), with the engineered feature TotalPagesViewed (+0.50) and ProductRelated (+0.39) also pushing toward purchase, partly offset by negative contributions from Region 3 (-0.32) and December (-0.26).
- This demonstrates that SHAP can explain not just overall feature importance, but also the specific trade-off of factors behind any single prediction — useful for building trust in individual model decisions (e.g., "why did the model flag this customer as likely to buy?"). 

## 6. Limitations

- **Missing month data:** The dataset has no sessions from January or April, meaning the model has never seen these months during training. Although `OneHotEncoder(handle_unknown="ignore")` prevents errors on unseen categories, the model has no learned signal for these months and may perform unpredictably if used for these periods.

- **Synthetic oversampling (SMOTE):** Class balancing was done using SMOTE, which generates synthetic minority-class samples. While this helped the model learn buyer patterns better, cross-validation performed on this synthetic balanced data (PR-AUC 0.9932) is noticeably higher than real-world test performance (PR-AUC 0.855) — a reminder that SMOTE-based CV metrics can be overly optimistic and shouldn't be used alone to judge real-world performance.

- **Threshold trade-off:** The tuned threshold (0.40) intentionally favors recall over precision, meaning the model will flag more false positives (non-buyers incorrectly predicted as buyers) in exchange for catching more true buyers. This trade-off may need reassessment if the cost of false positives (e.g., marketing spend) changes.

- **Static/session-based data:** The dataset captures session-level behavior only (e.g., page values, bounce rates) and does not include user history, demographics, or long-term behavior, which could further improve prediction accuracy if available.

- **Model generalization:** The model was trained and evaluated on a single dataset from one e-commerce platform's session logs. Performance may not generalize directly to other platforms or significantly different user bases without retraining.