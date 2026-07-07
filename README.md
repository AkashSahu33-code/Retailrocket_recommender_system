# Retailrocket E-Commerce Analytics: Abnormal User Detection & Item Property Prediction

A two-stage pipeline on real e-commerce clickstream data: first identifying and removing bot/abnormal traffic, then using the cleaned behavioral data to predict implicit item properties (category) from browsing behavior.

## Why This Pipeline Exists

Real e-commerce logs are noisy — industry estimates suggest **up to 40% of raw browsing traffic can be abnormal** (bots, scrapers, automated agents). This noise degrades any downstream model built on top of it, including recommendation systems. `Abnormal_user_detection` cleans the data; `The_Algorithm` then builds on that cleaned data to solve a genuinely hard prediction problem — inferring what a visitor is looking for before they tell you explicitly.

## Dataset

[Retailrocket E-commerce Dataset (Kaggle)](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) — anonymized real clickstream data:
- `events.csv` — 2,756,101 events (view/addtocart/transaction) from 1,407,580 unique visitors
- `item_properties_part1.csv` / `part2.csv` — time-varying item attributes, including category

---

## Part 1: `Abnormal_user_detection.ipynb`

**Goal:** Find abnormal (bot-like) users, generate the features and metric needed to justify the decision, and remove them without harming real conversion data.

### Pipeline
1. **Session construction** — 30-minute inactivity rule splits each visitor's activity into sessions
2. **Feature engineering** — 9 feature groups covering volume, item diversity, conversion rate, click velocity, timing regularity (coefficient of variation — bots tend to click at suspiciously constant intervals), repeated-item ratio, session behavior, night-activity ratio, and a sequential catalog-scan score (Spearman correlation between view order and item ID — a signature of systematic scraping)
3. **Neutral imputation** — visitors without enough activity to compute a given feature get the population median for that feature, rather than an artificial extreme value
4. **Primary model: Isolation Forest**, with its `contamination` parameter tuned using **synthetic bot injection** — 500 synthetic bot profiles with an unambiguous bot signature were injected into the feature matrix, and the contamination value was chosen as the smallest one that still recovered ≥95% of synthetic bots (avoiding over-flagging real users just to chase recall)
5. **Independent cross-validation** — two additional, independent detection methods were run to check agreement with Isolation Forest:
   - Local Outlier Factor (LOF): **92.1% agreement**
   - Strict rule-based conjunction (multiple suspicious signals required simultaneously): **45.1%** of rule-flagged users also caught by Isolation Forest, showing the model captures the same high-confidence cases rather than just noise
6. **Business-rule override** — any visitor who ever converted (`addtocart` or `transaction`) is protected from being flagged abnormal, regardless of model score, since removing real customers would be far more costly than leaving a few borderline bots in
7. **Event-level cleaning** — only `view` events from confirmed abnormal visitors are removed; all `addtocart`/`transaction` events are preserved

### Results

| Metric | Value |
|---|---|
| Selected contamination | 0.10 |
| Synthetic bot recall at selected contamination | 100% |
| Final abnormal users flagged (post business-override) | 7.4% |
| View events removed | 25.7% |
| Silhouette Score | 0.689 |
| Isolation Forest vs. LOF agreement | 92.1% |
| Isolation Forest vs. rule-based overlap | 45.1% |

Output: `events_clean_FINAL.csv` — feeds directly into the next stage.

---

## Part 2: `The_Algorithm.ipynb`

**Goal:** Predict the implicit properties (category) of items a visitor will `addtocart`, using only their prior `view` behavior — since these preferences aren't stated explicitly anywhere in the clickstream.

### Pipeline
1. **Leakage-safe category join** — item categories are joined to events using `merge_asof` with backward direction, ensuring only category information known *as of* each event's timestamp is used (categories can change over time in the raw data)
2. **Feature engineering** — last-viewed category, second-last-viewed category (trajectory signal), cumulative views, cumulative unique categories, category revisit count, running-mode category
3. **Time-based train/test split**, with the top-30 target categories selected **from the training set only** — an earlier version of this pipeline leaked future category-popularity information backward into training; this was identified and fixed
4. **Two-stage architecture**:
   - **Stage A** (binary LightGBM): will this visitor deviate from their last-viewed category? (~5% of cases do)
   - **Stage B** (multiclass LightGBM, trained only on Stage A's positive cases): if they deviate, which category?
   - Final prediction defaults to the last-viewed category unless Stage A triggers Stage B — this concentrates model capacity on the genuinely hard ~5% of cases instead of re-deriving "predict last-viewed" for the other 95%
5. **Hyperparameter tuning** — RandomizedSearchCV with TimeSeriesSplit for Stage B

### Results

| Model | Accuracy |
|---|---|
| Majority-class baseline | 71.4% |
| Same-as-last-viewed baseline | 94.42% |
| **Two-stage pipeline (final)** | **94.39%** (Top-1) |
| Two-stage pipeline — Top-3 | 97.24% |
| Two-stage pipeline — Top-5 | 97.92% |
| Macro F1 | 0.852 |

**Honest note:** the final Top-1 accuracy is marginally below the naive "same-as-last-viewed" baseline (94.39% vs. 94.42%) — reported transparently rather than adjusted. The value of the two-stage architecture is better seen in Top-3/Top-5 accuracy and Macro F1, which reflect the model's ability to rank the correct category highly even in the ~5.5% of cases where visitors genuinely change their mind — cases a single-baseline approach cannot address at all.

---

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · LightGBM · Isolation Forest · Local Outlier Factor · Matplotlib · Seaborn · SciPy

## Repository Structure

```
├── Abnormal_user_detection.ipynb   # Stage 1: bot/abnormal traffic detection & removal
├── The_Algorithm.ipynb             # Stage 2: addtocart category prediction from view behavior
├── data/                           # Raw CSVs (not included, see Kaggle link above)
└── README.md
```

## Limitations

- Bot labels are not available in the raw dataset; validation relies on synthetic injection and cross-method agreement rather than true ground truth.
- The two-stage prediction model is evaluated against a genuinely strong "same as last viewed" baseline (94.4%), reflecting how habitual real browsing behavior is — the harder, more valuable prediction task is specifically the ~5% of cases where visitors change category, which is what Stage A/B are built to target.
