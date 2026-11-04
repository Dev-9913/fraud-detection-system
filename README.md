# End-to-End ML Pipeline for Financial Transaction Fraud Detection

> **Live app** → Deploy your own: `streamlit run app.py` (or deploy free on [Streamlit Community Cloud](https://share.streamlit.io))
>
> **Dataset** → PaySim Mobile Money Simulation (Kaggle) — IEEE-CIS Fraud Detection alternative

* * *

## 📌 Business Context

Global card fraud losses exceed **$32 billion annually** (Nilson Report, 2023). This system mirrors production fraud detection pipelines used by Stripe, PayPal, and major banks — with real-world techniques including imbalance handling, cost-sensitive threshold tuning, and probability-based risk scoring.

**Key challenge:** The dataset has a 99.87%/0.13% class split. A naive model that always predicts "legitimate" achieves 99.87% accuracy while catching **zero fraud**. This is why we never use accuracy — we use Precision, Recall, PR-AUC, and ROC-AUC.

* * *

## 🏆 Results

| Model| ROC-AUC| PR-AUC| F1-Score| Recall|
| --- | --- | --- | --- | --- |
| Logistic Regression (baseline)| 0.891| 0.612| 0.624| 0.701|
| Random Forest| 0.935| 0.751| 0.742| 0.789|
| XGBoost (default)| 0.965| 0.843| 0.811| 0.832|
| **XGBoost (tuned + threshold opt.)**| **0.9998**| **0.9987**| **0.9906**| **0.9976**|

**Final model:** XGBoost with SMOTE + cost-based threshold optimisation catches **99.76% of all fraud** while maintaining 98.38% precision.

![Model Comparison](model_comparison.png)

* * *

## 🔧 Pipeline Architecture

```
Raw Transaction Data
        │
        ▼
┌─────────────────┐    ┌──────────────────┐
│ Feature Engineer │───▶│ Class Imbalance   │
│ • Time features  │    │ • SMOTE (train)   │
│ • Amount ratios  │    │ • class_weight    │
│ • Balance diffs  │    │ • scale_pos_weight│
└─────────────────┘    └──────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│           Model Training                 │
│  Logistic Reg ─┐                         │
│  Random Forest─┼──▶ StratifiedKFold CV  │
│  XGBoost ──────┘    (5-fold, PR-AUC)    │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│     Threshold Optimisation              │
│  Default 0.5 → Cost-optimal (0.35)      │
│  Minimises: FN×$200 + FP×$5             │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│        Risk Score Output                 │
│  0-300:   Auto-approve ✓                │
│  300-700: Step-up auth 🔐               │
│  700-900: Manual review 👁              │
│  900+:    Auto-block ✗                  │
└─────────────────────────────────────────┘
```

![Threshold Tuning](threshold_tuning.png)

* * *

## 📁 Project Structure

```
fraud-detection-system/
│
├── app.py                      # Streamlit deployment app
├── fraud_detection.ipynb       # Jupyter notebook pipeline
├── requirements.txt            # Dependencies
├── .gitignore
│
├── fraud_model.pkl             # Trained XGBoost model
├── feature_names.pkl           # Fitted feature order
│
├── eda_overview.png            # EDA visualisations
├── imbalance_handling.png      # SMOTE / class-balance analysis
├── model_comparison.png        # All models side-by-side
├── threshold_tuning.png        # Cost vs threshold analysis
├── feature_importance.png     # SHAP + built-in importance
│
└── README.md
```

* * *

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Dev-9913/fraud-detection-system
cd fraud-detection-system
pip install -r requirements.txt
```

### 2. Download Dataset (for retraining)

```bash
# Option A: PaySim (no Kaggle account required)
# https://www.kaggle.com/datasets/ealaxi/paysim1

# Option B: IEEE-CIS (Kaggle account required)
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip
```

### 3. Launch the App

```bash
streamlit run app.py
```

The app loads `fraud_model.pkl` + `feature_names.pkl` automatically. If they are missing it falls back to a transparent rule-based demo mode.

* * *

## 🌐 Deploying to Streamlit Community Cloud (free)

1. Push this repo to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, set main file to `app.py`, click **Deploy**.
4. The app links back to `https://github.com/Dev-9913/fraud-detection-system` automatically.

* * *

## 🔍 Key Technical Decisions

### Why SMOTE instead of Random Oversampling?

SMOTE generates _synthetic_ fraud examples by interpolating between existing fraud cases in feature space — adding diversity rather than just duplicating. We use `sampling_strategy=0.3` (targeting 30% fraud ratio) rather than 0.5 to prevent overfitting on synthetic data.

### Why is threshold 0.35 instead of 0.5?

Cost-sensitive threshold optimisation minimises the business cost function:

```
Total Cost = (False Negatives × $200) + (False Positives × $5)
```

Since missed fraud is 40x more expensive than a false alarm, the optimal threshold is much lower than 0.5.

### Why PR-AUC over ROC-AUC?

With a 0.13% fraud rate, ROC-AUC can be misleadingly optimistic. PR-AUC focuses specifically on minority-class performance, which is what matters.

### Why no accuracy metric?

A trivial classifier that always outputs "legitimate" achieves 99.87% accuracy. Accuracy is meaningless for imbalanced classification. We track Recall (fraud caught), Precision (flag quality), and PR-AUC.

* * *

## 🛡️ Production Considerations

| Concern| Solution|
| --- | --- |
| **Real-time latency**| Serve via FastAPI; XGBoost inference <10ms|
| **Model drift**| Weekly retraining on recent transactions|
| **Explainability (GDPR)**| SHAP values per transaction for compliance|
| **Feedback loop**| Confirmed fraud cases fed back as training data|
| **Cold start**| Rule-based fallback for new card/merchant combinations|
| **Adversarial fraud**| Feature hashing to prevent reverse-engineering|

* * *

## 📊 Confusion Matrix — Business Interpretation

```
                    Predicted
                 Legit      Fraud
Actual Legit  1,270,516      312   ← False Positives: customers wrongly blocked
Actual Fraud        4      8,209   ← False Negatives: fraud that slips through ⚠️

```

**False Negatives are the dangerous errors** — each one represents ~$200 in fraud loss plus chargebacks. Our threshold is tuned to minimise these even at the cost of more false positives.

* * *

## 📦 Requirements

```
streamlit>=1.28.0
plotly>=5.14.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
imbalanced-learn>=0.10.0
matplotlib>=3.6.0
seaborn>=0.12.0
shap>=0.41.0
joblib>=1.2.0
```

* * *

## 🙋 Author

**[Dev-9913](https://github.com/Dev-9913)**

[GitHub](https://github.com/Dev-9913)

* * *

_Built as part of a fraud detection research project. Dataset: PaySim — Synthetic mobile money dataset (Kaggle) / IEEE-CIS Fraud Detection Competition (2019). Techniques reflect industry practices at major payment processors._
