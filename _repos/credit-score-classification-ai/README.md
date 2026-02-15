# Credit Score Classification

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

Machine learning system for automated credit score classification based on financial behaviour data. Compares **5 models** (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost) across **3 risk categories** on **100,000+ customer records**.

> **Why it matters:** Banks process millions of credit applications. Manual assessment is slow and inconsistent. This pipeline automates scoring into Poor / Standard / Good categories, helping lenders make faster, data-driven decisions while reducing human bias.

## Key Results

| Metric | Value |
|--------|-------|
| Best model | LightGBM |
| Classes | 3 (Poor, Standard, Good) |
| Dataset | 100,000+ customer records, 28 raw features |
| Engineered features | 21 after cleaning |
| Class balancing | Stratified splits |

### Model Comparison

![Model Comparison](assets/model_comparison.png)

### Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

### Per-Class Accuracy

![Per-Class Accuracy](assets/per_class_accuracy.png)

### Feature Importance

![Feature Importance](assets/feature_importance.png)

## Credit Score Categories

| Class | Description | Typical profile |
|-------|-------------|-----------------|
| **Good** | Low risk | Low debt, timely payments, long credit history |
| **Standard** | Medium risk | Moderate debt, occasional delays |
| **Poor** | High risk | High debt, frequent delays, low income |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare dataset

Download the [Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) dataset from Kaggle and place `train.csv` into the `data/` directory:

```
data/
└── train.csv
```

### 3. Train

```bash
python train.py --data-dir data --model all
```

### 4. Evaluate

```bash
python evaluate.py --model all --data-dir data
```

### 5. Inference

```bash
python inference.py --model lightgbm --csv data/test.csv
```

### 6. Generate visuals

```bash
python generate_visuals.py
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Test split | 20% (stratified) |
| Validation split | 10% (stratified) |
| Feature scaling | StandardScaler |
| Seed | 42 |

### Models & Hyperparameters

| Model | Key hyperparameters |
|-------|-------------------|
| Logistic Regression | C=1.0, solver=lbfgs, max_iter=1000 |
| Random Forest | n_estimators=300, max_depth=20 |
| XGBoost | n_estimators=300, max_depth=8, lr=0.1 |
| LightGBM | n_estimators=300, max_depth=15, num_leaves=63 |
| CatBoost | iterations=300, depth=8, lr=0.1 |

### Dataset Challenges

- **Noisy data**: special characters in numeric columns, inconsistent formatting
- **Missing values**: ~5-15% missing across key features
- **Mixed types**: Credit_History_Age stored as "22 Years and 1 Months"
- **Class imbalance**: unequal distribution across Poor / Standard / Good

### Data Preprocessing

- Strip non-numeric characters from financial columns
- Parse Credit_History_Age into total months
- Label-encode categorical features (Occupation, Credit_Mix, Payment_Behaviour)
- Fill missing values with column medians
- StandardScaler normalization for all features

## Key Features

- **Modular architecture** — clean separation of config, data, model, training, and evaluation
- **5 model comparison** — train and compare LogReg, RF, XGBoost, LightGBM, CatBoost in one command
- **Robust preprocessing** — handles noisy Kaggle data with missing values and mixed types
- **Feature importance** — visual ranking of most predictive financial indicators
- **Reproducibility** — fixed random seeds across all libraries
- **CLI interface** — train any model or all models via command-line arguments

## Project Structure

```
credit-score-classification-ai/
├── config.py            # Centralized configuration & CLI arguments
├── dataset.py           # Data loading, cleaning, feature engineering
├── model.py             # Model factory (LogReg, RF, XGBoost, LightGBM, CatBoost)
├── train.py             # Training loop with multi-model support
├── evaluate.py          # Confusion matrix, classification report, plots
├── inference.py         # Batch prediction from CSV
├── generate_visuals.py  # Generate presentation-ready plots for README
├── requirements.txt     # Dependencies
└── README.md
```

## Tech Stack

- Python 3.12
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- pandas / numpy
- matplotlib / seaborn

## Dataset

[Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) — 100,000+ customer records with 28 features including income, debt, payment history, and credit utilization. Published on Kaggle by Paris Rohan.

## Author

Built by **Nikolai Shatikhin** — ML Engineer specializing in Machine Learning and data-driven solutions.

Open to freelance projects. Reach out via [GitHub](https://github.com/shatini) or [Telegram](https://t.me/That_Special_Someone).
