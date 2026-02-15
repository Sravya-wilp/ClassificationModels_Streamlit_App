# ClassificationModels_Streamlit_App
End-to-end Machine Learning classification project implementing six models (Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, and XGBoost) with performance comparison and interactive Streamlit deployment.


# Breast Cancer Wisconsin (Diagnostic) – ML Assignment 2

> Streamlit demo + 6 classifiers + full metrics and comparison

## Problem Statement
Predict whether a breast tumor is **malignant** or **benign** based on features computed from digitized images of fine needle aspirate (FNA).

## Dataset Description
- **Source:** UCI Machine Learning Repository (Breast Cancer Wisconsin – Diagnostic) and mirrored on Kaggle.
- **Instances:** 569, **Features:** 30 numeric, **Target:** binary (malignant / benign)
- No missing values. Features include radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension (mean, error, worst for each).

## Models Implemented
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

## How to Run (Locally / BITS Lab)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

This will train all models, save pickled artifacts to `model_artifacts/`, and create a `sample_test.csv` you can upload in the app.

## Comparison Table (Test Split, random_state=42, test_size=0.2)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| kNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest (Ensemble) | 0.9737 | 0.9944 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| XGBoost (Ensemble) | 0.9649 | 0.9931 | 1.0000 | 0.9048 | 0.9500 | 0.9258 |


## Observations
Add your concise observations here after inspecting the filled table.

## Deploying to Streamlit Community Cloud
1. Push this repo to GitHub
2. Visit https://streamlit.io/cloud and sign in with GitHub
3. Click **New app** → select repo/branch → choose `app.py` → **Deploy**

Deployed Link: https://ml-classification-model-comparision.streamlit.app/
