
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             classification_report, confusion_matrix)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

ARTIFACTS = Path('model_artifacts')
ARTIFACTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

BUNCH = load_breast_cancer(as_frame=True)
X = BUNCH.data.copy()
y = (BUNCH.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

sample = X_test.copy(); sample['target'] = y_test.values
sample.to_csv(ARTIFACTS / 'sample_test.csv', index=False)

scaler = StandardScaler()
num_transformer = Pipeline(steps=[('scaler', scaler)])
preprocess = ColumnTransformer(transformers=[('num', num_transformer, list(X.columns))])

models = {
    'Logistic Regression': Pipeline([
        ('preprocess', preprocess),
        ('clf', LogisticRegression(max_iter=1000, solver='lbfgs', random_state=RANDOM_STATE))
    ]),
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'kNN': Pipeline([
        ('preprocess', preprocess),
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ]),
    'Naive Bayes (Gaussian)': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
}

if HAS_XGB:
    models['XGBoost'] = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        tree_method='hist'
    )

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None
    y_pred = (y_prob >= 0.5).astype(int) if y_prob is not None else model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    rows.append({'Model': name, 'Accuracy': acc, 'AUC': auc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'MCC': mcc})

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(ARTIFACTS / 'metrics.csv', index=False)
print(metrics_df)
