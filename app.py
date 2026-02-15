
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             classification_report, confusion_matrix)

st.set_page_config(page_title="Breast Cancer Classifier – FAST", layout="wide")
st.title("Breast Cancer Wisconsin – Model Explorer (FAST)")
st.caption("Lazy-training per model + caching. Upload CSV, pick model, evaluate. Includes ROC + Threshold analysis.")

# ------------------ Sidebar options ------------------
st.sidebar.header("Settings")
fast_mode = st.sidebar.checkbox("Fast mode (skip XGBoost)", value=True)
rf_trees = st.sidebar.slider("Random Forest trees", 100, 400, 200, 50)
xgb_trees = st.sidebar.slider("XGBoost trees", 100, 400, 200, 50)
threshold = st.sidebar.slider("Decision Threshold (malignant=1)", 0.10, 0.90, 0.50, 0.05)

# ------------------ Cached data load/split ------------------
@st.cache_data(show_spinner=False)
def load_split():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = (data.target == 0).astype(int)  # 1=malignant, 0=benign
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    sample = Xte.copy(); sample['target'] = yte.values
    return Xtr, Xte, ytr, yte, list(X.columns), sample.to_csv(index=False)

Xtr, Xte, ytr, yte, FEATURE_NAMES, SAMPLE_CSV = load_split()

# ------------------ Cached builder per model ------------------
@st.cache_resource(show_spinner=False)
def build_model(name: str, rf_trees: int, xgb_trees: int, enable_xgb: bool):
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier

    if name == 'logistic_regression':
        preprocess = ColumnTransformer([("num", Pipeline([("scaler", StandardScaler())]), FEATURE_NAMES)])
        return Pipeline([("preprocess", preprocess),
                        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42))])
    if name == 'decision_tree':
        return DecisionTreeClassifier(random_state=42)
    if name == 'knn':
        preprocess = ColumnTransformer([("num", Pipeline([("scaler", StandardScaler())]), FEATURE_NAMES)])
        return Pipeline([("preprocess", preprocess), ("clf", KNeighborsClassifier(n_neighbors=5))])
    if name == 'naive_bayes_gaussian':
        return GaussianNB()
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=rf_trees, random_state=42)
    if name == 'xgboost':
        if not enable_xgb:
            raise RuntimeError("XGBoost disabled in fast mode")
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=xgb_trees, max_depth=4, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, eval_metric='logloss',
                             random_state=42, tree_method='hist')
    raise ValueError("Unknown model")

MODEL_LABELS = {
    'logistic_regression': 'Logistic Regression',
    'decision_tree': 'Decision Tree',
    'knn': 'kNN',
    'naive_bayes_gaussian': 'Naive Bayes (Gaussian)',
    'random_forest': 'Random Forest (Ensemble)',
    'xgboost': 'XGBoost (Ensemble)'
}

available = list(MODEL_LABELS.keys()) if not fast_mode else [k for k in MODEL_LABELS if k != 'xgboost']

left, right = st.columns([1,1])
with left:
    chosen = st.selectbox("Select Model", options=available, format_func=lambda k: MODEL_LABELS[k])
with right:
    st.download_button("Download Sample Test CSV", data=SAMPLE_CSV.encode('utf-8'), file_name='sample_test.csv', mime='text/csv')

uploaded = st.file_uploader("Upload CSV (features + optional 'target' or 'diagnosis')", type=['csv'])

# Train on demand (first use)
@st.cache_resource(show_spinner=False)
def fit_model(name, rf_trees, xgb_trees, enable_xgb):
    mdl = build_model(name, rf_trees, xgb_trees, enable_xgb)
    t0 = time.time()
    mdl.fit(Xtr, ytr)
    train_time = time.time() - t0
    return mdl, train_time

if uploaded is not None and chosen is not None:
    df = pd.read_csv(uploaded)
    y_true = None
    if 'target' in df.columns:
        y_true = df['target'].astype(int).values
        X = df.drop(columns=['target'])
    elif 'diagnosis' in df.columns:
        y_true = (df['diagnosis'].map({'M':1,'B':0})).astype(int).values
        X = df.drop(columns=['diagnosis','id'], errors='ignore')
    else:
        X = df

    # Fit or reuse cached model
    with st.spinner(f"Training {MODEL_LABELS[chosen]} (first time only)…"):
        try:
            model, ttrain = fit_model(chosen, rf_trees, xgb_trees, not fast_mode)
        except RuntimeError as e:
            st.warning(str(e))
            st.stop()

    st.success(f"{MODEL_LABELS[chosen]} is ready (trained in {ttrain:.2f}s; cached for reuse).")

    # Predict
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:,1]
    elif hasattr(model, 'decision_function'):
        from sklearn.preprocessing import MinMaxScaler
        scores = model.decision_function(X).reshape(-1,1)
        y_prob = MinMaxScaler().fit_transform(scores).ravel()

    y_pred = (y_prob >= threshold).astype(int) if y_prob is not None else model.predict(X)

    st.subheader("Predictions Preview")
    preview = pd.DataFrame({'pred_label(malignant=1)': y_pred, 'prob_malignant': y_prob if y_prob is not None else np.nan})
    st.dataframe(pd.concat([X.reset_index(drop=True).head(10), preview.head(10)], axis=1), width='stretch')

    if y_true is not None:
        st.subheader("Evaluation Metrics")
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        mcc  = matthews_corrcoef(y_true, y_pred)
        auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
        mdf = pd.DataFrame({'Metric': ['Accuracy','AUC','Precision','Recall','F1','MCC'],
                            'Value': [acc, auc, prec, rec, f1, mcc]})
        st.dataframe(mdf.style.format({'Value': '{:.4f}'}), width='stretch')

        if y_prob is not None:
            from sklearn.metrics import roc_curve, auc as _auc
            fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
            auc_val = _auc(fpr, tpr)
            idx = (np.abs(roc_thr - threshold)).argmin() if len(roc_thr)>0 else None
            st.subheader("ROC Curve (AUC)")
            fig2, ax2 = plt.subplots(figsize=(4,4))
            ax2.plot(fpr, tpr, label=f'ROC (AUC = {auc_val:.4f})')
            ax2.plot([0,1],[0,1],'k--', alpha=0.5)
            if idx is not None and idx < len(fpr):
                ax2.scatter([fpr[idx]],[tpr[idx]], color='red', zorder=5, label=f'Threshold ≈ {threshold:.2f}')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.legend(loc='lower right')
            st.pyplot(fig2)

            st.subheader("Threshold Analysis")
            grid = np.linspace(0.05, 0.95, 19)
            rows = []
            for t in grid:
                yp = (y_prob >= t).astype(int)
                pa = precision_score(y_true, yp, zero_division=0)
                ra = recall_score(y_true, yp, zero_division=0)
                fa = f1_score(y_true, yp, zero_division=0)
                rows.append({'Threshold': round(float(t),2), 'Precision': pa, 'Recall': ra, 'F1': fa})
            tdf = pd.DataFrame(rows).set_index('Threshold')
            st.line_chart(tdf, width='stretch')
            st.dataframe(tdf.reset_index().style.format({'Precision':'{:.3f}','Recall':'{:.3f}','F1':'{:.3f}'}), width='stretch')

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticklabels(['benign(0)','malignant(1)'])
        ax.set_yticklabels(['benign(0)','malignant(1)'])
        st.pyplot(fig)

        st.subheader("Classification Report")
        rep = classification_report(y_true, y_pred, target_names=['benign(0)','malignant(1)'], output_dict=False)
        st.text(rep)
    else:
        st.info("No ground-truth labels found. Include 'target' (1=malignant,0=benign) or 'diagnosis' (M/B) to see metrics.")
else:
    st.info("Upload a CSV and select a model to evaluate. Use 'Download Sample Test CSV' to get a ready-made file.")
