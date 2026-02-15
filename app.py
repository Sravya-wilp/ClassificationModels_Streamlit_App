
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             classification_report, confusion_matrix)

st.set_page_config(page_title="Breast Cancer Classifier Explorer", layout="wide")
st.title("Breast Cancer Wisconsin (Diagnostic) – Model Explorer")
st.caption("Upload test CSV, select a model, and view metrics, ROC curve, threshold analysis, confusion matrix, and classification report.")

ART = Path(__file__).resolve().parent / 'model_artifacts'

# Load available models dynamically
model_files = list(ART.glob('*.pkl'))
model_names = [f.stem for f in model_files]
name_map = {f.stem: f for f in model_files}

col_left, col_right = st.columns([1,1])
with col_left:
    chosen = st.selectbox("Select Trained Model", options=model_names, index=0 if model_names else None,
                          format_func=lambda s: s.replace('_',' ').title())
    thr = st.slider("Decision Threshold (for malignant=1)", 0.1, 0.9, 0.5, 0.05)
with col_right:
    st.download_button(
        label="Download Sample Test CSV",
        data=(ART / 'sample_test.csv').read_bytes() if (ART/'sample_test.csv').exists() else b'',
        file_name='sample_test.csv',
        mime='text/csv',
        disabled=not (ART/'sample_test.csv').exists()
    )

uploaded = st.file_uploader("Upload CSV (must contain the model feature columns; optional labels in 'target' or 'diagnosis')", type=['csv'])

@st.cache_data
def load_metrics():
    p = ART / 'metrics.json'
    if p.exists():
        with open(p, 'r') as f:
            return json.load(f)
    return None

meta = load_metrics()
feature_names = meta.get('feature_names') if meta else None

if uploaded is not None and chosen is not None:
    df = pd.read_csv(uploaded)
    # Try to infer labels
    y_true = None
    if 'target' in df.columns:
        y_true = df['target'].astype(int).values
        X = df.drop(columns=['target'])
    elif 'diagnosis' in df.columns:
        # Expect 'M'/'B'
        y_true = (df['diagnosis'].map({'M':1,'B':0})).astype(int).values
        X = df.drop(columns=['diagnosis','id'], errors='ignore')
    else:
        X = df

    model = joblib.load(name_map[chosen])

    # Predict prob if available
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:,1]
    elif hasattr(model, 'decision_function'):
        from sklearn.preprocessing import MinMaxScaler
        scores = model.decision_function(X).reshape(-1,1)
        y_prob = MinMaxScaler().fit_transform(scores).ravel()

    if y_prob is not None:
        y_pred = (y_prob >= thr).astype(int)
    else:
        y_pred = model.predict(X)

    st.subheader("Predictions Preview")
    preview = pd.DataFrame({
        'pred_label(malignant=1)': y_pred,
        'prob_malignant': y_prob if y_prob is not None else np.nan
    })
    st.dataframe(pd.concat([X.reset_index(drop=True).head(10), preview.head(10)], axis=1))

    if y_true is not None:
        st.subheader("Evaluation Metrics")
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
        mdf = pd.DataFrame({
            'Metric': ['Accuracy','AUC','Precision','Recall','F1','MCC'],
            'Value': [acc, auc, prec, rec, f1, mcc]
        })
        st.dataframe(mdf.style.format({'Value': '{:.4f}'}), use_container_width=True)

        # --- AUC ROC Curve ---
        if y_prob is not None:
            from sklearn.metrics import roc_curve, auc as _auc
            fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
            auc_val = _auc(fpr, tpr)
            import numpy as np
            idx = (np.abs(roc_thr - thr)).argmin() if len(roc_thr) > 0 else None
            st.subheader("ROC Curve (AUC)")
            fig2, ax2 = plt.subplots(figsize=(4,4))
            ax2.plot(fpr, tpr, label=f'ROC (AUC = {auc_val:.4f})')
            ax2.plot([0,1],[0,1],'k--', alpha=0.5)
            if idx is not None and idx < len(fpr):
                ax2.scatter([fpr[idx]],[tpr[idx]], color='red', zorder=5, label=f'Threshold ≈ {thr:.2f}')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('Receiver Operating Characteristic')
            ax2.legend(loc='lower right')
            st.pyplot(fig2)

            # --- Threshold Analysis ---
            st.subheader("Threshold Analysis")
            grid = np.linspace(0.05, 0.95, 19)
            rows = []
            for t in grid:
                yp = (y_prob >= t).astype(int)
                pa = precision_score(y_true, yp, zero_division=0)
                ra = recall_score(y_true, yp, zero_division=0)
                fa = f1_score(y_true, yp, zero_division=0)
                rows.append({'Threshold': round(float(t),2), 'Precision': pa, 'Recall': ra, 'F1': fa})
            tdf = pd.DataFrame(rows)
            c1, c2 = st.columns([1.2,1])
            with c1:
                st.line_chart(tdf.set_index('Threshold'))
            with c2:
                st.dataframe(tdf.style.format({'Precision':'{:.3f}','Recall':'{:.3f}','F1':'{:.3f}'}), use_container_width=True)

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
        st.info("No ground-truth labels found in the uploaded CSV. Include a 'target' column (1=malignant,0=benign) or 'diagnosis' column (M/B) to view evaluation metrics.")
else:
    st.info("Upload a CSV and pick a model to evaluate. You can also download a sample test CSV from the right panel.")
