import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

# Page configuration

st.set_page_config(
    page_title="Bank Marketing ML Classifier",
    layout="centered"
)

# Title and description

st.title("Bank Marketing Classification App")
st.caption(
    "Compare multiple machine learning models to predict whether a customer "
    "will subscribe to a term deposit using the Bank Marketing dataset."
)

st.markdown("---")

# DOWNLOAD SECTION

st.subheader("Download Sample Test Dataset")

default_data = None

try:
    default_data = pd.read_csv("test_labeled.csv")

    st.download_button(
        label="Download test_labeled.csv",
        data=default_data.to_csv(index=False),
        file_name="test_labeled.csv",
        mime="text/csv"
    )
except FileNotFoundError:
    st.warning("Default test dataset not found in repository.")

st.markdown("---")

# UPLOAD SECTION

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file (If not uploaded, default dataset will be used)",
    type=["csv"]
)

# Decide which dataset to use
data = None

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success(f"File {uploaded_file.name} successfully uploaded")
else:
    data = default_data


# Preview only if dataset exists
if data is not None:
    with st.expander("Dataset Preview", expanded=False):

        st.write(f"Shape: {data.shape}")

        st.dataframe(data.head())

st.markdown("---")

# MODEL SELECTION

st.subheader("Select Machine Learning Model")

model_name = st.selectbox(
    "Choose a model",
    [
        "logistic_regression",
        "decision_tree",
        "knn",
        "naive_bayes",
        "random_forest",
        "xgboost"
    ]
)

model = joblib.load(f"model/{model_name}.pkl")

st.markdown("---")

# RUN EVALUATION BUTTON

if st.button("Run Evaluation"):

    if uploaded_file:
        st.info("Using uploaded dataset")
    else:
        st.info("No dataset uploaded. Using default test_labeled.csv.")

    if data is None:
        st.error("No dataset available.")
    else:
        # Separate features and labels
        if "y" in data.columns:
            y_true = data["y"]
            X = data.drop("y", axis=1)
            labeled_data = True
        else:
            X = data
            y_true = None
            labeled_data = False
        
        # Predictions
        
        y_pred = model.predict(X)

        st.subheader("Predictions")

        result_df = X.copy()
        result_df["Prediction"] = y_pred
        result_df["Prediction Label"] = result_df["Prediction"].map(
            {0: "No Subscription", 1: "Subscription"}
        )

        cols = ["Prediction", "Prediction Label"] + [
            c for c in result_df.columns
            if c not in ["Prediction", "Prediction Label"]
        ]

        result_df = result_df[cols]

        st.info("Prediction Legend: 1 = Subscription, 0 = No Subscription")
        st.dataframe(result_df.head(20))
       
        # METRICS DISPLAY
        
        if labeled_data:
            st.markdown("---")
            st.subheader(
                f"Evaluation Metrics for: {model_name.replace('_',' ').title()}"
            )

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)

            y_prob = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y_true, y_prob)

            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)

            col1.metric("Accuracy", f"{accuracy:.3f}")
            col2.metric("AUC", f"{auc:.3f}")
            col3.metric("Precision", f"{precision:.3f}")

            col4.metric("Recall", f"{recall:.3f}")
            col5.metric("F1 Score", f"{f1:.3f}")
            col6.metric("MCC", f"{mcc:.3f}")
           
            # Confusion Matrix
            
            st.markdown("---")
            st.subheader(
                f"Confusion Matrix for: {model_name.replace('_',' ').title()}"
            )

            cm = confusion_matrix(y_true, y_pred)

            cm_df = pd.DataFrame(
                cm,
                index=["Actual No", "Actual Yes"],
                columns=["Predicted No", "Predicted Yes"]
            )

            st.dataframe(cm_df)

        else:
            st.warning(
                "Dataset does not contain labels (`y`). Metrics cannot be computed."
            )