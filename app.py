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
    classification_report
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Bank Marketing ML Classifier",
    layout="centered"
)

# --------------------------------------------------
# Title and description
# --------------------------------------------------
st.title("Bank Marketing Classification App")
st.caption(
    "Compare multiple machine learning models to predict whether a customer "
    "will subscribe to a term deposit using the Bank Marketing dataset."
)

st.markdown("---")

# --------------------------------------------------
# Download test dataset
# --------------------------------------------------
st.subheader("Download Sample Test Dataset")

try:
    test_df = pd.read_csv("test_labeled.csv")
    st.download_button(
        label="Download test_labeled.csv",
        data=test_df.to_csv(index=False),
        file_name="test_labeled.csv",
        mime="text/csv"
    )
except FileNotFoundError:
    st.warning("Test dataset not found in repository.")

st.markdown("---")

# --------------------------------------------------
# Model selection
# --------------------------------------------------
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

# Load selected model
model = joblib.load(f"model/{model_name}.pkl")

st.markdown("---")

# --------------------------------------------------
# Upload test dataset
# --------------------------------------------------
st.subheader("Upload Test Dataset (CSV)")

uploaded_file = st.file_uploader(
    "Upload the test CSV file (encoded features)",
    type=["csv"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(data.head())

    # Separate features and target (if available)
    if "y" in data.columns:
        y_true = data["y"]
        X = data.drop("y", axis=1)
        labeled_data = True
    else:
        X = data
        y_true = None
        labeled_data = False

    st.markdown("---")

    # --------------------------------------------------
    # Run prediction
    # --------------------------------------------------
    if st.button("Run Prediction"):
        y_pred = model.predict(X)

        # --------------------------------------------------
        # Display predictions
        # --------------------------------------------------
        st.subheader("Predictions")
        
        # Combine input features with predictions
        result_df = X.copy()
        result_df["Prediction"] = y_pred
        result_df["Prediction Label"] = result_df["Prediction"].map(
            {0: "No Subscription", 1: "Subscription"})
        
        # Reorder columns to show prediction first
        cols = ["Prediction", "Prediction Label"] + [
            col for col in result_df.columns
            if col not in ["Prediction", "Prediction Label"]
        ]
        
        result_df = result_df[cols]
        
        st.info("Prediction Legend: 1 = Subscription, 0 = No Subscription")
        
        # Show only first few rows for readability
        st.dataframe(result_df.head(20))


        # --------------------------------------------------
        # Evaluation metrics (only if labels exist)
        # --------------------------------------------------
        if labeled_data:
            st.markdown("---")
            st.subheader(f"Evaluation Metrics for: {model_name.replace('_', ' ').title()}")

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            
            # Probabilities for AUC
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

            # --------------------------------------------------
            # Confusion matrix
            # --------------------------------------------------
            st.markdown("---")
            st.subheader(f"Confusion Matrix for: {model_name.replace('_', ' ').title()}")

            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=["Actual No", "Actual Yes"],
                columns=["Predicted No", "Predicted Yes"]
            )

            st.dataframe(cm_df)

            
        else:
            st.warning(
                "Uploaded dataset does not contain labels (`y`). "
                "Evaluation metrics cannot be computed."
            )