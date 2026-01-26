import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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

        pred_df = pd.DataFrame({
            "Prediction": y_pred
        })

        pred_df["Prediction Label"] = pred_df["Prediction"].map(
            {0: "No Subscription", 1: "Subscription"}
        )

        st.info("Prediction Legend: 1 = Subscription, 0 = No Subscription")
        st.dataframe(pred_df.head(20))

        # --------------------------------------------------
        # Evaluation metrics (only if labels exist)
        # --------------------------------------------------
        if labeled_data:
            st.markdown("---")
            st.subheader("Evaluation Metrics")

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Accuracy", f"{accuracy:.3f}")
            col2.metric("Precision", f"{precision:.3f}")
            col3.metric("Recall", f"{recall:.3f}")
            col4.metric("F1 Score", f"{f1:.3f}")

            # --------------------------------------------------
            # Confusion matrix
            # --------------------------------------------------
            st.markdown("---")
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=["Actual No", "Actual Yes"],
                columns=["Predicted No", "Predicted Yes"]
            )

            st.dataframe(cm_df)

            # --------------------------------------------------
            # Classification report
            # --------------------------------------------------
            st.markdown("---")
            st.subheader("Classification Report")
            st.text(classification_report(y_true, y_pred))

        else:
            st.warning(
                "Uploaded dataset does not contain labels (`y`). "
                "Evaluation metrics cannot be computed."
            )