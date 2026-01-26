
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

st.set_page_config(page_title="Bank Marketing ML Classifier")
st.title("Bank Marketing Classification App")

# Download test dataset
try:
    test_df = pd.read_csv("test_labeled.csv")
    st.download_button(
        "Download Test Dataset",
        test_df.to_csv(index=False),
        "test_labeled.csv",
        "text/csv"
    )
except:
    pass

model_name = st.selectbox(
    "Select Model",
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

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.dataframe(data.head())

    if "y" in data.columns:
        y_true = data["y"]
        X = data.drop("y", axis=1)
    else:
        y_true = None
        X = data

    if st.button("Run Prediction"):
        y_pred = model.predict(X)

        st.subheader("Predictions")
        st.write(y_pred)

        if y_true is not None:
            st.subheader("Evaluation Metrics")
            st.write("Accuracy:", accuracy_score(y_true, y_pred))
            st.write("Precision:", precision_score(y_true, y_pred))
            st.write("Recall:", recall_score(y_true, y_pred))
            st.write("F1 Score:", f1_score(y_true, y_pred))

            st.subheader("Confusion Matrix")
            st.write(confusion_matrix(y_true, y_pred))

            st.subheader("Classification Report")
            st.text(classification_report(y_true, y_pred))
