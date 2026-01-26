from xgboost import XGBClassifier

def build_model():
    return XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )