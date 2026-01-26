import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

data = pd.read_csv("../bank-full.csv", sep=";")
data["y"] = data["y"].replace({"yes": 1, "no": 0})
data = pd.get_dummies(data, drop_first=True)

X = data.drop("y", axis=1)
y = data["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier()
model.fit(X_train_scaled, y_train)

pred = model.predict(X_test_scaled)
prob = model.predict_proba(X_test_scaled)[:, 1]

print("KNN")
print("Accuracy:", accuracy_score(y_test, pred))
print("AUC:", roc_auc_score(y_test, prob))
print("Precision:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
print("F1:", f1_score(y_test, pred))
print("MCC:", matthews_corrcoef(y_test, pred))

joblib.dump(model, "knn.pkl")
