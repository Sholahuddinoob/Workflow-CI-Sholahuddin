import argparse
import pandas as pd
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("--C", type=float, default=1.0)
args = parser.parse_args()

# Load data
df = pd.read_csv("employee_data_preprocessing/employee_data_preprocessing.csv")
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
with mlflow.start_run():
    mlflow.log_param("C", args.C)
    model = LogisticRegression(C=args.C, max_iter=1000)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", acc)
    print(f"Accuracy: {acc}")
