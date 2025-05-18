# modelling.py
import argparse
import pandas as pd
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature  

def main(data_path, C):
    df = pd.read_csv(data_path)  
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with mlflow.start_run():
        mlflow.log_param("C", C)
        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", acc)

        
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)

        print(f"Accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    main(args.data_path, args.C)
