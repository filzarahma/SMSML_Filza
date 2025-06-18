import os
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, log_loss, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn import set_config
import dagshub
dagshub.init(repo_owner='filzarahma', repo_name='heart-disease-prediction', mlflow=True)

# Set URI tracking dan experiment
mlflow.set_tracking_uri("https://dagshub.com/filzarahma/heart-disease-prediction.mlflow") # for DagsHub MLflow server
# mlflow.set_tracking_uri("http://127.0.0.1:5000/") # for local MLflow server
mlflow.set_experiment("Tuning - Heart Disease Prediction")

df = pd.read_csv("heart_preprocessing.csv")
X = df.drop(columns=["HeartDisease"])
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

n_estimators_range = [50, 100, 200, 300, 400, 500]
max_depth_range = [5, 10, 15, 20, None]

best_accuracy = 0
best_params = {}

os.makedirs("mlruns_artifacts", exist_ok=True)

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"rf_n{n_estimators}_depth{max_depth}"):
            # Log parameter manual
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("random_state", 42)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            model.fit(X_train, y_train)

            # Test set predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            accuracy = accuracy_score(y_test, y_pred)
            rf_score = model.score(X_test, y_test)

            # Training set predictions
            y_train_pred = model.predict(X_train)
            y_train_pred_proba = model.predict_proba(X_train) if hasattr(model, "predict_proba") else None
            training_accuracy = accuracy_score(y_train, y_train_pred)
            training_f1 = f1_score(y_train, y_train_pred, zero_division=0)
            training_precision = precision_score(y_train, y_train_pred, zero_division=0)
            training_recall = recall_score(y_train, y_train_pred, zero_division=0)
            training_score = model.score(X_train, y_train)
            training_logloss = log_loss(y_train, y_train_pred_proba) if y_train_pred_proba is not None else None
            try:
                training_roc_auc = roc_auc_score(y_train, y_train_pred_proba[:, 1]) if y_train_pred_proba is not None else None
            except Exception:
                training_roc_auc = None

            # Tambahan metric: test_f1_score dan test_roc_auc
            test_f1_score = f1_score(y_test, y_pred, zero_division=0)
            try:
                test_roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1]) if y_pred_proba is not None else None
            except Exception:
                test_roc_auc = None

            # Logging metrics manual (tambahan)
            mlflow.log_metric("test_f1_score", test_f1_score)
            if test_roc_auc is not None:
                mlflow.log_metric("test_roc_auc", test_roc_auc)

            # Logging metrics manual
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("RandomForestClassifier_score_X_test", rf_score)
            mlflow.log_metric("training_accuracy_score", training_accuracy)
            mlflow.log_metric("training_f1_score", training_f1)
            if training_logloss is not None:
                mlflow.log_metric("training_logloss", training_logloss)
            mlflow.log_metric("training_precision_score", training_precision)
            mlflow.log_metric("training_recall_score", training_recall)
            if training_roc_auc is not None:
                mlflow.log_metric("training_roc_auc", training_roc_auc)
            mlflow.log_metric("training_score", training_score)

            # Save confusion matrix plot
            fig_cm, ax_cm = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, X_train, y_train, ax=ax_cm)
            cm_path = "mlruns_artifacts/training_confusion_matrix.png"
            fig_cm.savefig(cm_path)
            plt.close(fig_cm)
            mlflow.log_artifact(cm_path)  # log to root artifacts

            # Save ROC curve plot
            if hasattr(model, "predict_proba"):
                fig_roc, ax_roc = plt.subplots()
                RocCurveDisplay.from_estimator(model, X_train, y_train, ax=ax_roc)
                roc_path = "mlruns_artifacts/training_roc_curve.png"
                fig_roc.savefig(roc_path)
                plt.close(fig_roc)
                mlflow.log_artifact(roc_path)  # log to root artifacts

                # Save Precision-Recall curve plot
                fig_pr, ax_pr = plt.subplots()
                PrecisionRecallDisplay.from_estimator(model, X_train, y_train, ax=ax_pr)
                pr_path = "mlruns_artifacts/training_precision_recall_curve.png"
                fig_pr.savefig(pr_path)
                plt.close(fig_pr)
                mlflow.log_artifact(pr_path)  # log to root artifacts

            # Save metric_info.json
            metric_info = {
                "accuracy": accuracy,
                "RandomForestClassifier_score_X_test": rf_score,
                "training_accuracy_score": training_accuracy,
                "training_f1_score": training_f1,
                "training_log_loss": training_logloss if training_logloss is not None else "NA",
                "training_precision_score": training_precision,
                "training_recall_score": training_recall,
                "training_roc_auc": training_roc_auc if training_roc_auc is not None else "NA",
                "training_score": training_score
            }
            metric_info_path = "mlruns_artifacts/metric_info.json"
            with open(metric_info_path, "w") as f:
                json.dump(metric_info, f, indent=2)
            mlflow.log_artifact(metric_info_path)  # log to root artifacts

            # Logging classification report sebagai artifact
            report = classification_report(y_train, y_train_pred)
            with open("mlruns_artifacts/classification_report.txt", "w") as f:
                f.write(report)
            mlflow.log_artifact("mlruns_artifacts/classification_report.txt")

            # Logging dataset ke MLflow UI bagian dataset (bukan artifact) dengan label Train/Test
            mlflow.log_input(
                mlflow.data.from_pandas(
                    X_train.assign(HeartDisease=y_train),
                    name="train-set"
                ),
                context="Train"
            )
            mlflow.log_input(
                mlflow.data.from_pandas(
                    X_test.assign(HeartDisease=y_test),
                    name="test-set"
                ),
                context="Test"
            )

            # Logging model (seperti autolog, hanya model terbaik yang di-register)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth
                }                
                # Membuat signature dengan nama kolom output "HeartDisease" (bukan hanya double)
                output_sample = pd.DataFrame(model.predict(X_train.sample(5, random_state=42)), columns=["HeartDisease"])
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=X_train.sample(5, random_state=42),
                    registered_model_name="rf-heart-disease-best",
                    signature=infer_signature(X_train, output_sample)
                )
            else:                
                # Menggunakan signature yang sama dengan nama kolom output "HeartDisease"
                output_sample = pd.DataFrame(model.predict(X_train.sample(5, random_state=42)), columns=["HeartDisease"])
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=X_train.sample(5, random_state=42),
                    signature=infer_signature(X_train, output_sample)
                )

            # Save estimator.html (model visualization)
            estimator_html_path = "mlruns_artifacts/estimator.html"
            set_config(display="diagram")
            try:
                from sklearn.utils import estimator_html_repr
                with open(estimator_html_path, "w", encoding="utf-8") as f:
                    f.write(estimator_html_repr(model))
                mlflow.log_artifact(estimator_html_path)
            except Exception:
                pass  # estimator_html_repr not available or error

