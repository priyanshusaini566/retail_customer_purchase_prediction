import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

RANDOM_STATE = 42

df = pd.read_csv("data/customer_data.csv")

FEATURES = [
    "pages_viewed", "time_spent_sec", "previous_visits",
    "products_viewed", "added_to_cart", "device_type", "previous_purchases"
]
TARGET = "purchase"

X = df[FEATURES]
y = df[TARGET]

numeric_features = ["pages_viewed", "time_spent_sec", "previous_visits",
                     "products_viewed", "previous_purchases"]
binary_features = ["added_to_cart"]
categorical_features = ["device_type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("bin", "passthrough", binary_features),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
])

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE, class_weight="balanced"
    ),
}

results = {}
fitted_pipelines = {}

for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    fitted_pipelines[name] = pipe

print("\n=== Model Comparison ===")
results_df = pd.DataFrame(results).T.round(4)
print(results_df)

best_model_name = results_df["roc_auc"].idxmax()
best_pipeline = fitted_pipelines[best_model_name]
print(f"\nBest model: {best_model_name} (ROC-AUC = {results_df.loc[best_model_name, 'roc_auc']})")

print(f"\n=== Classification Report ({best_model_name}) ===")
y_pred_best = best_pipeline.predict(X_test)
print(classification_report(y_test, y_pred_best, target_names=["No Purchase", "Purchase"]))


# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Purchase", "Purchase"],
            yticklabels=["No Purchase", "Purchase"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png", dpi=150)
plt.close()

# ROC curve 
plt.figure(figsize=(5, 4))
for name, pipe in fitted_pipelines.items():
    y_proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = results[name]["roc_auc"]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("model/roc_curve.png", dpi=150)
plt.close()

# Feature importance 
if best_model_name == "RandomForest":
    ohe = best_pipeline.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + binary_features + cat_names
    importances = best_pipeline.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": all_feature_names, "importance": importances}) \
        .sort_values("importance", ascending=True)

    plt.figure(figsize=(6, 4))
    plt.barh(imp_df["feature"], imp_df["importance"], color="#2E86AB")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("model/feature_importance.png", dpi=150)
    plt.close()

joblib.dump(best_pipeline, "model/purchase_model.pkl")
joblib.dump(best_model_name, "model/best_model_name.pkl")
results_df.to_csv("model/model_comparison.csv")

print("\nSaved: model/purchase_model.pkl")
print("Saved: model/confusion_matrix.png, model/roc_curve.png")
if best_model_name == "RandomForest":
    print("Saved: model/feature_importance.png")
