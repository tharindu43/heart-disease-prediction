"""
=============================================================
  Heart Disease Prediction — Member 2: Random Forest
=============================================================
  Model   : Random Forest Classifier (Ensemble / Supervised)
  Purpose : Improved prediction with feature importance analysis
  Task    : Identify the most influential cardiac indicators
  Dataset : UCI Heart Disease — Cleveland (processed.cleveland.data)
  Output  : Feature rankings, prediction scores, saved model
=============================================================
"""

# ── Standard Library ─────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

# ── Third-Party ───────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import joblib


# ══════════════════════════════════════════════════════════════
#  1. PATHS
# ══════════════════════════════════════════════════════════════
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "Data_set", "processed.cleveland.data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "random_forest")
MODEL_DIR   = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  2. LOAD & PREPROCESS DATA
# ══════════════════════════════════════════════════════════════
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

FEATURE_DESCRIPTIONS = {
    "age"      : "Age (years)",
    "sex"      : "Sex (1=Male, 0=Female)",
    "cp"       : "Chest Pain Type",
    "trestbps" : "Resting Blood Pressure (mm Hg)",
    "chol"     : "Serum Cholesterol (mg/dl)",
    "fbs"      : "Fasting Blood Sugar > 120 mg/dl",
    "restecg"  : "Resting ECG Results",
    "thalach"  : "Max Heart Rate Achieved",
    "exang"    : "Exercise Induced Angina",
    "oldpeak"  : "ST Depression (Exercise vs Rest)",
    "slope"    : "Slope of Peak Exercise ST Segment",
    "ca"       : "Number of Major Vessels (Fluoroscopy)",
    "thal"     : "Thalassemia Type",
}

print("=" * 60)
print("  Heart Disease Prediction — Random Forest Model")
print("=" * 60)

# Load dataset (missing values marked as '?')
df = pd.read_csv(DATA_PATH, names=COLUMN_NAMES, na_values="?")

print(f"\n[INFO] Raw dataset shape : {df.shape}")
print(f"[INFO] Missing values    :\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ── Handle missing values (median imputation for numeric cols)
for col in df.columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# ── Binarise target: 0 = No Disease, 1 = Disease (values 1–4)
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

print(f"\n[INFO] Class distribution:\n{df['target'].value_counts().rename({0: 'No Disease', 1: 'Disease'})}")


# ══════════════════════════════════════════════════════════════
#  3. FEATURES & SPLIT
# ══════════════════════════════════════════════════════════════
FEATURES = [c for c in df.columns if c != "target"]
X = df[FEATURES]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[INFO] Train samples : {len(X_train)}")
print(f"[INFO] Test  samples : {len(X_test)}")


# ══════════════════════════════════════════════════════════════
#  4. HYPERPARAMETER TUNING via GridSearchCV
# ══════════════════════════════════════════════════════════════
print("\n[INFO] Running GridSearchCV for hyperparameter tuning …")

param_grid = {
    "n_estimators"      : [100, 200, 300],
    "max_depth"         : [None, 5, 10],
    "min_samples_split" : [2, 5],
    "min_samples_leaf"  : [1, 2],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"[INFO] Best parameters : {best_params}")


# ══════════════════════════════════════════════════════════════
#  5. TRAIN BEST MODEL
# ══════════════════════════════════════════════════════════════
rf_model = RandomForestClassifier(
    **best_params,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
rf_model.fit(X_train, y_train)

# Cross-validation score
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="accuracy")
print(f"\n[INFO] 5-Fold CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ══════════════════════════════════════════════════════════════
#  6. EVALUATE ON TEST SET
# ══════════════════════════════════════════════════════════════
y_pred     = rf_model.predict(X_test)
y_prob     = rf_model.predict_proba(X_test)[:, 1]

test_acc   = accuracy_score(y_test, y_pred)
roc_auc    = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print(f"  TEST ACCURACY  : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"  ROC-AUC SCORE  : {roc_auc:.4f}")
print("=" * 60)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))


# ══════════════════════════════════════════════════════════════
#  7. FEATURE IMPORTANCE RANKING
# ══════════════════════════════════════════════════════════════
importances = rf_model.feature_importances_
indices     = np.argsort(importances)[::-1]

print("\n[RESULT] Feature Importance Rankings (Most → Least Influential):")
print("-" * 55)
print(f"{'Rank':<5} {'Feature':<12} {'Description':<40} {'Importance':>10}")
print("-" * 55)
for rank, idx in enumerate(indices, 1):
    feat = FEATURES[idx]
    desc = FEATURE_DESCRIPTIONS.get(feat, feat)
    print(f"{rank:<5} {feat:<12} {desc:<40} {importances[idx]:>10.4f}")
print("-" * 55)


# ══════════════════════════════════════════════════════════════
#  8. VISUALISATIONS
# ══════════════════════════════════════════════════════════════
plt.style.use("seaborn-v0_8-darkgrid")
PALETTE = "#2563EB"

# ── 8a. Feature Importance Bar Chart ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
colors = [plt.cm.Blues_r(i / len(FEATURES)) for i in range(len(FEATURES))]
bars = ax.barh(
    [FEATURE_DESCRIPTIONS.get(FEATURES[i], FEATURES[i]) for i in indices[::-1]],
    importances[indices[::-1]],
    color=colors,
    edgecolor="white",
    linewidth=0.5
)
ax.set_xlabel("Feature Importance (Gini Impurity Decrease)", fontsize=12)
ax.set_title("Random Forest — Feature Importance for Heart Disease Prediction",
             fontsize=14, fontweight="bold", pad=15)
for bar, val in zip(bars, importances[indices[::-1]]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
plt.close(fig)
print(f"\n[INFO] Saved → feature_importance.png")

# ── 8b. Confusion Matrix ──────────────────────────────────────
cm  = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"],
    ax=ax, linewidths=0.5, linecolor="white"
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Confusion Matrix — Random Forest", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close(fig)
print(f"[INFO] Saved → confusion_matrix.png")

# ── 8c. ROC Curve ─────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color=PALETTE, lw=2,
        label=f"Random Forest (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — Random Forest", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
plt.close(fig)
print(f"[INFO] Saved → roc_curve.png")

# ── 8d. Cross-Validation Score Distribution ───────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(range(1, 6), cv_scores, color=PALETTE, edgecolor="white", linewidth=0.5)
ax.axhline(cv_scores.mean(), color="red", linestyle="--",
           linewidth=1.5, label=f"Mean = {cv_scores.mean():.4f}")
ax.set_xlabel("Fold", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("5-Fold Cross-Validation Accuracy", fontsize=14, fontweight="bold")
ax.set_ylim(0.7, 1.0)
ax.legend(fontsize=11)
ax.set_xticks(range(1, 6))
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "cv_scores.png"), dpi=150)
plt.close(fig)
print(f"[INFO] Saved → cv_scores.png")

# ── 8e. Top-5 Feature Correlation with Target ─────────────────
top5 = [FEATURES[i] for i in indices[:5]]
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
for ax, feat in zip(axes, top5):
    df.boxplot(column=feat, by="target", ax=ax,
               boxprops=dict(color=PALETTE),
               medianprops=dict(color="red", linewidth=2))
    ax.set_title(FEATURE_DESCRIPTIONS.get(feat, feat), fontsize=9, fontweight="bold")
    ax.set_xlabel("0 = No Disease | 1 = Disease", fontsize=8)
    ax.set_ylabel(feat, fontsize=9)
plt.suptitle("Top-5 Features vs Heart Disease Outcome", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "top5_features_boxplot.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"[INFO] Saved → top5_features_boxplot.png")


# ══════════════════════════════════════════════════════════════
#  9. SAVE MODEL & SUMMARY REPORT
# ══════════════════════════════════════════════════════════════
model_path = os.path.join(MODEL_DIR, "random_forest_model.pkl")
joblib.dump(rf_model, model_path)
print(f"\n[INFO] Model saved → {model_path}")

# ── Text report ───────────────────────────────────────────────
report_lines = [
    "=" * 60,
    "  Heart Disease Prediction – Random Forest Summary Report",
    "=" * 60,
    f"  Dataset        : processed.cleveland.data",
    f"  Train / Test   : {len(X_train)} / {len(X_test)} samples",
    f"  Best Parameters: {best_params}",
    f"  CV Accuracy    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
    f"  Test Accuracy  : {test_acc:.4f}  ({test_acc*100:.2f}%)",
    f"  ROC-AUC Score  : {roc_auc:.4f}",
    "",
    "  Feature Importance Ranking:",
    "-" * 60,
]
for rank, idx in enumerate(indices, 1):
    feat = FEATURES[idx]
    desc = FEATURE_DESCRIPTIONS.get(feat, feat)
    report_lines.append(f"  {rank:>2}. {feat:<10} | {desc:<38} | {importances[idx]:.4f}")

report_lines += [
    "-" * 60,
    "",
    "  Outputs:",
    f"  • feature_importance.png",
    f"  • confusion_matrix.png",
    f"  • roc_curve.png",
    f"  • cv_scores.png",
    f"  • top5_features_boxplot.png",
    f"  • random_forest_model.pkl",
    "=" * 60,
]

report_text = "\n".join(report_lines)
report_path = os.path.join(OUTPUT_DIR, "summary_report.txt")
with open(report_path, "w") as f:
    f.write(report_text)

print("\n" + report_text)
print(f"\n[INFO] Report saved → {report_path}")
print("\n[DONE] Random Forest analysis complete.")
