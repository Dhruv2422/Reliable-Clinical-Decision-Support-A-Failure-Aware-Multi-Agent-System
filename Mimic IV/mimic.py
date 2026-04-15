import pandas as pd
from pathlib import Path
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
from sklearn.decomposition import PCA
from scipy.stats import entropy

data_dir = Path("Business - Documents/mimic IV")

print("Loading core tables...")
admissions = pd.read_csv(data_dir / "hosp/admissions.csv", low_memory=False)
patients = pd.read_csv(data_dir / "hosp/patients.csv", low_memory=False)
icustays = pd.read_csv(data_dir / "icu/icustays.csv", low_memory=False)
drgcodes = pd.read_csv(data_dir / "hosp/drgcodes.csv", low_memory=False)
micro = pd.read_csv(data_dir / "hosp/microbiologyevents.csv", low_memory=False)
pharmacy = pd.read_csv(data_dir / "hosp/pharmacy.csv", low_memory=False)
transfers = pd.read_csv(data_dir / "hosp/transfers.csv", low_memory=False)
inputevents = pd.read_csv(data_dir / "icu/inputevents.csv", low_memory=False)
outputevents = pd.read_csv(data_dir / "icu/outputevents.csv", low_memory=False)
procedureevents = pd.read_csv(data_dir / "icu/procedureevents.csv", low_memory=False)

print("Creating mortality label...")
admissions["mortality"] = admissions["hospital_expire_flag"].astype(int)

print("Merging demographics...")
data = admissions.merge(
    patients[["subject_id", "gender", "anchor_age"]],
    on="subject_id",
    how="left"
)



print("Building admission-level features...")

icu_hadm = icustays["hadm_id"].dropna().unique()
data["icu_admission"] = data["hadm_id"].isin(icu_hadm).astype(int)

if "los" in icustays.columns:
    icu_los = (
        icustays.groupby("hadm_id")["los"]
        .max()
        .reset_index()
        .rename(columns={"los": "icu_los_days"})
    )
    data = data.merge(icu_los, on="hadm_id", how="left")

drg_feat = (
    drgcodes.groupby("hadm_id")
    .size()
    .reset_index(name="drg_count")
)
data = data.merge(drg_feat, on="hadm_id", how="left")

micro_feat = (
    micro.groupby("hadm_id")
    .size()
    .reset_index(name="micro_count")
)
data = data.merge(micro_feat, on="hadm_id", how="left")

pharm_feat = (
    pharmacy.groupby("hadm_id")
    .size()
    .reset_index(name="pharmacy_count")
)
data = data.merge(pharm_feat, on="hadm_id", how="left")

transfer_feat = (
    transfers.groupby("hadm_id")
    .size()
    .reset_index(name="transfer_count")
)
data = data.merge(transfer_feat, on="hadm_id", how="left")

input_feat = (
    inputevents.groupby("hadm_id")
    .size()
    .reset_index(name="inputevent_count")
)
data = data.merge(input_feat, on="hadm_id", how="left")

output_feat = (
    outputevents.groupby("hadm_id")
    .size()
    .reset_index(name="outputevent_count")
)
data = data.merge(output_feat, on="hadm_id", how="left")

proc_feat = (
    procedureevents.groupby("hadm_id")
    .size()
    .reset_index(name="procedureevent_count")
)
data = data.merge(proc_feat, on="hadm_id", how="left")

count_cols = [
    "icu_los_days",
    "drg_count",
    "micro_count",
    "pharmacy_count",
    "transfer_count",
    "inputevent_count",
    "outputevent_count",
    "procedureevent_count"
]
for col in count_cols:
    if col in data.columns:
        data[col] = data[col].fillna(0)


print("Removing leakage-prone columns...")

drop_cols = [
    "subject_id",
    "hadm_id",
    "admittime",
    "dischtime",
    "deathtime",
    "edregtime",
    "edouttime",
    "hospital_expire_flag"
]

data = data.drop(columns=[c for c in drop_cols if c in data.columns], errors="ignore")

target_col = "mortality"

safe_feature_cols = [
    "admission_type",
    "admit_provider_id",
    "admission_location",
    "insurance",
    "language",
    "marital_status",
    "race",
    "gender",
    "anchor_age",
]

X = data[safe_feature_cols].copy()
y = data[target_col]

print(f"Final dataset shape (safe features only): {X.shape}")
print("Feature columns:", X.columns.tolist())

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

risk_model = LogisticRegression(max_iter=2000, C=1.0)
risk_model.fit(X_train_proc, y_train)

probs_test = risk_model.predict_proba(X_test_proc)[:, 1]
preds_test = (probs_test >= 0.5).astype(int)

print("\n=== BASELINE RISK MODEL PERFORMANCE ===")
print(f"AUROC: {roc_auc_score(y_test, probs_test):.3f}")
print(f"Brier score: {brier_score_loss(y_test, probs_test):.3f}")
print("\nClassification report:")
print(classification_report(y_test, preds_test))

print("Running representation-based OOD check...")

X_train_dense = X_train_proc.toarray() if hasattr(X_train_proc, "toarray") else X_train_proc
X_test_dense = X_test_proc.toarray() if hasattr(X_test_proc, "toarray") else X_test_proc

n_components = min(10, X_train_dense.shape[1] - 1) if X_train_dense.shape[1] > 1 else 1
pca = PCA(n_components=n_components)

X_train_repr = pca.fit_transform(X_train_dense)
X_test_repr = pca.transform(X_test_dense)

train_center = X_train_repr.mean(axis=0)
train_std = X_train_repr.std(axis=0) + 1e-8

def representation_distance(x):
    z = (x - train_center) / train_std
    return np.sqrt(np.sum(z ** 2))

train_distances = np.array([representation_distance(x) for x in X_train_repr])
ood_threshold = np.percentile(train_distances, 95)

test_distances = np.array([representation_distance(x) for x in X_test_repr])
ood_flags = test_distances > ood_threshold

def proposer_agent(prob):
    if prob >= 0.7:
        return "High mortality risk"
    elif prob <= 0.3:
        return "Low mortality risk"
    else:
        return "Intermediate mortality risk"

def uncertainty_score(prob):
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    return entropy([p, 1 - p], base=2)

def skeptic_agent(prob, ood_flag):
    ent = uncertainty_score(prob)
    near_boundary = 0.4 <= prob <= 0.6
    high_uncertainty = ent > 0.90

    reasons = []
    if near_boundary:
        reasons.append("probability near decision boundary")
    if high_uncertainty:
        reasons.append("high predictive uncertainty")
    if ood_flag:
        reasons.append("case appears out-of-distribution")

    flag = len(reasons) > 0
    return flag, ent, reasons

def safety_harness(prob, skeptic_flag):
    if skeptic_flag:
        return "HANDOFF_TO_CLINICIAN"
    if prob >= 0.85:
        return "AI_HIGH_RISK_ALERT"
    if prob <= 0.15:
        return "AI_LOW_RISK_OK"
    return "HANDOFF_TO_CLINICIAN"

results = pd.DataFrame({
    "y_true": y_test.values,
    "p_mortality": probs_test,
    "y_pred": preds_test,
    "ood_score": test_distances,
    "ood_flag": ood_flags
})

results["proposer_decision"] = results["p_mortality"].apply(proposer_agent)

skeptic_outputs = results.apply(
    lambda row: skeptic_agent(row["p_mortality"], row["ood_flag"]),
    axis=1
)

results["skeptic_flag"] = [x[0] for x in skeptic_outputs]
results["uncertainty_entropy"] = [x[1] for x in skeptic_outputs]
results["skeptic_reasons"] = [x[2] for x in skeptic_outputs]

results["system_action"] = results.apply(
    lambda row: safety_harness(row["p_mortality"], row["skeptic_flag"]),
    axis=1
)

print("\n=== FAILURE-AWARE RESULTS ===")
print(results[[
    "p_mortality",
    "proposer_decision",
    "ood_flag",
    "uncertainty_entropy",
    "skeptic_flag",
    "system_action"
]].head(10))


handoff_rate = (results["system_action"] == "HANDOFF_TO_CLINICIAN").mean()
ai_autonomy_rate = 1 - handoff_rate

high_risk_cases = results["y_true"] == 1
high_risk_handoff_rate = (
    results.loc[high_risk_cases, "system_action"] == "HANDOFF_TO_CLINICIAN"
).mean() if high_risk_cases.sum() > 0 else np.nan

false_negative_mask = (results["y_true"] == 1) & (results["y_pred"] == 0)
fn_handoff_rate = (
    results.loc[false_negative_mask, "system_action"] == "HANDOFF_TO_CLINICIAN"
).mean() if false_negative_mask.sum() > 0 else np.nan

print("\nSafety metrics:")
print(f"Overall handoff rate: {handoff_rate * 100:.1f}%")
print(f"AI autonomous decisions: {ai_autonomy_rate * 100:.1f}%")
print(f"High-risk cases handed off: {high_risk_handoff_rate * 100:.1f}%")
print(f"False negatives safely handed off: {fn_handoff_rate * 100:.1f}%")

output_path = data_dir / "Final Results.csv"
results.to_csv(output_path, index=False)
print(f"\nSaved results to {output_path}")