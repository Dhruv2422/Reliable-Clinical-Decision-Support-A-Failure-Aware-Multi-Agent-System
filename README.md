# Failure-Aware Clinical Decision Support on BRSET and MIMIC-IV

## Overview

This repository contains two related projects demonstrating **engineering-focused AI/ML systems** in medical domains:

1. **BRSET Retinal Disease Classifier**  
   A PyTorch-based multilabel image classification pipeline that predicts five retinal diseases from fundus photographs.

2. **MIMIC-IV Failure-Aware Mortality Risk System**  
   An admission-time mortality prediction model on MIMIC-IV, wrapped in a **Proposer–Skeptic–Safety Harness** architecture that prioritizes safety and uncertainty awareness over blind confidence.

Both projects emphasize **clean data pipelines, modular code, reproducible training, and explicit handling of failure modes**, rather than chasing state-of-the-art metrics.

---

## 1. BRSET Retinal Disease Classifier

### Task

Given a retinal **fundus image**, predict the presence or absence of:

- diabetic_retinopathy  
- macular_edema  
- amd  
- hypertensive_retinopathy  
- hemorrhage  

The model is evaluated using **per-label and average AUROC** on a held-out test set.

### Data

- Dataset: **BRSET – A Brazilian Multilabel Ophthalmological Dataset of Retina Images**.[web:114][web:115]
- Inputs:
  - `fundus_photos/` – JPEG fundus images.
  - `labels_brset.csv` – CSV with one row per image and five binary labels.

The code assumes the following layout (paths can be configured):

```text
brset_project/
  main.py
  Business - Documents/
    a-brazilian-multilabel-ophthalmological-dataset-brset-1.0.1/
      fundus_photos/
      labels_brset.csv
```

### Method

- **Model:** ResNet-50 backbone (`torchvision.models.resnet50`) with the final fully-connected layer replaced by a 5-output head and sigmoid activation at evaluation time.
- **Loss:** Binary cross-entropy on probabilities (multi-label BCE).
- **Optimization:** Adam optimizer, mini-batch gradient descent on CPU.
- **Data pipeline:**
  - Custom `BRSETDataset` that reads images and labels from disk.
  - Train/validation/test split stratified by the number of positive labels per example.
  - Training transforms: resize, random resized crop, random horizontal flip, normalization.
  - Evaluation transforms: resize, center crop, normalization.

### Results

On the held-out test set, the model achieves approximately:

- **Average AUROC:** ~0.76  
- **Per-label AUROC:**
  - Diabetic retinopathy: ~0.83  
  - Macular edema: ~0.88  
  - AMD: ~0.87  
  - Hypertensive retinopathy: ~0.53  
  - Hemorrhage: ~0.70  

These results show strong performance on common diseases and weaker performance on rare labels, reflecting class imbalance and limited signal for hypertensive retinopathy and hemorrhage.

---

## 2. MIMIC-IV Failure-Aware Mortality Risk System

### Task

Predict **in-hospital mortality at admission time** using MIMIC-IV, and build a **failure-aware clinical decision support system** that:

- Produces a risk estimate (Proposer),
- Evaluates uncertainty and out-of-distribution (OOD) status (Skeptic),
- Decides whether to accept the AI decision or hand off to a clinician (Safety Harness).

### Data

- Dataset: **MIMIC-IV Clinical Database** (core `hosp` and `icu` tables)

  - `hosp/admissions.csv`
  - `hosp/patients.csv`
  - `hosp/drgcodes.csv`
  - `hosp/microbiologyevents.csv`
  - `hosp/pharmacy.csv`
  - `hosp/transfers.csv`
  - `icu/icustays.csv`
  - `icu/inputevents.csv`
  - `icu/outputevents.csv`
  - `icu/procedureevents.csv`

**Label:**  
`mortality = hospital_expire_flag` from `admissions.csv` (1 = died in hospital, 0 = survived).

### Feature Engineering

The system builds an **admission-level cohort** with:

- Demographics and metadata:
  - admission_type, admit_provider_id, admission_location
  - insurance, language, marital_status, race
  - gender, anchor_age
- ICU admission flag (whether the admission had an ICU stay).

To avoid **label leakage**, all post-outcome or post-discharge fields (e.g., `hospital_expire_flag`, `deathtime`, `dischtime`) and whole-stay utilization summaries are removed from the final feature set.

Final “safe” feature set used for the main model:

```text
['admission_type', 'admit_provider_id', 'admission_location',
 'insurance', 'language', 'marital_status', 'race',
 'gender', 'anchor_age']
```

### Model and Training

- **Model:** Logistic Regression.
- **Preprocessing:**
  - Numeric features: median imputation + standardization.
  - Categorical features: most-frequent imputation + one-hot encoding.
- **Training:** Stratified train/test split (70/30) over admissions.
- **Evaluation:**
  - AUROC
  - Brier score
  - Classification report (precision, recall, F1)

Example performance after leakage correction:

- **AUROC:** 0.871  
- **Brier score:** 0.019  
- Mortality prevalence: ~2.2%

### Representation Learning & OOD Detection

To approximate **representation learning and OOD detection**:

- The processed feature matrix is projected into a low-dimensional space using **PCA**.
- For each case, we compute the **distance to the training mean** in PCA space (normalized by feature-wise std).
- Cases above the 95th percentile of training distances are flagged as **OOD** (potentially out-of-distribution or rare).

### Multi-Agent Failure-Aware Layer

On top of the risk model:

**Proposer Agent**

- Takes mortality probability `p_mortality` and outputs:
  - `High mortality risk` if \(p \geq 0.7\),
  - `Low mortality risk` if \(p \leq 0.3\),
  - `Intermediate mortality risk` otherwise.

**Skeptic Agent**

- Computes **predictive entropy** for the binary prediction.
- Flags a case when:
  - Probability is near the decision boundary (e.g. 0.4–0.6), or
  - Entropy is high (uncertain prediction), or
  - The case is flagged as OOD by the PCA distance check.

Returns a boolean flag and a list of reasons.

**Safety Harness**

- If the Skeptic flags the case → **HANDOFF_TO_CLINICIAN**.
- Otherwise:
  - If `p >= 0.85` → `AI_HIGH_RISK_ALERT`.
  - If `p <= 0.15` → `AI_LOW_RISK_OK`.
  - Else → conservative **HANDOFF_TO_CLINICIAN**.

### Safety Metrics

The system reports:

- Overall handoff rate (fraction of cases routed to clinicians),
- AI autonomous decision rate,
- Fraction of true deaths (positives) that are escalated,
- Fraction of false negatives that are escalated.

In one run:

- Overall handoff rate ≈ 7.1%,
- AI-autonomous decisions ≈ 92.9%,
- ~29.7% of true deaths escalated,
- ~28.5% of false negatives escalated.

This demonstrates that the safety layer catches a meaningful fraction of dangerous errors, aligning with the “failure-aware” design goal.

---

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

You must separately obtain access to the **BRSET dataset** and **MIMIC-IV** and place them in the expected folder structure (see “Data” sections above).

---

## Usage

### BRSET

```bash
cd brset_project
python main.py
```

This will:

- Load `labels_brset.csv` and fundus images,
- Create train/val/test splits,
- Train ResNet-50 for multilabel disease prediction,
- Print validation + test AUROC metrics.

### MIMIC-IV Failure-Aware System

```bash
cd mimic_failure_aware
python mimic.py
```

This will:

- Load the relevant MIMIC-IV CSVs,
- Build an admission-level cohort and safe feature set,
- Train the logistic regression risk model,
- Compute AUROC, Brier score, and classification metrics,
- Apply the Proposer–Skeptic–Safety Harness logic,
- Print safety metrics and save a detailed results CSV.

---

## Ethical Considerations

These systems are **research prototypes**, not clinical tools:

- **MIMIC-IV** and **BRSET** are retrospective datasets and may not generalize to other hospitals, populations, or imaging devices.[web:114][web:178]
- The mortality model uses demographic and admission-time features and is intended only to study **failure-aware behavior**, not for deployment.
- The multi-agent safety layer is designed to highlight uncertainty and OOD cases and to **escalate rather than automate** in ambiguous scenarios.

Any real-world deployment would require extensive validation, calibration, and regulatory review.

---
