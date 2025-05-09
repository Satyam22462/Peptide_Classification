# 🧬 Peptide Classification Using ML (Kaggle Rank #1 🥇)

This repository contains our winning solution to the **Peptide Classification** competition hosted on Kaggle. Our team, **Group 48**, achieved **Rank 1** on the final leaderboard with an AUC of **0.9206** 🎉.

 ![image](https://github.com/user-attachments/assets/6fadc633-f8ec-4df2-a2b5-8b86d79a5a5b)

## 📝 Problem Statement

Given peptide sequences, the task was to classify each as **positive (+1)** or **negative (-1)**. We were provided:

- `train.csv`: Peptide sequences with labels
- `test.csv`: Peptide sequences without labels
- Evaluation Metric: **ROC AUC**

## 🧠 Approach Overview

Our solution integrates **traditional bioinformatics** with **modern deep learning** and ensemble modeling:

### 🔬 Feature Extraction

1. **PSI-BLAST PSSM Features**  
   - Generated using `run_psiblast` and `extract_pssm_features`  
   - Represent evolutionary information using Position Specific Scoring Matrices (PSSMs)

2. **ESM-2 Embeddings**  
   - Extracted using Facebook's `esm2_t6_8M_UR50D` model  
   - Captures contextual sequence information via transformer embeddings

3. **Final Features**  
   - Concatenation of PSSM vectors and ESM-2 embeddings  
   - Standardized using `StandardScaler`

### 🤖 Models Used

We experimented with multiple base learners:

- Random Forest, XGBoost, CatBoost, LightGBM, SVM, AdaBoost, Gradient Boosting, MLP

### 🧬 Ensembling

- **Stacking Ensemble (Best)**: LightGBM meta-learner with base models as inputs  
- **Voting Ensemble**: Weighted soft voting of base models

### 🔧 Hyperparameter Tuning

- Used `RandomizedSearchCV` for XGBoost  
- 3-fold Stratified Cross-Validation

## 📊 Results

| Model             | AUC Score |
|------------------|-----------|
| Random Forest     | 0.7988    |
| SVM               | 0.8243    |
| CatBoost          | 0.8467    |
| AdaBoost          | 0.8483    |
| XGBoost           | 0.8578    |
| Voting Ensemble   | 0.8341    |
| **Stacking Ensemble** | **0.8949** |

📌 **Final Stacking AUC on full training set: 0.901**  
📌 **Kaggle Final Score: 0.9206** → 🥇 **Rank 1**

## 💡 Strengths

- Combined **bioinformatics + deep learning**
- Robust preprocessing and error handling
- Model interpretability via ROC curves, feature importance
- Extensive cross-validation and tuning


## 🛠️ How to Run

```bash
python classification_peptide.py train.csv test.csv output_predictions.csv
