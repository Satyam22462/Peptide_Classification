import subprocess
import pandas as pd
import numpy as np
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import HashingVectorizer
import argparse

# ----- Feature Extraction Functions -----
def run_psiblast(sequence, db='nr', out_file='pssm.txt'):
    """Run PSI-BLAST to generate PSSM features"""
    # Write the query sequence to a FASTA file
    with open('query.fasta', 'w') as f:
        f.write(f">seq\n{sequence}\n")
    # Execute the PSI-BLAST command with the specified parameters
    subprocess.run(
        ["psiblast", "-query", "query.fasta", "-db", db,
         "-num_iterations", "3", "-out_ascii_pssm", out_file],
        check=True
    )
    # Load the PSSM scores from the output file (columns 22-41, skip header rows)
    pssm = pd.read_csv(out_file, skiprows=3, delim_whitespace=True, usecols=range(22, 42))
    # Return the average scores across the sequence positions (a vector of length 20)
    return pssm.mean().values

def extract_pssm_features(df):
    """Extract PSSM features for all sequences in a DataFrame"""
    pssm_features = []
    for seq in df['Sequence']:
        try:
            pssm_features.append(run_psiblast(seq))
        except Exception as e:
            # If error occurs, append a zero-vector (length 20)
            pssm_features.append(np.zeros(20))
    return pd.DataFrame(pssm_features)

def extract_esm2_embeddings(df):
    """Extract embeddings from the ESM-2 model"""
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()  # evaluation mode
    
    embeddings = []
    for seq in df['Sequence']:
        inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # Average over the sequence length (dim=1) to get a fixed-size embedding
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(emb)
    
    return pd.DataFrame(embeddings)

def extract_hv_features(df):
    """Extract sequence features using HashingVectorizer with character n-grams."""
    vectorizer = HashingVectorizer(n_features=100, analyzer='char', ngram_range=(2, 3))
    features = vectorizer.transform(df['Sequence']).toarray()
    return pd.DataFrame(features)

def load_data_and_run_model(train_path, test_path, output_path):
    # ----- Load Data and Extract Features -----
    print("Loading data...")
    train_df = pd.read_csv(train_path, comment='#', names=['Sequence', 'Label'])
    test_df = pd.read_csv(test_path, comment='#', names=['ID', 'Sequence'])

    print("Extracting PSSM features...")
    X_train_pssm = extract_pssm_features(train_df)
    X_test_pssm = extract_pssm_features(test_df)

    print("Extracting ESM-2 embeddings...")
    X_train_esm = extract_esm2_embeddings(train_df)
    X_test_esm = extract_esm2_embeddings(test_df)

    print("Extracting HashingVectorizer features...")
    X_train_hv = extract_hv_features(train_df)
    X_test_hv = extract_hv_features(test_df)

    # Combine both feature sets
    X_train = pd.concat([X_train_pssm, X_train_esm, X_train_hv], axis=1)
    X_test = pd.concat([X_test_pssm, X_test_esm, X_test_hv], axis=1)

    # Map training labels from {-1, 1} to {0, 1} and remove NaNs
    y = train_df['Label'].map({-1: 0, 1: 1})
    mask = ~y.isna()
    X_train = X_train[mask]
    y = y[mask]

    # Stratified train-validation split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ----- Define Base Estimators and Ensembles -----
    rf = RandomForestClassifier(n_estimators=500, max_depth=None, max_features='sqrt', 
                                class_weight='balanced', random_state=42)
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, 
                        colsample_bytree=0.8, subsample=0.8, scale_pos_weight=1.5, 
                        min_child_weight=3, random_state=42)
    svm = SVC(probability=True, kernel='rbf', C=10, gamma='scale', class_weight='balanced')
    catboost = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, 
                                  verbose=0, random_state=42)
    lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, 
                          class_weight='balanced', random_state=42)
    gbm = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, 
                                     random_state=42)
    ada = AdaBoostClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)

    stacking_clf = StackingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('svm', svm),
                    ('cat', catboost), ('gbm', gbm), ('ada', ada), ('mlp', mlp)],
        final_estimator=lgbm
    )

    voting_clf = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('svm', svm),
                    ('cat', catboost), ('gbm', gbm), ('ada', ada), ('mlp', mlp)],
        voting='soft', weights=[1, 2, 1, 2, 1, 1, 1]
    )

    models = {
        "Random Forest": rf,
        "XGBoost": xgb,
        "SVM": svm,
        "CatBoost": catboost,
        "GBM": gbm,
        "AdaBoost": ada,
        "MLP": mlp,
        "Stacking": stacking_clf,
        "Voting": voting_clf
    }

    # ----- Hyperparameter Tuning for XGBoost -----
    grid_params = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 6, 8],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'subsample': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5]
    }

    print("Tuning XGBoost hyperparameters...")
    random_search = RandomizedSearchCV(
        xgb, grid_params, 
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='roc_auc', n_iter=15, random_state=42
    )
    random_search.fit(X_train_scaled, y_train_split)
    print("Best XGBoost Params:", random_search.best_params_)
    xgb_best = random_search.best_estimator_
    models["XGBoost"] = xgb_best

    # ----- Evaluate Models on the Validation Set -----
    print("\nEvaluating models...")
    best_auc = 0
    best_model_name = None
    best_model = None

    for name, model in models.items():
        model.fit(X_train_scaled, y_train_split)
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        print(f"{name}: AUC = {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model = model

    print(f"\nBest Model: {best_model_name} with AUC = {best_auc:.4f}")

    # Save the best model as a pickle file
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("Best model saved as 'best_model.pkl'.")

    # ----- Generate Submission using Stacking Ensemble -----
    print("\nTraining stacking ensemble for submission predictions...")
    stacking_clf.fit(X_train_scaled, y_train_split)
    y_val_pred_proba = stacking_clf.predict_proba(X_val_scaled)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    print(f"Stacking Ensemble Validation AUC: {val_auc:.4f}")

    y_test_pred_proba = stacking_clf.predict_proba(X_test_scaled)[:, 1]
    submission = pd.DataFrame({'ID': test_df['ID'], 'Label': y_test_pred_proba})
    submission.to_csv(output_path, index=False)
    print(f"Submission file saved as '{output_path}'.")

def main():
    parser = argparse.ArgumentParser(description="Peptide Classification using Stacking Model")
    parser.add_argument("train", type=str, help="Path to training dataset (CSV)")
    parser.add_argument("test", type=str, help="Path to test dataset (CSV)")
    parser.add_argument("output", type=str, help="Path to save predictions (CSV)")
    
    args = parser.parse_args()

    # Load train data and run model
    load_data_and_run_model(args.train, args.test, args.output)

if __name__ == "__main__":
    main()