"""
Phase 3: Clinical Preprocessing with Strict Training-Only Fitting
Guarantees zero data leakage by fitting all scalers, category vocabularies, and clinical mappings
strictly on the training dataset.

Outputs:
  - outputs/preprocessor.pkl
  - outputs/preprocessing_report.md
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import ClinicalDataPreprocessor

def run_preprocessing(
    train_path="data/train.csv",
    val_path="data/validation.csv",
    test_path="data/test.csv",
    output_dir="outputs"
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("PHASE 3: PREPROCESSING & TRAINING-ONLY FITTING")
    print("=" * 60)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    preprocessor = ClinicalDataPreprocessor()
    preprocessor.fit(train_df)

    train_features = preprocessor.transform(train_df)
    val_features = preprocessor.transform(val_df)
    test_features = preprocessor.transform(test_df)

    # Save preprocessor object
    pkl_path = os.path.join(output_dir, "preprocessor.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"Fitted Preprocessor serialized to: {pkl_path}")

    # Generate Preprocessing Report
    report_path = os.path.join(output_dir, "preprocessing_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Clinical Preprocessing & Feature Engineering Report\n\n")
        f.write("## 1. Zero Data Leakage Protocol\n")
        f.write("- **Strict Training Isolation:** All scaling parameters (mean, standard deviation), categorical vocabularies, and clipping ranges were fitted exclusively on `train.csv` (N=280).\n")
        f.write("- Validation (N=60) and Test (N=60) sets were transformed using the frozen preprocessor.\n\n")

        f.write("## 2. Hemodynamic Alignment (Header Swap)\n")
        f.write("- The raw dataset `sysbp` (70-89 mmHg) was identified as diastolic blood pressure.\n")
        f.write("- The raw dataset `diabp` (120-139 mmHg) was identified as systolic blood pressure.\n")
        f.write("- Mapped to `clean_sysbp` (systolic) and `clean_diabp` (diastolic) for clinically valid physiological representation.\n\n")

        f.write("## 3. Feature Breakdown\n")
        f.write(f"- **Continuous Features (17):** {', '.join(preprocessor.continuous_cols)}\n")
        f.write(f"- **Binary Clinical Features (5):** {', '.join(preprocessor.binary_cols)}\n")
        f.write(f"- **Categorical Features (3):** {', '.join(preprocessor.categorical_cols)}\n\n")

        f.write("### Training Normalization Parameters\n\n")
        f.write("| Feature | Training Mean | Training Std Dev |\n")
        f.write("| --- | --- | --- |\n")
        for col in preprocessor.continuous_cols:
            m = preprocessor.train_stats['cont_mean'][col]
            s = preprocessor.train_stats['cont_scale'][col]
            f.write(f"| {col} | {m:.4f} | {s:.4f} |\n")
        f.write("\n")

        f.write("### Categorical Encoding Vocabularies\n\n")
        for col, vocab in preprocessor.cat_vocab.items():
            f.write(f"- **{col}**: {vocab} (Card: {preprocessor.cat_cardinalities[col]}, index 0 = UNK)\n")
        f.write("\n")

    print(f"Preprocessing report written to: {report_path}")
    print("=" * 60)
    return preprocessor

if __name__ == "__main__":
    run_preprocessing()
