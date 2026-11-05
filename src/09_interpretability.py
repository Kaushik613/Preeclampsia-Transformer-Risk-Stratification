"""
Phase 9: Feature Contribution & Interpretability Analysis
Evaluates feature importance through:
  1. Permutation Sensitivity Analysis (mean risk score shift)
  2. Feature Masking Impact (risk delta when feature is masked to mean/baseline)
  3. Patient Embedding Drift (mean Euclidean distance in 32-D latent space)
Saves ranked feature importances to: outputs/tables/feature_contributions.csv
"""

import os
import sys
import torch
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
risk_mod = importlib.import_module("src.07_risk_model")
PreeclampsiaRiskModel = risk_mod.PreeclampsiaRiskModel

def run_interpretability_analysis(
    test_path: str = "data/test.csv",
    output_dir: str = "outputs"
):
    print("=" * 60)
    print("PHASE 9: CLINICAL INTERPRETABILITY & SENSITIVITY ANALYSIS")
    print("=" * 60)

    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    risk_model = PreeclampsiaRiskModel()
    preprocessor = risk_model.preprocessor
    df_test = pd.read_csv(test_path)
    n_samples = len(df_test)

    feats = preprocessor.transform(df_test)
    cont_orig = torch.tensor(feats["continuous"], dtype=torch.float32)
    bin_orig = torch.tensor(feats["binary"], dtype=torch.long)
    cat_orig = torch.tensor(feats["categorical"], dtype=torch.long)

    # Baseline predictions
    with torch.no_grad():
        base_out = risk_model.model.forward_reconstruction(cont_orig, bin_orig, cat_orig)
        base_scores = base_out["predicted_risk"].cpu().numpy()
        base_embeds = base_out["patient_embedding"].cpu().numpy()

    all_features = preprocessor.continuous_cols + preprocessor.binary_cols + preprocessor.categorical_cols
    n_cont = len(preprocessor.continuous_cols)
    n_bin = len(preprocessor.binary_cols)
    n_cat = len(preprocessor.categorical_cols)

    importance_records = []

    # Map friendly clinical names
    name_map = {
        "plgf:sflt": "Angiogenic Ratio (plgf:sflt)",
        "SEng": "Soluble Endoglin (SEng)",
        "clean_sysbp": "Systolic Blood Pressure (mmHg)",
        "clean_diabp": "Diastolic Blood Pressure (mmHg)",
        "htn": "Chronic Hypertension (htn)",
        "diabetes": "Pre-existing Diabetes",
        "fam_htn": "Family History of Preeclampsia",
        "sp_art": "Uterine/Spiral Artery Resistance",
        "creatinine": "Serum Creatinine",
        "cysC": "Cystatin C",
        "platelet": "Platelet Count",
        "bmi": "Maternal BMI",
        "age": "Maternal Age",
        "pp_13": "Placental Protein 13",
        "glycerides": "Serum Triglycerides",
        "sleep": "Sleep Duration / Quality",
        "activity": "Physical Activity Level",
        "diet": "Dietary Pattern",
        "hb": "Hemoglobin Concentration",
        "pcv": "Hematocrit (PCV)",
        "tsh": "Thyroid Stimulating Hormone",
        "gest_age": "Gestational Age (weeks)",
        "occupation": "Occupational Category",
        "height": "Maternal Height",
        "weight": "Maternal Weight"
    }

    category_map = {
        "plgf:sflt": "Angiogenic",
        "SEng": "Angiogenic",
        "pp_13": "Placental",
        "clean_sysbp": "Cardiovascular",
        "clean_diabp": "Cardiovascular",
        "htn": "Cardiovascular / Maternal",
        "diabetes": "Maternal Clinical",
        "fam_htn": "Maternal Clinical",
        "sp_art": "Cardiovascular / Doppler",
        "creatinine": "Biochemical / Renal",
        "cysC": "Biochemical / Renal",
        "platelet": "Hematological",
        "hb": "Hematological",
        "pcv": "Hematological",
        "tsh": "Endocrine",
        "glycerides": "Metabolic",
        "bmi": "Anthropometric",
        "age": "Demographic",
        "height": "Anthropometric",
        "weight": "Anthropometric",
        "gest_age": "Obstetric",
        "sleep": "Lifestyle",
        "activity": "Lifestyle",
        "diet": "Lifestyle",
        "occupation": "Lifestyle / Social"
    }

    with torch.no_grad():
        # 1. Continuous Features
        for i, col in enumerate(preprocessor.continuous_cols):
            # Mask perturbation
            cont_m = cont_orig.clone()
            cont_m[:, i] = 0.0  # mean in z-scale
            out_m = risk_model.model.forward_reconstruction(cont_m, bin_orig, cat_orig)
            score_m = out_m["predicted_risk"].cpu().numpy()
            embed_m = out_m["patient_embedding"].cpu().numpy()

            score_delta = np.mean(np.abs(base_scores - score_m))
            embed_drift = np.mean(np.linalg.norm(base_embeds - embed_m, axis=1))

            # Permutation perturbation
            perm_idx = torch.randperm(n_samples)
            cont_p = cont_orig.clone()
            cont_p[:, i] = cont_orig[perm_idx, i]
            out_p = risk_model.model.forward_reconstruction(cont_p, bin_orig, cat_orig)
            score_p = out_p["predicted_risk"].cpu().numpy()
            perm_delta = np.mean(np.abs(base_scores - score_p))

            importance_records.append({
                "feature": col,
                "clinical_name": name_map.get(col, col),
                "clinical_domain": category_map.get(col, "General"),
                "masking_score_shift": round(float(score_delta), 5),
                "permutation_score_shift": round(float(perm_delta), 5),
                "latent_embedding_drift": round(float(embed_drift), 5)
            })

        # 2. Binary Features
        for j, col in enumerate(preprocessor.binary_cols):
            bin_m = bin_orig.clone()
            bin_m[:, j] = 0
            out_m = risk_model.model.forward_reconstruction(cont_orig, bin_m, cat_orig)
            score_m = out_m["predicted_risk"].cpu().numpy()
            embed_m = out_m["patient_embedding"].cpu().numpy()

            score_delta = np.mean(np.abs(base_scores - score_m))
            embed_drift = np.mean(np.linalg.norm(base_embeds - embed_m, axis=1))

            perm_idx = torch.randperm(n_samples)
            bin_p = bin_orig.clone()
            bin_p[:, j] = bin_orig[perm_idx, j]
            out_p = risk_model.model.forward_reconstruction(cont_orig, bin_p, cat_orig)
            score_p = out_p["predicted_risk"].cpu().numpy()
            perm_delta = np.mean(np.abs(base_scores - score_p))

            importance_records.append({
                "feature": col,
                "clinical_name": name_map.get(col, col),
                "clinical_domain": category_map.get(col, "General"),
                "masking_score_shift": round(float(score_delta), 5),
                "permutation_score_shift": round(float(perm_delta), 5),
                "latent_embedding_drift": round(float(embed_drift), 5)
            })

        # 3. Categorical Features
        for k, col in enumerate(preprocessor.categorical_cols):
            cat_m = cat_orig.clone()
            cat_m[:, k] = 0  # UNK token
            out_m = risk_model.model.forward_reconstruction(cont_orig, bin_orig, cat_m)
            score_m = out_m["predicted_risk"].cpu().numpy()
            embed_m = out_m["patient_embedding"].cpu().numpy()

            score_delta = np.mean(np.abs(base_scores - score_m))
            embed_drift = np.mean(np.linalg.norm(base_embeds - embed_m, axis=1))

            perm_idx = torch.randperm(n_samples)
            cat_p = cat_orig.clone()
            cat_p[:, k] = cat_orig[perm_idx, k]
            out_p = risk_model.model.forward_reconstruction(cont_orig, bin_orig, cat_p)
            score_p = out_p["predicted_risk"].cpu().numpy()
            perm_delta = np.mean(np.abs(base_scores - score_p))

            importance_records.append({
                "feature": col,
                "clinical_name": name_map.get(col, col),
                "clinical_domain": category_map.get(col, "General"),
                "masking_score_shift": round(float(score_delta), 5),
                "permutation_score_shift": round(float(perm_delta), 5),
                "latent_embedding_drift": round(float(embed_drift), 5)
            })

    importance_df = pd.DataFrame(importance_records)
    # Composite importance score
    importance_df["composite_importance"] = (
        0.5 * importance_df["masking_score_shift"] +
        0.3 * importance_df["permutation_score_shift"] +
        0.2 * (importance_df["latent_embedding_drift"] / importance_df["latent_embedding_drift"].max() * importance_df["masking_score_shift"].max())
    )
    importance_df = importance_df.sort_values(by="composite_importance", ascending=False).reset_index(drop=True)
    importance_df["importance_rank"] = range(1, len(importance_df) + 1)

    csv_path = os.path.join(tables_dir, "feature_contributions.csv")
    importance_df.to_csv(csv_path, index=False)
    print(f"Feature contributions saved to: {csv_path}")

    print("\nTop 5 Influential Risk Factors:")
    for _, row in importance_df.head(5).iterrows():
        print(f"  {row['importance_rank']}. {row['clinical_name']} ({row['clinical_domain']}) - Mask Shift: {row['masking_score_shift']:.4f}")

    print("=" * 60)
    return importance_df

if __name__ == "__main__":
    run_interpretability_analysis()
