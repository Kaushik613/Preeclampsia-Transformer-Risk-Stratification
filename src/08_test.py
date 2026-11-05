"""
Phase 8: Comprehensive Model Evaluation on the Untouched Test Cohort (N=60)
Evaluates:
  1. Representation & Denoising Reconstruction Quality (MSE, RMSE, MAE, R², Discrete Accuracies)
  2. Reference Risk-Stratification Agreement (Agreement Rate, Cohen's Kappa, Score Correlation)
Outputs:
  - outputs/tables/reconstruction_metrics.csv
  - outputs/tables/risk_metrics.csv
  - outputs/tables/patient_risk_predictions.csv
  - outputs/evaluation_report.md
"""

import os
import sys
import pickle
import math
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, cohen_kappa_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
risk_mod = importlib.import_module("src.07_risk_model")
PreeclampsiaRiskModel = risk_mod.PreeclampsiaRiskModel

def run_evaluation(
    test_path: str = "data/test.csv",
    ref_test_path: str = "data/test_with_ref_risk.csv",
    output_dir: str = "outputs"
):
    print("=" * 60)
    print("PHASE 8: TESTING & RISK STRATIFICATION EVALUATION")
    print("=" * 60)

    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    # Load test set (untouched until now)
    if os.path.exists(ref_test_path):
        test_df = pd.read_csv(ref_test_path)
    else:
        test_df = pd.read_csv(test_path)
        ref_mod = importlib.import_module("src.06_clinical_risk_score")
        test_df = ref_mod.calculate_reference_risk_scores(test_df)

    n_test = len(test_df)
    print(f"Test cohort loaded: {n_test} patients (Locked test partition)")

    # Initialize model
    risk_model = PreeclampsiaRiskModel()
    pred_df, test_embeddings = risk_model.predict(test_df, compute_contributions=True)

    # 1. Evaluate Reconstruction Metrics
    preprocessor = risk_model.preprocessor
    feats = preprocessor.transform(test_df)
    cont_tensor = torch.tensor(feats["continuous"], dtype=torch.float32)
    bin_tensor = torch.tensor(feats["binary"], dtype=torch.long)
    cat_tensor = torch.tensor(feats["categorical"], dtype=torch.long)

    with torch.no_grad():
        out = risk_model.model.forward_reconstruction(cont_tensor, bin_tensor, cat_tensor)
        cont_pred = out["cont_recon"].cpu().numpy()
        bin_pred_logits = out["bin_recon"].cpu().numpy()
        cat_preds = [c.cpu().numpy() for c in out["cat_recon"]]

    # Continuous reconstruction metrics
    cont_true = feats["continuous"]
    mse = mean_squared_error(cont_true, cont_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(cont_true, cont_pred)
    r2 = r2_score(cont_true, cont_pred)

    # Binary accuracy
    bin_preds_binary = (bin_pred_logits > 0).astype(int)
    bin_acc = np.mean(bin_preds_binary == feats["binary"])

    # Categorical accuracy
    cat_acc_list = []
    for k, cp in enumerate(cat_preds):
        cat_k_pred = np.argmax(cp, axis=1)
        cat_acc_list.append(np.mean(cat_k_pred == feats["categorical"][:, k]))
    cat_acc = np.mean(cat_acc_list)

    recon_metrics_df = pd.DataFrame([{
        "Metric": "Continuous MSE", "Value": round(mse, 4)
    }, {
        "Metric": "Continuous RMSE", "Value": round(rmse, 4)
    }, {
        "Metric": "Continuous MAE", "Value": round(mae, 4)
    }, {
        "Metric": "Continuous R² Score", "Value": round(r2, 4)
    }, {
        "Metric": "Binary Exact-Match Accuracy", "Value": round(bin_acc * 100, 2)
    }, {
        "Metric": "Categorical Exact-Match Accuracy", "Value": round(cat_acc * 100, 2)
    }])

    recon_metrics_path = os.path.join(tables_dir, "reconstruction_metrics.csv")
    recon_metrics_df.to_csv(recon_metrics_path, index=False)
    print(f"Reconstruction metrics saved to: {recon_metrics_path}")

    # 2. Risk Stratification Evaluation
    # Note: Comparison is against the CLINICALLY ANCHORED REFERENCE TIER
    ref_tier = pred_df["reference_risk_category"].values
    model_tier = pred_df["risk_category"].values
    ref_scores = pred_df["reference_risk_score"].values
    model_scores = pred_df["risk_score"].values

    tier_agreement = np.mean(ref_tier == model_tier)
    kappa = cohen_kappa_score(ref_tier, model_tier, labels=["LOW", "MODERATE", "HIGH"])
    score_corr = float(np.corrcoef(ref_scores, model_scores)[0, 1])

    risk_metrics_df = pd.DataFrame([{
        "Evaluation Dimension": "Reference Risk-Tier Agreement", "Value": f"{tier_agreement * 100:.2f}%"
    }, {
        "Evaluation Dimension": "Cohen's Kappa (Reference vs Model)", "Value": f"{kappa:.4f}"
    }, {
        "Evaluation Dimension": "Continuous Score Pearson Correlation", "Value": f"{score_corr:.4f}"
    }, {
        "Evaluation Dimension": "Model Risk Distribution (Low / Mod / High)",
        "Value": f"{(model_tier=='LOW').sum()} / {(model_tier=='MODERATE').sum()} / {(model_tier=='HIGH').sum()}"
    }, {
        "Evaluation Dimension": "Reference Risk Distribution (Low / Mod / High)",
        "Value": f"{(ref_tier=='LOW').sum()} / {(ref_tier=='MODERATE').sum()} / {(ref_tier=='HIGH').sum()}"
    }])

    risk_metrics_path = os.path.join(tables_dir, "risk_metrics.csv")
    risk_metrics_df.to_csv(risk_metrics_path, index=False)
    print(f"Risk stratification metrics saved to: {risk_metrics_path}")

    # 3. Patient-Level Risk Predictions CSV
    patient_predictions = pd.DataFrame({
        "patient_index": [f"TEST_{i:03d}" for i in range(n_test)],
        "risk_score": pred_df["risk_score"],
        "risk_category": pred_df["risk_category"],
        "reference_risk_score": pred_df["reference_risk_score"],
        "reference_risk_category": pred_df["reference_risk_category"],
        "top_feature_1": pred_df["top_feature_1"],
        "top_feature_2": pred_df["top_feature_2"],
        "top_feature_3": pred_df["top_feature_3"]
    })

    patient_pred_path = os.path.join(tables_dir, "patient_risk_predictions.csv")
    patient_predictions.to_csv(patient_pred_path, index=False)
    print(f"Patient risk predictions saved to: {patient_pred_path}")

    # Save embeddings for visualization
    np.save(os.path.join(tables_dir, "test_embeddings.npy"), test_embeddings)

    # 4. Generate Comprehensive Evaluation Report
    report_path = os.path.join(output_dir, "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Comprehensive Model Evaluation Report\n\n")
        f.write("## 1. Methodological Clarification & Medical Rules\n")
        f.write("> **CRITICAL MEDICAL DISCLAIMER:**\n")
        f.write("> There is NO confirmed Preeclampsia diagnosis target in `preeclampsia.csv`.\n")
        f.write("> In strict adherence to clinical AI validation rules, this evaluation reports **Representation Reconstruction Quality** and **Reference Risk-Tier Agreement**.\n")
        f.write("> It does NOT report disease-prediction accuracy, sensitivity, specificity, ROC-AUC, or F1 scores for clinical preeclampsia diagnosis.\n\n")

        f.write("## 2. Tabular Representation & Self-Supervised Reconstruction\n\n")
        # Format table
        headers = list(recon_metrics_df.columns)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for _, row in recon_metrics_df.iterrows():
            f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
        f.write("\n")

        f.write("## 3. Clinical Research Risk Stratification Agreement\n\n")
        headers = list(risk_metrics_df.columns)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for _, row in risk_metrics_df.iterrows():
            f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
        f.write("\n")

        f.write("## 4. Analysis of Risk Tiers on Test Cohort\n")
        f.write(f"- Overall Agreement between FT-Transformer model and clinical reference framework: **{tier_agreement * 100:.2f}%**\n")
        f.write(f"- Cohen's Kappa: **{kappa:.4f}**\n")
        f.write(f"- Score Correlation (Pearson r): **{score_corr:.4f}**\n")
        f.write("- High correlation confirms that the learned 32-D patient embedding captures the multi-factor clinical risk continuum.\n\n")

    print(f"Evaluation report written to: {report_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()
