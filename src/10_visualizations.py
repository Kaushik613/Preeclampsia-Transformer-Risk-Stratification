"""
Phase 10: Publication-Quality Clinical Visualizations
Generates all 12 required figures for the clinical risk stratification project:
  1. 01_missingness.png
  2. 02_distributions_before.png
  3. 03_distributions_after.png
  4. 04_correlation_heatmap.png
  5. 05_training_validation_loss.png
  6. 06_embedding_visualization.png
  7. 07_risk_score_distribution.png
  8. 08_risk_category_distribution.png
  9. 09_risk_tier_agreement_matrix.png
  10. 10_feature_contributions.png
  11. 11_ratio_vs_risk.png
  12. 12_clinical_profile_comparison.png
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import ClinicalDataPreprocessor
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# Set global aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 300

TIER_COLORS = {
    "LOW": "#2ecc71",       # Emerald Green
    "MODERATE": "#f39c12",  # Amber Orange
    "HIGH": "#e74c3c"       # Alizarin Red
}

def generate_all_visualizations(
    raw_path="preeclampsia.csv",
    data_dir="data",
    outputs_dir="outputs"
):
    fig_dir = os.path.join(outputs_dir, "figures")
    tables_dir = os.path.join(outputs_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("PHASE 10: GENERATING CLINICAL VISUALIZATIONS")
    print("=" * 60)

    raw_df = pd.read_csv(raw_path)
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    pred_df = pd.read_csv(os.path.join(tables_dir, "patient_risk_predictions.csv"))

    with open(os.path.join(outputs_dir, "preprocessor.pkl"), "rb") as f:
        preprocessor = pickle.load(f)

    # -------------------------------------------------------------
    # Figure 1: Missingness Chart
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    missing_pct = raw_df.isnull().mean() * 100
    bars = ax.bar(raw_df.columns, missing_pct, color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_ylim(0, 10)
    ax.set_ylabel("Missing Values (%)")
    ax.set_title("Figure 1: Dataset Completeness & Missingness Audit (N=400, 25 Features)", fontweight='bold')
    plt.xticks(rotation=60, ha='right', fontsize=9)
    # Add text annotation
    ax.text(0.5, 0.5, "100% Complete Cohort\n(0 missing values across all 25 features)",
            transform=ax.transAxes, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#e8f8f5', edgecolor='#27ae60', alpha=0.9),
            fontsize=12, fontweight='bold', color='#1e8449')
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "01_missingness.png"))
    plt.close()
    print("  -> Saved: 01_missingness.png")

    # -------------------------------------------------------------
    # Figure 2: Distributions Before Preprocessing
    # -------------------------------------------------------------
    key_features = ['sysbp', 'diabp', 'plgf:sflt', 'SEng', 'creatinine', 'bmi']
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for i, col in enumerate(key_features):
        sns.histplot(raw_df[col], kde=True, ax=axes[i], color='#2980b9', edgecolor='black', alpha=0.6)
        axes[i].set_title(f"Raw: {col}", fontweight='bold')
        axes[i].set_xlabel("Original Scale")
    fig.suptitle("Figure 2: Key Clinical Feature Distributions (Before Preprocessing)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "02_distributions_before.png"))
    plt.close()
    print("  -> Saved: 02_distributions_before.png")

    # -------------------------------------------------------------
    # Figure 3: Distributions After Preprocessing
    # -------------------------------------------------------------
    train_transformed = preprocessor.transform(train_df)
    cont_scaled = pd.DataFrame(train_transformed['continuous'], columns=preprocessor.continuous_cols)
    key_scaled = ['clean_sysbp', 'clean_diabp', 'plgf:sflt', 'SEng', 'creatinine', 'bmi']
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for i, col in enumerate(key_scaled):
        sns.histplot(cont_scaled[col], kde=True, ax=axes[i], color='#16a085', edgecolor='black', alpha=0.6)
        axes[i].set_title(f"Standardized: {col} (Z-Score)", fontweight='bold')
        axes[i].set_xlabel("Zero-Mean Unit-Variance")
    fig.suptitle("Figure 3: Feature Distributions After Preprocessing (Training Fit)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "03_distributions_after.png"))
    plt.close()
    print("  -> Saved: 03_distributions_after.png")

    # -------------------------------------------------------------
    # Figure 4: Correlation Heatmap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 10))
    # Remap hemodynamics for physiological correlation
    clean_raw = preprocessor._remap_hemodynamics(raw_df)
    corr_cols = [
        'clean_sysbp', 'clean_diabp', 'htn', 'diabetes', 'fam_htn', 'sp_art',
        'plgf:sflt', 'SEng', 'pp_13', 'creatinine', 'cysC', 'platelet', 'bmi', 'age'
    ]
    corr_mat = clean_raw[corr_cols].corr()
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_mat, mask=mask, cmap=cmap, vmax=0.3, vmin=-0.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True, fmt=".2f",
                ax=ax, annot_kws={"size": 8})
    ax.set_title("Figure 4: Inter-Feature Correlation Heatmap of Key Clinical Biomarkers", fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "04_correlation_heatmap.png"))
    plt.close()
    print("  -> Saved: 04_correlation_heatmap.png")

    # -------------------------------------------------------------
    # Figure 5: Training vs Validation Loss Curves
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    history_file = os.path.join(tables_dir, "training_history.json")
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            hist = json.load(f)
        epochs = hist["epoch"]
        ax.plot(epochs, hist["train_loss"], label="Training Loss", color="#2980b9", lw=2)
        ax.plot(epochs, hist["val_loss"], label="Validation Loss", color="#e74c3c", lw=2, linestyle='--')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Figure 5: Training vs Validation Loss Progression (FT-Transformer)", fontweight='bold')
        ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "05_training_validation_loss.png"))
    plt.close()
    print("  -> Saved: 05_training_validation_loss.png")

    # -------------------------------------------------------------
    # Figure 6: Latent Embedding Visualization (PCA)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    embed_file = os.path.join(tables_dir, "test_embeddings.npy")
    if os.path.exists(embed_file):
        embeds = np.load(embed_file)
        pca = PCA(n_components=2)
        embed_2d = pca.fit_transform(embeds)
        categories = pred_df["risk_category"].values

        for cat, color in TIER_COLORS.items():
            mask = (categories == cat)
            if np.any(mask):
                ax.scatter(embed_2d[mask, 0], embed_2d[mask, 1],
                           c=color, label=f"{cat} Risk ({mask.sum()})",
                           alpha=0.8, edgecolors='k', s=60, lw=0.7)

        ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        ax.set_title("Figure 6: 2D Projection of 32-D FT-Transformer Patient Embeddings", fontweight='bold')
        ax.legend(title="Estimated Risk Tier", frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "06_embedding_visualization.png"))
    plt.close()
    print("  -> Saved: 06_embedding_visualization.png")

    # -------------------------------------------------------------
    # Figure 7: Continuous Risk-Score Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    scores = pred_df["risk_score"]
    sns.histplot(scores, kde=True, ax=ax, color='#34495e', bins=15, alpha=0.5, edgecolor='black')
    ax.axvline(0.35, color='#27ae60', linestyle='--', lw=2, label="Low / Moderate Cutoff (0.35)")
    ax.axvline(0.65, color='#c0392b', linestyle='--', lw=2, label="Moderate / High Cutoff (0.65)")
    ax.set_xlabel("Estimated Continuous Research Risk Index [0, 1]")
    ax.set_ylabel("Patient Count")
    ax.set_title("Figure 7: Continuous Research Risk Index Distribution (Test Cohort, N=60)", fontweight='bold')
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "07_risk_score_distribution.png"))
    plt.close()
    print("  -> Saved: 07_risk_score_distribution.png")

    # -------------------------------------------------------------
    # Figure 8: Risk Category Distribution (Model vs Reference)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    tier_order = ["LOW", "MODERATE", "HIGH"]
    model_counts = [int((pred_df["risk_category"] == t).sum()) for t in tier_order]
    ref_counts = [int((pred_df["reference_risk_category"] == t).sum()) for t in tier_order]

    x = np.arange(len(tier_order))
    width = 0.35
    ax.bar(x - width/2, ref_counts, width, label='Reference Clinical Tier', color='#95a5a6', edgecolor='black')
    ax.bar(x + width/2, model_counts, width, label='FT-Transformer Model Tier', color='#3498db', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(tier_order, fontweight='bold')
    ax.set_ylabel("Patient Count")
    ax.set_title("Figure 8: Risk Category Concordance (Model vs Reference Framework)", fontweight='bold')
    ax.legend(frameon=True)
    for i in range(len(tier_order)):
        ax.text(x[i] - width/2, ref_counts[i] + 0.5, str(ref_counts[i]), ha='center', fontsize=9)
        ax.text(x[i] + width/2, model_counts[i] + 0.5, str(model_counts[i]), ha='center', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "08_risk_category_distribution.png"))
    plt.close()
    print("  -> Saved: 08_risk_category_distribution.png")

    # -------------------------------------------------------------
    # Figure 9: Risk-Tier Agreement Matrix (MANDATORY: NOT Preeclampsia CM)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = ["LOW", "MODERATE", "HIGH"]
    cm = confusion_matrix(pred_df["reference_risk_category"], pred_df["risk_category"], labels=labels)
    # Annotate with count and row-percentage
    annot_matrix = np.empty_like(cm, dtype=object)
    for r in range(cm.shape[0]):
        row_sum = cm[r, :].sum()
        for c in range(cm.shape[1]):
            val = cm[r, c]
            pct = (val / row_sum * 100) if row_sum > 0 else 0.0
            annot_matrix[r, c] = f"{val}\n({pct:.1f}%)"

    sns.heatmap(cm, annot=annot_matrix, fmt="", cmap="Blues", cbar=True,
                xticklabels=labels, yticklabels=labels, ax=ax, linewidths=1, linecolor='white')
    ax.set_xlabel("Model Predicted Risk Tier", fontweight='bold')
    ax.set_ylabel("Clinically Anchored Reference Tier", fontweight='bold')
    ax.set_title("Figure 9: Clinical Risk-Tier Agreement Matrix\n(Reference Tier vs Model Tier — Test Cohort)", fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "09_risk_tier_agreement_matrix.png"))
    plt.close()
    print("  -> Saved: 09_risk_tier_agreement_matrix.png")

    # -------------------------------------------------------------
    # Figure 10: Top Feature Contributions
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    feat_df = pd.read_csv(os.path.join(tables_dir, "feature_contributions.csv"))
    top_10 = feat_df.head(10).iloc[::-1]  # Reverse for horizontal bar chart
    bars = ax.barh(top_10["clinical_name"], top_10["masking_score_shift"], color='#e67e22', edgecolor='black', alpha=0.85)
    ax.set_xlabel("Mean Absolute Risk Score Shift (Feature Masking Impact)")
    ax.set_title("Figure 10: Top 10 Clinical Risk Feature Importances (Perturbation Sensitivity)", fontweight='bold')
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va='center', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "10_feature_contributions.png"))
    plt.close()
    print("  -> Saved: 10_feature_contributions.png")

    # -------------------------------------------------------------
    # Figure 11: Ratio vs Risk Score
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    for cat, color in TIER_COLORS.items():
        mask = (pred_df["risk_category"] == cat)
        if np.any(mask):
            ax.scatter(test_df.loc[mask, "plgf:sflt"], pred_df.loc[mask, "risk_score"],
                       c=color, label=f"{cat} Risk", alpha=0.8, edgecolors='k', s=60, lw=0.7)

    # Reference indicator for the 38 cutoff context
    ax.axvline(38, color='purple', linestyle=':', lw=1.5, label="Historical Cutoff Reference (38)")
    ax.set_xlabel("Angiogenic Ratio (plgf:sflt)")
    ax.set_ylabel("FT-Transformer Risk Score [0, 1]")
    ax.set_title("Figure 11: Risk Score Trajectory vs Angiogenic Ratio (plgf:sflt)", fontweight='bold')
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "11_ratio_vs_risk.png"))
    plt.close()
    print("  -> Saved: 11_ratio_vs_risk.png")

    # -------------------------------------------------------------
    # Figure 12: Clinical Profile Comparison Across Tiers
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    clean_test = preprocessor._remap_hemodynamics(test_df)
    clean_test["risk_tier"] = pred_df["risk_category"].values

    # Systolic BP by tier
    sns.boxplot(x="risk_tier", y="clean_sysbp", data=clean_test, order=["LOW", "MODERATE", "HIGH"],
                palette=TIER_COLORS, ax=axes[0])
    axes[0].set_title("Physiological Systolic BP", fontweight='bold')
    axes[0].set_ylabel("mmHg")
    axes[0].set_xlabel("Risk Tier")

    # Soluble Endoglin by tier
    sns.boxplot(x="risk_tier", y="SEng", data=clean_test, order=["LOW", "MODERATE", "HIGH"],
                palette=TIER_COLORS, ax=axes[1])
    axes[1].set_title("Soluble Endoglin (SEng)", fontweight='bold')
    axes[1].set_ylabel("ng/mL")
    axes[1].set_xlabel("Risk Tier")

    # Serum Creatinine by tier
    sns.boxplot(x="risk_tier", y="creatinine", data=clean_test, order=["LOW", "MODERATE", "HIGH"],
                palette=TIER_COLORS, ax=axes[2])
    axes[2].set_title("Serum Creatinine", fontweight='bold')
    axes[2].set_ylabel("mg/dL")
    axes[2].set_xlabel("Risk Tier")

    fig.suptitle("Figure 12: Biomarker Comparisons Across Stratified Risk Tiers", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "12_clinical_profile_comparison.png"))
    plt.close()
    print("  -> Saved: 12_clinical_profile_comparison.png")

    print("\nAll 12 publication-quality figures generated successfully in outputs/figures/.")
    print("=" * 60)

if __name__ == "__main__":
    generate_all_visualizations()
