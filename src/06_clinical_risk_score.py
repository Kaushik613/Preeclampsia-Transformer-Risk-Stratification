"""
Phase 6: Clinically Anchored Research Risk Index (RRI)
Constructs a transparent, literature-informed clinical reference risk score
based on:
  1. Angiogenic Component (30%)
  2. Maternal Clinical Risk Factor Component (25%)
  3. Cardiovascular / Hemodynamic Component (20%)
  4. Biochemical / Renal Component (15%)
  5. Lifestyle / Social Component (10%)

IMPORTANT:
This produces an estimated preeclampsia-related reference risk score in [0, 1],
NOT a confirmed disease diagnosis.
"""

import os
import yaml
import numpy as np
import pandas as pd

def calculate_reference_risk_scores(df: pd.DataFrame, config_path: str = "config/risk_thresholds.yaml") -> pd.DataFrame:
    """
    Calculates the clinically anchored continuous Research Risk Index and assigns
    LOW / MODERATE / HIGH risk categories.
    """
    # Load configuration
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    weights = cfg.get("weights", {
        "angiogenic_component": 0.30,
        "maternal_clinical_component": 0.25,
        "cardiovascular_component": 0.20,
        "biochemical_component": 0.15,
        "lifestyle_component": 0.10
    })

    thresholds = cfg.get("thresholds", {
        "low_risk_upper": 0.35,
        "moderate_risk_upper": 0.65
    })

    df_calc = df.copy()

    # Physiologically correct blood pressure orientation
    if (df_calc['sysbp'] < df_calc['diabp']).all():
        sys_vals = df_calc['diabp'].values.astype(float)  # 120-139
        dia_vals = df_calc['sysbp'].values.astype(float)  # 70-89
    else:
        sys_vals = df_calc['sysbp'].values.astype(float)
        dia_vals = df_calc['diabp'].values.astype(float)

    # 1. Angiogenic Component (30%)
    # plgf:sflt normalized sigmoid (training-based anchor)
    sflt_vals = df_calc['plgf:sflt'].values.astype(float)
    sflt_norm = (sflt_vals - 190.0) / 100.0
    sflt_sig = 1.0 / (1.0 + np.exp(-sflt_norm))

    # SEng (soluble endoglin) normalized
    seng_vals = df_calc['SEng'].values.astype(float)
    seng_norm = (seng_vals - 18.0) / 5.0
    seng_sig = 1.0 / (1.0 + np.exp(-seng_norm))

    comp_angiogenic = 0.70 * sflt_sig + 0.30 * seng_sig

    # 2. Maternal Clinical Component (25%)
    # NICE NG133 high & moderate risk factors
    htn_val = df_calc['htn'].values.astype(float)          # Chronic hypertension (major)
    diab_val = df_calc['diabetes'].values.astype(float)    # Diabetes (major)
    fam_val = df_calc['fam_htn'].values.astype(float)      # Family history (moderate/high)
    
    age_val = df_calc['age'].values.astype(float)
    age_score = np.where(age_val >= 35, 0.20, np.where(age_val >= 30, 0.10, 0.0))
    
    bmi_val = df_calc['bmi'].values.astype(float)
    bmi_score = np.where(bmi_val >= 35.0, 0.20, np.where(bmi_val >= 30.0, 0.10, 0.0))

    comp_maternal = 0.35 * htn_val + 0.25 * diab_val + 0.15 * fam_val + age_score + bmi_score
    comp_maternal = np.clip(comp_maternal, 0.0, 1.0)

    # 3. Cardiovascular / Hemodynamic Component (20%)
    # Systolic elevation above 120, diastolic elevation above 80
    sys_score = np.clip((sys_vals - 120.0) / (140.0 - 120.0), 0.0, 1.0)
    dia_score = np.clip((dia_vals - 70.0) / (90.0 - 70.0), 0.0, 1.0)
    sp_art_val = df_calc['sp_art'].values.astype(float)  # Uterine artery Doppler notch

    comp_cardio = 0.40 * sys_score + 0.30 * dia_score + 0.30 * sp_art_val
    comp_cardio = np.clip(comp_cardio, 0.0, 1.0)

    # 4. Biochemical / Renal Component (15%)
    creat_val = df_calc['creatinine'].values.astype(float)
    creat_score = np.clip((creat_val - 0.7) / (1.4 - 0.7), 0.0, 1.0)
    
    cysc_val = df_calc['cysC'].values.astype(float)
    cysc_score = np.clip((cysc_val - 0.7) / (1.5 - 0.7), 0.0, 1.0)
    
    plt_val = df_calc['platelet'].values.astype(float)
    plt_risk = np.where(plt_val < 180, 0.5, 0.0)

    comp_biochem = 0.40 * creat_score + 0.40 * cysc_score + 0.20 * plt_risk
    comp_biochem = np.clip(comp_biochem, 0.0, 1.0)

    # 5. Lifestyle / Social Component (10%)
    sleep_val = df_calc['sleep'].values.astype(float)      # higher indicates poor sleep
    act_val = df_calc['activity'].values.astype(float)      # 0 indicates sedentary
    diet_val = df_calc['diet'].values.astype(float)

    sleep_score = sleep_val / 2.0
    act_score = (2.0 - act_val) / 2.0  # sedentary gets 1.0
    comp_lifestyle = 0.40 * sleep_score + 0.40 * act_score + 0.20 * diet_val
    comp_lifestyle = np.clip(comp_lifestyle, 0.0, 1.0)

    # Weighted Composite Research Risk Index
    rri = (
        weights["angiogenic_component"] * comp_angiogenic +
        weights["maternal_clinical_component"] * comp_maternal +
        weights["cardiovascular_component"] * comp_cardio +
        weights["biochemical_component"] * comp_biochem +
        weights["lifestyle_component"] * comp_lifestyle
    )
    rri = np.clip(rri, 0.0, 1.0)

    # Categorization into 3 tiers
    low_upper = thresholds.get("low_risk_upper", 0.35)
    mod_upper = thresholds.get("moderate_risk_upper", 0.65)

    categories = []
    for score in rri:
        if score < low_upper:
            categories.append("LOW")
        elif score < mod_upper:
            categories.append("MODERATE")
        else:
            categories.append("HIGH")

    result_df = df.copy()
    result_df['reference_risk_score'] = np.round(rri, 4)
    result_df['reference_risk_category'] = categories
    result_df['comp_angiogenic'] = np.round(comp_angiogenic, 4)
    result_df['comp_maternal'] = np.round(comp_maternal, 4)
    result_df['comp_cardio'] = np.round(comp_cardio, 4)
    result_df['comp_biochem'] = np.round(comp_biochem, 4)
    result_df['comp_lifestyle'] = np.round(comp_lifestyle, 4)

    return result_df

def run_clinical_risk_scoring(
    data_dir="data",
    config_path="config/risk_thresholds.yaml"
):
    print("=" * 60)
    print("PHASE 6: CLINICALLY ANCHORED REFERENCE RISK INDEX")
    print("=" * 60)

    for split in ["train", "validation", "test"]:
        csv_file = os.path.join(data_dir, f"{split}.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            scored_df = calculate_reference_risk_scores(df, config_path)
            
            # Save enriched dataframe with reference scores
            out_file = os.path.join(data_dir, f"{split}_with_ref_risk.csv")
            scored_df.to_csv(out_file, index=False)
            
            dist = scored_df['reference_risk_category'].value_counts().to_dict()
            mean_score = scored_df['reference_risk_score'].mean()
            print(f"Split: {split.upper()} (N={len(scored_df)}) | Mean Score: {mean_score:.3f} | Distribution: {dist}")

    print("Reference risk scoring completed cleanly.")
    print("=" * 60)

if __name__ == "__main__":
    run_clinical_risk_scoring()
