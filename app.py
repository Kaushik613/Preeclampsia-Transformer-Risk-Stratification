"""Professional Gradio dashboard for the preeclampsia research-risk prototype.

Important: scores reproduce/compare against a project-generated Research Risk
Index (RRI). The supplied dataset has no verified clinical outcome labels.
This is a research demonstration, not a diagnostic or clinical decision tool.
"""
from __future__ import annotations

import csv
import html
import importlib
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import gradio as gr
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

REQUIRED_ARTIFACTS = [
    ROOT / "outputs" / "checkpoints" / "best_model.pt",
    ROOT / "outputs" / "preprocessor.pkl",
    ROOT / "config" / "risk_thresholds.yaml",
]
for artifact in REQUIRED_ARTIFACTS:
    if not artifact.exists():
        raise FileNotFoundError(
            f"Required project artifact missing: {artifact}. Extract the full upgraded project ZIP and run app.py from its top-level folder."
        )

risk_mod = importlib.import_module("src.07_risk_model")
ref_mod = importlib.import_module("src.06_clinical_risk_score")
MODEL = risk_mod.PreeclampsiaRiskModel(
    checkpoint_path=str(ROOT / "outputs" / "checkpoints" / "best_model.pt"),
    preprocessor_path=str(ROOT / "outputs" / "preprocessor.pkl"),
    config_path=str(ROOT / "config" / "risk_thresholds.yaml"),
)

with (ROOT / "config" / "risk_thresholds.yaml").open("r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f) or {}
LOW_UPPER = float(CONFIG.get("thresholds", {}).get("low_risk_upper", 0.35))
MOD_UPPER = float(CONFIG.get("thresholds", {}).get("moderate_risk_upper", 0.65))
PREPROCESSOR = MODEL.preprocessor

EXPECTED = [
    "age", "gest_age", "height", "weight", "bmi", "sysbp", "diabp", "hb", "pcv", "tsh",
    "platelet", "creatinine", "plgf:sflt", "SEng", "cysC", "pp_13", "glycerides", "htn",
    "diabetes", "fam_htn", "sp_art", "occupation", "diet", "activity", "sleep"
]
NUMERIC_FIELDS = [
    ("age", "Maternal age (years)", 26),
    ("gest_age", "Gestational age (weeks)", 12),
    ("height", "Height (cm)", 157),
    ("weight", "Weight (kg)", 78),
    ("bmi", "BMI (kg/m²; supplied value)", 26.3),
    ("sysbp", "Systolic BP field (mmHg; source mapping unverified)", 130),
    ("diabp", "Diastolic BP field (mmHg; source mapping unverified)", 80),
    ("hb", "Hemoglobin (dataset units)", 13.0),
    ("pcv", "PCV / hematocrit (dataset units)", 41.0),
    ("tsh", "TSH (dataset units)", 3.0),
    ("platelet", "Platelet count (dataset units)", 254),
    ("creatinine", "Creatinine (dataset units)", 1.05),
    ("plgf:sflt", "plgf:sflt (orientation/assay unverified)", 190.0),
    ("SEng", "SEng (dataset units)", 18.0),
    ("cysC", "Cystatin C / cysC (dataset units)", 1.1),
    ("pp_13", "PP_13 (dataset units)", 79.0),
    ("glycerides", "Glycerides (units unverified)", 222.0),
]
BINARY_FIELDS = [
    ("htn", "htn code (0/1)", 0), ("diabetes", "diabetes code (0/1)", 0),
    ("fam_htn", "fam_htn code (0/1)", 0), ("sp_art", "sp_art code (0/1)", 0),
    ("diet", "diet code (0/1; meaning unverified)", 0),
]
CAT_FIELDS = [
    ("occupation", "occupation code (0/1/2)", 1),
    ("activity", "activity code (0/1/2; mapping unverified)", 1),
    ("sleep", "sleep code (0/1/2; mapping unverified)", 1),
]
FORM_FIELDS = NUMERIC_FIELDS[:7] + BINARY_FIELDS[:4] + NUMERIC_FIELDS[7:] + BINARY_FIELDS[4:] + CAT_FIELDS

FEATURE_LABELS = {
    "age": "Maternal age", "gest_age": "Gestational age", "height": "Height", "weight": "Weight",
    "bmi": "Supplied BMI", "sysbp": "sysbp field", "diabp": "diabp field", "hb": "Hemoglobin",
    "pcv": "PCV / hematocrit", "tsh": "TSH", "platelet": "Platelet count", "creatinine": "Creatinine",
    "plgf:sflt": "plgf:sflt ratio field", "SEng": "SEng", "cysC": "Cystatin C", "pp_13": "PP_13",
    "glycerides": "Glycerides", "htn": "Hypertension code", "diabetes": "Diabetes code",
    "fam_htn": "Family history code", "sp_art": "sp_art code", "occupation": "Occupation code",
    "diet": "Diet code", "activity": "Activity code", "sleep": "Sleep code"
}


def _fit_ridge_reference_baseline():
    """Fit the same fixed Ridge baseline as the earlier comparison, on train only."""
    train_path = ROOT / "data" / "train_with_ref_risk.csv"
    if train_path.exists():
        train_scored = pd.read_csv(train_path)
    else:
        train_raw = pd.read_csv(ROOT / "data" / "train.csv")
        train_scored = ref_mod.calculate_reference_risk_scores(
            train_raw, config_path=str(ROOT / "config" / "risk_thresholds.yaml")
        )
    y = train_scored["reference_risk_score"].to_numpy(dtype=float)
    train_clean = PREPROCESSOR._remap_hemodynamics(train_scored)
    num_cols = list(PREPROCESSOR.continuous_cols)
    bin_cols = list(PREPROCESSOR.binary_cols)
    cat_cols = list(PREPROCESSOR.categorical_cols)
    feature_cols = num_cols + bin_cols + cat_cols
    transformers = [
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
        ("binary", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), bin_cols),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ]
    prep = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    pipeline = Pipeline([("prep", prep), ("model", Ridge(alpha=10.0))])
    pipeline.fit(train_clean[feature_cols], y)
    return pipeline, feature_cols


# Load a saved Ridge pipeline when available. On the first local run only,
# fit it from the authorized training split and save the fitted artifact. This
# allows a cloned repository to run without bundling raw training records.
RIDGE_ARTIFACT_PATH = ROOT / "outputs" / "ridge_baseline.joblib"


def _load_or_fit_ridge_baseline():
    if RIDGE_ARTIFACT_PATH.exists():
        payload = joblib.load(RIDGE_ARTIFACT_PATH)
        if not isinstance(payload, dict) or not {
            "pipeline", "feature_columns", "alpha"
        }.issubset(payload):
            raise RuntimeError(
                f"Invalid Ridge artifact: {RIDGE_ARTIFACT_PATH}. "
                "Remove it and launch once with the authorized local training split to rebuild it."
            )
        return payload["pipeline"], list(payload["feature_columns"])

    train_scored_path = ROOT / "data" / "train_with_ref_risk.csv"
    train_raw_path = ROOT / "data" / "train.csv"
    if not train_scored_path.exists() and not train_raw_path.exists():
        raise FileNotFoundError(
            "Ridge baseline artifact is missing and no local training split was found. "
            "On the first run, place your authorized training file at "
            "'data/train_with_ref_risk.csv' or 'data/train.csv'. "
            "The app will fit and save outputs/ridge_baseline.joblib. "
            "Do not upload raw patient-level data to GitHub."
        )

    pipeline, feature_columns = _fit_ridge_reference_baseline()
    RIDGE_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": feature_columns,
            "alpha": 10.0,
            "target": "reference_risk_score",
            "training_source": "local training split; not bundled",
        },
        RIDGE_ARTIFACT_PATH,
        compress=3,
    )
    return pipeline, feature_columns


RIDGE_MODEL, RIDGE_FEATURES = _load_or_fit_ridge_baseline()


def tier_for_score(score: float) -> str:
    if score < LOW_UPPER:
        return "LOW"
    if score < MOD_UPPER:
        return "MODERATE"
    return "HIGH"


def validate_records(df: pd.DataFrame) -> list[str]:
    issues = []
    missing_cols = [c for c in EXPECTED if c not in df.columns]
    if missing_cols:
        return ["Missing required columns: " + ", ".join(missing_cols)]
    if df.empty:
        return ["The CSV has no rows."]
    if df[EXPECTED].isna().any().any():
        counts = df[EXPECTED].isna().sum()
        bad = [f"{k}={int(v)}" for k, v in counts.items() if v]
        issues.append("Missing values are not accepted by this demo: " + ", ".join(bad))
    for col in EXPECTED:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().any():
            issues.append(f"{col} must contain numeric values only.")
        elif not np.isfinite(converted.to_numpy(dtype=float)).all():
            issues.append(f"{col} must contain finite values; NaN/Infinity is not accepted.")
    for col, _, _ in BINARY_FIELDS:
        vals = set(pd.to_numeric(df[col], errors="coerce").dropna().unique())
        if not vals.issubset({0, 1, 0.0, 1.0}):
            issues.append(f"{col} must contain only codes 0 or 1; found {sorted(vals)}")
    for col, _, _ in CAT_FIELDS:
        vals = set(pd.to_numeric(df[col], errors="coerce").dropna().unique())
        if not vals.issubset({0, 1, 2, 0.0, 1.0, 2.0}):
            issues.append(f"{col} must contain only codes 0, 1, or 2; found {sorted(vals)}")
    # The raw source appears to have swapped BP headers. Mixed conventions in one file are rejected.
    if {"sysbp", "diabp"}.issubset(df.columns) and len(df) > 1:
        sys_vals = pd.to_numeric(df["sysbp"], errors="coerce")
        dia_vals = pd.to_numeric(df["diabp"], errors="coerce")
        order_flags = sys_vals < dia_vals
        if bool(order_flags.any()) and not bool(order_flags.all()):
            issues.append("Blood-pressure ordering differs across rows. Verify the source dictionary and use one consistent column convention for the whole CSV.")
    return issues


def _build_profile(raw_df: pd.DataFrame, compute_contributions: bool = True) -> pd.DataFrame:
    issues = validate_records(raw_df)
    if issues:
        raise ValueError("\n".join(issues))
    raw = raw_df[EXPECTED].copy().reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference_df = ref_mod.calculate_reference_risk_scores(
            raw, config_path=str(ROOT / "config" / "risk_thresholds.yaml")
        )
        ft_out, _ = MODEL.predict(reference_df, compute_contributions=compute_contributions)
    clean = PREPROCESSOR._remap_hemodynamics(raw)
    ridge_scores = np.clip(RIDGE_MODEL.predict(clean[RIDGE_FEATURES]), 0.0, 1.0)
    out = ft_out.copy()
    out["ridge_score"] = np.round(ridge_scores, 4)
    out["ridge_category"] = [tier_for_score(float(s)) for s in ridge_scores]
    out["ft_abs_gap_to_reference"] = np.abs(out["risk_score"].astype(float) - out["reference_risk_score"].astype(float)).round(4)
    out["ridge_abs_gap_to_reference"] = np.abs(out["ridge_score"].astype(float) - out["reference_risk_score"].astype(float)).round(4)
    out["ft_reference_match"] = np.where(out["risk_category"] == out["reference_risk_category"], "Same tier", "Different tier")
    out["ridge_reference_match"] = np.where(out["ridge_category"] == out["reference_risk_category"], "Same tier", "Different tier")
    return out


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return all project-score outputs for one or more raw input rows."""
    return _build_profile(df, compute_contributions=True)


def _scores_table(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([
        {"Method": "FT-Transformer", "Project score (0–1)": float(row["risk_score"]), "Tier": row["risk_category"], "Absolute gap to reference": float(row["ft_abs_gap_to_reference"])},
        {"Method": "Ridge baseline", "Project score (0–1)": float(row["ridge_score"]), "Tier": row["ridge_category"], "Absolute gap to reference": float(row["ridge_abs_gap_to_reference"])},
        {"Method": "Reference index", "Project score (0–1)": float(row["reference_risk_score"]), "Tier": row["reference_risk_category"], "Absolute gap to reference": 0.0},
    ])


def _gauge_html(score: float, tier: str, label: str = "PROJECT SCORE") -> str:
    score = float(np.clip(score, 0.0, 1.0))
    circumference = 2 * np.pi * 54
    offset = circumference * (1 - score)
    color = {"LOW": "#0f9f8f", "MODERATE": "#d99316", "HIGH": "#d94a62"}.get(tier, "#5367db")
    safe_tier = html.escape(str(tier))
    return f"""<div class='score-card'>
      <div class='score-ring-wrap'><svg viewBox='0 0 150 150' role='img' aria-label='{html.escape(label)} {score:.4f}'>
        <circle cx='75' cy='75' r='54' fill='none' stroke='#e7ebf3' stroke-width='12'/>
        <circle cx='75' cy='75' r='54' fill='none' stroke='{color}' stroke-width='12' stroke-linecap='round' stroke-dasharray='{circumference:.2f}' stroke-dashoffset='{offset:.2f}' transform='rotate(-90 75 75)'/>
        <text x='75' y='70' text-anchor='middle' font-size='25' font-weight='750' fill='#172033'>{score:.3f}</text>
        <text x='75' y='92' text-anchor='middle' font-size='10' fill='#64748b'>OUT OF 1.0</text>
      </svg></div>
      <div class='score-card-label'>{html.escape(label)}</div>
      <div class='score-tier' style='color:{color};'>{safe_tier}</div>
      <div class='score-note'>Project-defined score, not a disease probability</div>
    </div>"""


def _comparison_plot(row: pd.Series):
    methods = ["FT-Transformer", "Ridge baseline", "Reference index"]
    scores = [float(row["risk_score"]), float(row["ridge_score"]), float(row["reference_risk_score"])]
    tiers = [str(row["risk_category"]), str(row["ridge_category"]), str(row["reference_risk_category"])]
    colors = ["#5367db", "#0f9f8f", "#94a3b8"]
    fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=110)
    bars = ax.barh(methods, scores, color=colors, height=0.58)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Project score (0–1; not a clinical probability)")
    ax.set_title("Current record: model scores vs. reference index", loc="left", pad=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    for bar, score, tier in zip(bars, scores, tiers):
        ax.text(min(score + 0.025, 0.86), bar.get_y() + bar.get_height()/2,
                f"{score:.3f}  |  {tier}", va="center", fontsize=9, color="#172033")
    fig.tight_layout()
    return fig


def _placeholder_plot(title: str = "Your score comparison will appear here"):
    """Draw a helpful empty state instead of showing a blank plot before prediction."""
    fig, ax = plt.subplots(figsize=(8.5, 2.7), dpi=110)
    fig.patch.set_facecolor("#101e32")
    ax.set_facecolor("#101e32")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(.5, .58, title, ha="center", va="center", fontsize=13, fontweight="bold", color="#eaf2ff")
    ax.text(.5, .40, "Run a research profile to populate this chart with calculated scores.", ha="center", va="center", fontsize=9.5, color="#a7bad2", wrap=True)
    fig.tight_layout(pad=1.4)
    return fig


def _empty_scores_table() -> pd.DataFrame:
    return pd.DataFrame(columns=["Method", "Project score (0–1)", "Tier", "Absolute gap to reference"])


def _empty_whatif_table() -> pd.DataFrame:
    return pd.DataFrame(columns=["Method", "Original score", "Scenario score", "Change", "Scenario tier"])


def _ensure_export_dir() -> Path:
    path = ROOT / "upgrades" / "results" / "app_reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_report(raw_row: pd.Series, result: pd.Series, prefix: str = "patient_report") -> tuple[str, str]:
    out_dir = _ensure_export_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    token = uuid4().hex[:8]
    base = f"{prefix}_{stamp}_{token}"
    csv_path = out_dir / f"{base}.csv"
    html_path = out_dir / f"{base}.html"
    exported = {col: raw_row[col] for col in EXPECTED}
    exported.update({
        "ft_transformer_score": float(result["risk_score"]),
        "ft_transformer_tier": str(result["risk_category"]),
        "ridge_baseline_score": float(result["ridge_score"]),
        "ridge_baseline_tier": str(result["ridge_category"]),
        "reference_index_score": float(result["reference_risk_score"]),
        "reference_index_tier": str(result["reference_risk_category"]),
        "ft_absolute_gap_to_reference": float(result["ft_abs_gap_to_reference"]),
        "ridge_absolute_gap_to_reference": float(result["ridge_abs_gap_to_reference"]),
        "top_score_sensitive_feature_1": str(result.get("top_feature_1", "N/A")),
        "top_score_sensitive_feature_2": str(result.get("top_feature_2", "N/A")),
        "top_score_sensitive_feature_3": str(result.get("top_feature_3", "N/A")),
        "interpretation": "Research-only project score; not a clinically validated diagnosis or calibrated disease probability.",
    })
    pd.DataFrame([exported]).to_csv(csv_path, index=False)

    input_rows = "".join(
        f"<tr><th>{html.escape(FEATURE_LABELS.get(col, col))}<br><small>{html.escape(col)}</small></th><td>{html.escape(str(raw_row[col]))}</td></tr>"
        for col in EXPECTED
    )
    score_rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{score:.4f}</td><td>{html.escape(tier)}</td></tr>"
        for name, score, tier in [
            ("FT-Transformer", float(result["risk_score"]), str(result["risk_category"])),
            ("Ridge baseline", float(result["ridge_score"]), str(result["ridge_category"])),
            ("Project reference index", float(result["reference_risk_score"]), str(result["reference_risk_category"])),
        ]
    )
    tops = ", ".join(str(result.get(f"top_feature_{i}", "N/A")) for i in range(1, 4))
    doc = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
      <title>Preeclampsia research risk profile</title><style>
      body{{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 22px;color:#172033;background:#f5f7fb}}
      header,.panel{{background:white;border:1px solid #e3e8f2;border-radius:16px;padding:22px;margin:16px 0}}
      h1{{margin:0 0 8px}} .warning{{background:#fff7e6;border-left:5px solid #d99316;padding:13px 16px;border-radius:8px}}
      table{{width:100%;border-collapse:collapse;margin-top:10px;background:white}} th,td{{padding:9px 11px;border-bottom:1px solid #e8edf5;text-align:left}}
      th{{width:55%;color:#475569}} small{{font-weight:400;color:#64748b}} .muted{{color:#64748b;font-size:13px}}
      </style></head><body><header><h1>Preeclampsia Research Risk Profile</h1>
      <p class='muted'>Generated locally: {html.escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>
      <div class='warning'><strong>Research demonstration only.</strong> This dataset does not contain verified preeclampsia outcome labels. These are project-defined scores, not calibrated disease probabilities, diagnosis, or advice for patient care.</div></header>
      <section class='panel'><h2>Score comparison</h2><table><thead><tr><th>Method</th><th>Project score (0–1)</th><th>Tier</th></tr></thead><tbody>{score_rows}</tbody></table>
      <p><strong>Absolute gap to reference:</strong> FT-Transformer {float(result['ft_abs_gap_to_reference']):.4f}; Ridge {float(result['ridge_abs_gap_to_reference']):.4f}</p>
      <p><strong>Top model-sensitive features:</strong> {html.escape(tops)}</p><p class='muted'>Feature sensitivity is not evidence of medical causation. Tier boundaries are project-configured.</p></section>
      <section class='panel'><h2>Input record</h2><table><tbody>{input_rows}</tbody></table></section>
      <footer class='muted'>For academic demonstration only. Do not use to make clinical decisions.</footer></body></html>"""
    html_path.write_text(doc, encoding="utf-8")
    return str(html_path), str(csv_path)


def _form_values_to_df(values) -> pd.DataFrame:
    mapping = {field[0]: value for field, value in zip(FORM_FIELDS, values)}
    missing = [col for col in EXPECTED if mapping.get(col) is None]
    if missing:
        raise ValueError("Please provide values for: " + ", ".join(missing))
    return pd.DataFrame([{col: mapping[col] for col in EXPECTED}], columns=EXPECTED)


def single_prediction(*values):
    try:
        raw_df = _form_values_to_df(values)
        result_df = _build_profile(raw_df, compute_contributions=True)
        row = result_df.iloc[0]
        summary = (
            "### Model comparison\n"
            f"- **FT-Transformer:** `{float(row['risk_score']):.4f}` — **{row['risk_category']}**\n"
            f"- **Ridge baseline:** `{float(row['ridge_score']):.4f}` — **{row['ridge_category']}**\n"
            f"- **Reference index:** `{float(row['reference_risk_score']):.4f}` — **{row['reference_risk_category']}**\n\n"
            f"**Absolute gap to reference:** FT-Transformer `{float(row['ft_abs_gap_to_reference']):.4f}`; Ridge `{float(row['ridge_abs_gap_to_reference']):.4f}`.\n\n"
            f"**Top score-sensitive features:** {row.get('top_feature_1', 'N/A')}, {row.get('top_feature_2', 'N/A')}, {row.get('top_feature_3', 'N/A')}.\n\n"
            "> These scores compare two models with the project's own heuristic reference index. They are not probabilities of developing preeclampsia and are not diagnostic results."
        )
        report_html, report_csv = _write_report(raw_df.iloc[0], row)
        return (
            _gauge_html(float(row["risk_score"]), str(row["risk_category"]), "FT-TRANSFORMER SCORE"),
            summary,
            _scores_table(row),
            _comparison_plot(row),
            report_html,
            report_csv,
        )
    except Exception as exc:
        return ("<div class='score-card'>Unable to calculate score</div>", f"**Input / error:** {exc}", pd.DataFrame(), None, None, None)


def _validate_whatif_value(feature: str, value):
    if feature in ("htn", "diabetes", "fam_htn", "sp_art", "diet"):
        if float(value) not in (0.0, 1.0):
            raise ValueError(f"{feature} is coded as 0 or 1 in this prototype.")
        return int(value)
    if feature in ("occupation", "activity", "sleep"):
        if float(value) not in (0.0, 1.0, 2.0):
            raise ValueError(f"{feature} is coded as 0, 1, or 2 in this prototype.")
        return int(value)
    return float(value)


def whatif_prediction(*args):
    try:
        form_values = args[:len(FORM_FIELDS)]
        feature = args[len(FORM_FIELDS)]
        new_value = args[len(FORM_FIELDS) + 1]
        if not feature:
            raise ValueError("Choose a feature to change.")
        if new_value is None or not np.isfinite(float(new_value)):
            raise ValueError("Enter a finite numeric value for the scenario.")
        original_df = _form_values_to_df(form_values)
        new_value = _validate_whatif_value(feature, new_value)
        baseline_df = _build_profile(original_df, compute_contributions=False)
        scenario_df = original_df.copy()
        old_value = scenario_df.iloc[0][feature]
        scenario_df.loc[scenario_df.index[0], feature] = new_value
        changed_df = _build_profile(scenario_df, compute_contributions=False)
        b = baseline_df.iloc[0]
        c = changed_df.iloc[0]
        summary = (
            f"### What-if scenario: {FEATURE_LABELS.get(feature, feature)}\n"
            f"Changed `{feature}` from **{old_value}** to **{new_value}**. Only this input was changed.\n\n"
            f"- **FT-Transformer:** {float(b['risk_score']):.4f} → {float(c['risk_score']):.4f} (Δ {float(c['risk_score'] - b['risk_score']):+.4f})\n"
            f"- **Ridge baseline:** {float(b['ridge_score']):.4f} → {float(c['ridge_score']):.4f} (Δ {float(c['ridge_score'] - b['ridge_score']):+.4f})\n"
            f"- **Reference index:** {float(b['reference_risk_score']):.4f} → {float(c['reference_risk_score']):.4f} (Δ {float(c['reference_risk_score'] - b['reference_risk_score']):+.4f})\n\n"
            f"This is model sensitivity to a changed input, not a medical recommendation or a causal effect. Category codes and clinical feature conventions must be verified against original source documentation."
        )
        compare = pd.DataFrame([
            {"Method": "FT-Transformer", "Original score": float(b["risk_score"]), "Scenario score": float(c["risk_score"]), "Change": float(c["risk_score"] - b["risk_score"]), "Scenario tier": c["risk_category"]},
            {"Method": "Ridge baseline", "Original score": float(b["ridge_score"]), "Scenario score": float(c["ridge_score"]), "Change": float(c["ridge_score"] - b["ridge_score"]), "Scenario tier": c["ridge_category"]},
            {"Method": "Reference index", "Original score": float(b["reference_risk_score"]), "Scenario score": float(c["reference_risk_score"]), "Change": float(c["reference_risk_score"] - b["reference_risk_score"]), "Scenario tier": c["reference_risk_category"]},
        ])
        return (
            _gauge_html(float(b["risk_score"]), str(b["risk_category"]), "ORIGINAL FT SCORE"),
            _gauge_html(float(c["risk_score"]), str(c["risk_category"]), "WHAT-IF FT SCORE"),
            summary,
            compare,
        )
    except Exception as exc:
        return ("<div class='score-card'>—</div>", "<div class='score-card'>—</div>", f"**Input / error:** {exc}", pd.DataFrame())


def batch_prediction(file_obj):
    try:
        if file_obj is None:
            raise ValueError("Upload a CSV file first.")
        path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
        if not path:
            raise ValueError("Could not read the uploaded file path.")
        df = pd.read_csv(path)
        result = _build_profile(df[EXPECTED] if all(c in df.columns for c in EXPECTED) else df, compute_contributions=True)
        display_cols = [
            "risk_score", "risk_category", "ridge_score", "ridge_category", "reference_risk_score",
            "reference_risk_category", "ft_abs_gap_to_reference", "ridge_abs_gap_to_reference",
            "top_feature_1", "top_feature_2", "top_feature_3"
        ]
        table = pd.concat([df[EXPECTED].reset_index(drop=True), result[display_cols].reset_index(drop=True)], axis=1)
        out_dir = _ensure_export_dir()
        dest = out_dir / f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}.csv"
        table.to_csv(dest, index=False)
        return table, str(dest), f"Processed {len(table)} rows. Scores compare against the project-defined reference index, not verified clinical outcomes."
    except Exception as exc:
        return pd.DataFrame(), None, f"Error: {exc}"


CUSTOM_CSS = r"""
:root {
  --studio-bg: #07111f;
  --studio-panel: rgba(15, 29, 49, .82);
  --studio-panel-strong: #102039;
  --studio-border: rgba(139, 169, 205, .20);
  --studio-text: #eaf2ff;
  --studio-muted: #a7bad2;
  --studio-cyan: #4de2d0;
  --studio-blue: #7b8cff;
  --studio-violet: #b38aff;
}

html { scroll-behavior: smooth; }
body {
  background:
    radial-gradient(ellipse at 12% 0%, rgba(41, 105, 171, .25), transparent 34%),
    radial-gradient(ellipse at 92% 15%, rgba(81, 55, 169, .19), transparent 30%),
    linear-gradient(160deg, #07111f 0%, #081525 48%, #0b1224 100%) !important;
  background-attachment: fixed !important;
}
.gradio-container {
  max-width: 1480px !important;
  width: calc(100% - 52px) !important;
  margin: 0 auto !important;
  padding: 28px 0 54px !important;
  background: transparent !important;
  color: var(--studio-text) !important;
  font-size: 15px !important;
}
.gradio-container, .gradio-container * { box-sizing: border-box; }
.gradio-container p { line-height: 1.65; }
.gradio-container h2, .gradio-container h3, .gradio-container h4 {
  letter-spacing: -.025em;
}

/* Premium hero banner */
#hero {
  position: relative;
  isolation: isolate;
  overflow: hidden;
  padding: 36px 42px 30px !important;
  border-radius: 28px !important;
  border: 1px solid rgba(137, 189, 255, .25) !important;
  background:
    radial-gradient(circle at 88% 0%, rgba(123, 140, 255, .27), transparent 36%),
    radial-gradient(circle at 0% 100%, rgba(77, 226, 208, .12), transparent 36%),
    linear-gradient(120deg, rgba(12, 29, 51, .98), rgba(21, 39, 82, .96) 60%, rgba(26, 35, 84, .96)) !important;
  box-shadow: 0 24px 70px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.06) !important;
  margin: 0 0 18px !important;
  transition: border-color .35s ease, box-shadow .35s ease, transform .35s ease;
}
#hero:hover {
  border-color: rgba(77, 226, 208, .48) !important;
  box-shadow: 0 28px 80px rgba(0,0,0,.32), 0 0 36px rgba(77,226,208,.08), inset 0 1px 0 rgba(255,255,255,.08) !important;
  transform: translateY(-2px);
}
#hero::before {
  content: "";
  position: absolute;
  width: 320px; height: 320px;
  right: -90px; top: -170px;
  border-radius: 50%;
  border: 1px solid rgba(135, 178, 255, .18);
  box-shadow: 0 0 0 28px rgba(135,178,255,.025), 0 0 0 58px rgba(135,178,255,.02);
  pointer-events: none;
  z-index: -1;
}
#hero > * { position: relative; }
#hero p:first-child {
  color: var(--studio-cyan) !important;
  text-transform: uppercase;
  font-size: 11px !important;
  font-weight: 800 !important;
  letter-spacing: .20em !important;
  margin-bottom: 10px !important;
}
#hero h1 {
  color: #f7fbff !important;
  font-size: clamp(30px, 3.5vw, 48px) !important;
  line-height: 1.12 !important;
  letter-spacing: -.045em !important;
  margin: 0 0 12px !important;
  max-width: 900px;
}
#hero h1 + p {
  color: #c7d7ef !important;
  font-size: clamp(14px, 1.2vw, 17px) !important;
  max-width: 900px;
}
#hero code {
  border: 1px solid rgba(119, 155, 205, .27);
  color: #b5fff3 !important;
  background: rgba(5, 14, 28, .4) !important;
  border-radius: 8px;
  padding: 4px 8px;
}
#hero blockquote {
  background: rgba(3, 11, 23, .38) !important;
  border-left: 3px solid #f3bd62 !important;
  border-radius: 0 12px 12px 0;
  color: #d8e5f7 !important;
  padding: 10px 14px !important;
  margin: 18px 0 0 !important;
  max-width: 1040px;
}
#hero blockquote p { color: #d8e5f7 !important; margin: 0 !important; font-size: 13px !important; }

/* The four feature cards beneath the hero */
#stats-strip { margin: 0 0 22px !important; }
.stats-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  width: 100%;
}
.stat-card {
  position: relative;
  overflow: hidden;
  min-height: 102px;
  padding: 18px 20px;
  border-radius: 18px;
  border: 1px solid var(--studio-border);
  background: linear-gradient(145deg, rgba(19, 37, 61, .92), rgba(13, 27, 46, .86));
  box-shadow: 0 8px 24px rgba(0,0,0,.10), inset 0 1px 0 rgba(255,255,255,.035);
  transition: transform .28s ease, border-color .28s ease, box-shadow .28s ease, background .28s ease;
}
.stat-card::after {
  content: ""; position: absolute; left: 0; right: 0; top: 0; height: 2px;
  background: linear-gradient(90deg, var(--studio-cyan), var(--studio-blue), var(--studio-violet));
  opacity: .72; transform: scaleX(.32); transform-origin: left;
  transition: transform .35s ease, opacity .35s ease;
}
.stat-card:hover {
  transform: translateY(-5px);
  border-color: rgba(77,226,208,.43);
  box-shadow: 0 18px 35px rgba(0,0,0,.22), 0 0 28px rgba(77,226,208,.06);
  background: linear-gradient(145deg, rgba(23, 46, 72, .97), rgba(17, 32, 57, .94));
}
.stat-card:hover::after { transform: scaleX(1); opacity: 1; }
.stat-value { display:block; color:#f4fbff; font-size:23px; font-weight:850; letter-spacing:-.04em; line-height:1.2; }
.stat-label { display:block; margin-top:7px; color:#9bb1cc; font-size:10px; font-weight:800; letter-spacing:.15em; text-transform:uppercase; }
.stat-note { color:#c1d1e6; font-size:12px; margin-top:5px; }

/* Tabs as a modern glass navigation bar */
.gradio-container [role="tablist"], .gradio-container .tab-nav {
  gap: 8px !important;
  border: 1px solid rgba(139,169,205,.16) !important;
  border-radius: 17px !important;
  padding: 7px !important;
  margin: 0 0 20px !important;
  background: rgba(9, 22, 39, .77) !important;
  backdrop-filter: blur(18px);
}
.gradio-container [role="tab"] {
  border: 1px solid transparent !important;
  border-radius: 12px !important;
  padding: 11px 16px !important;
  color: #a9bed8 !important;
  background: transparent !important;
  font-weight: 700 !important;
  transition: all .22s ease !important;
}
.gradio-container [role="tab"]:hover {
  color: #efffff !important;
  background: rgba(77,226,208,.08) !important;
  border-color: rgba(77,226,208,.22) !important;
  transform: translateY(-2px);
}
.gradio-container [role="tab"][aria-selected="true"] {
  color: #071821 !important;
  background: linear-gradient(110deg, #67eddb, #9deee8) !important;
  border-color: rgba(255,255,255,.4) !important;
  box-shadow: 0 5px 18px rgba(77,226,208,.17) !important;
}
.gradio-container .tabitem { animation: paneIn .32s ease both; }
@keyframes paneIn { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }

/* Inputs grouped into panels */
.glass-panel, .input-panel, .control-panel {
  border: 1px solid var(--studio-border) !important;
  border-radius: 20px !important;
  padding: 20px !important;
  background: linear-gradient(145deg, rgba(17, 33, 55, .86), rgba(11, 25, 43, .80)) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.035), 0 10px 32px rgba(0,0,0,.08) !important;
  transition: border-color .28s ease, box-shadow .28s ease, transform .28s ease;
}
.glass-panel:hover, .input-panel:hover, .control-panel:hover {
  border-color: rgba(123,140,255,.36) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.045), 0 15px 40px rgba(0,0,0,.16) !important;
}
.section-intro { color: #b9cbe2 !important; margin: 7px 0 16px !important; }
.gradio-container h3, .gradio-container h4 { color:#eaf4ff !important; }
.gradio-container label, .gradio-container .label-wrap { color:#cbd9ec !important; font-weight:650 !important; }
.gradio-container input, .gradio-container textarea, .gradio-container select {
  color: #eef6ff !important;
  background: rgba(5, 17, 32, .72) !important;
  border: 1px solid rgba(139,169,205,.23) !important;
  border-radius: 11px !important;
  transition: border-color .2s ease, box-shadow .2s ease, background .2s ease !important;
}
.gradio-container input:hover, .gradio-container textarea:hover, .gradio-container select:hover {
  border-color: rgba(77,226,208,.48) !important;
  background: rgba(7, 22, 39, .92) !important;
}
.gradio-container input:focus, .gradio-container textarea:focus, .gradio-container select:focus {
  outline: none !important;
  border-color: var(--studio-cyan) !important;
  box-shadow: 0 0 0 3px rgba(77,226,208,.13) !important;
}

/* Buttons */
.gradio-container .gr-button, .gradio-container button.primary, .gradio-container button.secondary {
  border-radius: 12px !important;
  font-weight: 800 !important;
  letter-spacing: .01em;
  transition: transform .2s ease, box-shadow .2s ease, filter .2s ease, border-color .2s ease !important;
}
.gradio-container .gr-button-primary, .gradio-container button.primary {
  color: #061620 !important;
  border: 1px solid rgba(255,255,255,.26) !important;
  background: linear-gradient(105deg, #5ce4d2 0%, #8ba0ff 100%) !important;
  box-shadow: 0 8px 20px rgba(88,133,240,.18) !important;
}
.gradio-container .gr-button-primary:hover, .gradio-container button.primary:hover {
  transform: translateY(-3px) !important;
  filter: brightness(1.07) saturate(1.08);
  box-shadow: 0 13px 28px rgba(88,133,240,.29), 0 0 23px rgba(77,226,208,.12) !important;
}
.gradio-container .gr-button:active, .gradio-container button.primary:active { transform: translateY(0) scale(.99) !important; }

/* Score gauge card: subtle lift on hover */
.score-card {
  position: relative;
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  padding:24px 16px 22px;
  border:1px solid rgba(141,165,205,.22);
  border-radius:22px;
  background: radial-gradient(circle at 50% 0%, rgba(100,146,255,.10), transparent 45%), linear-gradient(180deg, rgba(18,36,59,.98), rgba(11,25,44,.98));
  min-height:255px; text-align:center;
  box-shadow: 0 14px 36px rgba(0,0,0,.15), inset 0 1px 0 rgba(255,255,255,.04);
  transition: transform .28s ease, border-color .28s ease, box-shadow .28s ease;
}
.score-card:hover {
  transform: translateY(-5px) scale(1.012);
  border-color: rgba(77,226,208,.42);
  box-shadow: 0 21px 44px rgba(0,0,0,.24), 0 0 28px rgba(77,226,208,.07);
}
.score-ring-wrap { width:160px; height:160px; filter: drop-shadow(0 7px 16px rgba(77,226,208,.07)); }
.score-ring-wrap svg circle { transition: stroke-dashoffset .85s cubic-bezier(.22,.68,0,1.0), stroke .3s ease; }
.score-card-label { font-size:11px; font-weight:850; letter-spacing:1.65px; color:#9db4d1; margin-top:7px; }
.score-tier { font-size:20px; font-weight:900; margin-top:7px; letter-spacing:.08em; }
.score-note { font-size:11px; color:#93a9c4; max-width:250px; margin-top:7px; line-height:1.5; }
.small-note { color:#9db4d1; font-size:12px; }

/* Tables and plots */
.gradio-container table { border-collapse: separate !important; border-spacing: 0 !important; overflow: hidden; border-radius: 12px !important; }
.gradio-container thead th { background: rgba(77,226,208,.10) !important; color: #c6fff5 !important; font-weight: 800 !important; }
.gradio-container tbody tr { transition: background .18s ease; }
.gradio-container tbody tr:hover { background: rgba(77,226,208,.07) !important; }
.gradio-container th, .gradio-container td { border-color: rgba(139,169,205,.15) !important; }
.gradio-container .plot-container, .gradio-container .dataframe-container {
  border-radius: 16px !important; overflow: hidden;
  border: 1px solid rgba(139,169,205,.16) !important;
}
.gradio-container .form, .gradio-container .panel { border-radius: 18px !important; }
.gradio-container .prose { color: #c4d4e8 !important; }
.gradio-container a { color: #6de6d5 !important; transition: color .2s ease; }
.gradio-container a:hover { color: #c0fff6 !important; }

/* Footer and responsive layout */
.app-footer { color:#7f96b5 !important; font-size:12px !important; text-align:center; padding:24px 0 0; }
@media (max-width: 900px) {
  .gradio-container { width: calc(100% - 28px) !important; padding-top: 14px !important; }
  #hero { padding: 26px 24px !important; border-radius: 22px !important; }
  .stats-strip { grid-template-columns: repeat(2, minmax(0,1fr)); gap:10px; }
  .stat-card { padding:15px; min-height:90px; }
  .gradio-container [role="tab"] { padding:9px 11px !important; }
}
@media (max-width: 520px) {
  .stats-strip { grid-template-columns: 1fr 1fr; }
  .stat-value { font-size:20px; }
  .gradio-container [role="tablist"], .gradio-container .tab-nav { overflow-x:auto; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior:auto !important; animation:none !important; transition:none !important; }
}

/* Studio layout correction: override Gradio's default narrow content wrapper. */
html, body { max-width: 100%; overflow-x: hidden !important; }
.gradio-container {
  max-width: none !important;
  width: 100% !important;
  padding: 26px clamp(18px, 3.2vw, 56px) 54px !important;
  overflow-x: clip !important;
}
.gradio-container > .main,
.gradio-container .main,
.gradio-container .contain,
.gradio-container .wrap {
  max-width: 1480px !important;
  width: 100% !important;
  min-width: 0 !important;
  margin-left: auto !important;
  margin-right: auto !important;
}
.gradio-container .prose,
.gradio-container .markdown,
#hero, #hero .prose, #hero > div {
  min-width: 0 !important;
  max-width: 100% !important;
  overflow-wrap: anywhere !important;
  word-break: normal !important;
}
#hero {
  width: 100% !important;
  overflow: hidden !important;
  padding: clamp(26px, 3.4vw, 46px) clamp(22px, 3.4vw, 48px) !important;
}
#hero pre, #hero table, #hero code { max-width: 100% !important; white-space: normal !important; }
#stats-strip { width: 100% !important; min-width: 0 !important; }
.gradio-container .tabitem, .gradio-container [role="tabpanel"] { min-width: 0 !important; }
.gradio-container .row, .gradio-container .column { min-width: 0 !important; }
.gradio-container input, .gradio-container select, .gradio-container textarea { max-width: 100% !important; }
.gradio-container .plot-container, .gradio-container .dataframe-container { max-width: 100% !important; }
@media (min-width: 1200px) {
  .gradio-container { padding-left: clamp(32px, 4vw, 72px) !important; padding-right: clamp(32px, 4vw, 72px) !important; }
}
@media (max-width: 760px) {
  .gradio-container { padding: 16px 14px 36px !important; }
  #hero { padding: 24px 20px !important; }
  .stats-strip { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
}

/* UI V2: accessibility, clear reading hierarchy, and robust full-width layout. */
html, body, gradio-app { width: 100% !important; max-width: 100% !important; overflow-x: hidden !important; }
body { font-size: 16px !important; }
.gradio-container,
.gradio-container.svelte-1ed2p3z,
.gradio-container > .main,
.gradio-container .main,
.gradio-container .main > .wrap,
.gradio-container .main .wrap,
.gradio-container .contain,
.gradio-container .main .contain,
.gradio-container .app,
.gradio-container #root {
  max-width: 1680px !important;
  width: 100% !important;
  min-width: 0 !important;
  margin-left: auto !important;
  margin-right: auto !important;
}
.gradio-container {
  width: 100% !important;
  padding-left: clamp(18px, 4vw, 72px) !important;
  padding-right: clamp(18px, 4vw, 72px) !important;
  font-size: 16px !important;
  line-height: 1.6 !important;
}
.gradio-container p,
.gradio-container .prose,
.gradio-container .markdown { font-size: 15px !important; line-height: 1.7 !important; }
.gradio-container h1 { font-size: clamp(30px, 3.2vw, 48px) !important; line-height: 1.15 !important; }
.gradio-container h2 { font-size: clamp(23px, 2vw, 30px) !important; line-height: 1.25 !important; }
.gradio-container h3 { font-size: 20px !important; line-height: 1.35 !important; }
.gradio-container h4 { font-size: 17px !important; line-height: 1.4 !important; }
.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .svelte-1gfkn6j { font-size: 14px !important; line-height: 1.45 !important; }
.gradio-container input,
.gradio-container select,
.gradio-container textarea,
.gradio-container button,
.gradio-container .wrap { min-width: 0; }
.gradio-container input,
.gradio-container select,
.gradio-container textarea { min-height: 42px !important; font-size: 15px !important; }
.gradio-container button { min-height: 42px; }
.gradio-container .gr-button-primary,
.gradio-container button.primary { min-height: 48px !important; }
.gradio-container :focus-visible,
.gradio-container button:focus-visible,
.gradio-container a:focus-visible,
.gradio-container input:focus-visible,
.gradio-container select:focus-visible,
.gradio-container textarea:focus-visible,
.gradio-container [role="tab"]:focus-visible {
  outline: 3px solid #93fff0 !important;
  outline-offset: 3px !important;
  box-shadow: 0 0 0 5px rgba(77, 226, 208, .20) !important;
}
.gradio-container [role="tablist"] { flex-wrap: wrap !important; }
.gradio-container [role="tab"] { min-height: 42px !important; font-size: 14px !important; }
.gradio-container .block { min-width: 0 !important; }
.gradio-container .plot-container,
.gradio-container .dataframe-container,
.gradio-container .file-preview,
.gradio-container [data-testid="block-info"] { max-width: 100% !important; min-width: 0 !important; }
.gradio-container .section-intro { font-size: 15px !important; max-width: 1100px; }
.gradio-container details,
.gradio-container .accordion { border-radius: 16px !important; }
.gradio-container .accordion { border: 1px solid var(--studio-border) !important; background: rgba(15,29,49,.72) !important; }
.gradio-container .dashboard-results { align-items: stretch !important; }
.gradio-container .dashboard-results > * { min-width: 0 !important; }
.ui-help-card {
  border: 1px solid rgba(139,169,205,.22); border-radius: 16px;
  padding: 16px 18px; background: rgba(10,24,42,.72); color: #dbe7f7;
}
.ui-help-card strong { color: #f1fbff; }
.ui-empty-state {
  border: 1px dashed rgba(139,169,205,.36); border-radius: 18px;
  background: rgba(13,28,47,.48); color: #b8cbe2;
  min-height: 145px; padding: 25px; display:flex; flex-direction:column;
  justify-content:center; align-items:center; text-align:center; gap:8px;
}
.ui-empty-state .empty-icon { font-size: 25px; line-height: 1; color: var(--studio-cyan); }
.ui-empty-state .empty-title { font-size: 16px; font-weight: 800; color: #eaf4ff; }
.ui-empty-state .empty-copy { font-size: 14px; max-width: 560px; line-height: 1.6; }
.ui-status-note { border-left: 3px solid var(--studio-cyan); padding: 10px 13px; background: rgba(77,226,208,.06); border-radius: 0 10px 10px 0; color: #c9d8e9; }
.app-footer { font-size: 13px !important; line-height: 1.7 !important; }
@media (min-width: 1500px) {
  .gradio-container .main > .wrap, .gradio-container .main .wrap { max-width: 1560px !important; }
}
@media (max-width: 900px) {
  .gradio-container { padding-left: 20px !important; padding-right: 20px !important; }
  .gradio-container .dashboard-results { flex-wrap: wrap !important; }
  .gradio-container .dashboard-results > * { flex-basis: 100% !important; }
}
@media (max-width: 600px) {
  .gradio-container { padding: 12px 12px 30px !important; }
  #hero { padding: 24px 18px !important; border-radius: 18px !important; }
  .stats-strip { grid-template-columns: 1fr 1fr !important; }
  .stat-card { padding: 12px !important; min-height: 88px !important; }
  .stat-value { font-size: 19px !important; }
  .score-card { min-height: 220px; }
  .score-ring-wrap { width: 140px; height: 140px; }
}
@media (prefers-contrast: more) {
  .gradio-container label, .gradio-container p, .small-note, .stat-note,
  .section-intro, .score-note { color: #eef6ff !important; }
  .glass-panel, .input-panel, .control-panel, .stat-card, .score-card { border-color: #a8c8ed !important; }
}
"""


def build_app():
    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="teal", neutral_hue="slate")
    with gr.Blocks(title="Preeclampsia Research Risk Studio — Accessible UI v2", fill_width=True) as demo:
        gr.Markdown(
            "**PREECLAMPSIA RESEARCH LAB  /  MODEL EXPLORER**\n\n"
            "# Preeclampsia Research Risk Dashboard\n"
            "Explore a Transformer-based patient representation, compare it with a regularized baseline, and inspect how input changes affect the project-defined score.\n\n"
            "`FT-TRANSFORMER`  `RIDGE BASELINE`  `WHAT-IF ANALYSIS`\n\n"
            "> ⚠️ **Research prototype only.** The supplied dataset has no verified clinical outcome labels. Scores are not calibrated disease probabilities, a diagnosis, or a basis for patient-care decisions.",
            elem_id="hero",
        )
        with gr.Accordion("Quick start — how to use the studio", open=False):
            gr.Markdown(
                "<div class='ui-help-card'>"
                "<strong>1. Patient Profile:</strong> review the illustrative defaults or enter a record you are authorized to use, then select <strong>Generate research profile</strong>.<br><br>"
                "<strong>2. Compare:</strong> review the FT-Transformer score, Ridge baseline, and project reference index together. These are project-defined scores, not disease probabilities.<br><br>"
                "<strong>3. What-if Lab:</strong> change one field to examine model sensitivity. This does not establish cause or recommend a clinical intervention.<br><br>"
                "<strong>4. Batch CSV:</strong> upload a CSV with the 25 required columns to generate a downloadable score table.<br><br>"
                "<strong>Accessibility:</strong> use Tab and Shift+Tab to move between controls; Enter or Space activates focused buttons and tabs. Animations are reduced when your operating system requests reduced motion."
                "</div>"
            )
        gr.HTML(
            "<div class='stats-strip'>"
            "<div class='stat-card'><span class='stat-value'>25</span><span class='stat-label'>Input features</span><div class='stat-note'>Maternal, biochemical &amp; lifestyle</div></div>"
            "<div class='stat-card'><span class='stat-value'>3 views</span><span class='stat-label'>Score comparison</span><div class='stat-note'>Transformer · Ridge · reference index</div></div>"
            "<div class='stat-card'><span class='stat-value'>32-D</span><span class='stat-label'>Patient embedding</span><div class='stat-note'>Compact learned representation</div></div>"
            "<div class='stat-card'><span class='stat-value'>Research</span><span class='stat-label'>Validation status</span><div class='stat-note'>Not a clinically validated diagnostic tool</div></div>"
            "</div>",
            elem_id="stats-strip",
        )
        with gr.Tab("Patient profile"):
            gr.Markdown(
                "### Build a patient research profile\n"
                "Enter the 25 required fields, then generate a profile to populate the score cards and report downloads. "
                "Initial values are illustrative demo defaults, not a real patient record. Categorical codes remain numeric because the source mapping was not supplied.",
                elem_classes=["section-intro"],
            )
            gr.Markdown("<div class='ui-status-note'><strong>Before you begin:</strong> use authorized data only. Verify the units and source mapping for blood-pressure, BMI, and angiogenic features before interpreting results.</div>")
            inputs = []
            with gr.Row(elem_id="profile-inputs"):
                with gr.Column(scale=1, elem_classes=["glass-panel", "input-panel"]):
                    gr.Markdown("#### Maternal & cardiovascular fields")
                    for field in NUMERIC_FIELDS[:7]:
                        inputs.append(gr.Number(label=field[1], value=field[2], precision=3))
                    for field in BINARY_FIELDS[:4]:
                        inputs.append(gr.Dropdown(label=field[1], choices=[0, 1], value=field[2], interactive=True))
                with gr.Column(scale=1, elem_classes=["glass-panel", "input-panel"]):
                    gr.Markdown("#### Biochemical, angiogenic & lifestyle fields")
                    for field in NUMERIC_FIELDS[7:]:
                        inputs.append(gr.Number(label=field[1], value=field[2], precision=3))
                    for field in BINARY_FIELDS[4:]:
                        inputs.append(gr.Dropdown(label=field[1], choices=[0, 1], value=field[2], interactive=True))
                    for field in CAT_FIELDS:
                        inputs.append(gr.Dropdown(label=field[1], choices=[0, 1, 2], value=field[2], interactive=True))
            run_btn = gr.Button("Generate research profile", variant="primary", size="lg")
            with gr.Row(equal_height=True, elem_classes=["dashboard-results"]):
                gauge = gr.HTML(
                    value="<div class='ui-empty-state'><div class='empty-icon' aria-hidden='true'>◉</div><div class='empty-title'>Your score will appear here</div><div class='empty-copy'>Generate a research profile to display the FT-Transformer score and project-defined tier.</div></div>",
                    label="FT-Transformer score",
                )
                with gr.Column(scale=2):
                    summary = gr.Markdown(
                        value="<div class='ui-empty-state'><div class='empty-title'>Model comparison</div><div class='empty-copy'>The FT-Transformer, Ridge baseline, and reference index will be compared here after a successful run.</div></div>",
                        label="Summary",
                    )
                    comparison = gr.Dataframe(value=_empty_scores_table(), label="Score comparison", interactive=False, wrap=True)
            plot = gr.Plot(value=_placeholder_plot(), label="Score comparison chart")
            gr.Markdown("### Export your results\nDownloads become available after a successful prediction. Reports include the inputs, all three project scores, feature-sensitivity notes, and a research-only disclaimer.", elem_classes=["section-intro"])
            with gr.Row():
                report_html = gr.File(label="Download formatted HTML patient report", interactive=False)
                report_csv = gr.File(label="Download input + scores CSV", interactive=False)
            run_btn.click(single_prediction, inputs=inputs, outputs=[gauge, summary, comparison, plot, report_html, report_csv])

        with gr.Tab("What-if lab"):
            gr.Markdown(
                "### Change one input and compare scores\n"
                "This uses the current values in the Patient profile tab, changes only the selected feature, and recalculates FT-Transformer, Ridge, and the reference index. It explains model sensitivity only—not medical causation or advice."
            )
            with gr.Row(elem_classes=["control-panel"]):
                whatif_feature = gr.Dropdown(
                    label="Feature to change", choices=EXPECTED, value="age", interactive=True
                )
                whatif_value = gr.Number(label="New value (respect the feature's numeric code if categorical)", value=30, precision=4)
            whatif_btn = gr.Button("Run what-if comparison", variant="primary")
            with gr.Row(equal_height=True, elem_classes=["dashboard-results"]):
                whatif_before = gr.HTML(value="<div class='ui-empty-state'><div class='empty-title'>Original score</div><div class='empty-copy'>Your current Patient Profile score will appear here.</div></div>")
                whatif_after = gr.HTML(value="<div class='ui-empty-state'><div class='empty-title'>Scenario score</div><div class='empty-copy'>Choose a feature and run a what-if comparison.</div></div>")
            whatif_summary = gr.Markdown(value="<div class='ui-status-note'>Tip: change one feature at a time. The output describes model sensitivity only, not medical causality.</div>")
            whatif_table = gr.Dataframe(value=_empty_whatif_table(), label="Original vs scenario score changes", interactive=False, wrap=True)
            whatif_btn.click(
                whatif_prediction,
                inputs=inputs + [whatif_feature, whatif_value],
                outputs=[whatif_before, whatif_after, whatif_summary, whatif_table],
            )

        with gr.Tab("Batch CSV"):
            gr.Markdown(
                "Upload a CSV containing all 25 expected columns. Missing values and invalid categorical codes are rejected. Scores compare models against the project's own reference index, not clinical outcome labels. Use only data you are authorized to process."
            )
            upload = gr.File(label="Patient input CSV", file_types=[".csv"], type="filepath")
            batch_btn = gr.Button("Process CSV", variant="primary")
            gr.Markdown("<div class='ui-help-card'><strong>CSV requirements</strong><br>Use the project's 25 expected column names. Categorical fields must use the documented numeric codes. Missing values and invalid category codes are rejected instead of silently corrected.</div>")
            batch_table = gr.Dataframe(value=pd.DataFrame(columns=EXPECTED), label="Output preview — ready for upload", interactive=False, wrap=True)
            batch_download = gr.File(label="Download batch scores CSV", interactive=False)
            batch_status = gr.Markdown(value="Upload a CSV file and select <strong>Process CSV</strong>. Results will appear here; no source data is uploaded to a remote service by this local app.")
            batch_btn.click(batch_prediction, inputs=[upload], outputs=[batch_table, batch_download, batch_status])

        with gr.Tab("Method & limitations"):
            gr.Markdown(
                f"""### How the studio works
25 input fields → saved training-fitted preprocessor → FT-Transformer → 32-D patient embedding → project score → Low / Moderate / High tier.

### Model comparison
- **FT-Transformer:** existing saved checkpoint; weights were not changed by this dashboard upgrade.
- **Ridge baseline:** loaded from the saved `outputs/ridge_baseline.joblib` artifact. On the first local run only, the app can fit and save this artifact from an authorized training split (`data/train_with_ref_risk.csv` or `data/train.csv`). Raw training data is not required after the artifact has been created and is not included in this repository.
- **Reference index:** deterministic score computed by the project's own heuristic formula.
- Tier boundaries configured in YAML: LOW < {LOW_UPPER:.2f}; MODERATE < {MOD_UPPER:.2f}; otherwise HIGH.

### Interpretation limits
- The training target is the project's own Research Risk Index (RRI), not a verified future preeclampsia outcome.
- Agreement with the index is not clinical diagnostic accuracy; the model may simply approximate the same heuristic.
- Supplied files do not verify the data source, blood-pressure column mapping, BMI consistency, or `plgf:sflt` ratio direction/assay.
- Feature masking / what-if tests describe model sensitivity, not medical causation.
- This prototype is not clinically validated. Do not use it for patient-care decisions.
"""
            )
        gr.Markdown("Made for research exploration · Reproducible scores, transparent comparisons, responsible interpretation", elem_classes=["app-footer"])
    return demo


if __name__ == "__main__":
    app_theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="teal", neutral_hue="slate")
    # Create the report directory before launch and explicitly allow Gradio to
    # serve generated CSV/HTML downloads from this project-owned folder.
    report_dir = _ensure_export_dir()
    build_app().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=app_theme,
        css=CUSTOM_CSS,
        allowed_paths=[str(report_dir.resolve())],
    )
