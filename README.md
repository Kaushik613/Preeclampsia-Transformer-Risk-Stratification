# Transformer-Based Early Risk Stratification for Preeclampsia

**Research prototype — not a clinical diagnostic or decision-support tool.**

This project explores tabular patient representation learning with an FT-Transformer and compares its project-defined score with a Ridge regression baseline.

## Important limitations

- The supplied development dataset contains 400 records and 25 predictor fields. Its provenance is not independently verified in the project materials.
- The dataset does **not** contain verified longitudinal preeclampsia outcomes. The project therefore does not establish disease-prediction accuracy.
- The Research Risk Index (RRI) is a project-defined reference target derived from selected input features and configured weights. It is not a validated clinical score.
- Reported model metrics measure agreement with that project-defined reference target, **not** sensitivity, specificity, ROC-AUC, or clinical diagnostic accuracy.
- Blood-pressure source-column mapping, BMI consistency, angiogenic-ratio orientation/assay, units, and category-code meanings require verification against the original data dictionary.
- The test partition has been used in exploratory model comparison and ablation work. It should not be described as an untouched final test set for all experiments.
- Do not use the dashboard for patient care or clinical decisions.

## Project overview

The dashboard provides:

- A 25-feature patient-profile interface with illustrative default values.
- FT-Transformer project score and 32-dimensional patient representation.
- Ridge regression baseline score.
- Comparison against the project-defined RRI.
- One-feature what-if sensitivity analysis.
- Batch CSV processing and local report exports.
- A method and limitations section.

All displayed scores are project-defined values on a 0–1 scale. They are **not probabilities** of developing preeclampsia.

## Repository structure

```text
.
├── app.py
├── run_pipeline.py
├── requirements.txt
├── README.md
├── config/
│   └── risk_thresholds.yaml
├── src/
│   ├── 01_data_audit.py
│   ├── 02_train_test_split.py
│   ├── 03_preprocessing.py
│   ├── 04_model.py
│   ├── 05_train_representation.py
│   ├── 06_clinical_risk_score.py
│   ├── 07_risk_model.py
│   ├── 08_test.py
│   ├── 09_interpretability.py
│   ├── 10_visualizations.py
│   ├── model.py
│   └── preprocessing.py
├── outputs/
│   ├── checkpoints/
│   │   └── best_model.pt
│   ├── preprocessor.pkl
│   └── ridge_baseline.joblib
└── upgrades/
    ├── compare_baselines.py
    ├── ablation_robustness.py
    ├── validate_clinical_assumptions.py
    └── results/
```

Raw data, patient-level predictions, patient reports, split CSVs, and embedding arrays are intentionally excluded from this repository. The actual `data/` and most generated `outputs/` content remain local.

## Environment setup

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The saved preprocessor and Ridge pipeline use scikit-learn serialization. The requirements file pins scikit-learn to the version used for the supplied artifacts.

## Run the dashboard

The dashboard requires these artifacts:

```text
outputs/checkpoints/best_model.pt
outputs/preprocessor.pkl
config/risk_thresholds.yaml
outputs/ridge_baseline.joblib
```

On the first local launch only, the updated `app.py` can create `outputs/ridge_baseline.joblib` from an authorized local training file at either:

```text
data/train_with_ref_risk.csv
data/train.csv
```

That training file is used locally to fit the Ridge baseline and is not required after the Ridge artifact has been created. Do not add raw patient-level data to Git.

Start the dashboard:

```powershell
python app.py
```

Open `http://127.0.0.1:7860`.

If you clone this repository, the FT-Transformer checkpoint, fitted preprocessor, configuration, and saved Ridge artifact must be present. Model artifacts should only be shared if the dataset owner/institution permits distribution of models derived from the source data.

## Re-run the training pipeline

The full pipeline is launched with:

```powershell
python run_pipeline.py
```

Re-running training requires the authorized source dataset and intermediate data files in the paths expected by the scripts. Those files are not included in this repository. Check the scripts and local data configuration before running the pipeline.

## Reported model comparison

The reported comparison target is the project-defined RRI, not verified clinical outcomes.

| Model | MAE | RMSE | R² | RRI tier agreement | Cohen's κ |
|---|---:|---:|---:|---:|---:|
| Ridge | 0.0118 | 0.0145 | 0.9764 | 98.3% | 0.90 |
| HistGradientBoosting | 0.0299 | 0.0374 | 0.8430 | 95.0% | 0.65 |
| FT-Transformer | 0.0315 | 0.0392 | 0.8272 | 91.7% | 0.51 |
| Random Forest | 0.0429 | 0.0543 | 0.6690 | 90.0% | 0.00 |
| Mean predictor | 0.0755 | 0.0942 | approximately 0 | 90.0% | 0.00 |
| MLP | 0.1509 | 0.1822 | -2.7360 | 55.0% | 0.12 |

These values describe how closely each model reproduces the constructed RRI on the project's evaluation partition. The Ridge model performs best at this task; this does not establish superior clinical prediction. The high agreement values must also be interpreted with the small evaluation sample and class imbalance in mind.

## Privacy and responsible use

- Only upload or process data that you are authorized to use.
- Do not commit raw records, patient-level reports, prediction CSVs, embeddings, secrets, or private identifiers.
- Before publishing the pretrained model, preprocessor, or Ridge artifact, confirm that their distribution is permitted by the dataset owner and institution.
- This software is an academic research demonstration. It is not a medical device and must not be used to diagnose, screen, triage, or make treatment decisions.

## License

No project license has been selected yet. Unless a license is added, all rights remain with the respective copyright holders. Check the licenses of dependencies before redistribution.
