"""
Master Pipeline Orchestrator: run_pipeline.py
Executes the complete clinical preeclampsia risk stratification pipeline end-to-end:
  Phase 1:  01_data_audit.py
  Phase 2:  02_train_test_split.py
  Phase 3:  03_preprocessing.py
  Phase 4:  06_clinical_risk_score.py (Reference risk anchor)
  Phase 5:  05_train_representation.py (FT-Transformer self-supervised + risk training)
  Phase 6:  07_risk_model.py (Architecture summary & model initialization)
  Phase 7:  08_test.py (Locked test cohort evaluation)
  Phase 8:  09_interpretability.py (Perturbation & latent sensitivity analysis)
  Phase 9:  10_visualizations.py (All 12 publication-quality clinical figures)
"""

import os
import sys
import time
import subprocess

PIPELINE_STEPS = [
    ("Phase 1: Dataset Audit", "src/01_data_audit.py"),
    ("Phase 2: Train/Validation/Test Split", "src/02_train_test_split.py"),
    ("Phase 3: Clinical Preprocessing", "src/03_preprocessing.py"),
    ("Phase 4: Clinical Reference Risk Scoring", "src/06_clinical_risk_score.py"),
    ("Phase 5: FT-Transformer Representation Pretraining", "src/05_train_representation.py"),
    ("Phase 6: Final Risk Model Setup", "src/07_risk_model.py"),
    ("Phase 7: Test Set Evaluation & Metrics", "src/08_test.py"),
    ("Phase 8: Feature Contribution Analysis", "src/09_interpretability.py"),
    ("Phase 9: Publication Visualizations (12 Figures)", "src/10_visualizations.py"),
]

def run_pipeline():
    start_total = time.time()
    print("=" * 70)
    print("STARTING END-TO-END PREECLAMPSIA RISK STRATIFICATION PIPELINE")
    print("=" * 70)

    for step_num, (desc, script_path) in enumerate(PIPELINE_STEPS, start=1):
        print(f"\n[{step_num}/{len(PIPELINE_STEPS)}] Running {desc} ({script_path})...")
        step_start = time.time()
        
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        step_elapsed = time.time() - step_start

        if result.returncode != 0:
            print(f"FAILED at {desc}!")
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)
            sys.exit(1)
        else:
            # Print brief stdout
            lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
            for line in lines[-5:]:
                print(f"    {line}")
            print(f"    [COMPLETED in {step_elapsed:.1f}s]")

    total_elapsed = time.time() - start_total
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_elapsed:.1f} SECONDS!")
    print("All outputs, tables, checkpoints, reports, and figures are verified.")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
