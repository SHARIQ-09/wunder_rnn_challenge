"""Submission Packager for Wunder RNN Challenge.

Validates and packages submission/ into a competition-ready submission.zip:
- Verifies solution.py and model.onnx presence and integrity.
- Tests that PredictionModel instantiates and runs a dummy predict() call.
- Confirms archive size <= 20 MB.
- Packages files cleanly at the root of submission.zip.
"""
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import DataPoint

MAX_ZIP_SIZE_MB = 20.0


def package_submission(
    submission_dir: str = str(ROOT_DIR / "submission"),
    output_zip: str = str(ROOT_DIR / "submission.zip"),
):
    sub_dir = Path(submission_dir)
    out_zip = Path(output_zip)

    solution_file = sub_dir / "solution.py"
    model_file = sub_dir / "model.onnx"

    print("=" * 60)
    print("      Wunder RNN Challenge -- Submission Packager")
    print("=" * 60)

    if not solution_file.exists():
        print(f"[ERROR] Missing required file: {solution_file}")
        sys.exit(1)
    if not model_file.exists():
        print(f"[ERROR] Missing required model: {model_file}")
        print("Please run export_onnx.py first to generate model.onnx.")
        sys.exit(1)

    # 1. Test local instantiation and predict() call
    print("Testing submission/solution.py locally...")
    if str(sub_dir) not in sys.path:
        sys.path.insert(0, str(sub_dir))

    try:
        from solution import PredictionModel
        predictor = PredictionModel()
        dummy_point = DataPoint(
            seq_ix=0,
            step_in_seq=0,
            need_prediction=True,
            state=np.zeros(112, dtype=np.float32),
        )
        dummy_pred = predictor.predict(dummy_point)
        print(f"  [SUCCESS] predict(data_point) output shape: {dummy_pred.shape}, values: {dummy_pred}")
        assert dummy_pred.shape == (2,)
        assert np.isfinite(dummy_pred).all()
    except Exception as e:
        print(f"[ERROR] Verification failed on solution.py: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 2. Package into ZIP
    print(f"\nCreating submission archive: {out_zip}...")
    files_to_pack = [solution_file, model_file]

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files_to_pack:
            zf.write(f, arcname=f.name)
            print(f"  Added: {f.name} ({f.stat().st_size / (1024 * 1024):.2f} MB)")

    zip_size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"\nFinal Archive Size: {zip_size_mb:.2f} MB / {MAX_ZIP_SIZE_MB} MB limit")

    if zip_size_mb > MAX_ZIP_SIZE_MB:
        print(f"[ERROR] Archive exceeds {MAX_ZIP_SIZE_MB} MB limit!")
        sys.exit(1)
    else:
        print("[SUCCESS] Submission package verified and ready for upload!")
    print("=" * 60)


if __name__ == "__main__":
    package_submission()
