"""Compare denominator-weighted Ridge with an EXISTING saved training run.

python compare_weighted_batter.py
python compare_weighted_batter.py --run training_outputs\\YOUR_RUN
python compare_weighted_batter.py --self-test

Uses the saved dataset and exact train/validation split; no additional cases
are removed based on sample-size thresholds. Age must be <=25 on January 1.
For the minor PA >=50 cohort, pass --run with a newly filtered training run.
The default path remains the historical unfiltered run.
New training runs use major-PA weights for every target; they are labelled
major_pa_ridge here, NOT equal_case_ridge. This script is an optional comparison
against target-denominator weighting, not the primary PA-weighted training.
Weights: AVG/SLG=major AB, OBP=AB+BB+HBP+SF, HR/PA=major PA.
Weights are divided by their training-fold mean to keep alpha comparable.
Target exposure is a training weight, NEVER an input feature.
Training-only median imputation and unweighted Z-score scaling are retained.
Alpha is selected by denominator-weighted out-of-fold MAE within training.
The 20% validation set is NOT used to select alpha.
Reports equal-case AND denominator-weighted MAE/RMSE for both models,
original training-mean and weighted training-mean baselines, and HR zero.
Outputs to a NEW weighted_comparison_<timestamp> directory inside the run.
This is validation-driven exploratory analysis, not an untouched final test.
Weighting favors players with more opportunities; it does not prove that
short segments are invalid or eliminate player-level correlation.
"""

import argparse
import json
import logging
import math
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from train_batter import (
    ALPHAS, FEATURES, TARGETS, fit_weighted_ridge, require_columns,
    select_weighted_alpha,
)


DEFAULT_RUN = Path(__file__).resolve().parent / "training_outputs" / "20261008_230121_332186"


def denominator(frame, target):
    if target in ("major_AVG", "major_SLG"):
        weights = frame["major_AB"].to_numpy(dtype=float)
    elif target == "major_HR_PA":
        weights = frame["major_PA"].to_numpy(dtype=float)
    elif target == "major_OBP":
        weights = frame[["major_AB", "major_BB", "major_HBP", "major_SF"]].sum(axis=1).to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown target: {target}")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError(f"{target}: nonpositive/nonfinite denominator")
    return weights


def fit_weighted(frame, target, alpha):
    return fit_weighted_ridge(frame, target, alpha, denominator(frame, target))


def select_alpha(frame, target, seed):
    return select_weighted_alpha(frame, target, seed, denominator(frame, target))


def evaluation(actual, predicted, weights):
    if not np.isfinite(predicted).all():
        raise ValueError("Nonfinite predictions")
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(math.sqrt(mean_squared_error(actual, predicted))),
        "weighted_MAE": float(mean_absolute_error(actual, predicted, sample_weight=weights)),
        "weighted_RMSE": float(math.sqrt(mean_squared_error(actual, predicted, sample_weight=weights))),
    }


def run(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("features") != FEATURES:
        raise ValueError("Expected a complete compatible training run")
    source_weighting = manifest.get("sample_weighting")
    if source_weighting not in ("equal per case", "major_PA"):
        raise ValueError(f"Unsupported source sample weighting: {source_weighting}")
    source_model = "major_pa_ridge" if source_weighting == "major_PA" else "equal_case_ridge"
    data = pd.read_csv(folder / "dataset.csv", dtype={"player_id": str})
    saved = pd.read_csv(folder / "validation_predictions.csv", dtype={"player_id": str})
    require_columns(data, FEATURES + TARGETS + [
        "case_id", "split", "age_jan1", "birthday", "year", "role",
        "major_AB", "major_BB", "major_HBP", "major_SF", "major_PA",
    ], "dataset")
    require_columns(saved, ["case_id", "target", "actual", "predicted", "baseline"], "predictions")
    if (data["case_id"].duplicated().any() or
            saved.duplicated(["case_id", "target"]).any()):
        raise ValueError("Duplicate cases/predictions")
    if data["role"].ne("batter").any() or data["age_jan1"].isna().any():
        raise ValueError("Invalid batter age metadata")
    birthdays = pd.to_datetime(data["birthday"], format="%Y-%m-%d", errors="raise")
    expected_age = data["year"] - birthdays.dt.year - (
        (birthdays.dt.month > 1) | (birthdays.dt.day > 1)).astype(int)
    if (expected_age.isna().any() or expected_age.ne(data["age_jan1"]).any()
            or (expected_age < 0).any() or (expected_age > 25).any()):
        raise ValueError("Age must match birthday and be <=25 on January 1")
    if not data["split"].isin(["train", "validation"]).all():
        raise ValueError("Invalid saved split")
    if (int(data["split"].eq("train").sum()) != manifest["train_cases"] or
            int(data["split"].eq("validation").sum()) != manifest["validation_cases"]):
        raise ValueError("Saved split counts differ from manifest")
    results, predictions, models, cv_records = {}, [], {}, []
    for target in TARGETS:
        tr = data.loc[data["split"].eq("train")].dropna(subset=[target])
        va = data.loc[data["split"].eq("validation")].dropna(subset=[target])
        if len(tr) < 5 or len(va) < 2:
            raise ValueError(f"{target}: insufficient samples")
        old = saved.loc[saved["target"].eq(target)].set_index("case_id")
        if set(old.index) != set(va["case_id"]):
            raise ValueError(f"{target}: validation cases differ")
        old = old.loc[va["case_id"]]
        if not np.allclose(old["actual"].to_numpy(), va[target].to_numpy(), atol=1e-12, rtol=0):
            raise ValueError(f"{target}: targets differ from saved predictions")
        weights = denominator(va, target)
        alpha, cv = select_alpha(tr, target, manifest["seed"])
        for record in cv:
            cv_records.append(dict(target=target, **record))
        model = fit_weighted(tr, target, alpha)
        pred = model.predict(va[FEATURES])
        weighted_mean = float(np.average(tr[target], weights=denominator(tr, target)))
        estimates = {
            source_model: old["predicted"].to_numpy(),
            "weighted_ridge": pred,
            "train_mean_baseline": old["baseline"].to_numpy(),
            "weighted_train_mean_baseline": np.full(len(va), weighted_mean),
        }
        if target == "major_HR_PA":
            estimates["zero_baseline"] = np.zeros(len(va))
        results[target] = {
            "train_rows": len(tr), "validation_rows": len(va), "alpha": alpha,
            "scores": {name: evaluation(va[target], values, weights)
                       for name, values in estimates.items()},
            "out_of_range_weighted_predictions": int(
                ((pred < 0) | (pred > (4 if target == "major_SLG" else 1))).sum()),
        }
        models[target] = {"pipeline": model, "features": FEATURES, "target": target,
                          "weighting": "target denominator / training mean denominator"}
        for pos, (_, row) in enumerate(va.iterrows()):
            record = {k: row[k] for k in (
                "case_id", "player_id", "player_name", "year", "age_jan1", "minor_PA", "major_PA"
            )}
            record.update(target=target, actual=row[target], target_denominator=weights[pos])
            record.update({name: values[pos] for name, values in estimates.items()})
            predictions.append(record)
        score = results[target]["scores"]["weighted_ridge"]
        logging.info("%s alpha=%g: MAE=%.6f, weighted MAE=%.6f",
                     target, alpha, score["MAE"], score["weighted_MAE"])
        if alpha == max(ALPHAS):
            logging.warning("%s: best alpha is search upper bound", target)
    output = folder / ("weighted_comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(exist_ok=False)
    state = {"status": "writing", "source_run": str(folder.resolve()),
             "source_model_column": source_model, "source_sample_weighting": source_weighting,
             "split": "exact saved case-level 80/20", "seed": manifest["seed"],
             "features": FEATURES, "alphas": ALPHAS,
             "cv_selection": "training-only denominator-weighted pooled out-of-fold MAE",
             "warning": "Exploratory validation comparison; repeated players may overlap"}
    path = output / "manifest.json"
    def persist():
        path.write_text(json.dumps(state, indent=2, allow_nan=False), encoding="utf-8")
    persist()
    try:
        (output / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
        pd.DataFrame(predictions).to_csv(output / "validation_predictions.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(cv_records).to_csv(output / "training_cv.csv", index=False, encoding="utf-8-sig")
        for target, model in models.items():
            joblib.dump(model, output / f"{target}.joblib")
    except (OSError, ValueError) as exc:
        state.update(status="failed", error=str(exc))
        persist()
        raise
    state.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
    persist()
    logging.info("Saved comparison: %s", output.resolve())


class WeightedTests(unittest.TestCase):
    def test_denominators(self):
        data = pd.DataFrame({"major_AB": [4], "major_PA": [6],
                             "major_BB": [1], "major_HBP": [1], "major_SF": [0]})
        for target, expected in zip(TARGETS, [4, 6, 4, 6]):
            self.assertEqual(denominator(data, target)[0], expected)

    def test_weighted_error(self):
        result = evaluation([0, 1], np.array([0, 0]), np.array([9, 1]))
        self.assertEqual(result["MAE"], 0.5)
        self.assertEqual(result["weighted_MAE"], 0.1)

    def test_scaling_and_weights(self):
        data = pd.DataFrame({f: [1., 2., 3.] for f in FEATURES})
        data["major_AVG"] = [0., 0., 1.]
        data["major_AB"] = [1, 1, 100]
        model = fit_weighted(data, "major_AVG", 1000000)
        self.assertAlmostEqual(model.named_steps["scaler"].mean_[0], 2)
        self.assertGreater(model.predict(data[FEATURES]).mean(), 0.9)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(WeightedTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    try:
        run(args.run)
    except (OSError, ValueError, KeyError) as exc:
        logging.error("Comparison failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
