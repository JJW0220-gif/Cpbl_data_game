"""Compare full-feature Ridge with same-metric single-feature Ridge.

python compare_single_batter.py
python compare_single_batter.py --comparison PATH --bootstrap 2000 --seed 42
python compare_single_batter.py --self-test

Uses the ORIGINAL saved 80/20 split and predictions from a completed weighted
comparison. Trains four single-feature, denominator-weighted Ridge pipelines:
minor AVG->major AVG, OBP->OBP, SLG->SLG, HR/PA->HR/PA.
Undefined X is imputed using training-fold medians, keeping identical cases.
Z-score and imputation fit training only; alpha selection uses the same grid
and training-fold splits as the full-feature weighted model.

Paired player-cluster bootstrap samples validation players with replacement,
retaining ALL their cases across years and transitions. Both methods use the
same sample in each replicate. Reports equal-case and denominator-weighted
MAE differences, with percentile 95% intervals.
Delta = full-feature MAE - comparator MAE: negative favors full-feature.
Intervals describe fixed-model validation uncertainty, NOT uncertainty from
refitting/tuning, future-year generalization, or multiple-comparison-adjusted
significance. Repeated use of this validation set is exploratory.
No model is selected using validation. New outputs never overwrite old runs.
Saved source predictions retain their weighting label: historical equal-case
Ridge or newer major-PA-weighted Ridge. Full/single models here still use
target-denominator weights, rather than major-PA weights for every target.
"""

import argparse
import json
import logging
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from compare_weighted_batter import ALPHAS, DEFAULT_RUN, denominator, evaluation
from train_batter import FEATURES, TARGETS, pipeline, require_columns


DEFAULT_COMPARISON = DEFAULT_RUN / "weighted_comparison_20261008_232403_903492"


def fit_single(frame, target, feature, alpha):
    if frame[feature].notna().sum() == 0:
        raise ValueError(f"{target}: single feature entirely missing in training fold")
    weights = denominator(frame, target)
    model = pipeline().set_params(ridge__alpha=alpha)
    model.fit(frame[[feature]], frame[target],
              ridge__sample_weight=weights / weights.mean())
    return model


def train_single(frame, target, seed):
    feature = target.replace("major_", "minor_", 1)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(frame))
    weights = denominator(frame, target)
    cv_rows = []
    for alpha in ALPHAS:
        oof = np.empty(len(frame))
        for training, validation in folds:
            model = fit_single(frame.iloc[training], target, feature, alpha)
            oof[validation] = model.predict(frame.iloc[validation][[feature]])
        cv_rows.append({"target": target, "alpha": alpha,
                        "weighted_oof_MAE": float(mean_absolute_error(
                            frame[target], oof, sample_weight=weights))})
    alpha = min(cv_rows, key=lambda r: r["weighted_oof_MAE"])["alpha"]
    return fit_single(frame, target, feature, alpha), feature, alpha, cv_rows


def bootstrap_differences(players, actual, full, comparators, weights, repeats, seed):
    players = np.asarray(players)
    actual, full, weights = (np.asarray(x, dtype=float) for x in (actual, full, weights))
    if (len(actual) == 0 or len(players) != len(actual) or len(full) != len(actual)
            or len(weights) != len(actual) or repeats <= 0
            or not np.isfinite(actual).all() or not np.isfinite(full).all()
            or not np.isfinite(weights).all() or (weights <= 0).any()):
        raise ValueError("Invalid bootstrap arrays/weights/repeats")
    unique, inverse = np.unique(players, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("Need at least two validation player clusters")
    group_counts = np.bincount(inverse)
    group_weights = np.bincount(inverse, weights=weights)
    full_errors = np.abs(full - actual)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(unique), size=(repeats, len(unique)))
    results = []
    for name, predicted in comparators.items():
        predicted = np.asarray(predicted, dtype=float)
        if predicted.shape != actual.shape or not np.isfinite(predicted).all():
            raise ValueError(f"Invalid comparator predictions: {name}")
        delta = full_errors - np.abs(predicted - actual)
        group_delta = np.bincount(inverse, weights=delta)
        group_weighted_delta = np.bincount(inverse, weights=weights * delta)
        for metric, numerator, denominators, observed in (
            ("MAE", group_delta, group_counts, delta.mean()),
            ("weighted_MAE", group_weighted_delta, group_weights,
             np.average(delta, weights=weights)),
        ):
            draws = numerator[samples].sum(axis=1) / denominators[samples].sum(axis=1)
            lower, upper = np.quantile(draws, [0.025, 0.975])
            results.append({
                "comparator": name, "metric": metric,
                "delta_full_minus_comparator": float(observed),
                "ci95_lower": float(lower), "ci95_upper": float(upper),
                "interval_entirely_below_zero": bool(upper < 0),
                "interval_entirely_above_zero": bool(lower > 0),
                "players": len(unique), "cases": len(actual), "replicates": repeats,
            })
    return results


def run(args):
    folder = args.run
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    comparison = json.loads((args.comparison / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or comparison.get("status") != "complete"
            or manifest.get("features") != FEATURES or comparison.get("features") != FEATURES
            or comparison.get("alphas") != ALPHAS
            or comparison.get("seed") != manifest.get("seed")):
        raise ValueError("Expected complete, compatible original and weighted runs")
    source_model = comparison.get("source_model_column", "equal_case_ridge")
    if source_model not in ("equal_case_ridge", "major_pa_ridge"):
        raise ValueError(f"Unsupported source model column: {source_model}")
    data = pd.read_csv(folder / "dataset.csv", dtype={"player_id": str})
    saved = pd.read_csv(args.comparison / "validation_predictions.csv", dtype={"player_id": str})
    require_columns(data, FEATURES + TARGETS + [
        "case_id", "split", "role", "birthday", "age_jan1", "year", "player_id",
        "major_AB", "major_PA", "major_BB", "major_HBP", "major_SF",
    ], "dataset")
    require_columns(saved, ["case_id", "target", "actual", "target_denominator",
                           "weighted_ridge", source_model, "train_mean_baseline",
                           "weighted_train_mean_baseline", "zero_baseline"], "weighted predictions")
    if (data["case_id"].duplicated().any() or saved.duplicated(["case_id", "target"]).any()
            or data["player_id"].isna().any() or data["role"].ne("batter").any()
            or not data["split"].isin(["train", "validation"]).all()):
        raise ValueError("Invalid cases, roles, player IDs or split")
    birthday = pd.to_datetime(data["birthday"], format="%Y-%m-%d", errors="raise")
    age = data["year"] - birthday.dt.year - (
        (birthday.dt.month > 1) | (birthday.dt.day > 1)).astype(int)
    if age.isna().any() or age.ne(data["age_jan1"]).any() or ((age < 0) | (age > 25)).any():
        raise ValueError("Invalid age: must be completed age <=25 on Jan 1")
    if (int(data["split"].eq("train").sum()) != manifest["train_cases"] or
            int(data["split"].eq("validation").sum()) != manifest["validation_cases"]):
        raise ValueError("Split counts do not match original run")
    metrics, predictions, intervals, cv_rows, models = {}, [], [], [], {}
    for index, target in enumerate(TARGETS):
        training = data.loc[data["split"].eq("train")].dropna(subset=[target])
        validation = data.loc[data["split"].eq("validation")].dropna(subset=[target])
        if len(training) < 5 or len(validation) < 2:
            raise ValueError(f"{target}: insufficient cases")
        old = saved.loc[saved["target"].eq(target)].set_index("case_id")
        if set(old.index) != set(validation["case_id"]):
            raise ValueError(f"{target}: saved validation cases differ")
        old = old.loc[validation["case_id"]]
        actual = validation[target].to_numpy()
        weights = denominator(validation, target)
        if (not np.allclose(actual, old["actual"], atol=1e-12, rtol=0)
                or not np.array_equal(weights, old["target_denominator"].to_numpy())):
            raise ValueError(f"{target}: targets/denominators differ from saved run")
        if (not np.allclose(old["train_mean_baseline"], training[target].mean(),
                            atol=1e-12, rtol=0) or
                not np.allclose(old["weighted_train_mean_baseline"],
                                np.average(training[target], weights=denominator(training, target)),
                                atol=1e-12, rtol=0)):
            raise ValueError(f"{target}: saved baselines do not match training data")
        model, feature, alpha, cv = train_single(training, target, manifest["seed"])
        cv_rows.extend(cv)
        estimates = {
            "single_feature_weighted_ridge": model.predict(validation[[feature]]),
            source_model: old[source_model].to_numpy(),
            "train_mean_baseline": old["train_mean_baseline"].to_numpy(),
            "weighted_train_mean_baseline": old["weighted_train_mean_baseline"].to_numpy(),
        }
        if target == "major_HR_PA":
            estimates["zero_baseline"] = old["zero_baseline"].to_numpy()
        full = old["weighted_ridge"].to_numpy()
        comparisons = bootstrap_differences(
            validation["player_id"].to_numpy(), actual, full, estimates, weights,
            args.bootstrap, args.seed + index,
        )
        intervals.extend(dict(target=target, **row) for row in comparisons)
        scores = {name: evaluation(actual, predicted, weights)
                  for name, predicted in dict(full_feature_weighted_ridge=full, **estimates).items()}
        metrics[target] = {
            "feature": feature, "alpha": alpha, "train_rows": len(training),
            "validation_rows": len(validation), "scores": scores,
            "undefined_single_feature_train_rows": int(training[feature].isna().sum()),
            "undefined_single_feature_validation_rows": int(validation[feature].isna().sum()),
            "out_of_range_single_predictions": int((
                (estimates["single_feature_weighted_ridge"] < 0) |
                (estimates["single_feature_weighted_ridge"] > (4 if target == "major_SLG" else 1))
            ).sum()),
        }
        models[target] = {"pipeline": model, "features": [feature], "target": target,
                          "age_definition": "Jan 1 completed age <=25"}
        for position, (_, row) in enumerate(validation.iterrows()):
            record = {k: row[k] for k in ("case_id", "player_id", "player_name", "year", "age_jan1")}
            record.update(target=target, actual=actual[position], target_denominator=weights[position],
                          full_feature_weighted_ridge=full[position])
            record.update({name: value[position] for name, value in estimates.items()})
            predictions.append(record)
        logging.info("%s: single alpha=%g, weighted MAE single=%.6f full=%.6f",
                     target, alpha, scores["single_feature_weighted_ridge"]["weighted_MAE"],
                     scores["full_feature_weighted_ridge"]["weighted_MAE"])
        for row in comparisons:
            if row["comparator"] == "single_feature_weighted_ridge":
                logging.info("%s %s delta(full-single)=%.6f CI95=[%.6f, %.6f]",
                             target, row["metric"], row["delta_full_minus_comparator"],
                             row["ci95_lower"], row["ci95_upper"])
        if alpha == max(ALPHAS):
            logging.warning("%s: single-feature alpha reached search upper bound", target)
    output = args.comparison / ("single_comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(exist_ok=False)
    state = {
        "status": "writing", "original_run": str(folder.resolve()),
        "weighted_run": str(args.comparison.resolve()), "training_seed": manifest["seed"],
        "source_model_column": source_model,
        "bootstrap_seed": args.seed, "bootstrap_replicates": args.bootstrap,
        "bootstrap_unit": "validation player ID; all years/cases retained, models fixed",
        "ci": "paired percentile 95%, unadjusted, delta full minus comparator",
        "warning": "Exploratory reused validation, not a final future/unseen-player test",
    }
    path = output / "manifest.json"
    def persist():
        path.write_text(json.dumps(state, indent=2, allow_nan=False), encoding="utf-8")
    persist()
    try:
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
        pd.DataFrame(intervals).to_csv(output / "bootstrap_comparisons.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(predictions).to_csv(output / "validation_predictions.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(cv_rows).to_csv(output / "training_cv.csv", index=False, encoding="utf-8-sig")
        for target, bundle in models.items():
            joblib.dump(bundle, output / f"{target}_single.joblib")
    except (OSError, ValueError) as exc:
        state.update(status="failed", error=str(exc))
        persist()
        raise
    state.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
    persist()
    logging.info("Saved: %s", output.resolve())


class ComparisonTests(unittest.TestCase):
    def test_identical_predictions(self):
        rows = bootstrap_differences(
            ["a", "a", "b"], [0., 1., 0.], [.2, .7, .3],
            {"same": np.array([.2, .7, .3])}, [1., 10., 2.], 100, 42)
        for row in rows:
            self.assertEqual(row["delta_full_minus_comparator"], 0)
            self.assertEqual(row["ci95_lower"], 0)
            self.assertEqual(row["ci95_upper"], 0)

    def test_difference_sign(self):
        rows = bootstrap_differences(
            ["a", "a", "b"], [0., 0., 0.], [0., 0., 0.],
            {"worse": np.array([1., 1., 1.])}, [1., 9., 2.], 100, 42)
        for row in rows:
            self.assertEqual(row["delta_full_minus_comparator"], -1)
            self.assertTrue(row["interval_entirely_below_zero"])

    def test_player_clusters_not_individual_cases(self):
        rows = bootstrap_differences(
            ["a", "a", "b"], [0., 0., 0.], [0., 0., 0.],
            {"other": np.array([1., 1., 0.])}, [1., 1., 1.], 1000, 42)
        self.assertEqual(rows[0]["players"], 2)
        self.assertAlmostEqual(rows[0]["delta_full_minus_comparator"], -2 / 3)
        self.assertEqual(rows[0]["ci95_lower"], -1)
        self.assertEqual(rows[0]["ci95_upper"], 0)

    def test_reproducible(self):
        inputs = (["a", "b"], [0., 0.], [0., .2], {"x": [1., .1]}, [1., 2.], 100, 42)
        self.assertEqual(bootstrap_differences(*inputs), bootstrap_differences(*inputs))

    def test_single_scaler_uses_training(self):
        frame = pd.DataFrame({"minor_AVG": [.1, .2, .3], "major_AVG": [.2, .3, .4],
                              "major_AB": [10, 20, 30]})
        model = fit_single(frame, "major_AVG", "minor_AVG", 1.)
        model.predict(pd.DataFrame({"minor_AVG": [100.]}))
        self.assertAlmostEqual(model.named_steps["scaler"].mean_[0], .2)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ComparisonTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if args.bootstrap < 100 or args.seed < 0:
        parser.error("bootstrap must be >=100 and seed must be nonnegative")
    try:
        run(args)
    except (OSError, ValueError, KeyError) as exc:
        logging.error("Comparison failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
