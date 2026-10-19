"""Train four independent batter Ridge regressions (Python 3.9+).

Run from Windows or WSL:
    python train_batter.py
    python train_batter.py --episodes PATH --ages PATH --seed 42
    python train_batter.py --self-test

Dependencies: numpy, pandas, scikit-learn, joblib (see requirements_training.txt).
One row is a minor-league segment followed by a major-league segment.
Eligibility: completed age on January 1 of season year <=25, and minor_PA >=50
within the episode's minor segment. There is no major-segment PA threshold;
undefined targets (zero denominator) cannot be trained or evaluated.
The threshold includes 50; PA is never pooled across a player's episodes.
Raw episodes remain unchanged; rejected cases are saved with exclusion reasons.
X contains ALL currently reconstructable segment batting rates below, NOT
full-season advanced data. Missing source metrics (BABIP, BIP%, GB%, Whiff%,
wOBA, OPS+, RC, SB%, BB/K, WPA and RE24) are not replaced with season values.
BB/K is deliberately unavailable when K=0 and omitted from this version.
No age, identifiers, year, or major-league information is used as an X feature.

Random case-level 80/20 split is shared by all four targets, seed=42.
Each target excludes undefined y separately. Median imputation and Z-score
scaling are fitted within training folds, never on validation rows. Alpha is
selected by 5-fold shuffled CV within the 80% training portion, using pooled
out-of-fold major-PA-weighted MAE. For ALL four targets, Ridge sample weights
are major_PA / mean(major_PA in the current training fold). This is PA
weighting, not target-specific AB/OBP-denominator weighting. Preprocessing
remains unweighted and training-only; y remains in original units.
Cases with logically inconsistent counts are excluded and logged, not repaired.
The same player can occur on both sides: validation does NOT estimate accuracy
for unseen players or future seasons. No prediction clipping is performed.
Output: dataset/split audit, four fitted pipelines, validation predictions,
metrics (equal-case and major-PA-weighted, with ordinary and PA-weighted
train-mean baselines), coefficients and preprocessing values.
Pickle/joblib models must only be loaded from trusted sources.
"""

import argparse
import json
import logging
import math
import sys
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
AGE_DIR = ROOT / "data" / "rebas_20261008_202112_563439" / "age25_20261008_212142_214166"
DEFAULT_EPISODES = AGE_DIR / "transitions_20261008_214237_438309" / "episodes.csv"
FEATURES = [
    "minor_AVG", "minor_OBP", "minor_SLG", "minor_OPS", "minor_ISO",
    "minor_K_pct", "minor_BB_pct", "minor_HR_PA", "minor_HBP_PA",
    "minor_Double_AB", "minor_Triple_AB",
]
TARGETS = ["major_AVG", "major_OBP", "major_SLG", "major_HR_PA"]
MISSING_METRICS = [
    "BABIP", "BIP_pct", "GB_pct", "Whiff_pct", "wOBA", "OPSplus",
    "RC", "SB_pct", "BBK", "WPA", "RE24",
]
COUNTS = ["AB", "PA", "H", "BB", "HBP", "SF", "SO", "HR", "Double", "Triple"]
IDENTITY = ["year", "player_id", "transition_no"]
MIN_MINOR_PA = 50
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def require_columns(frame: pd.DataFrame, columns: list, label: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")


def divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator > 0)


def add_batting_rates(data: pd.DataFrame, level: str) -> pd.Series:
    """Reconstruct rates in place and return the inconsistent-count mask."""
    require_columns(data, [f"{level}_{count}" for count in COUNTS], level)
    for count in COUNTS:
        column = f"{level}_{count}"
        data[column] = pd.to_numeric(data[column], errors="raise")
        values = data[column]
        if (values.isna().any() or not np.isfinite(values).all()
                or (values < 0).any() or (values % 1 != 0).any()):
            raise ValueError(f"Invalid count column {column}; not filling source counts")
    c = {name: data[f"{level}_{name}"] for name in COUNTS}
    invalid = ((c["H"] > c["AB"]) | (c["AB"] > c["PA"]) |
               (c["Double"] + c["Triple"] + c["HR"] > c["H"]) |
               (c["SO"] > c["AB"]) |
               (c["AB"] + c["BB"] + c["HBP"] + c["SF"] > c["PA"]))
    tb = c["H"] + c["Double"] + 2 * c["Triple"] + 3 * c["HR"]
    data[f"{level}_AVG"] = divide(c["H"], c["AB"])
    data[f"{level}_OBP"] = divide(c["H"] + c["BB"] + c["HBP"],
                                 c["AB"] + c["BB"] + c["HBP"] + c["SF"])
    data[f"{level}_SLG"] = divide(tb, c["AB"])
    data[f"{level}_OPS"] = data[f"{level}_OBP"] + data[f"{level}_SLG"]
    data[f"{level}_ISO"] = divide(tb - c["H"], c["AB"])
    for metric, numerator, denominator, factor in (
        ("K_pct", "SO", "PA", 100), ("BB_pct", "BB", "PA", 100),
        ("HR_PA", "HR", "PA", 1), ("HBP_PA", "HBP", "PA", 1),
        ("Double_AB", "Double", "AB", 1), ("Triple_AB", "Triple", "AB", 1),
    ):
        data[f"{level}_{metric}"] = factor * divide(c[numerator], c[denominator])
    return invalid


def prepare(episodes: Path, ages: Path) -> tuple:
    data = pd.read_csv(episodes, dtype={"player_id": str})
    age_data = pd.read_csv(ages, dtype={"player_id": str})
    require_columns(data, IDENTITY + ["player_name", "role", "minor_start", "minor_end",
                                     "major_start", "major_end"] +
                    [f"{level}_{c}" for level in ("minor", "major") for c in COUNTS],
                    "episodes")
    require_columns(age_data, ["year", "player_id", "birthday", "role"], "ages")
    if data.empty or data["role"].ne("batter").any() or age_data["role"].ne("batter").any():
        raise ValueError("Inputs must be nonempty batter episodes and batter age metadata")
    for frame in (data, age_data):
        if frame[["year", "player_id"]].isna().any().any():
            raise ValueError("Missing player/year identifiers")
        year = pd.to_numeric(frame["year"], errors="raise")
        if ((year % 1 != 0) | (year < 1) | (year > 9999)).any():
            raise ValueError("Invalid season year")
        frame["year"] = year.astype(int)
    if data.duplicated(IDENTITY).any():
        raise ValueError("Duplicate episode identity")
    birthday_map = {}
    for row in age_data[["year", "player_id", "birthday"]].itertuples(index=False):
        if not isinstance(row.birthday, str):
            raise ValueError(f"Missing birthday for {row.player_id}")
        birthday = date.fromisoformat(row.birthday)
        key = (row.year, row.player_id)
        if key in birthday_map and birthday_map[key] != birthday:
            raise ValueError(f"Conflicting birthdays for {key}")
        birthday_map[key] = birthday
    computed_ages = []
    birthdays = []
    for row in data[["year", "player_id"]].itertuples(index=False):
        key = (row.year, row.player_id)
        if key not in birthday_map:
            raise ValueError(f"No birthday metadata for {key}")
        birthday = birthday_map[key]
        if birthday > date(row.year, 1, 1):
            raise ValueError(f"Birthday after season cutoff for {key}")
        computed_ages.append(row.year - birthday.year -
                             int((1, 1) < (birthday.month, birthday.day)))
        birthdays.append(birthday.isoformat())
    data["birthday"] = birthdays
    data["age_jan1"] = computed_ages
    excluded_age = int(data["age_jan1"].gt(25).sum())
    data = data.loc[data["age_jan1"].le(25)].copy()
    for column in ("minor_start", "minor_end", "major_start", "major_end"):
        data[column] = pd.to_datetime(data[column], format="%Y-%m-%d", errors="raise")
        if data[column].isna().any() or data[column].dt.year.ne(data["year"]).any():
            raise ValueError(f"Invalid date/year in {column}")
    if ((data["minor_start"] > data["minor_end"]) |
            (data["minor_end"] >= data["major_start"]) |
            (data["major_start"] > data["major_end"])).any():
        raise ValueError("Invalid episode chronology")
    inconsistent = pd.Series(False, index=data.index)
    for level in ("minor", "major"):
        inconsistent |= add_batting_rates(data, level)
    insufficient_pa = data["minor_PA"].lt(MIN_MINOR_PA)
    eligible = ~inconsistent & ~insufficient_pa & data[TARGETS].notna().any(axis=1)
    rejected = data.loc[~eligible].copy()
    rejected["reason"] = np.where(
        inconsistent.loc[rejected.index], "inconsistent_batting_counts",
        np.where(insufficient_pa.loc[rejected.index], "minor_pa_below_minimum",
                 "all_targets_undefined"),
    )
    if not rejected.empty:
        logging.warning("Excluding %d cases: %s", len(rejected),
                        rejected["reason"].value_counts().to_dict())
    data = data.loc[eligible].reset_index(drop=True)
    data["case_id"] = [f"{y}:{p}:{n}" for y, p, n in data[IDENTITY].itertuples(index=False, name=None)]
    return data, rejected, excluded_age


def pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ])


def fit_weighted_ridge(
    frame: pd.DataFrame, target: str, alpha: float, weights: np.ndarray,
) -> Pipeline:
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (len(frame),) or not np.isfinite(weights).all()
            or (weights <= 0).any()):
        raise ValueError(f"{target}: weights must be finite, positive and match rows")
    if frame[FEATURES].notna().sum().eq(0).any():
        raise ValueError(f"{target}: entirely missing feature in training fold")
    model = pipeline().set_params(ridge__alpha=alpha)
    model.fit(frame[FEATURES], frame[target],
              ridge__sample_weight=weights / weights.mean())
    return model


def select_weighted_alpha(
    frame: pd.DataFrame, target: str, seed: int, weights: np.ndarray,
) -> tuple:
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (len(frame),) or not np.isfinite(weights).all()
            or (weights <= 0).any()):
        raise ValueError(f"{target}: weights must be finite, positive and match rows")
    folds = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(frame))
    results = []
    for alpha in ALPHAS:
        oof = np.empty(len(frame))
        for train_indices, val_indices in folds:
            model = fit_weighted_ridge(
                frame.iloc[train_indices], target, alpha, weights[train_indices],
            )
            oof[val_indices] = model.predict(frame.iloc[val_indices][FEATURES])
        results.append({"alpha": alpha, "weighted_oof_MAE": float(
            mean_absolute_error(frame[target], oof, sample_weight=weights))})
    best = min(results, key=lambda row: row["weighted_oof_MAE"])
    return best["alpha"], results


def scores(
    actual: pd.Series, predicted: np.ndarray, weights: Optional[np.ndarray] = None,
) -> dict:
    return {
        "MAE": float(mean_absolute_error(actual, predicted, sample_weight=weights)),
        "RMSE": float(math.sqrt(mean_squared_error(actual, predicted, sample_weight=weights))),
        "R2": float(r2_score(actual, predicted, sample_weight=weights))
        if len(actual) >= 2 and actual.nunique() > 1 else None,
    }


def train(args: argparse.Namespace) -> None:
    data, rejected, excluded_age = prepare(args.episodes, args.ages)
    if len(data) < 10:
        raise ValueError("Need at least 10 eligible cases")
    train_idx, val_idx = train_test_split(
        np.arange(len(data)), test_size=0.2, random_state=args.seed, shuffle=True
    )
    data["split"] = "train"
    data.loc[val_idx, "split"] = "validation"
    overlap = set(data.loc[train_idx, "player_id"]) & set(data.loc[val_idx, "player_id"])
    logging.warning("Random case split: %d players appear in BOTH subsets; not a future/unseen-player test.",
                    len(overlap))
    fits, metrics, predictions, coefficient_rows, scaler_rows = {}, {}, [], [], []
    cv_rows = []
    for target in TARGETS:
        tr = data.loc[train_idx].dropna(subset=[target])
        va = data.loc[val_idx].dropna(subset=[target])
        if len(tr) < 5 or len(va) < 2:
            raise ValueError(f"{target}: insufficient train/validation rows")
        if tr[FEATURES].notna().sum().eq(0).any():
            raise ValueError(f"{target}: a feature is entirely missing in training")
        train_weights = tr["major_PA"].to_numpy(dtype=float)
        val_weights = va["major_PA"].to_numpy(dtype=float)
        alpha, cv = select_weighted_alpha(tr, target, args.seed, train_weights)
        cv_rows.extend(dict(target=target, **row) for row in cv)
        model = fit_weighted_ridge(tr, target, alpha, train_weights)
        pred = model.predict(va[FEATURES])
        baseline = np.full(len(va), float(tr[target].mean()))
        weighted_baseline = np.full(
            len(va), float(np.average(tr[target], weights=train_weights)),
        )
        upper = 4.0 if target == "major_SLG" else 1.0
        metrics[target] = {
            "train_rows": len(tr), "validation_rows": len(va),
            "undefined_train_targets": len(train_idx) - len(tr),
            "undefined_validation_targets": len(val_idx) - len(va),
            "alpha": alpha,
            "training_cv_weighted_MAE": min(row["weighted_oof_MAE"] for row in cv),
            "validation": scores(va[target], pred),
            "validation_major_pa_weighted": scores(va[target], pred, val_weights),
            "train_mean_baseline": scores(va[target], baseline),
            "train_mean_baseline_major_pa_weighted": scores(va[target], baseline, val_weights),
            "pa_weighted_train_mean_baseline": scores(va[target], weighted_baseline),
            "pa_weighted_train_mean_baseline_major_pa_weighted": scores(
                va[target], weighted_baseline, val_weights),
            "training_major_pa_sum": float(train_weights.sum()),
            "validation_major_pa_sum": float(val_weights.sum()),
            "out_of_range_predictions": int(((pred < 0) | (pred > upper)).sum()),
        }
        fitted = model.named_steps
        fits[target] = {
            "pipeline": model, "features": FEATURES, "target": target,
            "age_definition": "Jan 1 completed age <=25", "seed": args.seed,
            "min_minor_segment_pa": MIN_MINOR_PA,
            "min_major_segment_pa": None,
            "sample_weighting": "major_PA",
            "weight_normalization": "current training fold mean",
        }
        for feature, coef, mean, scale, median in zip(
            FEATURES, fitted["ridge"].coef_, fitted["scaler"].mean_,
            fitted["scaler"].scale_, fitted["imputer"].statistics_,
        ):
            coefficient_rows.append({"target": target, "feature": feature,
                                     "standardized_coefficient": float(coef),
                                     "intercept": float(fitted["ridge"].intercept_)})
            scaler_rows.append({"target": target, "feature": feature,
                                "training_median": float(median), "training_mean": float(mean),
                                "training_scale": float(scale)})
        for (_, row), estimate, base, weighted_base in zip(
            va.iterrows(), pred, baseline, weighted_baseline,
        ):
            predictions.append({
                "case_id": row["case_id"], "year": row["year"],
                "player_id": row["player_id"], "player_name": row["player_name"],
                "age_jan1": row["age_jan1"], "target": target,
                "actual": row[target], "predicted": float(estimate),
                "baseline": float(base),
                "pa_weighted_baseline": float(weighted_base),
                "major_PA": row["major_PA"],
            })
        logging.info("%s: alpha=%g, validation MAE=%.6f, PA-weighted MAE=%.6f",
                     target, metrics[target]["alpha"],
                     metrics[target]["validation"]["MAE"],
                     metrics[target]["validation_major_pa_weighted"]["MAE"])
        if alpha == max(ALPHAS):
            logging.warning("%s: best alpha is search upper bound", target)
    output = args.output or ROOT / "training_outputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "writing", "features": FEATURES, "targets": TARGETS,
        "unavailable_metrics": MISSING_METRICS,
        "episodes": str(args.episodes.resolve()), "ages": str(args.ages.resolve()),
        "age_excluded_cases": excluded_age, "excluded_cases": len(rejected),
        "min_minor_segment_pa": MIN_MINOR_PA,
        "min_major_segment_pa": None,
        "pa_filter": "minor segment PA >=50; no pooling; undefined targets excluded",
        "excluded_case_reasons": rejected["reason"].value_counts().to_dict(),
        "eligible_cases": len(data), "train_cases": len(train_idx), "validation_cases": len(val_idx),
        "seed": args.seed, "split": "random case-level 80/20; validation size rounded up",
        "overlapping_player_ids": sorted(overlap), "sample_weighting": "major_PA",
        "weight_normalization": "current training fold mean",
        "alpha_selection": "5-fold training-only pooled major-PA-weighted OOF MAE",
        "alphas": ALPHAS,
        "versions": {"python": sys.version, "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
    }
    def save_manifest() -> None:
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    save_manifest()
    try:
        data.to_csv(output / "dataset.csv", index=False, encoding="utf-8-sig")
        rejected.to_csv(output / "excluded_cases.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(predictions).to_csv(output / "validation_predictions.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(coefficient_rows).to_csv(output / "coefficients.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(scaler_rows).to_csv(output / "preprocessing.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(cv_rows).to_csv(output / "training_cv.csv", index=False, encoding="utf-8-sig")
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
        for target, bundle in fits.items():
            joblib.dump(bundle, output / f"{target}.joblib")
    except (OSError, ValueError) as exc:
        manifest.update(status="failed", error=str(exc))
        save_manifest()
        raise
    manifest.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
    save_manifest()
    logging.info("Saved four models: %s", output.resolve())


class TrainingTests(unittest.TestCase):
    def test_prepare_age_counts_and_targets(self) -> None:
        rows = []
        for player, birthday, pa in (
            ("a", "1999-01-02", 50), ("b", "1999-01-01", 50),
            ("c", "2000-01-01", 40),
        ):
            row = dict(year=2025, player_id=player, player_name=player,
                       transition_no=1, role="batter",
                       minor_start="2025-04-01", minor_end="2025-04-02",
                       major_start="2025-04-03", major_end="2025-04-04")
            for level in ("minor", "major"):
                row.update({f"{level}_{count}": 0 for count in COUNTS})
                row.update({f"{level}_AB": 40, f"{level}_PA": 50,
                            f"{level}_H": 20, f"{level}_BB": 10,
                            f"{level}_Double": 10, f"{level}_SO": 10})
            row["minor_PA"] = pa
            rows.append((row, birthday))
        with tempfile.TemporaryDirectory() as folder:
            episodes = Path(folder) / "episodes.csv"
            ages = Path(folder) / "ages.csv"
            pd.DataFrame([r for r, _ in rows]).to_csv(episodes, index=False)
            pd.DataFrame([
                {"year": 2025, "player_id": r["player_id"], "role": "batter", "birthday": b}
                for r, b in rows
            ]).to_csv(ages, index=False)
            data, rejected, excluded_age = prepare(episodes, ages)
        self.assertEqual(excluded_age, 1)
        self.assertEqual(data["player_id"].tolist(), ["a"])
        self.assertEqual(data["age_jan1"].tolist(), [25])
        self.assertEqual(rejected["reason"].tolist(), ["inconsistent_batting_counts"])
        self.assertAlmostEqual(data.iloc[0]["major_AVG"], 0.5)
        self.assertAlmostEqual(data.iloc[0]["major_OBP"], 0.6)
        self.assertAlmostEqual(data.iloc[0]["major_SLG"], 0.75)
        self.assertEqual(data.iloc[0]["major_HR_PA"], 0)

    def test_segment_pa_threshold_is_inclusive_and_per_episode(self) -> None:
        rows = []
        pairs = [(50, 50), (49, 50), (50, 49), (49, 49),
                 (51, 50), (50, 51), (25, 25), (25, 25),
                 (50, 1), (50, 0)]
        for transition, (minor_pa, major_pa) in enumerate(pairs, start=1):
            row = dict(year=2025, player_id="a", player_name="a",
                       transition_no=transition, role="batter",
                       minor_start="2025-04-01", minor_end="2025-04-02",
                       major_start="2025-04-03", major_end="2025-04-04")
            for level, pa in (("minor", minor_pa), ("major", major_pa)):
                row.update({f"{level}_{count}": 0 for count in COUNTS})
                row.update({f"{level}_AB": pa, f"{level}_PA": pa,
                            f"{level}_H": min(10, pa)})
            rows.append(row)
        with tempfile.TemporaryDirectory() as folder:
            episodes = Path(folder) / "episodes.csv"
            ages = Path(folder) / "ages.csv"
            pd.DataFrame(rows).to_csv(episodes, index=False)
            pd.DataFrame([{"year": 2025, "player_id": "a", "role": "batter",
                           "birthday": "2000-01-01"}]).to_csv(ages, index=False)
            data, rejected, excluded_age = prepare(episodes, ages)
        self.assertEqual(excluded_age, 0)
        self.assertEqual(data["transition_no"].tolist(), [1, 3, 5, 6, 9])
        self.assertEqual(rejected["transition_no"].tolist(), [2, 4, 7, 8, 10])
        self.assertEqual(rejected["reason"].tolist(),
                         ["minor_pa_below_minimum"] * 4 + ["all_targets_undefined"])

    def test_major_pa_changes_fit_and_normalizes_weights(self) -> None:
        data = pd.DataFrame({feature: [0., 0.] for feature in FEATURES})
        data["major_AVG"] = [0., 1.]
        weights = np.array([10., 100.])
        weighted = fit_weighted_ridge(data, "major_AVG", 1., weights)
        equal = fit_weighted_ridge(data, "major_AVG", 1., np.ones(2))
        self.assertAlmostEqual(weighted.predict(data[FEATURES])[0], 100 / 110)
        self.assertAlmostEqual(equal.predict(data[FEATURES])[0], 0.5)
        data.loc[1, FEATURES] = 1.
        weighted = fit_weighted_ridge(data, "major_AVG", 1., weights)
        scaled = fit_weighted_ridge(data, "major_AVG", 1., weights * 10)
        np.testing.assert_allclose(weighted.predict(data[FEATURES]),
                                   scaled.predict(data[FEATURES]))
        self.assertAlmostEqual(weighted.named_steps["scaler"].mean_[0], 0.5)
        self.assertNotIn("major_PA", FEATURES)

    def test_weighted_scores(self) -> None:
        actual = pd.Series([0., 1.])
        predicted = np.array([0., 0.])
        self.assertEqual(scores(actual, predicted)["MAE"], 0.5)
        self.assertAlmostEqual(
            scores(actual, predicted, np.array([10., 100.]))["MAE"], 100 / 110,
        )

    def test_weighted_cv_uses_training_fold_means(self) -> None:
        data = pd.DataFrame({feature: np.zeros(10) for feature in FEATURES})
        data["major_AVG"] = np.linspace(0., 1., 10)
        weights = np.arange(1., 11.)
        alpha, cv = select_weighted_alpha(data, "major_AVG", 42, weights)
        oof = np.empty(10)
        for training, validation in KFold(5, shuffle=True, random_state=42).split(data):
            oof[validation] = np.average(data.iloc[training]["major_AVG"],
                                         weights=weights[training])
        expected = mean_absolute_error(data["major_AVG"], oof, sample_weight=weights)
        for row in cv:
            self.assertAlmostEqual(row["weighted_oof_MAE"], expected)
        self.assertIn(alpha, ALPHAS)

    def test_invalid_weights_are_rejected(self) -> None:
        data = pd.DataFrame({feature: [0., 1.] for feature in FEATURES})
        data["major_AVG"] = [0., 1.]
        for weights in (np.array([0., 1.]), np.array([-1., 1.]),
                        np.array([np.nan, 1.]), np.array([1.])):
            with self.assertRaises(ValueError):
                fit_weighted_ridge(data, "major_AVG", 1., weights)

    def test_scaler_does_not_fit_validation(self) -> None:
        model = pipeline()
        model.fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), [0.1, 0.2, 0.3])
        model.predict(pd.DataFrame({"x": [1000.0]}))
        self.assertAlmostEqual(model.named_steps["scaler"].mean_[0], 2.0)

    def test_zero_denominator_is_missing(self) -> None:
        result = divide(pd.Series([1, 0]), pd.Series([2, 0]))
        self.assertEqual(result.iloc[0], 0.5)
        self.assertTrue(pd.isna(result.iloc[1]))

    def test_shared_split(self) -> None:
        tr, va = train_test_split(np.arange(100), test_size=0.2, random_state=42)
        self.assertEqual(len(tr), 80)
        self.assertEqual(len(va), 20)
        self.assertFalse(set(tr) & set(va))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--ages", type=Path, default=AGE_DIR / "filtered.csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TrainingTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    try:
        train(args)
    except (OSError, ValueError, KeyError) as exc:
        logging.error("Training failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
