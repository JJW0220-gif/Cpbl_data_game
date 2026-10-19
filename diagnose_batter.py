"""Diagnose an existing batter validation split without fitting new models.

python diagnose_batter.py
python diagnose_batter.py --run training_outputs\\YOUR_RUN

Standard library only, Python 3.9+. Writes diagnostics_<timestamp> inside
the training run: grouped_errors.csv and validation_details.csv.
Bins are descriptive, not eligibility thresholds. Reports model and saved
training-mean baseline, with equal-case, target-denominator weighted and
major-PA-weighted errors. The latter matches the new training objective.
HR/PA also reports a zero-prediction baseline. No tuning uses these results.
Small bins and correlated same-player cases require cautious interpretation.
"""

import argparse
import csv
import json
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


DEFAULT_RUN = Path(__file__).resolve().parent / "training_outputs" / "20261008_230121_332186"
TARGETS = ("major_AVG", "major_OBP", "major_SLG", "major_HR_PA")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"Empty/malformed CSV: {path}")
    return rows


def number(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key}")
    return value


def count(row, key):
    value = number(row, key)
    if value < 0 or value != int(value):
        raise ValueError(f"Invalid count {key}")
    return value


def bin_label(value):
    for upper, label in ((10, "0-9"), (30, "10-29"), (60, "30-59"), (100, "60-99")):
        if value < upper:
            return label
    return "100+"


def errors(rows, column, weighted=False, weight_column="target_denominator"):
    weights = [row[weight_column] if weighted else 1 for row in rows]
    total = sum(weights)
    mae = sum(w * abs(r[column] - r["actual"]) for r, w in zip(rows, weights)) / total
    mse = sum(w * (r[column] - r["actual"]) ** 2 for r, w in zip(rows, weights)) / total
    return mae, math.sqrt(mse)


def run(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Training run is not complete")
    dataset = read_csv(folder / "dataset.csv")
    cases = {r["case_id"]: r for r in dataset}
    if len(cases) != len(dataset):
        raise ValueError("Duplicate case IDs")
    predictions = read_csv(folder / "validation_predictions.csv")
    details, seen = [], set()
    for prediction in predictions:
        key = (prediction["case_id"], prediction["target"])
        if key in seen or prediction["target"] not in TARGETS:
            raise ValueError(f"Duplicate/invalid prediction: {key}")
        seen.add(key)
        case = cases[prediction["case_id"]]
        target = prediction["target"]
        if case["split"] != "validation":
            raise ValueError("Prediction includes a training case")
        actual = number(prediction, "actual")
        if not math.isclose(actual, number(case, target), abs_tol=1e-10):
            raise ValueError("Prediction target differs from dataset")
        denominator = (
            count(case, "major_AB") if target in ("major_AVG", "major_SLG")
            else count(case, "major_PA") if target == "major_HR_PA"
            else sum(count(case, k) for k in ("major_AB", "major_BB", "major_HBP", "major_SF"))
        )
        if denominator <= 0:
            raise ValueError("Undefined target denominator")
        detail = dict(
            case_id=key[0], target=target, player_id=case["player_id"],
            player_name=case["player_name"], year=case["year"],
            minor_PA=count(case, "minor_PA"), major_PA=count(case, "major_PA"),
            target_denominator=denominator, actual=actual,
            predicted=number(prediction, "predicted"),
            baseline=number(prediction, "baseline"), zero=0.0,
        )
        detail["absolute_error"] = abs(detail["predicted"] - actual)
        details.append(detail)
    for target in TARGETS:
        expected = {r["case_id"] for r in dataset
                    if r["split"] == "validation" and r[target] != ""}
        actual_ids = {r["case_id"] for r in details if r["target"] == target}
        if actual_ids != expected:
            raise ValueError(f"Incomplete validation predictions for {target}")
    groups = defaultdict(list)
    for row in details:
        for dimension, label in (
            ("all", "all"), ("minor_PA", bin_label(row["minor_PA"])),
            ("major_PA", bin_label(row["major_PA"])),
            ("target_denominator", bin_label(row["target_denominator"])),
        ):
            groups[(row["target"], dimension, label)].append(row)
    results = []
    for (target, dimension, label), rows in sorted(groups.items()):
        record = dict(
            target=target, dimension=dimension, bin=label, cases=len(rows),
            players=len({r["player_id"] for r in rows}),
            denominator_sum=sum(r["target_denominator"] for r in rows),
            zero_actual_fraction=sum(r["actual"] == 0 for r in rows) / len(rows),
        )
        columns = ("predicted", "baseline", "zero") if target == "major_HR_PA" else ("predicted", "baseline")
        for column in columns:
            for suffix, weighted, weight_column in (
                ("", False, "target_denominator"),
                ("_weighted", True, "target_denominator"),
                ("_major_pa_weighted", True, "major_PA"),
            ):
                mae, rmse = errors(rows, column, weighted, weight_column)
                prefix = column + suffix
                record[prefix + "_MAE"] = mae
                record[prefix + "_RMSE"] = rmse
        results.append(record)
        if dimension == "all":
            logging.info("%s: n=%d MAE=%.6f weighted MAE=%.6f baseline MAE=%.6f",
                         target, len(rows), record["predicted_MAE"],
                         record["predicted_weighted_MAE"], record["baseline_MAE"])
            if target == "major_HR_PA":
                logging.info("HR/PA zero baseline MAE=%.6f; actual zero=%.1f%%",
                             record["zero_MAE"], 100 * record["zero_actual_fraction"])
    output = folder / ("diagnostics_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(exist_ok=False)
    for name, rows in (("grouped_errors.csv", results), ("validation_details.csv", details)):
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (output / name).open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    logging.info("Saved diagnostics: %s", output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run(args.run)
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        logging.error("Diagnosis failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
