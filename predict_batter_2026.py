"""Predict all 2026 batters with minor-season PA >50 and Jan-1 age <=25.

First explicitly crawl 2026 with rebas_crawler.py (both levels, batter role).
Then run in your training environment:
    python predict_batter_2026.py --data DATA_FOLDER --run TRUSTED_MODEL_FOLDER
    python predict_batter_2026.py --self-test

Uses completed crawler outputs and existing major-PA-weighted models; no fit.
Birthdays are fetched from public APIs with robots checks.
Major appearances and PA do not affect eligibility. Major PA/games from the
saved leaderboard are reference fields only; absent records remain blank,
not inferred as zero. Inconsistent major records are flagged, not used to
exclude otherwise eligible minor batters. --cache can reuse profile JSON.

Features are reconstructed from full 2026 minor-season counts, summed across
teams, using the training formulas. This differs from the training episode
window. Full-season minor stats may overlap or follow major appearances.
Predictions are exploratory projections of major batting rates, NOT observed
2026 outcomes, promotion probabilities, or a scored test set. No MAE/R2 is
computed, and no prediction clipping is applied. Out-of-range rates are flagged.
Only load joblib models from a trusted run, using its scikit-learn version.
Outputs never overwrite an existing folder. Raw responses and exclusions are
retained alongside predictions.csv and manifest.json.
"""

import argparse
import json
import logging
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.robotparser import RobotFileParser

import joblib
import numpy as np
import pandas as pd
import sklearn

from filter_age import age_jan1, birthday_from_profile
from rebas_crawler import BASE_URL, SEASONS, Client, DataError, positive, write_json
from rebas_transitions import load_raw
from train_batter import COUNTS, FEATURES, TARGETS, add_batting_rates, require_columns


YEAR = 2026
WARNING = (
    "Exploratory full-minor-season inference, not a training episode. "
    "Major PA is unrestricted. Major outcomes are not scored; overlapping "
    "minor/major season windows are not a prospective validation."
)


def aggregate_season(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    required = ["year", "level", "role", "season_id", "player_id", "player_name",
                "team_id", "team_abbr", "standard_games"]
    require_columns(frame, required + [f"standard_{c}" for c in COUNTS], level)
    if (frame.empty or frame[required].isna().any().any()
            or frame["year"].ne(YEAR).any() or frame["level"].ne(level).any()
            or frame["role"].ne("batter").any()
            or frame["season_id"].ne(SEASONS[level][YEAR]).any()
            or frame.duplicated(["player_id", "team_id"]).any()):
        raise ValueError(f"Invalid/empty {YEAR} {level} season data")
    for player in frame["player_id"]:
        if not player or not all(c.isascii() and (c.isalnum() or c in "-_") for c in player):
            raise ValueError(f"Unsafe player ID: {player!r}")
    if frame.groupby("player_id")["player_name"].nunique().gt(1).any():
        raise ValueError(f"Conflicting player names in {level}")
    frame = frame.rename(columns={
        f"standard_{c}": f"{level}_{c}" for c in COUNTS + ["games"]
    }).copy()
    frame[f"{level}_invalid"] = add_batting_rates(frame, level)
    games = pd.to_numeric(frame[f"{level}_games"], errors="raise")
    if (games.isna().any() or not np.isfinite(games).all()
            or games.lt(0).any() or games.mod(1).ne(0).any()):
        raise ValueError(f"Invalid {level} games")
    frame[f"{level}_games"] = games
    aggregations = {f"{level}_{c}": "sum" for c in COUNTS + ["games"]}
    aggregations.update({
        "player_name": "first",
        "team_abbr": lambda values: "|".join(sorted(set(values))),
        f"{level}_invalid": "any",
    })
    result = frame.groupby("player_id", as_index=False).agg(aggregations)
    result[f"{level}_invalid"] |= add_batting_rates(result, level)
    return result


def select_candidates(minor: pd.DataFrame, major: pd.DataFrame) -> tuple:
    joined = minor.merge(
        major[["player_id", "major_PA", "major_games", "major_invalid"]],
        on="player_id", how="left", validate="one_to_one", indicator=True,
    )
    joined["major_table_status"] = np.where(
        joined["_merge"].eq("both"), "listed", "not_listed",
    )
    joined["exclusion_reason"] = np.select(
        [joined["minor_invalid"], joined["minor_PA"].le(50)],
        ["inconsistent_batting_counts", "minor_pa_not_over_50"],
        default="",
    )
    rejected = joined.loc[joined["exclusion_reason"].ne("")].copy()
    candidates = joined.loc[joined["exclusion_reason"].eq("")].copy()
    return candidates.drop(columns="_merge"), rejected.drop(columns="_merge")


def predict(frame: pd.DataFrame, models: dict) -> pd.DataFrame:
    result = frame.copy()
    for target in TARGETS:
        estimates = np.asarray(models[target].predict(frame[FEATURES]), dtype=float)
        if estimates.shape != (len(frame),) or not np.isfinite(estimates).all():
            raise ValueError(f"{target}: invalid predictions")
        result[f"predicted_{target}"] = estimates
        upper = 4 if target == "major_SLG" else 1
        outside = (estimates < 0) | (estimates > upper)
        result[f"{target}_out_of_range"] = outside
        if outside.any():
            logging.warning("%s: %d out-of-range predictions, not clipped", target, outside.sum())
    result["year"] = YEAR
    result["evaluation_status"] = "prediction_only_not_scored"
    result["feature_window"] = "full_2026_minor_regular_season"
    return result


def run(args: argparse.Namespace) -> None:
    source = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    if (source.get("status") != "complete" or YEAR not in source.get("years", [])
            or not {"minor", "major"}.issubset(source.get("levels", []))
            or "batter" not in source.get("roles", [])
            or source.get("source") != BASE_URL):
        raise ValueError("Need a complete Rebas crawl containing both 2026 batter levels")
    tables = {}
    for level in ("minor", "major"):
        name = f"{YEAR}_{level}_batter.csv"
        frame = pd.read_csv(args.data / name, dtype={"player_id": str, "team_id": str})
        outputs = [entry for entry in source["outputs"] if entry["file"] == name]
        if len(outputs) != 1 or outputs[0]["rows"] != len(frame):
            raise ValueError(f"{name}: row count differs from crawler manifest")
        tables[level] = aggregate_season(frame, level)
    candidates, rejected = select_candidates(tables["minor"], tables["major"])
    logging.info("Pre-age candidates: %d; exclusions: %s", len(candidates),
                 rejected["exclusion_reason"].value_counts().to_dict())
    invalid_major = candidates["major_invalid"].eq(True)
    if invalid_major.any():
        logging.warning("Inconsistent major reference counts for %s; not an eligibility filter",
                        candidates.loc[invalid_major, "player_id"].tolist())
    model_state = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    if (model_state.get("status") != "complete" or model_state.get("features") != FEATURES
            or model_state.get("targets") != TARGETS
            or model_state.get("sample_weighting") != "major_PA"
            or model_state.get("min_minor_segment_pa") != 50):
        raise ValueError("Need a complete minor-PA50, major-PA-weighted training run")
    if model_state["versions"]["sklearn"] != sklearn.__version__:
        raise ValueError("Use the training run's scikit-learn version to load its models")
    training = pd.read_csv(args.run / "dataset.csv", dtype={"player_id": str})
    require_columns(training, ["year", "player_id", "split"], "training dataset")
    if (training.empty or training["year"].isna().any()
            or not training["year"].between(2020, 2025).all()
            or not training["split"].isin(["train", "validation"]).all()):
        raise ValueError("Training dataset must contain only 2020-2025")
    trained_players = set(training.loc[training["split"].eq("train"), "player_id"])
    models = {}
    for target in TARGETS:
        bundle = joblib.load(args.run / f"{target}.joblib")
        if (bundle.get("features") != FEATURES or bundle.get("target") != target
                or bundle.get("sample_weighting") != "major_PA"):
            raise ValueError(f"{target}: incompatible model bundle")
        models[target] = bundle["pipeline"]
    output = args.output or args.data / (
        "predictions_2026_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    raw.mkdir()
    path = output / "manifest.json"
    state = {
        "status": "in_progress", "year": YEAR, "source_data": str(args.data.resolve()),
        "model_run": str(args.run.resolve()), "max_age_jan1": 25,
        "minor_pa_exclusive_minimum": 50, "major_pa_required": None,
        "major_pa_filter": "none",
        "major_appearances_allowed": True, "features": FEATURES, "targets": TARGETS,
        "major_reference_source": "saved season leaderboard; absent records remain missing",
        "feature_window": "full_2026_minor_regular_season",
        "source_crawl_started_at_utc": source.get("started_at_utc"),
        "source_crawl_finished_at_utc": source.get("finished_at_utc"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "warning": WARNING, "requests": [],
    }
    write_json(path, state)
    logging.warning(WARNING)
    try:
        client = Client(args.delay, args.timeout)
        robots_text, _ = client.fetch(f"{BASE_URL}/robots.txt")
        (raw / "robots.txt").write_text(robots_text, encoding="utf-8")
        robots = RobotFileParser()
        robots.parse(robots_text.splitlines())

        kept, exclusions = [], rejected.to_dict("records")
        for row in candidates.to_dict("records"):
            player = row["player_id"]
            profile_url = f"{BASE_URL}/api/players/{player}"
            text = load_raw(client, robots, raw, args.cache, state, path,
                            profile_url, f"profile_{player}.json")
            birthday = birthday_from_profile(text, player)
            age = age_jan1(birthday, YEAR) if birthday else None
            row.update(birthday=birthday.isoformat() if birthday else "",
                       age_jan1=age, birthday_source_url=profile_url)
            if age is None or age > 25:
                row["exclusion_reason"] = "missing_birthday" if age is None else "over_age"
                exclusions.append(row)
                logging.warning("%s: %s", player, row["exclusion_reason"])
                continue
            row["seen_in_training"] = player in trained_players
            kept.append(row)
        pd.DataFrame(exclusions, columns=list(dict.fromkeys(
            list(rejected.columns) + [key for row in exclusions for key in row]
        ))).to_csv(output / "excluded.csv", index=False, encoding="utf-8-sig")
        if not kept:
            raise ValueError(f"No eligible players with verified ages; see {output / 'excluded.csv'}")
        result = predict(pd.DataFrame(kept), models)
        result.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
        state.update(status="complete", predicted_players=len(result),
                     excluded_players=len(exclusions),
                     completed_at_utc=datetime.now(timezone.utc).isoformat())
        logging.info("Saved %d prediction-only players: %s", len(result), output.resolve())
    except (OSError, ValueError, KeyError, DataError) as exc:
        state.update(status="failed", error=str(exc))
        raise
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        raise
    finally:
        write_json(path, state)


class PredictionTests(unittest.TestCase):
    def season(self, level, entries):
        rows = []
        for player, team, pa in entries:
            row = dict(year=YEAR, level=level, role="batter",
                       season_id=SEASONS[level][YEAR], player_id=player,
                       player_name=player, team_id=team, team_abbr=team,
                       standard_games=1)
            row.update({f"standard_{c}": 0 for c in COUNTS})
            row.update(standard_PA=pa, standard_AB=pa, standard_H=min(pa, 10))
            rows.append(row)
        return pd.DataFrame(rows)

    def test_strict_minor_pa_without_major_pa_filter_across_teams(self):
        minor = aggregate_season(self.season("minor", [
            ("a", "t", 50), ("b", "t", 51), ("c", "t", 51),
            ("d", "t", 26), ("d", "u", 25), ("e", "t", 51),
        ]), "minor")
        major = aggregate_season(self.season("major", [
            ("b", "t", 0), ("c", "t", 0), ("c", "u", 1),
        ]), "major")
        kept, rejected = select_candidates(minor, major)
        self.assertEqual(kept["player_id"].tolist(), ["b", "c", "d", "e"])
        self.assertEqual(rejected["player_id"].tolist(), ["a"])
        self.assertEqual(kept.loc[kept["player_id"].eq("b"), "major_games"].iloc[0], 1)
        self.assertEqual(kept.loc[kept["player_id"].eq("c"), "major_PA"].iloc[0], 1)
        self.assertTrue(pd.isna(kept.loc[kept["player_id"].eq("e"), "major_PA"].iloc[0]))
        self.assertAlmostEqual(minor.loc[minor["player_id"].eq("d"), "minor_AVG"].iloc[0], 20 / 51)

    def test_major_quality_does_not_exclude_minor_candidates(self):
        minor = aggregate_season(self.season("minor", [("p", "t", 51)]), "minor")
        major_rows = self.season("major", [("p", "t", 1)])
        major_rows["standard_H"] = 2
        major = aggregate_season(major_rows, "major")
        kept, rejected = select_candidates(minor, major)
        self.assertEqual(kept["player_id"].tolist(), ["p"])
        self.assertTrue(kept["major_invalid"].iloc[0])
        self.assertTrue(rejected.empty)

    def test_age_boundary(self):
        from datetime import date
        self.assertEqual(age_jan1(date(2000, 1, 2), YEAR), 25)
        self.assertEqual(age_jan1(date(2000, 1, 1), YEAR), 26)

    def test_predict_does_not_fit_or_create_actual_targets(self):
        class FixedModel:
            def predict(self, frame):
                return np.full(len(frame), 0.2)

        frame = pd.DataFrame({feature: [0.1] for feature in FEATURES})
        result = predict(frame, {target: FixedModel() for target in TARGETS})
        for target in TARGETS:
            self.assertNotIn(target, result.columns)
            self.assertEqual(result[f"predicted_{target}"].iloc[0], 0.2)
        self.assertEqual(result["evaluation_status"].iloc[0],
                         "prediction_only_not_scored")

    def test_end_to_end_without_network_or_fitting(self):
        class FixedModel:
            def predict(self, frame):
                return np.full(len(frame), 0.2)

        def fetch(client, url):
            if url.endswith("robots.txt"):
                text = "User-agent: *\nAllow: /\n"
            elif "/api/players/" in url:
                player = url.rsplit("/", 1)[1]
                text = json.dumps({"error": False, "data": {
                    "uniqid": player,
                    "birthday": "2000-01-01" if player == "d" else "2000-01-02",
                }})
            else:
                raise AssertionError(f"Unexpected request: {url}")
            return text, "2026-10-19T00:00:00+00:00"

        def load_model(path):
            return {"features": FEATURES, "target": path.stem,
                    "sample_weighting": "major_PA", "pipeline": FixedModel()}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, model, output = root / "source", root / "model", root / "output"
            source.mkdir()
            model.mkdir()
            minor = self.season("minor", [
                ("a", "t", 50), ("b", "t", 51), ("c", "t", 51),
                ("d", "t", 51), ("e", "t", 51),
            ])
            major = self.season("major", [("b", "t", 0), ("e", "t", 1)])
            outputs = []
            for level, frame in (("minor", minor), ("major", major)):
                name = f"{YEAR}_{level}_batter.csv"
                frame.to_csv(source / name, index=False)
                outputs.append({"file": name, "rows": len(frame)})
            write_json(source / "manifest.json", {
                "status": "complete", "source": BASE_URL, "years": [YEAR],
                "levels": ["minor", "major"], "roles": ["batter"], "outputs": outputs,
            })
            write_json(model / "manifest.json", {
                "status": "complete", "features": FEATURES, "targets": TARGETS,
                "sample_weighting": "major_PA", "min_minor_segment_pa": 50,
                "versions": {"sklearn": sklearn.__version__},
            })
            pd.DataFrame({"year": [2025, 2025], "player_id": ["b", "c"],
                          "split": ["train", "validation"]}).to_csv(
                              model / "dataset.csv", index=False)
            args = argparse.Namespace(data=source, run=model, output=output,
                                      cache=None, delay=1.5, timeout=45)
            with patch.object(Client, "fetch", new=fetch), patch.object(
                joblib, "load", side_effect=load_model,
            ):
                run(args)
            result = pd.read_csv(output / "predictions.csv")
            self.assertEqual(result["player_id"].tolist(), ["b", "c", "e"])
            self.assertEqual(result.loc[result["player_id"].eq("b"), "major_PA"].iloc[0], 0)
            self.assertTrue(pd.isna(result.loc[result["player_id"].eq("c"), "major_PA"].iloc[0]))
            self.assertEqual(result.loc[result["player_id"].eq("e"), "major_PA"].iloc[0], 1)
            self.assertEqual(result["seen_in_training"].tolist(), [True, False, False])
            self.assertTrue(result["minor_PA"].gt(50).all())
            self.assertTrue(result["age_jan1"].le(25).all())
            self.assertFalse(set(TARGETS) & set(result.columns))
            rejected = pd.read_csv(output / "excluded.csv")
            self.assertEqual(set(rejected["exclusion_reason"]),
                             {"minor_pa_not_over_50", "over_age"})
            state = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["predicted_players"], 3)
            self.assertIsNone(state["major_pa_required"])
            self.assertEqual(state["major_pa_filter"], "none")
            self.assertFalse((output / "metrics.json").exists())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path, help="Optional cached profile JSON directory")
    parser.add_argument("--delay", type=positive, default=1.5)
    parser.add_argument("--timeout", type=positive, default=45)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(PredictionTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if args.data is None or args.run is None:
        parser.error("--data and --run are required")
    try:
        run(args)
    except (OSError, ValueError, KeyError, DataError) as exc:
        logging.error("Prediction failed: %s; do not use incomplete outputs", exc)
        return 1
    except KeyboardInterrupt:
        logging.error("Prediction interrupted; do not use incomplete outputs")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
