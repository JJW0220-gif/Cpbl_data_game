"""篩選同季在一軍、二軍皆出場的球員（Python 3.9+，僅用標準函式庫）。

python filter_same_season.py data\\資料夾\\all_batter.csv
python filter_same_season.py data\\資料夾\\all_batter.csv data\\資料夾\\all_pitcher.csv
python filter_same_season.py --self-test

輸出於來源旁：all_batter_same_season.csv／all_pitcher_same_season.csv。
原檔不變，輸出保留原欄位、原順序與所有符合條件的球隊列。
以 year + role + player_id 配對，不以姓名或 team_id 配對；
打者與投手分開篩選，且 standard_games > 0 才算出場。
不設最低打席／投球局數、不篩選新秀。此條件不能證明升降順序，
同季資料仍可能交錯，不能直接當成無時間洩漏的預測訓練集。
若輸出已存在，預設拒絕覆寫；確認後可加 --overwrite。
"""

import argparse
import csv
import logging
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


Row = Dict[str, str]
Key = Tuple[str, str, str]
REQUIRED = ("year", "role", "player_id", "level", "standard_games")


def select_rows(rows: Sequence[Row]) -> Tuple[List[Row], Dict[str, int]]:
    levels: Dict[Key, set] = {}
    appearances: List[Tuple[Row, Key, int]] = []
    for line, row in enumerate(rows, start=2):
        for field in REQUIRED:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Line {line}: missing {field}")
        if row["level"] not in ("major", "minor"):
            raise ValueError(f"Line {line}: invalid level {row['level']!r}")
        if row["role"] not in ("batter", "pitcher"):
            raise ValueError(f"Line {line}: invalid role {row['role']!r}")
        if not row["year"].isascii() or not row["year"].isdigit():
            raise ValueError(f"Line {line}: invalid year {row['year']!r}")
        games_text = row["standard_games"]
        if not games_text.isascii() or not games_text.isdigit():
            raise ValueError(f"Line {line}: invalid standard_games {games_text!r}")
        games = int(games_text)
        key = (row["year"], row["role"], row["player_id"])
        appearances.append((row, key, games))
        if games > 0:
            levels.setdefault(key, set()).add(row["level"])
    eligible = {key for key, values in levels.items() if values == {"major", "minor"}}
    selected = [row for row, key, games in appearances if key in eligible and games > 0]
    counts: Dict[str, int] = {}
    for year, role, _ in eligible:
        label = f"{year}/{role}"
        counts[label] = counts.get(label, 0) + 1
    return selected, dict(sorted(counts.items()))


def filter_file(source: Path, overwrite: bool = False) -> Path:
    destination = source.with_name(f"{source.stem}_same_season.csv")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"{source}: missing or duplicate column headers")
        missing = set(REQUIRED) - set(fields)
        if missing:
            raise ValueError(f"{source}: missing columns {sorted(missing)}")
        rows = []
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{source}: malformed CSV row {line}")
            rows.append(row)
    selected, counts = select_rows(rows)
    with destination.open(
        "w" if overwrite else "x", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    logging.info("%s: %d -> %d rows; %d player-seasons",
                 source.name, len(rows), len(selected), sum(counts.values()))
    for label, count in counts.items():
        logging.info("%s: %d players", label, count)
    if not selected:
        logging.warning("No qualifying player-seasons; output contains headers only.")
    logging.info("Saved: %s", destination.resolve())
    return destination


class FilterTests(unittest.TestCase):
    @staticmethod
    def row(level: str, year: str = "2025", player: str = "p1",
            role: str = "batter", games: str = "1") -> Row:
        return {"year": year, "role": role, "player_id": player,
                "level": level, "standard_games": games}

    def test_same_season(self) -> None:
        rows = [self.row("minor"), self.row("major"), self.row("minor", player="p2")]
        selected, counts = select_rows(rows)
        self.assertEqual(selected, rows[:2])
        self.assertEqual(counts, {"2025/batter": 1})

    def test_different_years(self) -> None:
        self.assertEqual(select_rows([
            self.row("minor", year="2024"), self.row("major")
        ])[0], [])

    def test_roles_are_separate(self) -> None:
        self.assertEqual(select_rows([
            self.row("minor"), self.row("major", role="pitcher")
        ])[0], [])

    def test_zero_games(self) -> None:
        self.assertEqual(select_rows([
            self.row("minor", games="0"), self.row("major")
        ])[0], [])

    def test_multiple_team_rows(self) -> None:
        rows = [dict(self.row("minor"), team_id="m1"),
                dict(self.row("major"), team_id="a1"),
                dict(self.row("major"), team_id="a2")]
        selected, counts = select_rows(rows)
        self.assertEqual(selected, rows)
        self.assertEqual(sum(counts.values()), 1)

    def test_invalid_input(self) -> None:
        for games in ("", "NaN", "-1", "1.5"):
            with self.assertRaises(ValueError):
                select_rows([self.row("major", games=games)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", type=Path, nargs="*")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(FilterTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if not args.files:
        parser.error("provide at least one all_batter.csv or all_pitcher.csv")
    try:
        for source in args.files:
            filter_file(source, args.overwrite)
    except (OSError, ValueError, csv.Error) as exc:
        logging.error("Stopped: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
