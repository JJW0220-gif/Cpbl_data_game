"""依球季當年 1 月 1 日實足年齡篩選（Python 3.9+，標準函式庫）。

python filter_age.py data\\資料夾\\all_batter_same_season.csv
python filter_age.py data\\資料夾\\all_pitcher_same_season.csv
python filter_age.py data\\transitions_資料夾\\episodes.csv
python filter_age.py --self-test

預設保留年齡 <= 25，包含 25 歲；--max-age 可修改上限。
適用含 year、player_id 的 CSV。輸出至來源旁 age25_<時間>：
filtered.csv 保留原欄位並新增 birthday、age_jan1、birthday_source_url；
excluded.csv 列出超齡／生日缺失者與原因；原始 CSV 不變。
raw 與 manifest.json 記錄公開生日來源及執行狀態。
生日缺失明確排除並警告；格式錯誤或 API 失敗則停止，不猜測。
--cache 可指定先前 age25_<時間>\\raw，避免重抓生日。
篩完一二軍同季資料後，可將 filtered.csv 交給 rebas_transitions.py，
如此只需下載符合年齡球員的逐場資料。打者、投手都適用。
"""

import argparse
import csv
import json
import logging
import sys
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.robotparser import RobotFileParser

from rebas_crawler import BASE_URL, USER_AGENT, Client, DataError, positive, write_json
from rebas_transitions import load_raw, save_csv


def birthday_from_profile(text: str, player: str) -> Optional[date]:
    payload = json.loads(text)
    if not isinstance(payload, dict) or payload.get("error") is not False:
        raise DataError(f"Profile API error: {player}")
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("uniqid") != player:
        raise DataError(f"Missing/mismatched profile: {player}")
    value = data.get("birthday")
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise DataError(f"Invalid birthday: {player}")
    try:
        parsed = datetime.fromisoformat(value) if "T" in value else date.fromisoformat(value)
    except ValueError as exc:
        raise DataError(f"Invalid birthday {value!r}: {player}") from exc
    birthday = parsed.date() if isinstance(parsed, datetime) else parsed
    if birthday > date.today():
        raise DataError(f"Future birthday: {player}")
    return birthday


def age_jan1(birthday: date, year: int) -> int:
    cutoff = date(year, 1, 1)
    if birthday > cutoff:
        raise DataError(f"Birthday {birthday} after season cutoff {cutoff}")
    return year - birthday.year - ((1, 1) < (birthday.month, birthday.day))


def run(args: argparse.Namespace) -> None:
    with args.source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if (not {"year", "player_id"}.issubset(fields) or
                len(fields) != len(set(fields))):
            raise DataError("Input needs unique headers including year and player_id")
        rows = list(reader)
    if not rows:
        raise DataError("Input CSV is empty")
    for line, row in enumerate(rows, 2):
        if None in row or any(v is None for v in row.values()):
            raise DataError(f"Malformed CSV line {line}")
        player = row["player_id"]
        if not player or not all(c.isascii() and (c.isalnum() or c in "-_") for c in player):
            raise DataError(f"Invalid player_id at line {line}")
        year = row["year"]
        if not year.isascii() or not year.isdigit() or not 1 <= int(year) <= 9999:
            raise DataError(f"Invalid year at line {line}")
    client = Client(args.delay, args.timeout)
    robots_text, _ = client.fetch(f"{BASE_URL}/robots.txt")
    robots = RobotFileParser()
    robots.parse(robots_text.splitlines())
    output = args.source.parent / (
        f"age{args.max_age}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(exist_ok=False)
    raw = output / "raw"
    raw.mkdir()
    manifest = {
        "status": "in_progress", "source_csv": str(args.source.resolve()),
        "age_definition": "completed years on January 1 of season year",
        "max_age_inclusive": args.max_age, "requests": [],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = output / "manifest.json"
    write_json(path, manifest)
    try:
        birthdays = {}
        for player in sorted({row["player_id"] for row in rows}):
            url = f"{BASE_URL}/api/players/{player}"
            text = load_raw(client, robots, raw, args.cache, manifest, path,
                            url, f"profile_{player}.json")
            birthdays[player] = birthday_from_profile(text, player)
            if birthdays[player] is None:
                logging.warning("%s: missing birthday; excluding all its rows", player)
        kept, excluded = [], []
        for row in rows:
            player = row["player_id"]
            birthday = birthdays[player]
            age = age_jan1(birthday, int(row["year"])) if birthday else None
            record = dict(row)
            record.update(
                birthday=birthday.isoformat() if birthday else "",
                age_jan1=age, birthday_source_url=f"{BASE_URL}/api/players/{player}",
            )
            if age is not None and age <= args.max_age:
                kept.append(record)
            else:
                record["exclusion_reason"] = "missing_birthday" if age is None else "over_age"
                excluded.append(record)
        columns = fields + [k for k in ("birthday", "age_jan1", "birthday_source_url")
                            if k not in fields]
        save_csv(output / "filtered.csv", kept, columns)
        save_csv(output / "excluded.csv", excluded,
                 columns + ([] if "exclusion_reason" in columns else ["exclusion_reason"]))
        manifest.update(
            status="complete", input_rows=len(rows), retained_rows=len(kept),
            excluded_rows=len(excluded),
            missing_birthday_players=sum(b is None for b in birthdays.values()),
        )
        logging.info("Retained %d/%d rows. Saved: %s", len(kept), len(rows), output.resolve())
        if not kept:
            logging.warning("No eligible rows; filtered.csv contains headers only.")
    except (OSError, ValueError, DataError, csv.Error) as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        raise
    finally:
        write_json(path, manifest)


class AgeTests(unittest.TestCase):
    def test_jan1_birthday(self) -> None:
        self.assertEqual(age_jan1(date(2000, 1, 1), 2025), 25)

    def test_later_birthday_boundary(self) -> None:
        self.assertEqual(age_jan1(date(1999, 1, 2), 2025), 25)
        self.assertEqual(age_jan1(date(1999, 1, 1), 2025), 26)

    def test_leap_day(self) -> None:
        self.assertEqual(age_jan1(date(2000, 2, 29), 2025), 24)

    def test_actual_profile(self) -> None:
        text = '{"error":false,"data":{"uniqid":"p","birthday":"1988-06-12T00:00:00+08:00"}}'
        self.assertEqual(birthday_from_profile(text, "p"), date(1988, 6, 12))

    def test_missing_and_invalid(self) -> None:
        self.assertIsNone(birthday_from_profile(
            '{"error":false,"data":{"uniqid":"p","birthday":null}}', "p"))
        with self.assertRaises(DataError):
            birthday_from_profile('{"error":false,"data":{"uniqid":"q"}}', "p")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", type=Path)
    parser.add_argument("--max-age", type=int, default=25)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--delay", type=positive, default=1.5)
    parser.add_argument("--timeout", type=positive, default=45)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(AgeTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if args.source is None:
        parser.error("provide a source CSV")
    if args.max_age < 0:
        parser.error("--max-age must be non-negative")
    try:
        run(args)
    except (OSError, ValueError, DataError, csv.Error) as exc:
        logging.error("Stopped: %s; do not use incomplete outputs.", exc)
        return 1
    except KeyboardInterrupt:
        logging.error("Interrupted; do not use incomplete outputs.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
