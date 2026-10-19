"""下載 Rebas 公開的一、二軍例行賽數據（Python 3.9+，不需第三方套件）。

執行：
    python rebas_crawler.py
    python rebas_crawler.py --years 2024 2025 --levels minor --roles batter
    python rebas_crawler.py --output my_data --delay 2
    python rebas_crawler.py --self-test

預設：2020–2025、一軍 major／二軍 minor、打者 batter／投手 pitcher。
輸出 UTF-8 BOM CSV（可用 Excel 開啟）、原始 JSON 與 manifest.json。
資料來自公開 API；不使用帳號、不繞過付費限制。使用前請確認網站
最新資料使用條款及研究／競賽使用權限，簡報請註明 Rebas 野球革命。

合併 CSV 的 standard_／advanced_／new_ 前綴表示來源數據表。
例如 advanced_BBp、advanced_Kp 是百分數（13.303 表示 13.303%），
不是 0–1 比例。IPOut 是出局數，投球局數 = IPOut / 3；不能把
棒球紀錄的 72.1 局解讀成十進位的 72.1。
保留所有樣本（含未達規定打席／局數）；未篩選年齡、國籍或新秀資格。
如需年齡篩選，使用 filter_age.py，預設只保留球季當年 1 月 1 日
實足年齡 <= 25 的球員，並附生日與年齡來源；原始爬取資料不變。
本程式只整理球季數據，不代表已建立無洩漏的預測訓練集：
同年二軍與一軍數據可能交錯，不能直接視為先後關係。
train_batter.py 建模時只要求二軍段 PA >= 50，並以接續一軍段 PA 加權，
不跨區間加總；未定義目標不建模，原始球季資料不受此門檻影響。
可用 rebas_transitions.py 補抓逐場紀錄，記錄每次二軍→一軍出賽轉換，
包含同季一軍→二軍→一軍的回歸案例，並計算每段的表現。
2026 可用 --years 2026 明確下載，預設仍只下載 2020–2025。
predict_batter_2026.py 可預測 2026 年二軍 PA > 50 且年初 <= 25 歲的打者，
不限制一軍打席；整季數據可能交錯，預測結果不當成前瞻驗證成績。
"""

import argparse
import csv
import json
import logging
import math
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser


BASE_URL = "https://www.rebas.tw"
USER_AGENT = "BaseballResearchCrawler/1.0 (public season stats)"
SEASONS = {
    "major": {
        2020: "CPBL-2020-KS",
        2021: "CPBL-2021-fi",
        2022: "CPBL-2022-dG",
        2023: "CPBL-2023-sk",
        2024: "CPBL-2024-xa",
        2025: "CPBL-2025-JO",
        2026: "CPBL-2026-oB",
    },
    "minor": {
        2020: "CPBLmi-2020-cP",
        2021: "CPBLmi-2021-Vw",
        2022: "CPBLmi-2022-C2",
        2023: "CPBLmi-2023-GP",
        2024: "CPBLmi-2024-4l",
        2025: "CPBLmi-2025-ED",
        2026: "CPBLmi-2026-Jn",
    },
}
SECTIONS = ("standard", "advanced", "new")
Row = Dict[str, Any]


class DataError(RuntimeError):
    """資料回應不符合已驗證的公開介面格式。"""


class Client:
    def __init__(self, delay: float, timeout: float) -> None:
        self.delay = delay
        self.timeout = timeout
        self.last_request = 0.0

    def fetch(self, url: str) -> Tuple[str, str]:
        for attempt in range(3):
            wait = self.delay - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            self.last_request = time.monotonic()
            try:
                request = Request(
                    url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"}
                )
                with urlopen(request, timeout=self.timeout) as response:
                    if response.status != 200:
                        raise DataError(
                            f"HTTP {response.status}: {url}; refusing partial data"
                        )
                    text = response.read().decode("utf-8-sig")
                return text, datetime.now(timezone.utc).isoformat()
            except HTTPError as exc:
                if exc.code in (401, 403):
                    raise DataError(
                        f"HTTP {exc.code}: {url}; access restricted, stopping"
                    ) from exc
                if exc.code not in (429, 500, 502, 503, 504):
                    raise
                if attempt == 2:
                    raise
                retry_after = exc.headers.get("Retry-After", "")
                backoff = max(5.0 * (2 ** attempt), self.delay)
                if retry_after.isdigit():
                    backoff = max(backoff, float(retry_after))
                logging.warning("HTTP %s; retry in %.1fs: %s", exc.code, backoff, url)
                time.sleep(backoff)
            except (URLError, TimeoutError) as exc:
                if attempt == 2:
                    raise
                backoff = 5.0 * (2 ** attempt)
                logging.warning("%s; retry in %.1fs: %s", exc, backoff, url)
                time.sleep(backoff)
        raise DataError(f"Request attempts exhausted: {url}")


def parse_rows(text: str, url: str) -> List[Row]:
    payload = json.loads(text)
    if not isinstance(payload, dict) or payload.get("error") is not False:
        raise DataError(f"API reports an error or unexpected envelope: {url}")
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        raise DataError(f"Missing or empty stats array: {url}")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise DataError(f"Non-object stats row: {url}")
        player = row.get("player")
        if not isinstance(player, dict):
            raise DataError(f"Missing player metadata: {url}")
        for key in ("uniqid", "team_uniqid", "name"):
            if not isinstance(player.get(key), str) or not player[key]:
                raise DataError(f"Missing player {key}: {url}")
        identity = (player["uniqid"], player["team_uniqid"])
        if identity in seen:
            raise DataError(f"Duplicate player/team {identity}: {url}")
        seen.add(identity)
        metrics = [key for key in row if key not in ("player", "reach_min")]
        if not metrics:
            raise DataError(f"Missing metrics for {identity}: {url}")
        for key in metrics:
            value = row[key]
            if isinstance(value, (dict, list)):
                raise DataError(f"Unexpected nested metric {key}: {url}")
            if isinstance(value, float) and not math.isfinite(value):
                raise DataError(f"Non-finite metric {key}: {url}")
    return rows


def merge_sections(
    sections: Dict[str, List[Row]], year: int, level: str, role: str, season: str
) -> List[Row]:
    merged: Dict[Tuple[str, str], Row] = {}
    expected = None
    for section in SECTIONS:
        rows = sections[section]
        identities = {(r["player"]["uniqid"], r["player"]["team_uniqid"]) for r in rows}
        if expected is None:
            expected = identities
        elif identities != expected:
            raise DataError(
                f"{season}/{role}/{section}: inconsistent player sets "
                f"(missing={len(expected - identities)}, extra={len(identities - expected)})"
            )
        for row in rows:
            player = row["player"]
            identity = (player["uniqid"], player["team_uniqid"])
            if identity not in merged:
                merged[identity] = {
                    "year": year,
                    "level": level,
                    "role": role,
                    "season_id": season,
                    "player_id": player["uniqid"],
                    "player_name": player["name"],
                    "team_id": player["team_uniqid"],
                    "team_abbr": player.get("team_abbr"),
                    "number": player.get("number"),
                }
            target = merged[identity]
            if target["player_name"] != player["name"]:
                raise DataError(f"Inconsistent player name for {identity}")
            for key, value in row.items():
                if key != "player":
                    target[f"{section}_{key}"] = value
    return [merged[key] for key in sorted(merged)]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: List[Row]) -> None:
    if not rows:
        raise DataError(f"Refusing to write empty CSV: {path}")
    columns = list(rows[0])
    columns.extend(sorted({key for row in rows for key in row} - set(columns)))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def positive(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def crawl(args: argparse.Namespace) -> None:
    client = Client(args.delay, args.timeout)
    robots_url = f"{BASE_URL}/robots.txt"
    robots_text, robots_time = client.fetch(robots_url)
    robots = RobotFileParser(robots_url)
    robots.parse(robots_text.splitlines())
    jobs = []
    for year in sorted(set(args.years)):
        for level in dict.fromkeys(args.levels):
            season = SEASONS[level][year]
            for role in dict.fromkeys(args.roles):
                jobs.append((year, level, role, season))
                for section in SECTIONS:
                    url = stats_url(season, role, section)
                    if not robots.can_fetch(USER_AGENT, url):
                        raise DataError(f"robots.txt disallows: {url}")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    raw.mkdir()
    (raw / "robots.txt").write_text(robots_text, encoding="utf-8")
    manifest: Row = {
        "status": "in_progress",
        "source": BASE_URL,
        "started_at_utc": robots_time,
        "years": sorted(set(args.years)),
        "levels": list(dict.fromkeys(args.levels)),
        "roles": list(dict.fromkeys(args.roles)),
        "sections": list(SECTIONS),
        "scope": "regular season, all available players, no rookie/age filter",
        "requests": [],
        "outputs": [],
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    all_rows: Dict[str, List[Row]] = {role: [] for role in args.roles}
    try:
        for year, level, role, season in jobs:
            sections = {}
            for section in SECTIONS:
                url = stats_url(season, role, section)
                logging.info("Fetching %s", url)
                text, fetched_at = client.fetch(url)
                filename = f"{year}_{level}_{role}_{section}.json"
                (raw / filename).write_text(text, encoding="utf-8")
                rows = parse_rows(text, url)
                sections[section] = rows
                manifest["requests"].append({
                    "url": url, "fetched_at_utc": fetched_at,
                    "rows": len(rows), "raw_file": f"raw/{filename}",
                })
                write_json(manifest_path, manifest)
            merged = merge_sections(sections, year, level, role, season)
            filename = f"{year}_{level}_{role}.csv"
            write_csv(output / filename, merged)
            manifest["outputs"].append({"file": filename, "rows": len(merged)})
            all_rows[role].extend(merged)
            logging.info("%s: %d player/team rows", filename, len(merged))
        for role, rows in all_rows.items():
            filename = f"all_{role}.csv"
            write_csv(output / filename, rows)
            manifest["outputs"].append({"file": filename, "rows": len(rows)})
        manifest["status"] = "complete"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    except (DataError, OSError, ValueError) as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        raise
    finally:
        write_json(manifest_path, manifest)
    logging.info("Completed: %s", output.resolve())


def stats_url(season: str, role: str, section: str) -> str:
    query = urlencode({"type": role, "section": section})
    return f"{BASE_URL}/api/seasons/{season}/leaders?{query}"


class CrawlerTests(unittest.TestCase):
    def row(self, value: float = 0.3) -> Row:
        return {
            "player": {"uniqid": "p1", "team_uniqid": "t1", "name": "Player"},
            "AVG": value, "PA": 100, "reach_min": False,
        }

    def test_envelope(self) -> None:
        text = json.dumps({"error": False, "data": [self.row()]})
        self.assertEqual(parse_rows(text, "fixture")[0]["PA"], 100)

    def test_errors_and_empty(self) -> None:
        for payload in ({"error": True, "data": []}, {"error": False, "data": []}):
            with self.assertRaises(DataError):
                parse_rows(json.dumps(payload), "fixture")

    def test_duplicate(self) -> None:
        with self.assertRaises(DataError):
            parse_rows(json.dumps({"error": False, "data": [self.row()] * 2}), "fixture")

    def test_merge_preserves_sections(self) -> None:
        sections = {section: [self.row()] for section in SECTIONS}
        rows = merge_sections(sections, 2025, "minor", "batter", "fixture")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["advanced_PA"], 100)
        self.assertFalse(rows[0]["new_reach_min"])

    def test_mismatched_sets(self) -> None:
        sections = {section: [self.row()] for section in SECTIONS}
        sections["new"][0]["player"]["uniqid"] = "p2"
        with self.assertRaises(DataError):
            merge_sections(sections, 2025, "minor", "batter", "fixture")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", choices=range(2020, 2027),
                        default=list(range(2020, 2026)))
    parser.add_argument("--levels", nargs="+", choices=tuple(SEASONS),
                        default=["major", "minor"])
    parser.add_argument("--roles", nargs="+", choices=("batter", "pitcher"),
                        default=["batter", "pitcher"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--delay", type=positive, default=1.5)
    parser.add_argument("--timeout", type=positive, default=45.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(CrawlerTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.output = Path(__file__).resolve().parent / "data" / f"rebas_{stamp}"
    try:
        crawl(args)
    except (DataError, OSError, ValueError) as exc:
        logging.error("Stopped: %s. Partial output is NOT a complete dataset.", exc)
        return 1
    except KeyboardInterrupt:
        logging.error("Interrupted. Partial output is NOT a complete dataset.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
