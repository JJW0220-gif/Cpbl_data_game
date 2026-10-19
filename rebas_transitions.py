"""記錄同季每次二軍→一軍的出賽轉換（Python 3.9+，無第三方套件）。

python rebas_transitions.py data\\資料夾\\all_pitcher_same_season.csv
python rebas_transitions.py data\\資料夾\\all_batter_same_season.csv
python rebas_transitions.py data\\資料夾\\all_pitcher_same_season.csv --years 2025 --limit 5
python rebas_transitions.py --self-test

與 rebas_crawler.py、filter_same_season.py 放在同一資料夾。
可先用 filter_age.py 篩選當年 1 月 1 日實足年齡 <= 25 的球員，
再將其 filtered.csv 作為本程式輸入；輸出的日期段仍依完整出賽排序。
僅存取公開逐場 API；遵守 robots.txt，不登入、不繞過存取限制。
預設輸出至來源旁 transitions_<時間>，保留原始逐場 JSON、出賽明細、
轉換案例、同日兩層級無法排序的球員球季清單，以及 manifest.json。
--cache 可指定既有輸出的 raw 資料夾，減少重複請求；快取仍會驗證。

定義：依實際出賽日期排序，連續同層級出賽構成一段。
每段二軍後接一段一軍，記為一次轉換；一軍→二軍→一軍也會記錄。
前段二軍／後段一軍的統計只使用該段內比賽，不使用整季進階指標。
return_after_major 表示同季較早已有一軍出賽，不代表生涯第一次升軍。
段落起訖是出賽日，不是正式升降登錄日；同日跨層級因缺精確時間，
整個球員球季不產生轉換案例，另列 ambiguous.csv，避免猜測順序。
零局投手或零打席打者仍按逐場出場紀錄保留；不設最低樣本量。
百分率 K_pct／BB_pct 是 0–100；IP = 出局數 / 3。
ERA／WHIP／OPS 等由段內基礎數據重算；分母為零時留白。
不重算缺乏充分定義／欄位的 FIP、wOBA、OPS+、揮空率等。
以可取得的逐場數據為準；整季場數只作比較，不一致時警告並列入
quality.csv，不補造比賽。這種樣本可能有來源缺漏，須留意研究限制。
打者逐場紀錄若無打擊統計（例如只有代跑），會查該場成績表補齊，
並記錄 stats_source_url；不把缺少統計直接當作零。
若成績表也找不到該球員打擊列，排除整個球員球季並列入 quality.csv，
不刪除中間一場後假裝出賽順序完整；其餘球員繼續處理。
這仍是「有兩層級出場者」的觀察樣本，不包含未獲一軍機會者。
"""

import argparse
import csv
import json
import logging
import sys
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.robotparser import RobotFileParser

from filter_same_season import select_rows
from rebas_crawler import BASE_URL, USER_AGENT, Client, DataError, positive, write_json


Row = Dict[str, Any]
CORE = {
    "pitcher": ("IPOut", "BF", "H", "BB", "SO", "HR", "ER", "R"),
    "batter": ("AB", "PA", "H", "BB", "HBP", "SO", "HR", "Double", "Triple", "SF"),
}
LOG_COLUMNS = [
    "year", "role", "player_id", "player_name", "level", "season_id",
    "date", "game_id", "seq", "opponent_abbr", "source_url", "stats_source_url",
] + sorted(set(CORE["pitcher"]) | set(CORE["batter"]))
SUMMARY_COLUMNS = sorted(set(CORE["pitcher"]) | set(CORE["batter"])) + [
    "IP", "ERA", "WHIP", "AVG", "OBP", "SLG", "OPS", "ISO", "K_pct", "BB_pct",
]
EPISODE_COLUMNS = [
    "year", "role", "player_id", "player_name", "transition_no",
    "transition_kind", "minor_start", "minor_end", "major_start", "major_end",
    "gap_days", "minor_games", "major_games",
] + [f"{level}_{metric}" for level in ("minor", "major") for metric in SUMMARY_COLUMNS]
AMBIGUOUS_COLUMNS = ["year", "role", "player_id", "player_name", "same_day_dates"]
QUALITY_COLUMNS = [
    "year", "role", "player_id", "player_name", "level", "issue",
    "season_games", "log_games", "included", "detail",
]


class MissingBattingRow(DataError):
    """逐場與成績表皆無法取得球員打擊統計。"""


def save_csv(path: Path, rows: List[Row], columns: List[str]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_source(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {"year", "role", "player_id", "player_name", "level",
                    "standard_games", "season_id"}
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            raise DataError("Missing or duplicate input CSV columns")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise DataError("Malformed input CSV")
    selected, _ = select_rows(rows)
    if not selected:
        raise DataError("No same-season major/minor appearances in input")
    return selected


def batting_from_box(text: str, player: str, game: str, url: str) -> Row:
    payload = json.loads(text)
    if not isinstance(payload, dict) or payload.get("error") is not False:
        raise DataError(f"Invalid box-score envelope: {url}")
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("uniqid") != game:
        raise DataError(f"Missing/mismatched box-score game: {url}")
    matches = []
    for side in ("home", "away"):
        team = data.get(side)
        if not isinstance(team, dict) or not isinstance(team.get("box"), dict):
            raise DataError(f"Missing {side} box score: {url}")
        battings = team["box"].get("battings")
        if not isinstance(battings, list):
            raise DataError(f"Missing {side} batting rows: {url}")
        for entry in battings:
            if not isinstance(entry, dict) or not isinstance(entry.get("player"), dict):
                raise DataError(f"Invalid batting box row: {url}")
            if entry["player"].get("uniqid") == player:
                matches.append(entry)
    if not matches:
        raise MissingBattingRow(f"No batting box row for {player}: {url}")
    if len(matches) != 1:
        raise DataError(f"Expected one batting box row for {player}, got {len(matches)}: {url}")
    stats = {}
    for metric in CORE["batter"]:
        box_key = {"Double": "2B", "Triple": "3B"}.get(metric, metric)
        value = matches[0].get(box_key)
        if type(value) is not int or value < 0:
            raise DataError(f"Missing/invalid box metric {box_key}: {url}")
        stats[metric] = value
    return stats


def load_raw(
    client: Client, robots: RobotFileParser, raw: Path, cache_dir: Optional[Path],
    manifest: Row, manifest_path: Path, url: str, filename: str,
) -> str:
    if not robots.can_fetch(USER_AGENT, url):
        raise DataError(f"robots.txt disallows {url}")
    cache = cache_dir / filename if cache_dir else None
    local = raw / filename
    cache_source = local if local.is_file() else cache
    from_cache = cache_source is not None and cache_source.is_file()
    if from_cache:
        text = cache_source.read_text(encoding="utf-8")
        fetched_at = None
    else:
        logging.info("Fetching %s", url)
        text, fetched_at = client.fetch(url)
    local.write_text(text, encoding="utf-8")
    manifest["requests"].append({
        "url": url, "raw_file": f"raw/{filename}",
        "fetched_at_utc": fetched_at, "cached": from_cache,
        "cache_path": str(cache_source.resolve()) if from_cache else None,
    })
    write_json(manifest_path, manifest)
    return text


def parse_logs(text: str, meta: Row, url: str) -> List[Row]:
    payload = json.loads(text)
    if not isinstance(payload, dict) or payload.get("error") is not False:
        raise DataError(f"API error or invalid envelope: {url}")
    data = payload.get("data")
    if not isinstance(data, list):
        raise DataError(f"Invalid logs array: {url}")
    role = meta["role"]
    field = "pitching" if role == "pitcher" else "batting"
    rows = []
    seen = set()
    for item in data:
        if not isinstance(item, dict):
            raise DataError(f"Invalid log item: {url}")
        if field not in item:
            if role == "batter":
                raise DataError(f"Unresolved missing batting stats: {url}")
            continue
        stats = item[field]
        if not isinstance(stats, dict):
            raise DataError(f"Invalid {field} stats: {url}")
        game_id = item.get("uniqid")
        day = item.get("date")
        if not isinstance(game_id, str) or not game_id or game_id in seen:
            raise DataError(f"Missing/duplicate game ID: {url}")
        seen.add(game_id)
        if not isinstance(day, str):
            raise DataError(f"Missing date: {url}")
        parsed = date.fromisoformat(day)
        if parsed.isoformat() != day or parsed.year != int(meta["year"]):
            raise DataError(f"Unexpected date {day}: {url}")
        row = {key: meta[key] for key in (
            "year", "role", "player_id", "player_name", "level", "season_id"
        )}
        row.update(date=day, game_id=game_id, seq=item.get("seq"),
                   opponent_abbr=item.get("opponent_abbr"), source_url=url,
                   stats_source_url=item.get("stats_source_url", url))
        for metric in CORE[role]:
            value = stats.get(metric)
            if type(value) is not int or value < 0:
                raise DataError(f"Missing/invalid {metric} in {game_id}: {url}")
            row[metric] = value
        rows.append(row)
    return rows


def ratio(numerator: float, denominator: float) -> Optional[float]:
    return round(numerator / denominator, 6) if denominator else None


def summarize(rows: List[Row], role: str) -> Row:
    total: Row = {metric: sum(row[metric] for row in rows) for metric in CORE[role]}
    if role == "pitcher":
        total.update(
            IP=ratio(total["IPOut"], 3),
            ERA=ratio(27 * total["ER"], total["IPOut"]),
            WHIP=ratio(3 * (total["H"] + total["BB"]), total["IPOut"]),
            K_pct=ratio(100 * total["SO"], total["BF"]),
            BB_pct=ratio(100 * total["BB"], total["BF"]),
        )
    else:
        tb = total["H"] + total["Double"] + 2 * total["Triple"] + 3 * total["HR"]
        obp = ratio(total["H"] + total["BB"] + total["HBP"],
                    total["AB"] + total["BB"] + total["HBP"] + total["SF"])
        slg = ratio(tb, total["AB"])
        total.update(
            AVG=ratio(total["H"], total["AB"]), OBP=obp, SLG=slg,
            OPS=round(obp + slg, 6) if obp is not None and slg is not None else None,
            ISO=ratio(tb - total["H"], total["AB"]),
            K_pct=ratio(100 * total["SO"], total["PA"]),
            BB_pct=ratio(100 * total["BB"], total["PA"]),
        )
    return total


def build_episodes(rows: List[Row]) -> Tuple[List[Row], List[str]]:
    ordered = sorted(rows, key=lambda r: (r["date"], str(r["seq"]), r["game_id"]))
    by_day: Dict[str, set] = {}
    for row in ordered:
        by_day.setdefault(row["date"], set()).add(row["level"])
    ambiguous = sorted(day for day, levels in by_day.items() if len(levels) > 1)
    if ambiguous:
        return [], ambiguous
    segments: List[List[Row]] = []
    for row in ordered:
        if not segments or segments[-1][-1]["level"] != row["level"]:
            segments.append([])
        segments[-1].append(row)
    episodes = []
    had_major = False
    for index, segment in enumerate(segments):
        if segment[0]["level"] == "major":
            had_major = True
            continue
        if index + 1 == len(segments):
            continue
        major = segments[index + 1]
        first = segment[0]
        episode = {key: first[key] for key in ("year", "role", "player_id", "player_name")}
        episode.update(
            transition_no=len(episodes) + 1,
            transition_kind="return_after_major" if had_major else "first_major_in_season",
            minor_start=first["date"], minor_end=segment[-1]["date"],
            major_start=major[0]["date"], major_end=major[-1]["date"],
            gap_days=(date.fromisoformat(major[0]["date"]) -
                      date.fromisoformat(segment[-1]["date"])).days,
            minor_games=len(segment), major_games=len(major),
        )
        for level, games in (("minor", segment), ("major", major)):
            episode.update({f"{level}_{key}": value
                            for key, value in summarize(games, first["role"]).items()})
        episodes.append(episode)
    return episodes, []


def run(args: argparse.Namespace) -> None:
    rows = read_source(args.source)
    grouped: Dict[Tuple[str, str, str], List[Row]] = {}
    for row in rows:
        if args.years and int(row["year"]) not in args.years:
            continue
        if args.player_id and row["player_id"] not in args.player_id:
            continue
        grouped.setdefault((row["year"], row["role"], row["player_id"]), []).append(row)
    keys = sorted(grouped)
    if args.limit:
        keys = keys[:args.limit]
    if not keys:
        raise DataError("No player-seasons match the requested filters")
    client = Client(args.delay, args.timeout)
    robots_text, _ = client.fetch(f"{BASE_URL}/robots.txt")
    robots = RobotFileParser()
    robots.parse(robots_text.splitlines())
    output = args.output or args.source.parent / (
        "transitions_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    raw.mkdir()
    manifest: Row = {
        "status": "in_progress", "source_csv": str(args.source.resolve()),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_player_seasons": len(keys), "completed_player_seasons": 0,
        "definition": "consecutive appearance-level segments; not registration dates",
        "requests": [],
    }
    path = output / "manifest.json"
    write_json(path, manifest)
    appearances: List[Row] = []
    episodes: List[Row] = []
    ambiguous: List[Row] = []
    quality: List[Row] = []
    try:
        for key in keys:
            logs = []
            input_rows = grouped[key]
            excluded = False
            for level in ("minor", "major"):
                level_rows = [r for r in input_rows if r["level"] == level]
                seasons = {r["season_id"] for r in level_rows}
                if len(seasons) != 1:
                    raise DataError(f"{key}/{level}: expected one regular-season ID")
                season = next(iter(seasons))
                meta = dict(level_rows[0])
                meta["expected_games"] = sum(int(r["standard_games"]) for r in level_rows)
                player = key[2]
                if not all(c.isascii() and (c.isalnum() or c in "-_") for c in player + season):
                    raise DataError(f"Unsafe player/season identifier: {key}")
                filename = f"{season}_{player}.json"
                url = f"{BASE_URL}/api/formal/players/{player}/seasons/{season}/logs"
                text = load_raw(client, robots, raw, args.cache, manifest, path, url, filename)
                if meta["role"] == "batter":
                    payload = json.loads(text)
                    if (not isinstance(payload, dict) or payload.get("error") is not False
                            or not isinstance(payload.get("data"), list)):
                        raise DataError(f"Invalid logs envelope: {url}")
                    for item in payload["data"]:
                        if not isinstance(item, dict):
                            raise DataError(f"Invalid log item: {url}")
                        if item.get("batting") is not None:
                            continue
                        game = item.get("uniqid")
                        if (not isinstance(game, str) or not game or
                                not all(c.isascii() and (c.isalnum() or c in "-_") for c in game)):
                            raise DataError(f"Unsafe/missing game ID: {url}")
                        box_url = f"{BASE_URL}/api/seasons/{season}/games/{game}"
                        box_text = load_raw(
                            client, robots, raw, args.cache, manifest, path,
                            box_url, f"box_{season}_{game}.json",
                        )
                        try:
                            item["batting"] = batting_from_box(box_text, player, game, box_url)
                        except MissingBattingRow as exc:
                            entry = {k: meta[k] for k in (
                                "year", "role", "player_id", "player_name", "level"
                            )}
                            entry.update(
                                issue="unresolved_batting_stats",
                                season_games=meta["expected_games"], log_games=None,
                                included=False, detail=str(exc),
                            )
                            quality.append(entry)
                            logging.warning("%s: excluding player-season: %s", key, exc)
                            excluded = True
                            break
                        item["stats_source_url"] = box_url
                        logging.info("Recovered batting appearance: %s %s %s",
                                     player, item.get("date"), game)
                    text = json.dumps(payload)
                if excluded:
                    break
                parsed_logs = parse_logs(text, meta, url)
                if len(parsed_logs) != meta["expected_games"]:
                    entry = {k: meta[k] for k in (
                        "year", "role", "player_id", "player_name", "level"
                    )}
                    entry.update(
                        issue="season_game_count_mismatch",
                        season_games=meta["expected_games"], log_games=len(parsed_logs),
                        included=True, detail="Using available role-specific game logs",
                    )
                    quality.append(entry)
                    logging.warning("%s/%s: season games=%d, logs=%d; using logs",
                                    key, level, meta["expected_games"], len(parsed_logs))
                logs.extend(parsed_logs)
            if excluded:
                for entry in quality:
                    if (entry["year"], entry["role"], entry["player_id"]) == key:
                        entry["included"] = False
                manifest["completed_player_seasons"] += 1
                continue
            found, dates = build_episodes(logs)
            appearances.extend(logs)
            episodes.extend(found)
            if dates:
                entry = {k: input_rows[0][k] for k in ("year", "role", "player_id", "player_name")}
                entry["same_day_dates"] = ";".join(dates)
                ambiguous.append(entry)
                logging.warning("%s: same-day levels %s; excluded from episodes", key, dates)
            manifest["completed_player_seasons"] += 1
            logging.info("%s: %d transitions", key, len(found))
        save_csv(output / "appearances.csv", appearances, LOG_COLUMNS)
        save_csv(output / "episodes.csv", episodes, EPISODE_COLUMNS)
        save_csv(output / "ambiguous.csv", ambiguous, AMBIGUOUS_COLUMNS)
        save_csv(output / "quality.csv", quality, QUALITY_COLUMNS)
        manifest.update(status="complete", transitions=len(episodes),
                        ambiguous_player_seasons=len(ambiguous),
                        quality_issues=len(quality),
                        excluded_player_seasons=sum(
                            r["issue"] == "unresolved_batting_stats" for r in quality
                        ),
                        data_basis="available role-specific game logs",
                        finished_at_utc=datetime.now(timezone.utc).isoformat())
        logging.info("Saved %d transitions to %s", len(episodes), output.resolve())
    except (OSError, ValueError, DataError) as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        raise
    finally:
        write_json(path, manifest)


class TransitionTests(unittest.TestCase):
    @staticmethod
    def game(level: str, day: int, outs: int = 3) -> Row:
        row: Row = {key: 0 for key in CORE["pitcher"]}
        row.update(year="2025", role="pitcher", player_id="p", player_name="P",
                   level=level, date=f"2025-04-{day:02d}", game_id=f"{level}{day}",
                   seq=day, IPOut=outs, BF=4, H=1, SO=1)
        return row

    def test_repeated_returns(self) -> None:
        rows = [self.game("major", 1), self.game("minor", 2),
                self.game("minor", 3), self.game("major", 4),
                self.game("minor", 5), self.game("major", 6)]
        episodes, ambiguous = build_episodes(list(reversed(rows)))
        self.assertEqual(ambiguous, [])
        self.assertEqual(len(episodes), 2)
        self.assertEqual([r["transition_kind"] for r in episodes],
                         ["return_after_major"] * 2)
        self.assertEqual(episodes[0]["minor_games"], 2)
        self.assertEqual(episodes[0]["major_end"], "2025-04-04")
        self.assertEqual(episodes[1]["transition_no"], 2)

    def test_first_then_return(self) -> None:
        episodes, _ = build_episodes([self.game("minor", 1), self.game("major", 2),
                                     self.game("minor", 3), self.game("major", 4)])
        self.assertEqual([r["transition_kind"] for r in episodes],
                         ["first_major_in_season", "return_after_major"])

    def test_trailing_minor(self) -> None:
        self.assertEqual(build_episodes([self.game("major", 1), self.game("minor", 2)])[0], [])

    def test_same_day_ambiguous(self) -> None:
        episodes, dates = build_episodes([self.game("minor", 1), self.game("major", 1)])
        self.assertEqual(episodes, [])
        self.assertEqual(dates, ["2025-04-01"])

    def test_zero_outs(self) -> None:
        total = summarize([self.game("major", 1, outs=0)], "pitcher")
        self.assertIsNone(total["ERA"])
        self.assertIsNone(total["WHIP"])
        self.assertEqual(total["K_pct"], 25)

    def test_batting_rates(self) -> None:
        row = {key: 0 for key in CORE["batter"]}
        row.update(AB=4, PA=5, H=2, BB=1, Double=1, SO=1)
        result = summarize([row], "batter")
        self.assertEqual(result["AVG"], 0.5)
        self.assertEqual(result["SLG"], 0.75)
        self.assertEqual(result["OPS"], 1.35)

    def test_empty_logs_override_season_count(self) -> None:
        meta = {"role": "pitcher", "expected_games": 1}
        self.assertEqual(parse_logs('{"error":false,"data":[]}', meta, "fixture"), [])

    def test_parse_log(self) -> None:
        game = self.game("minor", 1)
        meta = {key: game[key] for key in (
            "year", "role", "player_id", "player_name", "level"
        )}
        meta.update(season_id="season", expected_games=2)
        payload = {"error": False, "data": [{
            "uniqid": "g1", "date": game["date"], "seq": 1,
            "pitching": {key: game[key] for key in CORE["pitcher"]},
        }]}
        logs = parse_logs(json.dumps(payload), meta, "fixture")
        self.assertEqual(logs[0]["IPOut"], 3)
        self.assertEqual(logs[0]["source_url"], "fixture")
        self.assertEqual(len(logs), 1)

    def test_box_recovers_zero_pa_appearance(self) -> None:
        batting = {key: 0 for key in CORE["batter"]}
        batting["2B"] = batting.pop("Double")
        batting["3B"] = batting.pop("Triple")
        batting["player"] = {"uniqid": "p"}
        payload = {"error": False, "data": {
            "uniqid": "g", "home": {"box": {"battings": []}},
            "away": {"box": {"battings": [batting]}},
        }}
        stats = batting_from_box(json.dumps(payload), "p", "g", "box")
        self.assertEqual(stats["PA"], 0)
        meta = dict(year="2025", role="batter", player_id="p", player_name="P",
                    level="minor", season_id="s", expected_games=1)
        logs = {"error": False, "data": [{
            "uniqid": "g", "date": "2025-04-01", "batting": stats,
            "stats_source_url": "box",
        }]}
        parsed = parse_logs(json.dumps(logs), meta, "logs")
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["stats_source_url"], "box")

    def test_box_does_not_invent_stats(self) -> None:
        payload = {"error": False, "data": {
            "uniqid": "g", "home": {"box": {"battings": []}},
            "away": {"box": {"battings": []}},
        }}
        with self.assertRaises(DataError):
            batting_from_box(json.dumps(payload), "p", "g", "box")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--years", type=int, nargs="+")
    parser.add_argument("--player-id", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delay", type=positive, default=1.5)
    parser.add_argument("--timeout", type=positive, default=45)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TransitionTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if args.source is None:
        parser.error("provide the source CSV")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    try:
        run(args)
    except (OSError, ValueError, DataError, csv.Error) as exc:
        logging.error("Stopped: %s; incomplete outputs must not be used.", exc)
        return 1
    except KeyboardInterrupt:
        logging.error("Interrupted; incomplete outputs must not be used.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
