"""Real per-station delay history
for the corridor's 225 confirmed-real trains (data/derived/
pdf_confirmed_train_roster_2026.csv), sourced from etrain.info -- a
public, no-auth-required Indian train running-history site. Chosen over
RailRadar's own API host (api.railradar.in) because that host resets the
TLS connection on every attempt from this environment (confirmed with two
independent HTTP clients: "Connection reset by peer" during the
handshake) while the plain https://railradar.in marketing site connects
fine -- an environment-level block on that specific API subdomain, not a
key/quota problem. etrain.info has no such block and needs no API key at
all.

How a train's real URL slug is resolved: etrain.info's own sitemap
(https://etrain.info/sitemap-dynamic.xml, fetched fresh each run) lists
every train page it tracks; each corridor train's real 4-5 digit number
is matched against the sitemap's URLs (which all end in -<number>) to
get its real slug -- no name-guessing, no Selenium. 223 of the corridor's
225 real trains were confirmed present this way; the other 2 (43805,
43821 -- MEMU/suburban numbers) aren't tracked by the site at all and are
handled by _synthesize_missing_train, clearly tagged data_source=
SYNTHETIC in the output, never blended in as if real.

Each train's /history page embeds two real JS data arrays directly in
its static HTML (no JS execution needed to read them, confirmed by
`curl` alone returning them):
  - et.rsStat.primaryData: one row per real corridor station on this
    train's route, [station_code, right_time_count, slight_delay_count,
    significant_delay_count, cancelled_count, avg_delay_minutes] --
    an aggregate over roughly the last month of real running days.
  - et.rsStat.tooltipData: a header row naming each station (in column
    order) followed by one row per real calendar date actually observed,
    each with that date's real per-station delay in minutes -- genuine
    day-level history, not just an aggregate.

Resumable (same pattern as fetch_real_train_data.py): progress written
to DELAY_HISTORY_PROGRESS_JSON after every single train, so an
interrupted run (Ctrl+C, network blip) picks up where it left off rather
than re-fetching already-confirmed trains. Run:

    python -m railblock.integrations.fetch_train_delay_history

Writes TRAIN_DELAY_HISTORY_CSV (long format: one row per train/station/
date, real day-level delay) and TRAIN_DELAY_STATION_SUMMARY_CSV (one row
per train/station, the aggregate) every run, from whatever is currently
in the progress cache.
"""

from __future__ import annotations

import json
import random
import re
import time
from datetime import date

import httpx

from railblock.paths import (
    DELAY_HISTORY_PROGRESS_JSON,
    PROJECT_ROOT,
    TRAIN_DELAY_HISTORY_CSV,
    TRAIN_DELAY_STATION_SUMMARY_CSV,
)

ROSTER_CSV = PROJECT_ROOT / "data" / "derived" / "pdf_confirmed_train_roster_2026.csv"
SITEMAP_URL = "https://etrain.info/sitemap-dynamic.xml"
REQUEST_DELAY_S = 0.8  # politeness pacing -- 225 trains at this rate is a few minutes, not a hammering
TIMEOUT_S = 15.0
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RailBlockCoPilot-research/1.0)"}


def _load_roster() -> list[dict]:
    import csv
    with open(ROSTER_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_number_to_url(sitemap_xml: str) -> dict[str, str]:
    urls = set(re.findall(r"https://etrain\.info/train/[A-Za-z0-9-]+", sitemap_xml))
    by_number: dict[str, str] = {}
    for u in urls:
        m = re.search(r"-(\d{4,5})$", u)
        if m:
            by_number[m.group(1)] = u
    return by_number


def _extract_bracketed(html: str, marker: str) -> str | None:
    """These are JS array literals embedded in static HTML,
    not JSON (single quotes, `new Date(...)`, unquoted-ish bits) -- a
    hand-rolled bracket-depth scan is more robust here than a fragile
    single regex, since the arrays nest (row-of-rows) and a non-greedy
    regex would stop at the first inner `]`."""
    start = html.find(marker)
    if start == -1:
        return None
    bracket_start = html.find("[", start)
    if bracket_start == -1:
        return None
    depth = 0
    for i in range(bracket_start, len(html)):
        if html[i] == "[":
            depth += 1
        elif html[i] == "]":
            depth -= 1
            if depth == 0:
                return html[bracket_start:i + 1]
    return None


def _parse_primary_data(blob: str) -> list[dict]:
    """[station_code, right_time, slight_delay, significant_delay,
    cancelled, avg_delay_minutes] per real corridor station on this
    train's route."""
    rows = []
    for m in re.finditer(
        r"\['([A-Z]+)',\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+)\]",
        blob,
    ):
        station, right, slight, sig, cancelled, avg_delay = m.groups()
        rows.append({
            "station_code": station,
            "right_time_count": float(right),
            "slight_delay_count": float(slight),
            "significant_delay_count": float(sig),
            "cancelled_count": float(cancelled),
            "avg_delay_minutes": float(avg_delay),
        })
    return rows


def _parse_tooltip_data(blob: str) -> list[dict]:
    """Header row's {'label': 'XXX'} entries give real station order;
    each `[new Date(Y,M,D), v1, v2, ...]` data row is that REAL
    CALENDAR DATE's real per-station delay in minutes, same order.
    JS Date month is 0-indexed (7 == August), unlike Python's."""
    header_end = blob.find("]", blob.find("["))
    header = blob[:header_end + 1] if header_end != -1 else ""
    stations = re.findall(r"'label'\s*:\s*'([A-Z]+)'", header)
    if not stations:
        return []

    rows = []
    for m in re.finditer(r"\[new Date\((\d+),(\d+),(\d+)\),([^\]]+)\]", blob):
        y, mo, d, rest = m.groups()
        try:
            d_obj = date(int(y), int(mo) + 1, int(d))  # JS month is 0-indexed
        except ValueError:
            continue
        values = [v.strip() for v in rest.split(",") if v.strip() != ""]
        for station, val in zip(stations, values):
            try:
                delay = float(val)
            except ValueError:
                continue
            rows.append({"station_code": station, "date": d_obj.isoformat(), "delay_minutes": delay})
    return rows


def _fetch_train_history(client: httpx.Client, url: str) -> tuple[list[dict], list[dict]] | None | str:
    """Returns (summary, history) on success, None on a genuine fetch
    failure (network error / non-200 -- worth stopping and retrying
    later, same as the rest of this project's real-quota-stop
    convention), or the string "NO_DATA" when the page itself loaded
    fine (200, a real train) but simply has no running-history stats
    block at all -- confirmed real for some local/passenger services
    (e.g. 43892 PRES AVD LOCAL) that etrain.info tracks by page but
    doesn't compute punctuality stats for. That's a conclusive real
    answer, not a transient failure -- retrying won't change it, so the
    caller treats it as a synthesize-and-continue case, not a stop."""
    try:
        r = client.get(f"{url}/history", headers=HEADERS, timeout=TIMEOUT_S, follow_redirects=True)
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    html = r.text
    primary_blob = _extract_bracketed(html, "et.rsStat.primaryData")
    tooltip_blob = _extract_bracketed(html, "et.rsStat.tooltipData")
    if not primary_blob or not tooltip_blob:
        return "NO_DATA"
    return _parse_primary_data(primary_blob), _parse_tooltip_data(tooltip_blob)


# Illustrative-only fallback for trains etrain.info doesn't track at all
# (confirmed by sitemap absence, not a fetch failure) -- disclosed the
# same way this project's other synthetic data is (IMPACT_WEIGHT,
# DEFECT_DURATION_HOURS): a reasoned placeholder, never fabricated as if
# it were a real observation. Mean/spread are a rough, named guess for a
# local MEMU/passenger service, not derived from any real dataset --
# tighten or replace outright the moment real data for these two trains
# becomes available.
_SYNTHETIC_MEAN_DELAY_MIN = 12.0
_SYNTHETIC_STD_DELAY_MIN = 6.0


def _synthesize_missing_train(train_no: str, source: str, destination: str, rng: random.Random) -> list[dict]:
    rows = []
    for station in (source, destination):
        if not station:
            continue
        delay = max(0.0, rng.gauss(_SYNTHETIC_MEAN_DELAY_MIN, _SYNTHETIC_STD_DELAY_MIN))
        rows.append({
            "station_code": station,
            "avg_delay_minutes": round(delay, 1),
            "right_time_count": None, "slight_delay_count": None,
            "significant_delay_count": None, "cancelled_count": None,
        })
    return rows


def main() -> None:
    roster = _load_roster()
    print(f"Loaded {len(roster)} real corridor trains from {ROSTER_CSV.name}")

    progress: dict = {}
    if DELAY_HISTORY_PROGRESS_JSON.exists():
        progress = json.loads(DELAY_HISTORY_PROGRESS_JSON.read_text())
        print(f"Resuming: {len(progress)} trains already done in a previous run.")

    remaining = [row for row in roster if row["Train No"].strip() not in progress]
    if not remaining:
        print("Nothing left to fetch -- writing final CSVs from existing progress.")
    else:
        print(f"Fetching {len(remaining)} remaining trains from etrain.info...")
        with httpx.Client() as client:
            sitemap_resp = client.get(SITEMAP_URL, headers=HEADERS, timeout=30.0)
            by_number = _build_number_to_url(sitemap_resp.text)
            print(f"Sitemap: {len(by_number)} real train pages indexed.")

            rng = random.Random(42)
            for i, row in enumerate(remaining, 1):
                train_no = row["Train No"].strip()
                url = by_number.get(train_no)
                if url is None:
                    synth = _synthesize_missing_train(train_no, row.get("Source Station", ""), row.get("Destination Station", ""), rng)
                    progress[train_no] = {"data_source": "SYNTHETIC", "history": [], "summary": synth}
                    print(f"  [{i}/{len(remaining)}] {train_no}: not on etrain.info -- synthesized ({len(synth)} station rows)")
                    continue

                result = _fetch_train_history(client, url)
                if result is None:
                    print(f"  [{i}/{len(remaining)}] {train_no}: fetch failed (network/rate-limit) -- will retry next run, stopping here.")
                    break
                if result == "NO_DATA":
                    synth = _synthesize_missing_train(train_no, row.get("Source Station", ""), row.get("Destination Station", ""), rng)
                    progress[train_no] = {"data_source": "SYNTHETIC", "history": [], "summary": synth}
                    print(f"  [{i}/{len(remaining)}] {train_no}: real page, no tracked stats -- synthesized ({len(synth)} station rows)")
                else:
                    summary, history = result
                    progress[train_no] = {"data_source": "REAL_ETRAIN", "history": history, "summary": summary}
                    print(f"  [{i}/{len(remaining)}] {train_no}: {len(summary)} stations, {len(history)} real daily observations")
                DELAY_HISTORY_PROGRESS_JSON.write_text(json.dumps(progress))
                time.sleep(REQUEST_DELAY_S)

    DELAY_HISTORY_PROGRESS_JSON.write_text(json.dumps(progress))

    history_rows, summary_rows = [], []
    for train_no, entry in progress.items():
        for h in entry["history"]:
            history_rows.append({"train_no": train_no, "data_source": entry["data_source"], **h})
        for s in entry["summary"]:
            summary_rows.append({"train_no": train_no, "data_source": entry["data_source"], **s})

    import csv
    if history_rows:
        with open(TRAIN_DELAY_HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["train_no", "data_source", "station_code", "date", "delay_minutes"])
            w.writeheader()
            w.writerows(history_rows)
    if summary_rows:
        with open(TRAIN_DELAY_STATION_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "train_no", "data_source", "station_code", "avg_delay_minutes",
                "right_time_count", "slight_delay_count", "significant_delay_count", "cancelled_count",
            ])
            w.writeheader()
            w.writerows(summary_rows)

    n_real = sum(1 for e in progress.values() if e["data_source"] == "REAL_ETRAIN")
    n_synth = sum(1 for e in progress.values() if e["data_source"] == "SYNTHETIC")
    print(f"\nDone: {len(progress)}/{len(roster)} trains ({n_real} real from etrain.info, {n_synth} synthetic fallback)")
    print(f"  {TRAIN_DELAY_HISTORY_CSV.relative_to(PROJECT_ROOT)}: {len(history_rows)} real daily observations")
    print(f"  {TRAIN_DELAY_STATION_SUMMARY_CSV.relative_to(PROJECT_ROOT)}: {len(summary_rows)} station summary rows")


if __name__ == "__main__":
    main()
