"""Derive the real MAS-JTJ corridor station list from the timetable data.

Session 16, at explicit user request: project scope reduced from the full
MAS-CBE run to MAS-JTJ (Chennai Central to Jolarpettai Jn) only, to cut
scheduling problem size -- fewer sections means fewer CP-SAT
windows/variables and a materially faster solve. This is a scope
reduction, not a feature change: the derivation method below is unchanged,
only the far anchor moved from CBE to JTJ.

Method (per docs/MASTER_PROMPT_SIH_26027.md Section 2 and the Session 1 brief):

1. From Train_details_22122017.csv, find every train whose stop sequence
   contains both corridor anchor stations (AJJ, JTJ) as an ordered
   subsequence, in either direction (a train can run MAS->JTJ or
   JTJ->MAS; both must qualify since either direction proves the physical
   route passes through the whole corridor).
2. For each qualifying train, take every stop between AJJ and JTJ inclusive
   (not just the two anchors) and union the station codes across ALL
   qualifying trains, so minor halts that only local/passenger trains stop
   at are still captured.
3. MAS is added explicitly (distance 0) since it is the fixed Chennai-end
   corridor terminus, not one of the anchors.
4. Cross-validate against india_railway_stations.csv by geographic
   proximity to the MAS-JTJ line as a secondary, non-authoritative check
   (see geographic_cross_check) -- flagged candidates are reported, not
   silently added, because a straight line does not follow the real curved
   track and picks up branch-line/suburban stations near the two ends.

Run directly to (re)generate data/derived/corridor_stations.csv:
    python -m railblock.corridor.derive_stations
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from railblock.paths import (
    CORRIDOR_STATIONS_CSV,
    GEO_FLAGGED_CSV,
    STATIONS_CSV,
    TRAIN_DETAILS_CSV,
)

ANCHOR_CODES = ["AJJ", "JTJ"]
FAR_ANCHOR_CODE = "JTJ"  # Session 16: was "CBE" before the MAS-JTJ scope reduction
MAS_CODE = "MAS"

# Manually verified by the user before this derivation was implemented.
# The derivation is expected to reproduce (at least) this set -- see the
# Session 1 test that asserts this. Session 16: trimmed to the MAS-JTJ
# scope (first 9 of the original 19).
KNOWN_VERIFIED_9 = [
    "MAS", "AJJ", "WJR", "MCN", "KPD", "GYM", "AB", "VN", "JTJ",
]


def load_timetable(path=TRAIN_DETAILS_CSV) -> tuple[pd.DataFrame, int]:
    """Load Train_details_22122017.csv and drop malformed rows.

    A small number of rows (10 out of 186,124 in the Dec-2017 snapshot) have
    an embedded newline in a station name that shifts every following field
    by one column, so ``Distance``/``SEQ`` end up holding a time string.
    None of the affected trains (UBL/BJP/SUR/GDG, all in Karnataka) touch
    this corridor, so they are safely dropped. Returns (clean_df, n_dropped).
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    before = len(df)
    numeric = df["Distance"].str.match(r"^\d+$", na=False) & df["SEQ"].str.match(
        r"^\d+$", na=False
    )
    df = df[numeric].copy()
    df["SEQ"] = df["SEQ"].astype(int)
    df["Distance"] = df["Distance"].astype(int)
    return df, before - len(df)


def _find_qualifying_trains(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame, int]]:
    """Return (train_no, span_df, sign) for every train that stops at both
    anchors in order (either direction). span_df holds every stop between
    AJJ and JTJ inclusive; sign is +1 for MAS->JTJ orientation, -1 for JTJ->MAS.
    """
    qualifying = []
    for train_no, g in df.groupby("Train No"):
        g = g.sort_values("SEQ")
        codes = list(g["Station Code"])
        if not all(a in codes for a in ANCHOR_CODES):
            continue
        fwd_positions = [codes.index(a) for a in ANCHOR_CODES]
        is_fwd = all(
            fwd_positions[i] < fwd_positions[i + 1] for i in range(len(fwd_positions) - 1)
        )
        rev_positions = [codes.index(a) for a in reversed(ANCHOR_CODES)]
        is_rev = all(
            rev_positions[i] < rev_positions[i + 1] for i in range(len(rev_positions) - 1)
        )
        if not (is_fwd or is_rev):
            continue

        ajj_row = g[g["Station Code"] == "AJJ"].iloc[0]
        far_row = g[g["Station Code"] == FAR_ANCHOR_CODE].iloc[0]
        lo, hi = sorted([ajj_row["SEQ"], far_row["SEQ"]])
        span = g[(g["SEQ"] >= lo) & (g["SEQ"] <= hi)]
        sign = 1 if far_row["Distance"] > ajj_row["Distance"] else -1
        qualifying.append((train_no, span, sign, ajj_row["Distance"]))
    return qualifying


def _union_with_canonical_distance(qualifying) -> pd.DataFrame:
    """Union every station in every qualifying train's AJJ..CBE span, mapping
    each train's own (source-relative) Distance column onto a shared corridor
    kilometrage anchored at AJJ = 68 km (AJJ's real distance from MAS on the
    direct MAS-CBE trains). Per-station canonical distance is the median
    across all trains that report it, to smooth normal cross-train noise.
    """
    rows = []
    for train_no, span, sign, ajj_distance in qualifying:
        for _, row in span.iterrows():
            canonical_km = 68 + sign * (row["Distance"] - ajj_distance)
            rows.append(
                {
                    "code": row["Station Code"],
                    "name": row["Station Name"].strip(),
                    "canonical_km": canonical_km,
                    "train_no": train_no,
                }
            )
    union = pd.DataFrame(rows)
    summary = (
        union.groupby("code")
        .agg(
            name=("name", "first"),
            distance_km=("canonical_km", "median"),
            n_trains=("train_no", "nunique"),
            distance_std_km=("canonical_km", "std"),
        )
        .reset_index()
        .rename(columns={"code": "station_code"})
    )
    summary["distance_std_km"] = summary["distance_std_km"].fillna(0.0)
    return summary


def derive_corridor_stations(
    timetable_path=TRAIN_DETAILS_CSV,
) -> tuple[pd.DataFrame, int]:
    """Run the full derivation. Returns (stations_df, n_dropped_rows).

    stations_df columns: station_code, station_name, distance_km,
    sequence_order, n_trains, distance_std_km, source.
    """
    df, n_dropped = load_timetable(timetable_path)
    qualifying = _find_qualifying_trains(df)
    if not qualifying:
        raise ValueError(
            "No trains found containing both anchor stations "
            f"{ANCHOR_CODES} as an ordered subsequence -- check the input file."
        )
    union = _union_with_canonical_distance(qualifying)
    union["source"] = "train_union"

    mas_row = pd.DataFrame(
        [
            {
                "station_code": MAS_CODE,
                "name": "CHENNAI CENTRAL",
                "distance_km": 0.0,
                "n_trains": sum(1 for q in qualifying if q[2] == 1),
                "distance_std_km": 0.0,
                "source": "added_explicit",
            }
        ]
    )
    combined = pd.concat([mas_row, union], ignore_index=True)
    combined = combined.sort_values("distance_km").reset_index(drop=True)
    combined["sequence_order"] = combined.index + 1
    combined = combined.rename(columns={"name": "station_name"})
    combined = combined[
        [
            "sequence_order",
            "station_code",
            "station_name",
            "distance_km",
            "n_trains",
            "distance_std_km",
            "source",
        ]
    ]
    return combined, n_dropped


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def geographic_cross_check(
    corridor_stations: pd.DataFrame,
    stations_path=STATIONS_CSV,
    max_km_from_line: float = 8.0,
    bbox_margin_deg: float = 0.3,
) -> pd.DataFrame:
    """Flag real stations geographically close to the straight MAS-JTJ line
    that were NOT captured by the train-union derivation.

    This is a secondary, non-authoritative check: real track is curved and
    branches near both terminal city networks, so proximity to the straight
    line does not prove a station sits on this specific physical corridor.
    Results are returned for human review, never auto-merged into the
    corridor station list.
    """
    st = pd.read_csv(stations_path)
    st.columns = [c.strip() for c in st.columns]
    st = st.dropna(subset=["latitude", "longitude"]).copy()

    mas = st[st["station_code"] == MAS_CODE].iloc[0]
    far = st[st["station_code"] == FAR_ANCHOR_CODE].iloc[0]
    lat1, lon1 = mas["latitude"], mas["longitude"]
    lat2, lon2 = far["latitude"], far["longitude"]

    ts = np.linspace(0, 1, 300)
    line_lats = lat1 + ts * (lat2 - lat1)
    line_lons = lon1 + ts * (lon2 - lon1)

    def dist_to_line(lat, lon):
        return _haversine_km(lat, lon, line_lats, line_lons).min()

    st["dist_to_line_km"] = st.apply(
        lambda r: dist_to_line(r["latitude"], r["longitude"]), axis=1
    )

    latmin, latmax = sorted([lat1, lat2])
    lonmin, lonmax = sorted([lon1, lon2])
    near = st[
        (st["dist_to_line_km"] <= max_km_from_line)
        & st["latitude"].between(latmin - bbox_margin_deg, latmax + bbox_margin_deg)
        & st["longitude"].between(lonmin - bbox_margin_deg, lonmax + bbox_margin_deg)
    ]

    already_found = set(corridor_stations["station_code"])
    flagged = near[~near["station_code"].isin(already_found)].sort_values(
        "dist_to_line_km"
    )
    return flagged[
        ["station_code", "station_name", "latitude", "longitude", "dist_to_line_km", "is_junction"]
    ].reset_index(drop=True)


def main() -> None:
    stations, n_dropped = derive_corridor_stations()
    stations.to_csv(CORRIDOR_STATIONS_CSV, index=False)

    flagged = geographic_cross_check(stations)
    flagged.to_csv(GEO_FLAGGED_CSV, index=False)

    print(f"Dropped {n_dropped} malformed timetable rows before derivation.")
    print(f"Derived {len(stations)} corridor stations -> {CORRIDOR_STATIONS_CSV}")
    print(stations.to_string(index=False))

    known = set(KNOWN_VERIFIED_9)
    derived = set(stations["station_code"])
    missing = known - derived
    extra = derived - known
    if missing:
        print(f"\nWARNING: derivation is MISSING known-verified stations: {sorted(missing)}")
    if extra:
        print(f"\nFlag: derivation found {len(extra)} station(s) beyond the known-9 list: {sorted(extra)}")
    if not missing and not extra:
        print("\nDerived set exactly matches the known-verified 9-station (MAS-JTJ) list.")

    print(
        f"\nGeographic cross-check flagged {len(flagged)} additional real station(s) "
        f"near the straight MAS-JTJ line but not in the derived list "
        f"(see {GEO_FLAGGED_CSV} -- review manually, do not auto-merge)."
    )


if __name__ == "__main__":
    main()
