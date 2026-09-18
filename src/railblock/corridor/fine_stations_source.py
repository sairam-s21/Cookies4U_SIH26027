"""Raw transcription of train 12243 (MGR Chennai Central - Coimbatore
Shatabdi Express)'s full stop-by-stop timetable, source:
docs/Full Time-Table for MGR Chennai Central - Coimbatore Shatabdi
Express_12243_ MGR Chennai to Coimbatore - Railway Enquiry.pdf (site:
indiarailinfo.com, a crowd-sourced railfan enquiry site -- NOT an official
IR publication, same caveat as any secondary source; distances/station
existence are corroborated against the official Train_details_22122017.csv
derivation in fine_stations.py's cross-validation, see PROGRESS.md Session
4 for the comparison results).

This is the single most granular real point-list available for this
corridor, including signal cabins and block huts alongside commercial
halts -- these ARE real block-boundary points on the physical line, not
noise (see PROGRESS.md Session 4 for why they're kept rather than
filtered).

Session 16, at explicit user request: project scope reduced from the full
MAS-CBE run to MAS-JTJ only (Chennai Central to Jolarpettai Jn) -- the
original transcription covered all 102 points through Coimbatore Jn; this
list is now truncated at JTJ (57 points, MAS through JTJ inclusive) to cut
scheduling problem size (fewer sections -> fewer CP-SAT windows/variables).
Everything originally transcribed past JTJ (Jolarpettai Chord Cabin
onward) was simply dropped, not renumbered or otherwise altered -- the
kept rows are a verbatim prefix of the original real transcription.

Each row: (station_code, station_name, x_o_flag, distance_km, code_source)
  - x_o_flag: "X" = crossing station with loop line, "O" = originating-type,
    "XO" = both, "" = plain halt/pass-through (per the source site's own
    legend) -- kept as-is, not reinterpreted.
  - distance_km: cumulative km from MAS, AS RECORDED BY THIS TRAIN'S OWN
    TIMETABLE. Train 12243 originates at MAS (distance 0 there), so unlike
    Train_details_22122017.csv's cross-train stations (each measured
    relative to that train's own, possibly different, source station),
    every one of these km values is already corridor-referenced from
    Chennai directly -- no offset/anchoring math needed, a cleaner distance
    source than derive_stations.py's cross-train median approach.
  - code_source: "timetable" for codes as printed in the source; four cabin
    entries had their code truncated in the source page's rendering
    ("XX-I...", "XX-J...", "XX-...", "XX-S..." for Integral Coach Factory,
    Jolarpettai Chord Cabin, Magnesite Bypass Cabin, and Salem Outer Cabin
    respectively) -- Integral Coach Factory's code is filled in as "ICF"
    (a well-known, publicly documented real IR station code, not a guess);
    the other three have no publicly documented short code available, so a
    readable synthetic code is used instead and tagged code_source=
    "inferred_truncated" so it is never silently presented as verified.
"""

from __future__ import annotations

# fmt: off
TRAIN_12243_STOPS: list[tuple[str, str, str, float, str]] = [
    ("MAS",  "MGR Chennai Central",          "",   0.0, "timetable"),
    ("BBQ",  "Basin Bridge Jn",               "X",  2.1, "timetable"),
    ("VPY",  "Vyasarpadi",                    "X",  3.4, "timetable"),
    ("VJM",  "Vyasarpadi Jeeva",              "",   4.1, "timetable"),
    ("PER",  "Perambur",                      "",   5.5, "timetable"),
    ("PCW",  "Perambur Carriage Works",       "",   6.1, "timetable"),
    ("PEW",  "Perambur Locomotive Works",     "X",  7.5, "timetable"),
    ("ICF",  "Integral Coach Factory",        "",   8.3, "inferred_truncated"),
    ("VLK",  "Villivakkam",                   "",   9.3, "timetable"),
    ("KOTR", "Korattur",                      "",  12.1, "timetable"),
    ("PVM",  "Pattaravakkam",                 "",  14.0, "timetable"),
    ("ABU",  "Ambattur",                      "O", 15.5, "timetable"),
    ("TMVL", "Tirumullaivayil",               "",  17.2, "timetable"),
    ("ANNR", "Annanur",                       "",  18.3, "timetable"),
    ("AVD",  "Avadi",                         "",  21.0, "timetable"),
    ("HC",   "Hindu College",                 "",  23.9, "timetable"),
    ("PAB",  "Pattabiram",                    "",  24.9, "timetable"),
    ("NEC",  "Nemilichery",                   "",  27.1, "timetable"),
    ("TI",   "Tiruninravur",                  "",  29.0, "timetable"),
    ("VEU",  "Veppampattu",                   "",  32.3, "timetable"),
    ("SVR",  "Sevvapet Road",                 "",  35.7, "timetable"),
    ("PTLR", "Putlur Halt",                   "",  38.9, "timetable"),
    ("TRL",  "Tiruvallur",                    "",  41.6, "timetable"),
    ("EGT",  "Egattur",                       "X", 44.9, "timetable"),
    ("KBT",  "Kadambattur",                   "",  47.3, "timetable"),
    ("SPAM", "Senji Panambakkam",             "",  50.3, "timetable"),
    ("MAF",  "Manavur",                       "",  54.1, "timetable"),
    ("TO",   "Tiruvalangadu",                 "X", 58.2, "timetable"),
    ("MSU",  "Mosur",                         "",  63.8, "timetable"),
    ("PLMG", "Puliyamangalam",                "",  66.8, "timetable"),
    ("AJLS", "Arakkonam Electric Loco Shed",  "X", 67.3, "timetable"),
    ("AJJ",  "Arakkonam Jn",                  "X", 68.6, "timetable"),
    ("MLPM", "Melpakkam Cabin",               "X", 71.3, "timetable"),
    ("CTRE", "Chitteri",                      "X", 76.4, "timetable"),
    ("AVN",  "Anavardikhanpettai",            "",  80.3, "timetable"),
    ("MDVE", "Mahendra Vadi",                 "X", 86.6, "timetable"),
    ("SHU",  "Sholinghur",                    "X", 90.0, "timetable"),
    ("TUG",  "Thalangai",                     "",  97.1, "timetable"),
    ("MRLM", "Marudalam",                     "",  99.6, "timetable"),
    ("WJR",  "Walajah Road Jn",               "X", 105.0, "timetable"),
    ("MCN",  "Mukundarayapuram",              "X", 112.8, "timetable"),
    ("THL",  "Tiruvalam",                     "X", 117.7, "timetable"),
    ("SVUR", "Sevur",                         "XO", 123.8, "timetable"),
    ("KPD",  "Katpadi Jn",                    "X", 129.6, "timetable"),
    ("LTI",  "Latteri",                       "",  137.4, "timetable"),
    ("VJ",   "Virinchipuram",                 "X", 141.5, "timetable"),
    ("KVN",  "Kavanur",                       "X", 148.1, "timetable"),
    ("GYM",  "Gudiyattam",                    "X", 154.1, "timetable"),
    ("MEH",  "Melalathur",                    "X", 159.0, "timetable"),
    ("VLT",  "Valathoor",                     "X", 164.0, "timetable"),
    ("MPI",  "Melpatti",                      "X", 170.0, "timetable"),
    ("PCKM", "Pachchakuppam",                 "",  175.7, "timetable"),
    ("AB",   "Ambur",                         "",  181.7, "timetable"),
    ("VGM",  "Vinnamangalam",                 "X", 189.6, "timetable"),
    ("VN",   "Vaniyambadi",                   "XO", 197.8, "timetable"),
    ("KDY",  "Kettandapatti",                 "X", 206.8, "timetable"),
    ("JTJ",  "Jolarpettai Jn",                "",  214.1, "timetable"),
]
# fmt: on

# Session 16, at explicit user request: scope reduced to MAS-JTJ (see
# module docstring) -- truncated from the original 102-point MAS-CBE
# transcription to these 57.
assert len(TRAIN_12243_STOPS) == 57, f"expected 57 points, got {len(TRAIN_12243_STOPS)}"
