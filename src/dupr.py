"""Join DUPR ratings onto the drafted players.

DUPR supplied this file directly (2026-11-08) in response to a data request --
it is NOT scraped, and their terms prohibit automated access to their site.

Two rules this module enforces, both of them leakage controls:

  1. A rating dated on or after DRAFT_DATE is discarded. `rating_date` is the
     day the rating last moved, so a rating stamped on draft day reflects
     matches the bidders had not seen when they bid.
  2. A rating of exactly 0.000 is DUPR's "no rating" placeholder, not a skill
     of zero. It becomes NaN.

The raw file is gitignored on purpose: the data request said this project was
not looking to republish ratings, and the published app honours that. See
data/DATA.md.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
RATINGS = DATA / "dupr_ratings.csv"

sys.path.insert(0, str(Path(__file__).resolve().parent / "scrape"))


def load_ratings(cutoff):
    """DUPR rows that predate `cutoff`, indexed for name matching."""
    if not RATINGS.exists():
        return None

    rows = list(csv.DictReader(RATINGS.open()))
    for r in rows:
        r["player"] = r["name"]          # build_index() expects this key

    kept = [r for r in rows if r["rating_date"] < cutoff]
    if any(r["rating_date"] >= cutoff for r in kept):
        raise ValueError("post-draft rating leaked past the cutoff")
    if not kept:
        raise ValueError("no DUPR ratings survived the date filter")

    return rows, kept


def attach(df, cutoff):
    """Add dupr_doubles / dupr_match_kind to df. Returns (df, report).

    Matching reuses build_dataset's exact -> initial+surname -> fuzzy ladder so
    a DUPR join is auditable the same way the DinkBank join is. Every row
    records how it matched.
    """
    from build_dataset import build_index, match

    df = df.copy()
    loaded = load_ratings(cutoff)
    if loaded is None:
        df["dupr_doubles"] = np.nan
        df["dupr_match_kind"] = "absent"
        return df, {"available": False}

    rows, kept = loaded
    index = build_index(kept)

    doubles, kinds, dates = [], [], []
    for name in df["player"]:
        r, kind = match(name, *index)
        kinds.append(kind)
        if r is None:
            doubles.append(np.nan)
            dates.append("")
            continue
        v = float(r["doubles"])
        doubles.append(v if v > 0 else np.nan)   # 0.000 is "unrated"
        dates.append(r["rating_date"])

    df["dupr_doubles"] = doubles
    df["dupr_match_kind"] = kinds
    df["dupr_rating_date"] = dates

    matched = df["dupr_doubles"].notna()
    report = {
        "available": True,
        "pool_rows": len(rows),
        "pool_after_cutoff": len(kept),
        "cutoff": cutoff,
        "matched": int(matched.sum()),
        "n": int(len(df)),
        "unmatched_players": sorted(df.loc[~matched, "player"]),
        "match_kinds": df["dupr_match_kind"].value_counts().to_dict(),
        "date_min": min((d for d in dates if d), default=""),
        "date_max": max(dates) if any(dates) else "",
    }
    return df, report


if __name__ == "__main__":
    import features
    d, rep = attach(features.load(), features.DRAFT_DATE)
    for k, v in rep.items():
        print(f"  {k:<20} {v}")
