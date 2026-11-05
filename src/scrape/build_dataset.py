"""Join raw scrapes into the analysis-ready data/drafts.csv + data/players.csv.

Draft coverage and DinkBank names disagree ("Alex Crum" vs "Alexander Crum"), so
matching runs exact -> first-initial+surname -> fuzzy, and every row records how
it matched (match_kind) plus a stable player_key both files share. features.py
should merge on player_key alone.
"""
import csv, re, sys, unicodedata
from difflib import get_close_matches
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]+", " ", s.lower()).split()


def key(s):
    return "-".join(norm(s))


def build_index(players):
    exact, initial = {}, {}
    for r in players:
        n = norm(r["player"])
        if n:
            exact.setdefault(" ".join(n), r)
        if len(n) >= 2:
            initial.setdefault(f"{n[0][0]}|{n[-1]}", []).append(r)
    return exact, initial


def match(name, exact, initial):
    n = norm(name)
    k = " ".join(n)
    if r := exact.get(k):
        return r, "exact"
    if len(n) >= 2:
        c = initial.get(f"{n[0][0]}|{n[-1]}", [])
        if len(c) == 1:
            return c[0], "initial_surname"
    if g := get_close_matches(k, exact.keys(), n=1, cutoff=0.80):
        return exact[g[0]], "fuzzy"
    return None, "unmatched"


def main():
    drafts = list(csv.DictReader((DATA / "drafts_raw.csv").open()))
    players = list(csv.DictReader((DATA / "players_raw.csv").open()))
    exact, initial = build_index(players)

    out_drafts, used = [], {}
    for d in drafts:
        p, kind = match(d["player"], exact, initial)
        pk = key(p["player"]) if p else key(d["player"])
        out_drafts.append({**d, "player_key": pk, "match_kind": kind,
                           "gender": (p or {}).get("gender") or ""})
        if p:
            used[pk] = p

    with (DATA / "drafts.csv").open("w", newline="") as f:
        cols = ["season", "pick_no", "player", "player_key", "team", "price_raw",
                "price_unit", "gender", "match_kind", "source_url"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_drafts)

    pcols = ["player_key"] + [c for c in players[0] if c != "slug"] + ["slug"]
    with (DATA / "players.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pcols, extrasaction="ignore")
        w.writeheader()
        for r in players:
            w.writerow({**r, "player_key": key(r["player"])})

    d26 = [d for d in out_drafts if d["season"] == "2026"]
    matched = [d for d in d26 if d["match_kind"] != "unmatched"]
    print(f"drafts.csv: {len(out_drafts)} rows  |  players.csv: {len(players)} rows", file=sys.stderr)
    print(f"2026: {len(d26)} picks, {len(matched)} joined to features", file=sys.stderr)
    for k in ["career_earnings", "earnings_rank", "total_2025", "gender", "card_tier"]:
        n = sum(1 for d in matched if used.get(d["player_key"], {}).get(k))
        print(f"    {k:18s} {n}/{len(matched)}", file=sys.stderr)
    miss = [d["player"] for d in d26 if d["match_kind"] == "unmatched"]
    print(f"    unmatched: {miss}", file=sys.stderr)


if __name__ == "__main__":
    main()
