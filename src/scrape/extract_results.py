"""Scrape per-event tournament results from DinkBank into data/results_raw.csv.

The player pages embed an `event_played` array in their Next.js payload: one row
per player per bracket, with tour, placement, prize and -- the part that matters
-- `start_date`. That date is what makes on-court form usable as a feature at
all: results before the draft are legitimate inputs, results after it are the
answer leaking backwards.

Two limits found while building this, both carried downstream into DATA.md:
  * DinkBank's results table only covers 2026. There is no 2023-2025 match
    history here, so "form" means the 2026 season up to draft day, not a career.
  * The page renders 10 rows at a time; `?page=N` walks the rest.

Regenerate: python3 src/scrape/extract_results.py
(Pages cache to .cache/dinkbank/, so re-parsing never re-crawls.)
"""
import csv, json, re, ssl, sys, time, urllib.request
from pathlib import Path

import certifi

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
SSL_CTX = ssl.create_default_context(cafile=certifi.where())
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = DATA / "results_raw.csv"
CACHE = ROOT / ".cache" / "dinkbank"
DELAY = 0.4
MAX_PAGES = 12

# The 2026 MLP draft. Dated from the live-updates article that drafts_raw.csv
# cites (datePublished 2026-02-27). Everything on or after this is post-draft.
DRAFT_DATE = "2026-02-27"

FIELDS = ["player_key", "slug", "event_id", "event_name", "seo_slug", "tour",
          "bracket", "placement", "amount_usd", "start_date", "event_year",
          "location", "source", "badge", "pre_draft"]


def get(url):
    f = CACHE / (re.sub(r"[^a-z0-9]+", "_", url.lower().split("//")[-1]) + ".html")
    if f.exists():
        return f.read_text(encoding="utf-8", errors="ignore")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    raw = urllib.request.urlopen(req, timeout=30, context=SSL_CTX).read().decode("utf-8", "ignore")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(raw, encoding="utf-8")
    time.sleep(DELAY)
    return raw


def extract_array(raw, key):
    """Pull one JSON array out of the escaped Next.js flight payload."""
    u = raw.replace('\\"', '"')
    i = u.find('"%s":[' % key)
    if i < 0:
        return None
    j = u.index("[", i)
    depth = 0
    for k in range(j, len(u)):
        if u[k] == "[":
            depth += 1
        elif u[k] == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(u[j:k + 1])
                except json.JSONDecodeError:
                    return None
    return None


def player_events(slug):
    """Every event row for one player, walking ?page=N until nothing new."""
    rows, seen = [], set()
    for page in range(1, MAX_PAGES + 1):
        url = f"https://www.dinkbank.com/en/players/{slug}"
        if page > 1:
            url += f"?page={page}"
        try:
            evs = extract_array(get(url), "event_played") or []
        except Exception as e:
            print(f"    {slug} page {page}: {type(e).__name__}", file=sys.stderr)
            break
        # An event appears once per bracket entered, so the pair is the identity.
        new = [e for e in evs if (e.get("event_id"), e.get("bracket")) not in seen]
        for e in evs:
            seen.add((e.get("event_id"), e.get("bracket")))
        rows += new
        if not evs or not new:
            break
    return rows


def main():
    drafts = list(csv.DictReader((DATA / "drafts.csv").open()))
    players = {r["player_key"]: r for r in csv.DictReader((DATA / "players.csv").open())}

    targets = {}
    for d in drafts:
        if d["season"] == "2026" and d["match_kind"] != "unmatched":
            p = players.get(d["player_key"])
            if p and p.get("slug"):
                targets[d["player_key"]] = p["slug"]

    print(f"{len(targets)} drafted players to crawl", file=sys.stderr)
    out, no_results = [], []
    for i, (key, slug) in enumerate(sorted(targets.items()), 1):
        evs = player_events(slug)
        if not evs:
            no_results.append(slug)
        for e in evs:
            sd = e.get("start_date") or ""
            out.append({
                "player_key": key, "slug": slug,
                "event_id": e.get("event_id"), "event_name": e.get("event_name"),
                "seo_slug": e.get("seo_slug"), "tour": e.get("tour"),
                "bracket": e.get("bracket"), "placement": e.get("placement"),
                "amount_usd": e.get("amount_usd"), "start_date": sd,
                "event_year": e.get("event_year"), "location": e.get("location"),
                "source": e.get("source"), "badge": e.get("badge"),
                "pre_draft": int(bool(sd) and sd < DRAFT_DATE),
            })
        if i % 10 == 0:
            print(f"  {i}/{len(targets)}", file=sys.stderr, flush=True)

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)

    pre = [r for r in out if r["pre_draft"]]
    with_pre = len({r["player_key"] for r in pre})
    print(f"\nwrote {len(out)} event rows -> {OUT}", file=sys.stderr)
    print(f"  players crawled:        {len(targets)}", file=sys.stderr)
    print(f"  players with any result:{len(targets) - len(no_results)}", file=sys.stderr)
    print(f"  PRE-DRAFT rows (< {DRAFT_DATE}): {len(pre)}", file=sys.stderr)
    print(f"  players with pre-draft results: {with_pre}/{len(targets)}", file=sys.stderr)
    if no_results:
        print(f"  no results at all: {no_results[:8]}", file=sys.stderr)


if __name__ == "__main__":
    main()
