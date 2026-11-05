"""Scrape player attributes from DinkBank into data/players_raw.csv.

DinkBank republishes DUPR ratings alongside prize money, birth year, gender and
UPA contract status. Their robots.txt allows all crawlers (ClaudeBot named
explicitly), which is why features come from here and never from DUPR directly --
DUPR's own terms prohibit automated access.

Caveat carried downstream: the DUPR ratings here are CURRENT, not as-of any past
draft date. See the leakage note in plan.md.
"""
import csv, html, json, re, ssl, sys, time, unicodedata, urllib.request
from pathlib import Path

import certifi

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
SSL_CTX = ssl.create_default_context(cafile=certifi.where())
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "players_raw.csv"
SITEMAP = "https://www.dinkbank.com/sitemap.xml"
DIRECTORY = "https://www.dinkbank.com/en/players"
DELAY = 0.4  # be polite

YEARS = (2023, 2024, 2025, 2026)
# DUPR ratings are parsed off the page but deliberately NOT written out: this
# project does not redistribute them, and the model no longer uses them.
FIELDS = (["slug", "player", "gender", "country", "birth_year",
           "contract_status", "card_tier", "earnings_ytd",
           "earnings_annual", "career_earnings", "earnings_rank", "events"]
          + [f"{k}_{y}" for y in YEARS for k in ("total", "contracts", "prize")]
          + ["source_url"])

CACHE = ROOT / ".cache" / "dinkbank"


def get(url, cache=True):
    """Fetch, caching page HTML so re-parsing never re-crawls the site."""
    f = CACHE / (re.sub(r"[^a-z0-9]+", "_", url.lower().split("//")[-1]) + ".html")
    if cache and f.exists():
        return f.read_text(encoding="utf-8", errors="ignore")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    raw = urllib.request.urlopen(req, timeout=30, context=SSL_CTX).read().decode("utf-8", "ignore")
    if cache:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(raw, encoding="utf-8")
        time.sleep(DELAY)  # only sleep on a real fetch
    return raw


def text_lines(raw):
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", raw, flags=re.S)
    return [l.strip() for l in html.unescape(re.sub(r"<[^>]+>", "\n", t)).split("\n") if l.strip()]


def money(s):
    return int(re.sub(r"[^\d]", "", s)) if s and re.search(r"\d", s) else None


def after(lines, label, pat=r".+"):
    """Value on the line following `label` (DinkBank renders label/value as siblings)."""
    for i, l in enumerate(lines):
        if l.rstrip(":").lower() == label.lower() and i + 1 < len(lines):
            if re.fullmatch(pat, lines[i + 1]):
                return lines[i + 1]
    return None


def parse_year_table(L):
    """'Career Earnings by Year' renders as: <year> <total> 'Contracts' <contract$>.

    Prize money is the remainder, and matches the page's own 'This Year's Prize
    Money' figure. These per-year splits are what make leak-free features
    possible: 2025-and-earlier values predate the Feb 2026 draft.
    """
    out = {}
    try:
        i = L.index("Career Earnings by Year")
    except ValueError:
        return out
    for j in range(i + 1, min(i + 40, len(L) - 3)):
        if re.fullmatch(r"20\d\d", L[j]) and L[j + 2] == "Contracts":
            year, total, contracts = int(L[j]), money(L[j + 1]), money(L[j + 3])
            if year in YEARS and total is not None:
                out[f"total_{year}"] = total
                out[f"contracts_{year}"] = contracts
                if contracts is not None:
                    out[f"prize_{year}"] = total - contracts
    return out


def parse_player(url, raw):
    L = text_lines(raw)
    rec = {f: None for f in FIELDS}
    rec["slug"], rec["source_url"] = url.rstrip("/").split("/")[-1], url

    blocks = re.findall(r'<script[^>]*application/ld\+json[^>]*>(.*?)</script>', raw, flags=re.S)
    if blocks:
        try:
            person = next((n for n in json.loads(blocks[0])["@graph"] if n.get("@type") == "Person"), {})
        except (json.JSONDecodeError, KeyError):
            person = {}
        rec["player"] = person.get("name")
        rec["country"] = person.get("nationality")
        # only some bios carry the event count
        if m := re.search(r"across (\d+) events", person.get("description", "")):
            rec["events"] = int(m.group(1))

    if not rec["player"]:
        # <title> is "Name | Earnings, Rankings & Career Stats | DinkBank".
        # Don't use the breadcrumb: the nav also renders a "Players" link.
        if m := re.search(r"<title>(.*?)</title>", raw, flags=re.S | re.I):
            rec["player"] = html.unescape(m.group(1)).split("|")[0].strip() or None

    for l in L:
        if m := re.match(r"^(Male|Female)(?:\s*\u2022\s*(.+))?$", l):
            rec["gender"] = m.group(1)
            rec["country"] = rec["country"] or m.group(2)
            break
    if m := re.search(r"\(born (\d{4})\)", " ".join(L)):
        rec["birth_year"] = int(m.group(1))

    if (i := next((i for i, l in enumerate(L) if l.startswith("UPA Contract Status")), None)) is not None:
        tail = L[i + 1:i + 4]
        rec["contract_status"] = next((t.rstrip(" \u2022") for t in tail if "Signed" in t or "Unsigned" in t), None)
        rec["card_tier"] = next((t for t in tail if re.fullmatch(r"Gold Card|Silver Card|Standard|Bronze Card", t)), None)

    # body values are present on every page; ld+json is not
    rec["earnings_ytd"] = money(after(L, "This Year's Prize Money", r"\$[\d,]+") or "")
    rec["earnings_annual"] = money(after(L, "Current Annual Earnings", r"\$[\d,]+") or "")
    rec["career_earnings"] = money(after(L, "Career Earnings", r"\$[\d,]+") or "")
    if rank := after(L, "Career Earnings Rank", r"#\d+"):
        rec["earnings_rank"] = int(rank.lstrip("#"))

    rec.update(parse_year_table(L))
    return rec


def slugify(name):
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", n.lower()).strip("-")


def player_urls():
    """Neither the sitemap (218) nor the directory page (401) is complete, so
    also seed a slug for every drafted player -- those are the rows we need."""
    urls = {u for u in re.findall(r"<loc>(.*?)</loc>", get(SITEMAP, cache=False)) if "/players/" in u}
    urls |= {f"https://www.dinkbank.com/en/players/{s}"
             for s in re.findall(r"/en/players/([a-z0-9\-]+)", get(DIRECTORY, cache=False))}
    drafts = ROOT / "data" / "drafts_raw.csv"
    if drafts.exists():
        with drafts.open() as f:
            urls |= {f"https://www.dinkbank.com/en/players/{slugify(r['player'])}"
                     for r in csv.DictReader(f) if r["player"]}
    return sorted(urls)


def main():
    urls = player_urls()
    print(f"{len(urls)} player pages", file=sys.stderr)
    rows, failed = [], []
    for i, u in enumerate(urls, 1):
        try:
            rec = parse_player(u, get(u))
            if rec["player"] and (rec["career_earnings"] is not None or rec["earnings_rank"]):
                rows.append(rec)
            else:
                failed.append((u, "no player payload"))
        except Exception as e:
            failed.append((u, repr(e)[:60]))
        if i % 25 == 0:
            print(f"  {i}/{len(urls)}", file=sys.stderr, flush=True)

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    have = lambda k: sum(1 for r in rows if r[k] not in (None, ""))
    print(f"wrote {len(rows)} rows -> {OUT}", file=sys.stderr)
    print("coverage: " + "  ".join(f"{k}={have(k)}" for k in
          ["player", "gender", "career_earnings", "earnings_rank",
           "events", "card_tier", "total_2025", "contracts_2025", "prize_2025"]), file=sys.stderr)
    for u, e in failed[:5]:
        print("FAIL", u, e, file=sys.stderr)


if __name__ == "__main__":
    main()
