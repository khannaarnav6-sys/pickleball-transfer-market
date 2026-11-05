"""Extract MLP draft picks + prices from published coverage into data/drafts_raw.csv.

Prices live only in journalism; each season has its own layout, so each gets its
own parser. Every row keeps its source_url so any number can be re-checked.

2023 and 2025 are deliberately absent: their prices were never published in a
form we can reach (Forbes carried them and blocks automated fetches).
"""
import csv, html, re, ssl, sys, urllib.request
from pathlib import Path

import certifi

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
OUT = Path(__file__).resolve().parents[2] / "data" / "drafts_raw.csv"

SOURCES = {
    2026: "https://thekitchenpickle.com/blogs/news/2026-major-league-pickleball-draft-live-updates/",
    2024: "https://pickleball.com/news/live-coverage-mlp-premier-level-draft",
}


# python.org builds ship without a wired-up trust store; point at certifi's.
SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def lines(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    raw = urllib.request.urlopen(req, timeout=30, context=SSL_CTX).read().decode("utf-8", "ignore")
    raw = re.sub(r"<script.*?</script>|<style.*?</style>", "", raw, flags=re.S)
    text = html.unescape(re.sub(r"<[^>]+>", "\n", raw))
    return [l.strip() for l in text.split("\n") if l.strip()]


# 2026: one line per pick -- "1. Anna Bright, St. Louis Shock, $1.23 million."
P2026 = re.compile(
    r"^(\d{1,2})\.\s+([A-Z][A-Za-z'\-\. ]+?),\s+(.+?),\s+\$([\d,\.]+)\s*(million|k)?\b", re.I
)


def parse_2026(url):
    out = {}
    for line in lines(url):
        m = P2026.match(line)
        if not m:
            continue
        pick, player, team, amt, scale = m.groups()
        amt = float(amt.replace(",", ""))
        if scale and scale.lower() == "million":
            amt *= 1_000_000
        elif scale and scale.lower() == "k":
            amt *= 1_000
        out.setdefault(int(pick), (player.strip(), team.strip(), int(amt)))
    return out


# 2024: three lines per pick -- "1." / "Ben Johns" / ", Seattle Pioneers-840,000"
P2024_TAIL = re.compile(r"^,\s*(.+?)(?:-([\d,]{4,}))?$")


def parse_2024(url):
    L, out = lines(url), {}
    for i, line in enumerate(L):
        m = re.match(r"^(\d{1,2})\.$", line)
        if not m or i + 2 >= len(L):
            continue
        pick, player = int(m.group(1)), L[i + 1]
        t = P2024_TAIL.match(L[i + 2])
        if not t or not re.match(r"^[A-Z][A-Za-z'\-\. ]+$", player):
            continue
        team, amt = t.groups()
        out.setdefault(pick, (player.strip(), team.strip(), int(amt.replace(",", "")) if amt else None))
    return out


def main():
    rows = []
    for season, url in SOURCES.items():
        picks = (parse_2026 if season == 2026 else parse_2024)(url)
        unit = "usd" if season == 2026 else "draft_points"
        for pick in sorted(picks):
            player, team, price = picks[pick]
            rows.append(dict(season=season, pick_no=pick, player=player, team=team,
                             price_raw=price if price is not None else "",
                             price_unit=unit, source_url=url))
        priced = sum(1 for p in picks.values() if p[2] is not None)
        gaps = [i for i in range(1, max(picks) + 1) if i not in picks] if picks else []
        print(f"{season}: {len(picks)} picks, {priced} priced, gaps={gaps or 'none'}", file=sys.stderr)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
