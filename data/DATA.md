# Dataset

Regenerate: `python3 src/scrape/extract_drafts.py && python3 src/scrape/extract_players.py && python3 src/scrape/build_dataset.py && python3 src/scrape/extract_results.py`
(Page HTML caches to `.cache/dinkbank/`, so only the first run crawls.)

## Files

| File | Rows | What |
|---|---|---|
| `drafts_raw.csv` | 114 | Draft picks straight from coverage |
| `players_raw.csv` | 409 | Player attributes straight from DinkBank |
| `drafts.csv` | 114 | + `player_key`, `gender`, `match_kind` |
| `players.csv` | 409 | + `player_key` |
| `results_raw.csv` | 765 | One row per drafted player per bracket entered, 2026 |

Join on **`player_key`** (slugified name). `match_kind` records how each pick
resolved: `exact` / `initial_surname` / `fuzzy` / `unmatched`.

## The modeling set: 2026 only, n=60

| Season | Picks | Priced | Usable |
|---|---|---|---|
| 2026 | 66 | **66** | **60** (6 unjoined) |
| 2024 | 48 | 17 | **0 — do not train on these** |

**2024 is excluded on purpose.** Its source published prices only through pick 19,
so all 17 priced rows are top-19 picks. That's selection on the dependent
variable — training on it teaches the model that mid-tier picks don't exist and
biases every coefficient upward. Keep them as a held-out rank-order check only.

Because there's a single season, prices are directly comparable: **model dollars,
skip the `price_share` normalization** the original plan called for.

2023 and 2025 have no reachable published prices (Forbes carried them and blocks
automated fetching).

## Feature coverage (of the 60)

| Field | Coverage | Notes |
|---|---|---|
| `career_earnings` | 60/60 | |
| `earnings_rank` | 60/60 | Career earnings rank, lower = better |
| `total_2025` | 58/60 | **Pre-draft. The best feature you have.** |
| `contracts_2025`, `prize_2025` | partial | `total` splits into contract + prize |
| `gender` | 55/60 | |
| `card_tier` | 47/60 | UPA contract tier |
| ~~`dupr_doubles`~~ | — | **Removed, see below** |
| ~~`dupr_singles`~~ | — | **Removed, see below** |
| `birth_year` | 5/60 | **Unusable — drop age entirely** |
| `events` | partial | Only where the bio states it |

## On-court results (`results_raw.csv`)

Added after the first model came back near-useless on earnings alone. DinkBank's
player pages embed an `event_played` array in their Next.js payload: one row per
player per bracket, carrying tour, placement, prize and **`start_date`**. The
page renders 10 rows at a time; `?page=N` walks the rest.

**`start_date` is the whole point.** The 2026 draft ran **2026-02-27** (dated
from the live-updates article `drafts_raw.csv` cites). Results before that are
legitimate features; results after it are the outcome leaking backwards.
`features.load_form()` re-applies the cutoff itself rather than trusting the
scraper's `pre_draft` column, and raises if a post-draft row survives.

| | |
|---|---|
| Total rows scraped | 765 |
| **Pre-draft rows (< 2026-02-27)** | **293** |
| Post-draft rows, discarded | 472 |
| Players with pre-draft form | **52 / 60** |
| Window | 2026-01-12 to 2026-02-16 |
| Brackets per player | median 6, range 1-12 |

Placements map onto a 1-7 ladder (`Round of 32` = 1 ... `Champion` = 7) in
`features.PLACEMENT_SCORE`; both spellings of each finish are covered, and an
unrecognised label raises rather than silently scoring as missing.

### Three limits that matter

1. **2026 only.** DinkBank's results table carries no 2023-2025 matches, so
   "form" means the current season up to draft day, not career history. It is
   also why the 2024 rank-order check had to be retired: the form features
   postdate that draft by two years.
2. **One tour, seven weeks.** Every pre-draft bracket in the window is PPA.
   No APP or MLP play falls inside it.
3. **No strength-of-field adjustment.** A Round of 32 exit at a stacked major
   scores the same 1 as one at a thin draw. This is the most obvious remaining
   improvement.

## Measured signal (log price, n=60)

```
corr(log price, log career_earnings) = +0.33   n=60
corr(log price, log total_2025)      = +0.33   n=58
corr(log price, earnings_rank)       = -0.20   n=60
corr(log price, dupr_doubles)        = +0.29   n=29
mean log-price:  Female 10.00 (n=29)   Male 9.58 (n=26)
```

Real but modest. **Expect R² around 0.2–0.4, not the 0.5–0.75 in plan.md** — that
estimate was written before any data existed.

> **Both numbers above were wrong, in opposite directions.** A correlation of
> 0.33 caps R² near **0.11**, so 0.2–0.4 was never reachable from earnings: the
> earnings-only model cross-validates at **+0.059** and the original four-feature
> model at **+0.033**, losing to its own baseline. Adding pre-draft form changed
> the picture — the current model reaches **+0.344** (nested CV **+0.304**).
>
> ```
> corr(log price, pd_doubles_mean) = +0.63   n=52   <- strongest feature found
> corr(log price, pd_round_max)    = +0.56   n=52
> corr(log price, pd_podiums)      = +0.56   n=52
> corr(log price, pd_prize)        = +0.44   n=52
> corr(log price, pd_events)       = +0.30   n=52
> ```
>
> **Gender and DUPR were dropped.** The female premium is two rows: the F-M gap
> in mean log price is +0.42 across all 60 and **+0.14** once Anna Bright and
> Jorja Johnson are set aside. Dropping gender improved both R² and rank
> correlation. `has_dupr` carried a negative sign that only ever meant "DinkBank
> catalogues this player".

**Correction to plan.md:** it asserted that without a gender term the model would
mostly learn "is this a man." The opposite holds in 2026 — women were bid *higher*
on average, because MLP starting fours are 2M/2W and the two priciest picks
(Anna Bright $1.23M, Jorja Johnson $800k) were women. Keep the gender term, but
for the female premium.

## Why there are no DUPR ratings in this repo

The published dataset carries **no DUPR values**, and `extract_players.py` no
longer writes them out. Two reasons, in order:

1. **The model stopped using them.** Once pre-draft form was added, `has_dupr`
   was dropped from the feature set; the rating value itself was never used,
   because the only ratings reachable were current ones carrying seven months of
   post-draft results.
2. **This project does not redistribute DUPR ratings.** They were only ever
   reachable second-hand through DinkBank, and a public repo mirroring them
   would be republishing another party's ratings. The request for proper
   as-of-draft-date ratings is in `docs/` and is not part of the public repo.

The measured correlation of +0.29 noted above is kept as a recorded finding. If
DUPR grants historical access, ratings come back under whatever terms they set.

## Caveats that matter

1. **DUPR ratings here are current (Sept 2026), not as-of the Feb 2026 draft.**
   They carry ~7 months of post-draft information, so any model using them is
   partly predicting the past from the future and will look better than it is.
   `total_2025` and earlier are genuinely pre-draft and safe. This is the single
   strongest argument for the DUPR API request — historical ratings fix it.
2. **`pick_no` is an outcome, not a feature.** It's produced by the same auction
   that sets price. Using it leaks the target.
3. **Source is DinkBank, never DUPR directly.** DUPR's terms prohibit automated
   access; DinkBank republishes ratings and its robots.txt allows all crawlers.
4. Some DinkBank figures are their own estimates rather than confirmed values;
   they label which on-site but the label isn't captured here.
