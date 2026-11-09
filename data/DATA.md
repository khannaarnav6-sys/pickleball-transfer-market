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
| `dupr_doubles` | `data/dupr_ratings.csv` (gitignored) | Supplied by DUPR on request; tested, not shipped — see below |
| `dupr_singles` | `data/dupr_ratings.csv` (gitignored) | Not used; correlates +0.08 with log price |
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

## DUPR ratings: granted, tested, not used

**DUPR supplied ratings directly on 2026-11-08**, in response to the access
request in `docs/`. 153 players with singles and doubles ratings and a
`rating_date`. This replaced the earlier situation, where the only reachable
ratings were current ones carrying seven months of post-draft results.

The file lives at `data/dupr_ratings.csv` and is **gitignored**. The request
said this project was not looking to republish ratings, so the file stays local
and `app/data.json` carries no rating, no DUPR id, and no per-gender mean/sd —
without those moments a normalised term cannot be inverted back to a rating.

### What was done with it

`src/dupr.py` joins on name using the same exact → initial+surname → fuzzy
ladder as the DinkBank join, and enforces two leakage controls: a rating dated
on or after `DRAFT_DATE` is discarded (`rating_date` is the day the rating last
moved, so a draft-day stamp reflects matches bidders had not seen), and a rating
of exactly `0.000` is DUPR's "unrated" placeholder, not a skill of zero.

| | |
|---|---|
| rows supplied | 153 |
| after date cutoff | 152 (drops Chao Yi Wang, dated 2026-02-27) |
| matched to the 60 drafted | 57 (50 exact, 6 initial+surname, 1 fuzzy) |
| not in the file | Ava Cavataio, Genie Bouchard, Tom Protzek |
| matched rating dates | 2025-11-22 to 2026-02-23 |

Every non-exact match was checked by hand and all seven are correct (Mari /
Mariana Humberg, Alli / Allison Phillips, Victoria Di Muzio / DiMuzio, Alex /
Alexander Crum, Chris / Christopher Crouch, Chris / Christopher Haworth, John
Lucian Goins / John Goins). The matcher correctly refused to pair **Genie
Bouchard** with **Genie Erokhina**.

### The result: no measurable gain over recent form

Raw, a DUPR doubles rating correlates +0.34 with log price — real, but weaker
than `pd_doubles_mean` at +0.61, and 0.42 correlated with it. Added as a fifth
feature it made the model *worse* (+0.354 → +0.317): at n=60 it cost more in
variance than it returned in signal.

The specification that looked right is a **within-field z-score interacted with
`is_female`**, because the rating behaves completely differently by field:

```
corr(log price, dupr_doubles)   women  +0.675   n=31
                                men    +0.216   n=29
```

That term reached CV R² **+0.407** against form's +0.354, with p = 0.034 — and
then a data fix removed it. See the next section. The honest final numbers:

| | CV R² | Spearman | paired vs form |
|---|---|---|---|
| form only (shipped) | **+0.354** | +0.448 | — |
| form + DUPR term | +0.359 | +0.492 | better in 13/20 repeats, p = 0.45 |

The coefficient's 95% interval is [−0.138, +0.721], p = 0.18. **Not shipped.** A
term that clears no significance bar, found after a dozen specifications on 60
rows, is exactly the overfitting-by-search this project exists to avoid. It is
kept wired up as `features.DUPR_CANDIDATE` and re-measured on every export, so
the published page reports the finding rather than a number typed in once.

Worth retesting on the next draft: the within-field asymmetry is large, and
dropping the three best-known racket-sport crossovers makes the men's
correlation *worse* (+0.22 → +0.09), so celebrity pricing is not the
explanation.

## The gender bug that changed the DUPR answer

Draft coverage records gender for 55 of 60 and DinkBank did not fill the rest,
so five picks defaulted to not-female. **Two of them are women** (Elsie
Hendershot, Zoey Weil; the other three — Jack Sock, Camden Chaffin, Len Yang —
are men). That mattered only once `is_female` started gating a feature.

`features.infer_gender()` now reads gender off the **gendered brackets a player
actually entered**: MD/MS is a man, WD/WS a woman. This agrees with the recorded
value on every player who has both, and nobody appears in both MD and WD, which
is what makes it safe — `load()` raises on any disagreement rather than quietly
preferring one source.

Filling those five moved the DUPR term's coefficient from +0.402 to +0.291, its
interval back across zero, and its paired gain to nothing. **The significant
result was an artifact of two misclassified women.**

## Caveats that matter

1. **DUPR ratings are now as-of a pre-draft date, and still unused.** The file
   DUPR supplied carries a `rating_date` per player and anything dated on or
   after draft day is dropped, so the old "predicting the past from the future"
   problem is gone. The ratings simply did not add predictive power over recent
   form — see above.
2. **`pick_no` is an outcome, not a feature.** It's produced by the same auction
   that sets price. Using it leaks the target.
3. **Scraped features come from DinkBank, never from DUPR.** DUPR's terms
   prohibit automated access; DinkBank republishes ratings and its robots.txt
   allows all crawlers. The DUPR ratings in this project were **given on
   request**, not taken, and are not redistributed.
4. Some DinkBank figures are their own estimates rather than confirmed values;
   they label which on-site but the label isn't captured here.
