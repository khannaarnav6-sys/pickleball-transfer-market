# Pickleball Transfer Market — MLP Draft Price Model

> **Status: built.** Model, export and app are done and the app is published.
> `data/` holds the assembled dataset; see `data/DATA.md` for schemas, coverage
> and caveats. **Measured result is weaker than this plan assumed** - see
> "What it actually found" at the bottom before reading the estimates below.

## Context

Soccer has Transfermarkt: every player carries a market value anchored in real
transfer fees. Pickleball already has the equivalent — Major League Pickleball
runs an open auction draft where teams bid real money. The 2026 draft ran 66
picks from $1.23M (Anna Bright) down to $1,000.

So this is real supervised learning: **given what was publicly knowable before
the draft, predict what the market paid.** The interesting output isn't the
prediction, it's the residual — who went cheap relative to what they should
have. That's the thing worth shipping.

**Scope:** pro-only (no amateur valuation — the training set is 60 people in the
top 0.01% of the sport, so pricing a rec player would be pure extrapolation).
Features come from DinkBank, never DUPR directly, because DUPR's terms prohibit
automated access. Deliverable is a Transfermarkt-style web app over a Python model.

## What the data turned out to be

**n = 60. One season (2026). Model dollars directly** — with a single season
there's no cross-currency problem, so the `price_share` normalization originally
planned is unnecessary.

2024 is deliberately excluded: its source published prices only through pick 19,
making all 17 priced rows top-19 picks. That's selection on the dependent
variable and would bias every coefficient upward. 2023/2025 prices aren't
publicly reachable at all.

Best features: `total_2025` (pre-draft earnings, 58/60), `career_earnings`
(60/60), `earnings_rank` (60/60), `gender` (55/60), `card_tier` (47/60).
`dupr_doubles` is only 29/60, and `birth_year` is 5/60 — **drop age entirely.**

Measured correlations with log price are ~0.33 for earnings and ~0.29 for DUPR,
so **expect R² of roughly 0.2–0.4.** Full detail and caveats in `data/DATA.md`.

> **Corrected after building.** That estimate was arithmetically wrong: a
> correlation of 0.33 caps R² near **0.11** in-sample, before any out-of-sample
> penalty. Measured in-sample R² is 0.134; cross-validated is **+0.033.**

---

## What to build

### `src/features.py`
Merge `drafts.csv` + `players.csv` on `player_key`, filter to
`season == 2026 and match_kind != 'unmatched'`, and produce an X matrix.

- Log-transform the money columns (`log1p`) — all are heavily right-skewed.
- Decide missingness per feature: median-impute plus a `*_missing` indicator
  beats dropping rows when n is 60. For `dupr_doubles` at 29/60, the indicator
  may carry as much signal as the value.
- **Assert no post-draft data enters X.** `pick_no` is the trap: it's an outcome
  of the same auction that sets price, so using it leaks the target. Current-date
  DUPR is a softer version of the same problem (see caveat 1 in `data/DATA.md`).

### `src/model.py`
- **Target:** `log(price_raw)`. Prices span $1k–$1.23M.
- **Model:** Ridge, standardized features. At n=60 keep it to **4–6 features**;
  more will just fit noise. `statsmodels` alongside `sklearn` for coefficient
  CIs — the app needs to explain *why* a player is worth what they're worth.
- **Gender:** keep the term, but note the direction is a *female* premium
  (mean log-price 10.00 vs 9.58) — MLP starting fours are 2M/2W and the two
  priciest 2026 picks were women. plan.md originally predicted the opposite.
- **Validation:** repeated K-fold (LOSO is impossible with one season). Report
  the spread across folds, not a single number — at n=60 a lucky split means
  nothing.
- **Baselines to beat:** predict-the-mean, and earnings-only univariate. If the
  full model doesn't clear them, say so in the writeup rather than burying it.
- Report MAE in dollars alongside R² — R² on a log target is easy to misread.

### `src/export.py`
Coefficients, per-player predictions, residuals and per-feature contributions
→ `app/data.json`.

### `app/index.html`
Single file reading `data.json`:
- **Player search** → valuation card: predicted vs actual, confidence band,
  per-feature contribution breakdown
- **Bargains & overpays** — sorted residuals. The headline feature.
- **What-if slider** — move a feature, watch the valuation move
- Publish as an Artifact for a shareable link

Given n=60 and R² in the 0.2–0.4 range, the app should show uncertainty
honestly — wide bands, and language like "modeled value" rather than a hard price.

---

## Later: DUPR API

Once this is functional, submit it to DUPR for API access. That fixes the two
real weaknesses at once: `dupr_doubles` coverage (29/60 → complete) and the
as-of-draft-date problem, since historical ratings would replace today's.

## Verification

- Leakage assertion passes; `pick_no` absent from X
- K-fold R² and dollar MAE reported against both baselines, with fold spread
- Sanity check by hand: Anna Bright and Jorja Johnson should price at the top
- Residual plot vs. predicted and vs. earnings
- A player's card in the app matches the model's number for that player

---

## What it actually found

Run `python3 src/model.py` to reproduce all of this.

### First attempt: earnings, gender, DUPR

| Model | CV R² | MAE |
|---|---|---|
| Predict the mean | -0.007 | $58,291 |
| 2025 earnings only | **+0.059** | $56,557 |
| Four-feature ridge (earnings, gender, DUPR) | +0.033 | $56,356 |

**It lost to its own baseline**, and worse, the whole score rested on about
three rows: dropping any one of Anna Bright, Dylan Frazier or Jack Sock pushed
CV R² negative. A learning curve over n=20..40 was flat, so more rows of the
same features were not the answer. Meanwhile `log(pick_no)` alone explains
**R² = 0.813** of log price — the information existed, it just wasn't in the
public record.

### What fixed it: pre-draft on-court form

DinkBank's player pages embed per-bracket tournament results with a
`start_date`, so results from before draft day (2026-02-27) are usable as
features. See `data/DATA.md` for the scrape and its limits.

| Model | CV R² | Spearman |
|---|---|---|
| Predict the mean | -0.007 | — |
| 2025 earnings only | +0.059 | +0.249 |
| **Doubles form + best finish + brackets + earnings** | **+0.354** | **+0.448** |

Nested cross-validation — inner loop choosing among five feature sets and five
ridge strengths inside each training fold — gives **+0.304**, so the gain is not
an artifact of the search. The inner loop never once chose an earnings-only or
pre-form set.

**`pd_doubles_mean` is the first feature in this project whose confidence
interval clears zero** (p = 0.001). That fits the format: MLP starting fours are
2M/2W playing team doubles, so teams bid on current doubling ability. Once form
is in the model, the 2025-earnings coefficient goes slightly negative and
insignificant — earnings stopped mattering once form was measured.

It also survives the test that killed the first model: **+0.233 without Anna
Bright**, where earnings-only went to -0.038.

### Honest caveats on the improvement

- **Mean dollar error rose** ($56K → $83K) because two picks hold half the
  draft's dollars and the model now predicts boldly enough to miss them by large
  absolute amounts. The robust measures improved: median AE $15.4K → $14.3K,
  mean log error 1.13 → 0.97.
- **The residual leaderboard is still partly price in disguise** (-0.82, down
  from -0.93). The $1,000 rows are the auction's minimum bid, not discoveries.
- **Form is seven weeks of one tour**, with no strength-of-field weighting.
- Gender and DUPR were dropped; the "female premium" was two rows.

### What would move it next

1. **Strength-of-field weighting** on each result — cheapest real gain.
2. **Earlier seasons of results**, which would also restore the 2024 check.
3. **More seasons of prices** (2023/2025 sit behind Forbes).
4. **As-of-draft-date DUPR** via the API request below.

---

## Round three: DUPR answered the email (November 2026)

The data request in `docs/` worked. DUPR supplied a 153-player file of singles
and doubles ratings with a `rating_date` on each one, which solved the problem
that killed the first attempt: the only ratings previously reachable were
*current* ones carrying seven months of post-draft results.

**The ratings did not improve the model.** Full workings in `data/DATA.md`; the
short version:

- Raw, a doubles rating correlates +0.34 with log price — weaker than
  `pd_doubles_mean` (+0.61) and 0.42 correlated with it. As a fifth feature it
  made things worse (+0.354 → +0.317).
- The rating behaves very differently by field: **+0.68 within the women,
  +0.22 within the men**. Carried as a within-field z-score interacted with
  `is_female`, it reached CV R² **+0.407**, p = 0.034.
- That result was an artifact. Five picks had no recorded gender and defaulted
  to not-female; **two were women**. Reading gender off the gendered brackets
  players actually entered moved the coefficient from +0.402 to +0.291, the
  interval back across zero (p = 0.18), and the paired gain to nothing (better
  in 13 of 20 independent repeats, p = 0.45).

So the term is measured on every export and left out of the model.

### Two things that did survive from this round

1. **Prediction capping.** Never predict above the priciest player in the
   training fold. The top pick went for 54% more than the next, so held out she
   had no comparable and the model said $7.2M. Moves 1 of 60 predictions; mean
   absolute error $129K → $38K, CV R² +0.344 → +0.354.
2. **Gender inference from brackets.** MD/MS is a man, WD/WS a woman. Agrees
   with the recorded value on every player who has both, and nobody appears in
   both draws, so `load()` raises on disagreement instead of preferring a
   source.

### What this says about the original question

The project set out asking whether a skill rating would price players. The
answer the data gives is that **a rating adds nothing once you can see seven
weeks of actual results** — the rating and the results are measuring the same
thing, and the results are more current. That is a more useful finding than a
small R² bump would have been, and it is the honest thing to send back to DUPR.
