# Pickleball Transfer Market

**[Live app](https://khannaarnav6-sys.github.io/pickleball-transfer-market/)**

Soccer has Transfermarkt, where every player carries a market value. Pickleball
has something better: Major League Pickleball runs an **open auction draft**
where teams bid real money. The 2026 draft ran 66 picks from $1.23M down to
$1,000.

So it's a real supervised learning problem: *given what was publicly knowable
before the draft, predict what the market paid.*

## The result

**Recent doubles form predicts draft price. Career earnings barely do.**

| Model | CV R² | Spearman |
|---|---|---|
| Predict the mean | −0.007 | — |
| 2025 earnings only | +0.059 | +0.249 |
| **Doubles form + best finish + brackets entered + earnings** | **+0.344** | **+0.448** |

Nested cross-validation — an inner loop choosing among five feature sets and
five ridge strengths *inside each training fold* — gives **+0.304**, so the gain
isn't an artifact of the feature search. The inner loop never once selected an
earnings-only feature set.

The strongest feature is how deep a player ran in **doubles brackets in the
seven weeks before the draft**, and it is the only term whose confidence
interval clears zero (p = 0.001). That fits the format: MLP starting fours are
two men and two women playing team doubles, so teams bid on current doubling
ability rather than career prize money. Once form is in the model, the earnings
coefficient goes slightly negative and insignificant.

An earlier version using earnings, gender and DUPR scored **+0.033** and lost to
its own one-variable baseline. Writing up why that failed is in `plan.md`.

## The part that took the most care: not leaking the answer

With a draft, almost everything interesting is downstream of the thing you're
predicting.

- **`pick_no` is the trap.** It explains R² = 0.813 of log price on its own, but
  it's an output of the same auction that set the price. `features.py` runs a
  substring assertion that fails the build if any banned column reaches the
  feature matrix — `log_pick_no` fails too, not just `pick_no`.
- **Form is controlled by a date, not a flag.** Tournament results are only
  usable if they finished before draft day (2026-02-27). Of 765 scraped result
  rows, **293 are kept and 472 discarded**. `features.load_form()` re-applies
  that cutoff itself rather than trusting the scraper's `pre_draft` column, and
  raises if a single post-draft row survives.
- **A held-out check was retired rather than fudged.** A 2024 rank-order test
  became invalid once form arrived, because 2026 brackets postdate that draft by
  two years. It's a documented stub, not a deleted function.

## Honest limits

This is 60 players. The app says all of this on the page rather than burying it.

- **48% of squared error sits in the 8 players who went at ≤$2,000**, all of them
  picks #58–#66. That's the auction running out of money and roster slots, not
  underpriced talent — and nothing in a player's record predicts it (AUC 0.557).
- **Mean dollar error rose** ($56K → $83K) because two picks hold half the
  draft's dollars. The robust measures improved: median AE $15.4K → $14.3K, mean
  log error 1.13 → 0.97.
- **Form is seven weeks of one tour** (PPA, 12 Jan – 16 Feb 2026) with no
  strength-of-field weighting. Weighting by field strength was tested and gained
  nothing (+0.344 → +0.347).
- **Gender was dropped.** The apparent female premium is two rows: the gap in
  mean log price is +0.42 across all 60 and **+0.14** without Anna Bright and
  Jorja Johnson.
- **No DUPR ratings.** See `data/DATA.md`.

## Running it

```bash
pip install -r requirements.txt

# 1. scrape (caches to .cache/, so only the first run crawls)
python3 src/scrape/extract_drafts.py
python3 src/scrape/extract_players.py
python3 src/scrape/build_dataset.py
python3 src/scrape/extract_results.py

# 2. model
python3 src/features.py     # build + assert no leakage
python3 src/model.py        # scores against both baselines, with fold spread
python3 src/export.py       # writes app/data.json

# 3. app
python3 -m http.server 8000 --directory app
```

| File | What |
|---|---|
| `src/features.py` | Merge, log-transform, impute, and the leakage assertions |
| `src/model.py` | Ridge + repeated K-fold, baselines, statsmodels CIs |
| `src/export.py` | Writes `app/data.json`, asserting contributions reconstruct each prediction |
| `app/index.html` | Single-file app: search, valuation cards, residual leaderboard, diagnostics |
| `data/DATA.md` | Schemas, coverage, and every caveat |
| `plan.md` | Original plan, plus what the data actually showed |

## Data

Player attributes and tournament results from
**[DinkBank](https://www.dinkbank.com)**, whose robots.txt permits crawling.
Draft prices from published 2026 draft coverage. Nothing is sourced from DUPR
directly — their terms prohibit automated access.

Built with pandas, scikit-learn and statsmodels.
