"""STEP 1 OF 4 -- build the table the model learns from.

Goal: one row per drafted 2026 player, with a few numeric columns (X) and the
price they sold for (y).

Run:  python3 src/features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import dupr

DATA = Path(__file__).resolve().parents[1] / "data"

# These can never be used to predict. pick_no is the big one: it comes out of
# the same auction that set the price, so it's basically the answer in disguise.
LEAKY_COLS = {"pick_no", "price_raw", "price_unit", "earnings_ytd",
              "earnings_annual", "total_2026", "prize_2026", "contracts_2026"}

# The 2026 MLP draft. Any on-court result dated on or after this is an outcome
# the bidders could not have seen, so it can never become a feature.
DRAFT_DATE = "2026-02-27"

# How far a player got in a bracket. DinkBank spells the same finish several
# ways across tours, so both spellings map to one rung.
PLACEMENT_SCORE = {
    "Champion": 7, "Winner": 7,
    "Runner-up": 6, "Runner-Up": 6,
    "Third Place": 5, "3rd Place": 5,
    "Fourth Place": 4, "4th Place": 4, "Semifinalist": 4,
    "Quarterfinalist": 3,
    "9th-12th Place": 2, "Round of 16": 2,
    "Round of 32": 1,
}

# MLP is played as team doubles, so doubles brackets are the relevant form.
DOUBLES_BRACKETS = {"MD", "WD", "XD"}

# Gendered brackets. A player entered in MD/MS is a man, WD/WS a woman; XD and
# Team say nothing. Used only to fill gender where the draft coverage left it
# blank -- see infer_gender().
MENS_BRACKETS = {"MD", "MS"}
WOMENS_BRACKETS = {"WD", "WS"}

# DUPR ratings are NOT in here. They arrive via src/dupr.py from the file DUPR
# supplied directly, are joined by name, and never get redistributed. See DATA.md.
NUMERIC_COLS = ["career_earnings", "total_2025", "prize_2025", "contracts_2025",
                "earnings_rank"]

# The four form/earnings columns. Chosen by nested cross-validation over five
# candidate sets: the inner loop never once picked an earnings-only or pre-form
# set. See DATA.md.
FEATURES = ["pd_doubles_mean", "pd_round_max", "pd_events", "log_total_2025"]

# Raw inputs for the DUPR experiment. model.py's GenderRelativeDupr step turns
# these two into one within-field interaction column, fitting the per-gender
# mean and sd inside each CV fold so test rows never set their own baseline.
DUPR_INPUTS = ["dupr_doubles", "is_female"]

# The DUPR candidate set. TESTED AND NOT SHIPPED -- see data/DATA.md.
# DUPR supplied their ratings on request in November 2026. Within the women's
# field the doubles rating correlates +0.68 with log price and within the men's
# +0.22, which looked like a real find; expressed as a within-field z-score
# interacted with is_female it reached CV R2 +0.407 against form's +0.354.
#
# That gain did not survive a data fix. Five picks had no recorded gender and
# were silently counted as men, two of them women. Filling gender from the
# gendered brackets they actually entered moved the term's coefficient from
# +0.402 to +0.291, its interval back across zero (p = 0.18), and the paired
# gain over form-only to nothing (better in 13 of 20 repeats, p = 0.45).
#
# Kept wired up so the experiment reruns, but not in MODEL_INPUTS: a term that
# clears no significance bar, found after testing a dozen specifications on 60
# rows, is the overfitting-by-search this project exists to avoid.
DUPR_CANDIDATE = FEATURES + DUPR_INPUTS

# What actually gets handed to the pipeline, and the names the coefficients line
# up with. The DUPR step passes these through untouched.
MODEL_INPUTS = FEATURES
FINAL_FEATURES = FEATURES


def load_form():
    """Pre-draft on-court form, one row per player.

    Reads the per-bracket results scraped by src/scrape/extract_results.py and
    keeps only what finished before draft day. This is the one place in the
    project where a date filter IS the leakage control, so the cutoff is
    re-applied here rather than trusting the scraper's `pre_draft` column.
    """
    path = DATA / "results_raw.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing -- run: python3 src/scrape/extract_results.py")

    r = pd.read_csv(path)
    r["start_date"] = r["start_date"].astype(str)
    pre = r[r["start_date"] < DRAFT_DATE].copy()

    if pre.empty:
        raise ValueError("no pre-draft results survived the date filter")
    if (pre["start_date"] >= DRAFT_DATE).any():
        raise ValueError("post-draft result leaked past the cutoff")

    pre["score"] = pre["placement"].map(PLACEMENT_SCORE)
    unknown = set(pre.loc[pre["score"].isna(), "placement"].dropna())
    if unknown:
        raise ValueError(f"unscored placement labels: {sorted(unknown)}")

    g = pre.groupby("player_key")
    dbl = pre[pre["bracket"].isin(DOUBLES_BRACKETS)].groupby("player_key")
    form = pd.DataFrame({
        "pd_events": g.size(),
        "pd_round_mean": g["score"].mean(),
        "pd_round_max": g["score"].max(),
        "pd_podiums": g["score"].apply(lambda s: int((s >= 4).sum())),
        "pd_titles": g["score"].apply(lambda s: int((s == 7).sum())),
        "pd_prize": g["amount_usd"].sum(),
        "pd_doubles_mean": dbl["score"].mean(),
    })
    form.attrs["cutoff"] = DRAFT_DATE
    form.attrs["last_event"] = pre["start_date"].max()
    return form.reset_index()


FORM_COLS = ["pd_events", "pd_round_mean", "pd_round_max", "pd_podiums",
             "pd_titles", "pd_prize", "pd_doubles_mean"]


def infer_gender():
    """player_key -> 'Male'/'Female', read off the gendered brackets they entered.

    The draft coverage leaves gender blank for a handful of picks, and DinkBank
    does not always fill the gap. It matters more than it used to: is_female
    gates the DUPR term, so a woman recorded as blank silently loses it.

    Entering a men's singles or doubles draw is about as unambiguous a signal as
    this data has. Where both sources exist they agree on every player and
    nobody appears in both MD and WD, which is what makes this safe -- load()
    asserts that rather than assuming it.
    """
    path = DATA / "results_raw.csv"
    if not path.exists():
        return pd.Series(dtype=object)

    r = pd.read_csv(path)
    r = r[r["bracket"].isin(MENS_BRACKETS | WOMENS_BRACKETS)]
    sex = np.where(r["bracket"].isin(MENS_BRACKETS), "Male", "Female")
    g = pd.DataFrame({"player_key": r["player_key"], "sex": sex})

    # Anyone who somehow appears on both sides is dropped, not guessed at.
    per = g.groupby("player_key")["sex"].agg(lambda s: set(s))
    clean = per[per.apply(len) == 1].apply(lambda s: next(iter(s)))
    return clean


def load():
    """One row per matched 2026 pick, with DinkBank attributes merged on."""
    drafts = pd.read_csv(DATA / "drafts.csv")
    players = pd.read_csv(DATA / "players.csv")

    drafts = drafts[drafts["season"] == 2026]
    drafts = drafts[drafts["match_kind"] != "unmatched"]

    # Empty suffix on the left keeps `gender` / `player` pointing at the draft
    # coverage; DinkBank's copies land under *_dinkbank.
    df = drafts.merge(players, on="player_key", how="left",
                      suffixes=("", "_dinkbank"))

    for name in NUMERIC_COLS:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df["price_raw"] = pd.to_numeric(df["price_raw"], errors="coerce")

    # Gender is 55/60 in the draft coverage; fall back to DinkBank's column
    # where it's blank (today they agree, but don't rely on that), then to the
    # gendered brackets the player actually entered.
    df["gender"] = df["gender"].fillna(df["gender_dinkbank"])

    inferred = infer_gender()
    if len(inferred):
        from_brackets = df["player_key"].map(inferred)
        both = df["gender"].notna() & from_brackets.notna()
        clash = df.loc[both & (df["gender"] != from_brackets), "player_key"]
        if len(clash):
            raise ValueError(
                "bracket-inferred gender contradicts the recorded gender for "
                f"{sorted(clash)} -- do not fill from brackets until that is "
                "understood")
        df["gender_inferred"] = df["gender"].isna() & from_brackets.notna()
        df["gender"] = df["gender"].fillna(from_brackets)
    else:
        df["gender_inferred"] = False

    if len(df) != 60:
        raise ValueError(f"expected 60 modeling rows, got {len(df)}")
    if df["player_key"].duplicated().any():
        raise ValueError("duplicate player_key after merge -- would double-count")
    if df["price_raw"].isna().any():
        raise ValueError("a 2026 pick is missing price_raw")

    df = df.merge(load_form(), on="player_key", how="left")
    if len(df) != 60:
        raise ValueError(f"form merge changed row count to {len(df)}")

    df, rep = dupr.attach(df, DRAFT_DATE)
    if len(df) != 60:
        raise ValueError(f"DUPR join changed row count to {len(df)}")
    df.attrs["dupr"] = rep

    return df.reset_index(drop=True)


def add_features(df):
    """Add the columns the model will actually use. Return a modified copy.

    Money columns are lopsided -- most players earned a little, a few earned
    millions. Taking a log evens that out so a straight line can fit it.
    """
    df = df.copy()

    # 52 of 60 played a rated bracket before draft day; the rest get the median
    # and are flagged, same policy as the money columns.
    df["has_results"] = df["pd_events"].notna().astype(int)
    for col in FORM_COLS:
        df[col] = df[col].fillna(df[col].median())
    df["log_pd_prize"] = np.log1p(df["pd_prize"])

    df["log_total_2025"] = np.log1p(df["total_2025"])
    df["log_career_earnings"] = np.log1p(df["career_earnings"])
    df["is_female"] = (df["gender"] == "Female").astype(int)

    # Only 58 of 60 have total_2025. Median-impute rather than drop rows: at
    # n=60 two rows is 3% of the training set.
    for col in ["log_total_2025", "log_career_earnings"]:
        df[col] = df[col].fillna(df[col].median())

    # 57 of 60 are in the DUPR file; the other three get the median and a flag.
    # Same policy as the money and form columns.
    df["has_dupr"] = df["dupr_doubles"].notna().astype(int)
    df["dupr_doubles"] = df["dupr_doubles"].fillna(df["dupr_doubles"].median())

    left = df[MODEL_INPUTS].isna().sum().sum()
    if left:
        raise ValueError(f"{left} missing values left in MODEL_INPUTS")

    return df


def assert_no_leakage(feature_names):
    """Crash if a banned column snuck into the feature list.

    Substring, not equality -- "log_pick_no" has to fail too.
    """
    for name in feature_names:
        for bad in LEAKY_COLS:
            if bad in name:
                raise ValueError(
                    f"leakage: feature {name!r} contains banned column {bad!r} "
                    "-- it is an outcome of the draft, not an input to it"
                )


def make_xy(df, features):
    """Return (X, y) with y on the log-dollar scale.

    Prices run $1,000 to $1,230,000. Without the log, the two most expensive
    players would drown out the other 58.

    Nothing is scaled here -- model.py does that inside its pipeline.
    """
    assert_no_leakage(features)
    X = df[features]
    y = np.log(df["price_raw"])
    return X, y


def _raises(fn):
    """True if fn() raises anything. Helper for the progress check."""
    try:
        fn()
    except NotImplementedError:
        raise
    except Exception:
        return True
    return False


def _step(name, fn):
    """Run one step, print what happened. Used by the progress check below."""
    try:
        out = fn()
    except NotImplementedError:
        print(f"  [ ] {name:<20} not written yet  <-- you are here")
        return None
    except Exception as e:
        print(f"  [!] {name:<20} CRASHED: {type(e).__name__}: {e}")
        return None
    print(f"  [x] {name:<20} {out[1]}")
    return out[0]


if __name__ == "__main__":
    print("\nfeatures.py  (step 1 of 4)\n")

    df = _step("load()", lambda: (lambda d: (d, f"{len(d)} rows, {len(d.columns)} cols"))(load()))
    if df is None:
        raise SystemExit("\nWrite load() first, then run this again.\n")

    df = _step("add_features()", lambda: (lambda d: (d, f"{d[MODEL_INPUTS].isna().sum().sum()} missing values left"))(add_features(df)))
    if df is None:
        raise SystemExit("\nWrite add_features() next.\n")

    ok = _step("assert_no_leakage()", lambda: (True, "catches banned columns")
               if _raises(lambda: assert_no_leakage(["pick_no"])) else (None, ""))
    if ok is None:
        raise SystemExit("\nWrite assert_no_leakage() next (it must raise on 'pick_no').\n")

    xy = _step("make_xy()", lambda: (lambda t: (t, f"X={t[0].shape}, y from {t[1].min():.1f} to {t[1].max():.1f}"))(make_xy(df, MODEL_INPUTS)))
    if xy is None:
        raise SystemExit("\nWrite make_xy() last.\n")

    print("\nAll four done. Move on to src/model.py\n")
