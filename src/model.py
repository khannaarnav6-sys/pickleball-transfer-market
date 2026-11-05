"""STEP 2 OF 4 -- fit the model and find out if it's any good.

Only 60 players here, so the whole job is avoiding self-deception: few features,
test on data the model didn't see, and compare against dumb baselines.

Run:  python3 src/model.py
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features

CV = dict(n_splits=5, n_repeats=20, random_state=0)


def build_pipeline(alpha=10.0):
    """Scale the features, then ridge regression.

    Why a pipeline and not two separate steps: the scaler has to be fit
    separately inside each test fold. If you scale everything up front, the
    model peeks at the test data and your scores come out too good.
    """
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def evaluate(pipe, X, y):
    """Score the model on data it hasn't seen. Return a dict.

    R2 = 0.35 sounds respectable; the dollar error tells you what it actually
    means, and it will be large. That's the honest number.
    """
    cv = RepeatedKFold(**CV)

    # Every player gets predicted by a model trained without them.
    preds = cross_val_predict(pipe, X, y, cv=RepeatedKFold(n_splits=CV["n_splits"],
                                                           n_repeats=1,
                                                           random_state=0))
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(np.exp(y), np.exp(preds))

    # One number from one split means nothing at n=60. Carry the spread too.
    folds = cross_validate(pipe, X, y, cv=cv, scoring="r2")["test_score"]

    return {
        "r2": float(r2),
        "mae_dollars": float(mae),
        "fold_r2_mean": float(np.mean(folds)),
        "fold_r2_sd": float(np.std(folds)),
        "fold_r2_p10": float(np.percentile(folds, 10)),
        "fold_r2_p90": float(np.percentile(folds, 90)),
        "n_folds": int(len(folds)),
    }


def cv_predictions(pipe, X, y):
    """Out-of-fold log predictions, one per player. What the app may display."""
    return cross_val_predict(pipe, X, y,
                             cv=RepeatedKFold(n_splits=CV["n_splits"],
                                              n_repeats=1, random_state=0))


def baselines(X, y):
    """Two dumb models the real one has to beat.

    If the 4-feature model can't beat earnings_only, that's a real result worth
    reporting -- not a bug to hide.
    """
    return {
        "mean_only": evaluate(DummyRegressor(strategy="mean"), X, y),
        "earnings_only": evaluate(build_pipeline(), X[["log_total_2025"]], y),
    }


def fit_final(X, y):
    """Fit on all 60 rows. This one ships to the app.

    Its score on its own training data will look great -- ignore that, and
    quote evaluate()'s number instead.
    """
    pipe = build_pipeline()
    pipe.fit(X, y)
    return pipe


def coefficient_table(pipe, feature_names):
    """feature -> coefficient, biggest effect first.

    These are on the standardized scale: the change in log price for a
    one-SD move in that feature. The app uses them to explain a valuation.
    """
    coefs = pipe.named_steps["ridge"].coef_
    out = pd.DataFrame({"feature": list(feature_names), "coef": coefs})
    return (out.reindex(out["coef"].abs().sort_values(ascending=False).index)
               .reset_index(drop=True))


def confidence_intervals(X, y):
    """OLS coefficient CIs, for the app's explanation panel.

    statsmodels on the standardized features, so the numbers line up with
    coefficient_table(). Unregularized, so read these as "is this term
    distinguishable from zero", not as the shipped model's coefficients.
    """
    import statsmodels.api as sm

    Z = (X - X.mean()) / X.std(ddof=0)
    fit = sm.OLS(np.asarray(y), sm.add_constant(Z.to_numpy())).fit()
    names = ["intercept"] + list(X.columns)
    lo, hi = fit.conf_int(alpha=0.05).T
    return pd.DataFrame({
        "feature": names,
        "coef": fit.params,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_value": fit.pvalues,
    })


def holdout_2024_rank_check(pipe):
    """Retired. Returns None.

    This used to score the model against 2024's 17 priced picks. It cannot run
    any more: the form features are built from January-February 2026 brackets,
    which postdate the 2024 draft by two years. Scoring 2024 prices with them
    would be predicting the past from the future -- the same error the DUPR
    caveat warns about, just larger. Kept as a stub so the shape of the check
    is on record for when earlier seasons of results become available.
    """
    return None


def alpha_sensitivity(X, y, alphas=(0.1, 1, 3, 10, 30, 100, 300)):
    """CV R2 across ridge strengths, reported so the alpha choice is auditable.

    alpha is NOT tuned on this table. It is fixed at the value the nested
    selection in DATA.md picked inside its training folds.
    """
    return [{"alpha": float(a), "r2": evaluate(build_pipeline(a), X, y)["r2"]}
            for a in alphas]


def _fmt(s):
    return (f"R2 {s['r2']:+.3f}   MAE ${s['mae_dollars']:,.0f}   "
            f"folds {s['fold_r2_mean']:+.2f} +/- {s['fold_r2_sd']:.2f} "
            f"[p10 {s['fold_r2_p10']:+.2f}, p90 {s['fold_r2_p90']:+.2f}]")


if __name__ == "__main__":
    df = features.add_features(features.load())
    X, y = features.make_xy(df, features.FEATURES)
    print(f"\nmodel.py  (step 2 of 4)   n={len(X)}  features={list(X.columns)}\n")

    base = baselines(X, y)
    scores = evaluate(build_pipeline(), X, y)
    print("  baseline  mean_only      ", _fmt(base["mean_only"]))
    print("  baseline  earnings_only  ", _fmt(base["earnings_only"]))
    print("  model     4-feature ridge", _fmt(scores))

    beat = [k for k, v in base.items() if scores["r2"] > v["r2"]]
    lost = [k for k, v in base.items() if scores["r2"] <= v["r2"]]
    print(f"\n  beats:      {', '.join(beat) if beat else 'NOTHING'}")
    if lost:
        print(f"  LOSES TO:   {', '.join(lost)}  <-- report this, do not bury it")

    print("\n  alpha sensitivity (reported, not tuned on):")
    for row in alpha_sensitivity(X, y):
        print(f"    alpha={row['alpha']:<6g} R2 {row['r2']:+.3f}")

    pipe = fit_final(X, y)
    print("\n  coefficients (standardized, ridge):")
    for _, r in coefficient_table(pipe, features.FEATURES).iterrows():
        print(f"    {r['feature']:<22} {r['coef']:+.3f}")

    print("\n  OLS 95% CIs (is the term distinguishable from zero?):")
    for _, r in confidence_intervals(X, y).iterrows():
        star = "  *" if r["p_value"] < 0.05 else ""
        print(f"    {r['feature']:<22} {r['coef']:+.3f}  "
              f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  p={r['p_value']:.3f}{star}")

    rank = holdout_2024_rank_check(pipe)
    if rank:
        print(f"\n  2024 held-out rank check (n={rank['n']}, DV-selected): "
              f"spearman {rank['spearman']:+.2f}  p={rank['p_value']:.3f}")

    out = df[["player", "team", "pick_no", "price_raw"]].copy()
    out["modeled"] = np.exp(pipe.predict(X))
    print("\n  top 8 by modeled value (sanity: Bright & Johnson should be up here):")
    for _, r in out.sort_values("modeled", ascending=False).head(8).iterrows():
        print(f"    {r['player']:<24} modeled ${r['modeled']:>10,.0f}   "
              f"actual ${r['price_raw']:>10,.0f}   pick {int(r['pick_no']):>2}")
    print()
