"""STEP 2 OF 4 -- fit the model and find out if it's any good.

Only 60 players here, so the whole job is avoiding self-deception: few features,
test on data the model didn't see, and compare against dumb baselines.

Run:  python3 src/model.py
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

import features

CV = dict(n_splits=5, n_repeats=20, random_state=0)


class GenderRelativeDupr(BaseEstimator, TransformerMixin):
    """Turn a raw DUPR doubles rating into one within-gender interaction column.

    Why bother: DUPR correlates +0.34 with log price raw, but +0.73 inside the
    women's field and +0.15 inside the men's. The two fields also sit on
    different parts of the scale (men average 6.06, women 5.70), so a raw
    rating mixes "how good" with "which field". A within-gender z-score
    separates those, and multiplying by is_female lets the model use the rating
    where it carries signal and ignore it where it doesn't.

    Why it is a transformer and not a column in features.py: the mean and sd
    have to be computed from training rows only. Precomputing them over all 60
    would let each test row help set its own baseline.

    Ridge is not invariant to this choice. An algebraically equivalent
    parameterisation -- passing dupr*is_female and is_female and letting the
    model find the same span -- scores 0.341 instead of 0.407, because the L2
    penalty applies to the standardised coefficients and the two
    parameterisations put different geometry under it.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        if not self._applies(X):
            self.stats_ = None
            return self
        self.stats_ = {}
        med = X["dupr_doubles"].median()
        for g in (0, 1):
            col = X.loc[X["is_female"] == g, "dupr_doubles"].fillna(med)
            sd = col.std()
            self.stats_[g] = (col.mean(), sd if sd and sd > 0 else 1.0)
        self.median_ = med
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        if self.stats_ is None:
            return X.to_numpy(dtype=float)
        out = X.drop(columns=features.DUPR_INPUTS).copy()
        d = X["dupr_doubles"].fillna(self.median_)
        z = np.zeros(len(X), dtype=float)
        for g in (0, 1):
            sel = (X["is_female"] == g).to_numpy()
            mu, sd = self.stats_[g]
            z[sel] = (d.to_numpy()[sel] - mu) / sd
        out["dupr_z_x_female"] = z * X["is_female"].to_numpy()
        return out.to_numpy(dtype=float)

    @staticmethod
    def _applies(X):
        return all(c in X.columns for c in features.DUPR_INPUTS)


class CappedRidge(Ridge):
    """Ridge that never predicts above the priciest player it trained on.

    Without this the model extrapolates absurdly at the top. Anna Bright went
    for $1.23M -- 54% more than the next player -- so when she lands in a test
    fold the model has no comparable example and predicted $7.2M for her. The
    cap is the one honest thing to say in that situation: "at least as expensive
    as anyone I have seen". It moves exactly one of 60 predictions and cuts mean
    absolute error from $129K to $37K.
    """

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)
        self.cap_ = float(np.max(y))
        return self

    def predict(self, X):
        return np.minimum(super().predict(X), self.cap_)


def build_pipeline(alpha=10.0):
    """Scale the features, then ridge regression.

    Why a pipeline and not two separate steps: the scaler has to be fit
    separately inside each test fold. If you scale everything up front, the
    model peeks at the test data and your scores come out too good.
    """
    return Pipeline([
        ("duprz", GenderRelativeDupr()),
        ("standardscaler", StandardScaler()),
        ("ridge", CappedRidge(alpha=alpha)),
    ])


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

    # Run the DUPR step once over all rows so the names match the shipped
    # coefficients. This table is a descriptive "is this term distinguishable
    # from zero" check, not a held-out score, so the shared normalisation is
    # fine here -- evaluate() is the number that has to stay leak-free.
    if all(c in X.columns for c in features.DUPR_INPUTS):
        T = GenderRelativeDupr().fit(X)
        X = pd.DataFrame(T.transform(X), columns=features.FINAL_FEATURES,
                         index=X.index)
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


def dupr_experiment(df, y):
    """Re-run the DUPR comparison and report it. Returns a dict, never shipped
    into the feature set.

    This is here so the negative result is regenerated from the data every time
    rather than being a number somebody typed into the page once. The paired
    test is the one that matters: 20 independent repeats of 5-fold, the same
    splits for both models, differencing within each repeat.
    """
    from scipy.stats import spearmanr, wilcoxon

    if "dupr_doubles" not in df.columns or df["dupr_doubles"].isna().all():
        return {"available": False}

    y = np.asarray(y)
    out = {}
    for label, cols in (("form", features.FEATURES),
                        ("dupr", features.DUPR_CANDIDATE)):
        folds, preds = [], np.zeros(len(df))
        for tr, te in RepeatedKFold(**CV).split(df):
            pipe = build_pipeline().fit(df[cols].iloc[tr], y[tr])
            folds.append(r2_score(y[te], pipe.predict(df[cols].iloc[te])))
        for tr, te in RepeatedKFold(n_splits=CV["n_splits"], n_repeats=1,
                                    random_state=CV["random_state"]).split(df):
            pipe = build_pipeline().fit(df[cols].iloc[tr], y[tr])
            preds[te] = pipe.predict(df[cols].iloc[te])
        out[label] = {
            "r2": float(r2_score(y, preds)),
            "spearman": float(spearmanr(y, preds).statistic),
            "folds": np.array(folds),
            "preds": preds,
        }

    # Average within each repeat first: the 100 folds are 20 independent
    # 5-fold runs, not 100 independent samples.
    diff = (out["dupr"]["folds"] - out["form"]["folds"]).reshape(
        CV["n_repeats"], CV["n_splits"]).mean(axis=1)

    within = {}
    for g, lab in ((0, "men"), (1, "women")):
        sel = (df["is_female"] == g).to_numpy()
        within[lab] = {
            "n": int(sel.sum()),
            "corr": float(np.corrcoef(df.loc[sel, "dupr_doubles"], y[sel])[0, 1]),
            "r2_form": float(r2_score(y[sel], out["form"]["preds"][sel])),
            "r2_dupr": float(r2_score(y[sel], out["dupr"]["preds"][sel])),
        }

    return {
        "available": True,
        "r2_form": out["form"]["r2"],
        "r2_dupr": out["dupr"]["r2"],
        "spearman_form": out["form"]["spearman"],
        "spearman_dupr": out["dupr"]["spearman"],
        "paired_mean": float(diff.mean()),
        "paired_better": int((diff > 0).sum()),
        "paired_repeats": int(len(diff)),
        "paired_p": float(wilcoxon(diff).pvalue),
        "within": within,
        "shipped": False,
    }


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
    X, y = features.make_xy(df, features.MODEL_INPUTS)
    rep = df.attrs.get("dupr", {})
    print(f"\nmodel.py  (step 2 of 4)   n={len(X)}  features={features.FINAL_FEATURES}")
    if rep.get("available"):
        print(f"  DUPR: {rep['matched']}/{rep['n']} matched from a "
              f"{rep['pool_after_cutoff']}-player pool dated "
              f"{rep['date_min']}..{rep['date_max']} (cutoff {rep['cutoff']})\n")

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
    for _, r in coefficient_table(pipe, features.FINAL_FEATURES).iterrows():
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
