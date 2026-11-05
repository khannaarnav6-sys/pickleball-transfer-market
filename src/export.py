"""STEP 3 OF 4 -- write app/data.json for the web page.

Run:  python3 src/export.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import features
import model

OUT = Path(__file__).resolve().parents[1] / "app" / "data.json"

# How each feature is presented and manipulated in the app. `kind` tells the
# page what control to draw and how to get from a display value back to the
# model's input: money features are entered in dollars and log1p'd.
FEATURE_SPEC = {
    "pd_doubles_mean": {
        "label": "Doubles form",
        "blurb": "Average round reached in doubles brackets in the weeks before the draft, "
                 "on a 1-7 ladder (Round of 32 = 1, Champion = 7). MLP is played as team "
                 "doubles, so this is the most direct evidence bidders had.",
        "kind": "linear", "source_col": "pd_doubles_mean", "decimals": 2,
    },
    "pd_round_max": {
        "label": "Best finish",
        "blurb": "Deepest run in any bracket before the draft, same 1-7 ladder. Catches the "
                 "player who broke through once but is otherwise unremarkable.",
        "kind": "linear", "source_col": "pd_round_max", "decimals": 0,
    },
    "pd_events": {
        "label": "Brackets entered",
        "blurb": "How many brackets the player entered before draft day. Partly form, partly "
                 "a measure of who was healthy and active on tour.",
        "kind": "linear", "source_col": "pd_events", "decimals": 0,
    },
    "log_total_2025": {
        "label": "2025 earnings",
        "blurb": "Contracts plus prize money the season before the draft. Carried the model "
                 "before form was available; now adds almost nothing on top of it.",
        "kind": "money", "source_col": "total_2025",
    },
}


def build_payload(df, X, y, pipe, scores):
    """One dict holding everything the web page needs."""
    ridge = pipe.named_steps["ridge"]
    scaler = pipe.named_steps["standardscaler"]
    names = list(X.columns)

    pred_log = pipe.predict(X)
    predicted = np.exp(pred_log)
    actual = df["price_raw"].to_numpy(dtype=float)

    # positive -> model thinks they're worth MORE than they sold for = a BARGAIN
    # negative -> model thinks they're worth LESS = an OVERPAY
    # Flipping this sign produces a confident, backwards leaderboard.
    residual_log = pred_log - y.to_numpy()

    # Out-of-fold: each player scored by a model that never saw them. The
    # leaderboard's honest twin -- the card shows the fitted number, this says
    # how much of it survives the player being held out.
    oof_log = model.cv_predictions(model.build_pipeline(), X, y)
    residual_log_oof = oof_log - y.to_numpy()

    # Uncertainty band from out-of-fold error, not training error. Wide on
    # purpose: at R2 this low the band IS the result.
    resid_sd = float(np.std(residual_log_oof, ddof=1))
    z80 = 1.2816

    Z = scaler.transform(X)
    contrib = Z * ridge.coef_          # per-player, per-feature push in log dollars

    cis = model.confidence_intervals(X, y).set_index("feature")

    players = []
    for i, row in df.iterrows():
        players.append({
            "player_key": row["player_key"],
            "player": row["player"],
            "team": row["team"],
            # pick_no is fine HERE -- displayed, never predicted from.
            "pick_no": int(row["pick_no"]),
            "gender": None if pd.isna(row["gender"]) else row["gender"],
            "actual": int(round(actual[i])),
            "predicted": int(round(predicted[i])),
            "predicted_lo": int(round(np.exp(pred_log[i] - z80 * resid_sd))),
            "predicted_hi": int(round(np.exp(pred_log[i] + z80 * resid_sd))),
            "predicted_oof": int(round(np.exp(oof_log[i]))),
            "residual_log": float(residual_log[i]),
            "residual_log_oof": float(residual_log_oof[i]),
            "features": {n: float(X.iloc[i][n]) for n in names},
            "contributions": {n: float(contrib[i, j]) for j, n in enumerate(names)},
            "display": {
                "total_2025": _num(row.get("total_2025")),
                "career_earnings": _num(row.get("career_earnings")),
                "earnings_rank": _num(row.get("earnings_rank")),
                "card_tier": None if pd.isna(row.get("card_tier")) else row.get("card_tier"),
                "pd_events": _num(row.get("pd_events")),
                "pd_doubles_mean": _num(row.get("pd_doubles_mean")),
                "pd_round_max": _num(row.get("pd_round_max")),
                "pd_titles": _num(row.get("pd_titles")),
                "pd_podiums": _num(row.get("pd_podiums")),
                "pd_prize": _num(row.get("pd_prize")),
                "has_results": int(row.get("has_results", 0)),
                "imputed_total_2025": bool(pd.isna(row.get("total_2025"))),
            },
        })

    feats = []
    for j, n in enumerate(names):
        spec = FEATURE_SPEC[n]
        col = spec["source_col"]
        if spec["kind"] in ("money", "linear"):
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            dmin, dmax = float(vals.min()), float(vals.max())
        else:
            dmin, dmax = 0.0, 1.0
        feats.append({
            "name": n,
            "label": spec["label"],
            "blurb": spec["blurb"],
            "kind": spec["kind"],
            "coef": float(ridge.coef_[j]),
            "mean": float(scaler.mean_[j]),
            "std": float(scaler.scale_[j]),
            # Clamp the slider to what was observed. Extrapolating past the
            # training data is the failure mode we cut the amateur tier to avoid.
            "display_min": dmin,
            "display_max": dmax,
            "decimals": spec.get("decimals", 0),
            "ols_coef": float(cis.loc[n, "coef"]),
            "ci_lo": float(cis.loc[n, "ci_lo"]),
            "ci_hi": float(cis.loc[n, "ci_hi"]),
            "p_value": float(cis.loc[n, "p_value"]),
        })

    base = model.baselines(X, y)
    rank24 = model.holdout_2024_rank_check(pipe)

    # The honesty block. With R2 this low the fitted values barely move off the
    # mean, so the residual ranking collapses toward "sorted by price ascending"
    # -- which restates the answer instead of explaining it. Measure that and
    # ship the number, so the page can say it out loud instead of implying a
    # bargain-hunting power the model does not have.
    resid_vs_actual = float(np.corrcoef(residual_log, y.to_numpy())[0, 1])
    diagnostics = {
        "predicted_min": float(predicted.min()),
        "predicted_max": float(predicted.max()),
        "predicted_spread_x": float(predicted.max() / predicted.min()),
        "actual_min": float(actual.min()),
        "actual_max": float(actual.max()),
        "actual_spread_x": float(actual.max() / actual.min()),
        "resid_vs_actual_corr": resid_vs_actual,
        "oof_vs_insample_resid_corr": float(
            np.corrcoef(residual_log, residual_log_oof)[0, 1]),
        "in_sample_r2": float(
            1 - np.sum((y.to_numpy() - pred_log) ** 2)
            / np.sum((y.to_numpy() - y.to_numpy().mean()) ** 2)),
    }

    import features as _f
    form_meta = {
        "cutoff": _f.DRAFT_DATE,
        "covered": int(df["has_results"].sum()),
        "imputed": int((1 - df["has_results"]).sum()),
    }

    return {
        "model": {
            "n": int(len(df)),
            "season": 2026,
            "form": form_meta,
            "target": "log(price_raw)",
            "alpha": float(ridge.alpha),
            "intercept": float(ridge.intercept_),
            "r2": scores["r2"],
            "mae_dollars": scores["mae_dollars"],
            "fold_r2_mean": scores["fold_r2_mean"],
            "fold_r2_sd": scores["fold_r2_sd"],
            "fold_r2_p10": scores["fold_r2_p10"],
            "fold_r2_p90": scores["fold_r2_p90"],
            "n_folds": scores["n_folds"],
            "resid_sd_log": resid_sd,
            "band_z": z80,
            "band_label": "80%",
            "coefficients": [{"name": f["name"], "label": f["label"],
                              "coef": f["coef"]} for f in feats],
            "features": feats,
            "baselines": {
                k: {"r2": v["r2"], "mae_dollars": v["mae_dollars"],
                    "fold_r2_mean": v["fold_r2_mean"], "fold_r2_sd": v["fold_r2_sd"]}
                for k, v in base.items()
            },
            "beats_earnings_only": bool(scores["r2"] > base["earnings_only"]["r2"]),
            "beats_mean_only": bool(scores["r2"] > base["mean_only"]["r2"]),
            "alpha_sensitivity": model.alpha_sensitivity(X, y),
            "holdout_2024_rank": rank24,
            "diagnostics": diagnostics,
        },
        "players": players,
    }


def _num(v):
    """None for anything pandas calls missing; a plain float otherwise."""
    if v is None or pd.isna(v):
        return None
    return float(v)


def main():
    df = features.add_features(features.load())
    X, y = features.make_xy(df, features.FEATURES)

    pipe = model.fit_final(X, y)
    # From cross-validation, not from pipe's own training data. That's the
    # number the app is allowed to display.
    scores = model.evaluate(model.build_pipeline(), X, y)

    payload = build_payload(df, X, y, pipe, scores)

    # The decomposition the app relies on must actually add up.
    for p in payload["players"]:
        recon = payload["model"]["intercept"] + sum(p["contributions"].values())
        if abs(np.exp(recon) - p["predicted"]) > max(1.0, 0.001 * p["predicted"]):
            raise ValueError(f"contributions don't reconstruct {p['player']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    m = payload["model"]
    top = max(payload["players"], key=lambda p: p["predicted"])
    print(f"\nexport.py  (step 3 of 4)")
    print(f"  wrote {len(payload['players'])} players -> {OUT}")
    print(f"  CV R2 {m['r2']:+.3f}   MAE ${m['mae_dollars']:,.0f}   "
          f"band +/-{m['band_z'] * m['resid_sd_log']:.2f} log dollars")
    print(f"  highest modeled value: {top['player']} "
          f"(${top['predicted']:,} vs ${top['actual']:,} actual)")
    print(f"  beats mean_only: {m['beats_mean_only']}   "
          f"beats earnings_only: {m['beats_earnings_only']}\n")


if __name__ == "__main__":
    main()
