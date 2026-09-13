"""Run the first development-only AI Infrastructure timing baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def performance(returns: pd.Series, periods_per_year: float) -> dict[str, float | int]:
    returns = returns.dropna().astype(float)
    wealth = (1.0 + returns).cumprod()
    annual_return = float(wealth.iloc[-1] ** (periods_per_year / len(returns)) - 1.0) if len(returns) else np.nan
    annual_vol = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 else np.nan
    sharpe = annual_return / annual_vol if annual_vol and np.isfinite(annual_vol) and annual_vol > 0 else np.nan
    drawdown = wealth / wealth.cummax() - 1.0
    return {"observations": int(len(returns)), "total_return": float(wealth.iloc[-1] - 1.0), "annual_return": annual_return, "annual_volatility": annual_vol, "sharpe_zero_rf": float(sharpe), "max_drawdown": float(drawdown.min())}


def run(args: argparse.Namespace) -> Path:
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    panel_run = Path(cfg["panel_run"])
    panel_summary = json.loads((panel_run / "summary.json").read_text(encoding="utf-8"))
    if panel_summary.get("status") != "complete_development_only":
        raise ValueError("Configured panel is not explicitly development-only")
    panel = pd.read_csv(panel_run / "timing_panel_dev.csv", parse_dates=["Date"])
    maximum_development_date = pd.Timestamp(cfg.get("maximum_development_date", "2022-12-31"))
    if panel["Date"].max() > maximum_development_date:
        raise ValueError("Configured panel contains data after allowed development period")
    output = Path(args.output_root) / args.run_id
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    features = cfg["features"]
    model_data = panel.loc[panel["model_eligible"]].copy()
    train = model_data.loc[model_data["split"].eq("training")].copy()
    valid = model_data.loc[model_data["split"].eq("validation")].copy()
    if train.empty or valid.empty or train["y"].nunique() != 2 or valid["y"].nunique() != 2:
        raise ValueError("Training/validation sets must both contain two target classes")
    if train["Date"].max() >= valid["Date"].min():
        raise ValueError("Time split overlap")
    if train[features].isna().any().any() or valid[features].isna().any().any():
        raise ValueError("Baseline expects the panel's complete-case eligible observations")

    model = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(**cfg["logistic"])),
    ])
    model.fit(train[features], train["y"].astype(int))
    valid["p_ai_outperform"] = model.predict_proba(valid[features])[:, 1]
    valid["logistic_hold_ai"] = valid["p_ai_outperform"].ge(float(cfg["signal_threshold"]))
    valid["momentum_hold_ai"] = valid["ai_excess_momentum_20"].gt(0)

    n = int(cfg["execution"]["rebalance_every_sessions"])
    valid["execution_event"] = False
    valid.loc[valid.index[::n], "execution_event"] = True
    execution = valid.loc[valid["execution_event"]].copy()
    execution["always_ai_return"] = execution["forward_ai_return"]
    execution["always_spy_return"] = execution["forward_spy_return"]
    execution["momentum_return"] = np.where(execution["momentum_hold_ai"], execution["forward_ai_return"], execution["forward_spy_return"])
    execution["logistic_return"] = np.where(execution["logistic_hold_ai"], execution["forward_ai_return"], execution["forward_spy_return"])
    execution["logistic_trade"] = execution["logistic_hold_ai"].ne(execution["logistic_hold_ai"].shift()).fillna(True)
    execution["momentum_trade"] = execution["momentum_hold_ai"].ne(execution["momentum_hold_ai"].shift()).fillna(True)
    periods = 252 / n
    rows = []
    for name, return_col, position_col in [
        ("always_ai", "always_ai_return", None), ("always_spy", "always_spy_return", None),
        ("momentum_20", "momentum_return", "momentum_hold_ai"), ("logistic_l2", "logistic_return", "logistic_hold_ai"),
    ]:
        record = {"strategy": name, **performance(execution[return_col], periods)}
        record["ai_exposure_fraction"] = float(execution[position_col].mean()) if position_col else (1.0 if name == "always_ai" else 0.0)
        record["switches_including_initial"] = int(execution[f"{name.split('_')[0]}_trade"].sum()) if name in {"momentum_20", "logistic_l2"} else 1
        record["transaction_costs_applied"] = False
        rows.append(record)
    strategy_metrics = pd.DataFrame(rows)
    classifier_metrics = pd.DataFrame([{
        "model": "logistic_l2", "validation_rows": int(len(valid)), "positive_rate": float(valid["y"].mean()),
        "roc_auc": float(roc_auc_score(valid["y"], valid["p_ai_outperform"])),
        "brier": float(brier_score_loss(valid["y"], valid["p_ai_outperform"])),
        "directional_accuracy": float((valid["logistic_hold_ai"].astype(int) == valid["y"].astype(int)).mean()),
    }])
    valid.to_csv(output / "validation_predictions_daily.csv", index=False)
    execution.to_csv(output / "validation_execution_events.csv", index=False)
    classifier_metrics.to_csv(output / "validation_classifier_metrics.csv", index=False)
    strategy_metrics.to_csv(output / "validation_strategy_metrics_gross.csv", index=False)
    joblib.dump(model, output / "logistic_l2.joblib")
    coefficients = pd.DataFrame({"feature": features, "standardised_coefficient": model.named_steps["logistic"].coef_.ravel()})
    coefficients["absolute_coefficient"] = coefficients["standardised_coefficient"].abs()
    coefficients.sort_values("absolute_coefficient", ascending=False).to_csv(output / "logistic_coefficients.csv", index=False)
    summary = {
        "schema_version": cfg["schema_version"], "run_id": args.run_id, "status": "complete_development_only_baseline",
        "config_path": str(cfg_path), "config_sha256": digest(cfg_path), "runner_path": str(Path(__file__)), "runner_sha256": digest(Path(__file__)),
        "panel_run": str(panel_run), "panel_summary_sha256": digest(panel_run / "summary.json"),
        "train_rows": int(len(train)), "validation_rows": int(len(valid)), "execution_events": int(len(execution)),
        "features": features, "classifier_metrics": classifier_metrics.iloc[0].to_dict(),
        "strategy_metrics": strategy_metrics.to_dict(orient="records"),
        "test_policy_verified": f"Read only configured development-only panel ending {panel['Date'].max().date()}; no test target path is configured.",
        "limitations": ["This is a first gross-return signal baseline; no transaction costs are applied.", "Validation is used only for reporting, not for refitting or parameter selection.", "Static-role universe limitation remains."],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ai_infrastructure_timing_baseline_v1_config.json")
    parser.add_argument("--output-root", default="data/model_runs/ai_infrastructure_timing_baseline_v1")
    parser.add_argument("--run-id", required=True)
    print(run(parser.parse_args()))
