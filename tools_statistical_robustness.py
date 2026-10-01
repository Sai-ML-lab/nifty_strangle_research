from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _weekly_block_ci(trades: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    x = trades.copy()
    x["entry_timestamp"] = pd.to_datetime(x["entry_timestamp"], errors="coerce")
    x["net_pnl"] = pd.to_numeric(x["net_pnl"], errors="coerce")
    x = x.dropna(subset=["entry_timestamp", "net_pnl"]).copy()
    weekly = x.set_index("entry_timestamp")["net_pnl"].resample("W-FRI").sum()
    weekly = weekly.reindex(pd.date_range(weekly.index.min(), weekly.index.max(), freq="W-FRI"), fill_value=0.0)
    return _mean_ci(weekly.to_numpy(dtype=float), rng, n_boot)


def robustness_report(trades: pd.DataFrame, slippage: pd.DataFrame | None = None, seed: int = 42, n_boot: int = 20000) -> pd.DataFrame:
    x = trades.copy()
    x["entry_timestamp"] = pd.to_datetime(x["entry_timestamp"], errors="coerce")
    x["net_pnl"] = pd.to_numeric(x["net_pnl"], errors="coerce")
    x = x.dropna(subset=["entry_timestamp", "net_pnl"]).sort_values("entry_timestamp")
    p = x["net_pnl"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    rows = []
    mean, lo, hi = _mean_ci(p, rng, n_boot)
    rows.append({"metric": "trade_mean_pnl", "estimate": mean, "ci_low_2_5": lo, "ci_high_97_5": hi, "probability_positive_bootstrap": np.nan})

    weekly_mean, wlo, whi = _weekly_block_ci(x, rng, n_boot)
    rows.append({"metric": "weekly_mean_pnl_block_bootstrap", "estimate": weekly_mean, "ci_low_2_5": wlo, "ci_high_97_5": whi, "probability_positive_bootstrap": np.nan})

    if len(p):
        draws = rng.integers(0, len(p), size=(n_boot, len(p)))
        boot_means = p[draws].mean(axis=1)
        rows[0]["probability_positive_bootstrap"] = float((boot_means > 0).mean())

    losses = p[p < 0]
    total_loss = abs(losses.sum()) if len(losses) else 0.0
    top1_share = abs(np.min(p)) / total_loss if total_loss else np.nan
    top5_share = abs(np.sort(losses)[:min(5, len(losses))].sum()) / total_loss if total_loss else np.nan

    row = {"metric": "loss_concentration_top1_share", "estimate": top1_share, "ci_low_2_5": np.nan, "ci_high_97_5": np.nan, "probability_positive_bootstrap": np.nan}
    rows.append(row)
    rows.append({"metric": "loss_concentration_top5_share", "estimate": top5_share, "ci_low_2_5": np.nan, "ci_high_97_5": np.nan, "probability_positive_bootstrap": np.nan})

    if "exit_reason" in x.columns:
        stop_loss = x.loc[x["exit_reason"].eq("stop"), "net_pnl"]
        total_losses = abs(x.loc[x["net_pnl"] < 0, "net_pnl"].sum())
        stop_loss_total = abs(stop_loss[stop_loss < 0].sum())
        rows.append({
            "metric": "stop_share_of_total_losses",
            "estimate": stop_loss_total / total_losses if total_losses else np.nan,
            "ci_low_2_5": np.nan, "ci_high_97_5": np.nan, "probability_positive_bootstrap": np.nan,
        })

    if slippage is not None and not slippage.empty:
        s = slippage.sort_values("slippage_points_per_leg")
        below = s[s["total_net_pnl"] < 0]
        above = s[s["total_net_pnl"] >= 0]
        if not below.empty and not above.empty:
            hi = above.iloc[-1]
            lo2 = below.iloc[0]
            x0, y0 = float(hi["slippage_points_per_leg"]), float(hi["total_net_pnl"])
            x1, y1 = float(lo2["slippage_points_per_leg"]), float(lo2["total_net_pnl"])
            zero = x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else np.nan
            rows.append({"metric": "linear_interpolated_zero_pnl_slippage_points_per_leg", "estimate": zero, "ci_low_2_5": np.nan, "ci_high_97_5": np.nan, "probability_positive_bootstrap": np.nan})

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap and tail concentration checks for a frozen NIFTY strategy")
    ap.add_argument("--trades", required=True)
    ap.add_argument("--slippage", default=None)
    ap.add_argument("--out", default="results/statistical_robustness.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap-reps", type=int, default=20000)
    args = ap.parse_args()

    trades = pd.read_csv(args.trades)
    slip = pd.read_csv(args.slippage) if args.slippage else None
    result = robustness_report(trades, slip, seed=args.seed, n_boot=args.bootstrap_reps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(result.to_string(index=False))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
