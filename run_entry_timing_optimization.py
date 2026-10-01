from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.entry_timing_optimizer import (
    DEFAULT_DTES,
    DEFAULT_PROFIT_CAPTURES,
    DEFAULT_SDS,
    DEFAULT_STOP_MULTIPLES,
    add_lot_metrics,
    build_candidate_trades,
    evaluate_policies,
    load_spot,
    report,
    walk_forward_select,
)
from src.strangle_backtest import load_config


def _floats(values: list[str]) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def _ints(values: list[str]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def _required_credit_points(
    *,
    target_rupees: float,
    lots: int,
    lot_size: int,
    capture: float,
    costs,
) -> float:
    qty = int(lots) * int(lot_size)

    def net_for_credit(credit: float) -> float:
        sell = credit * qty
        buy = credit * (1.0 - float(capture)) * qty
        gross = float(capture) * credit * qty
        return gross - (
            costs.brokerage_per_order * 4
            + sell * 0.0015
            + (sell + buy) * costs.exchange_txn_pct
            + (sell + buy) * costs.sebi_turnover_pct
            + buy * costs.stamp_duty_buy_option_pct
            + (
                costs.brokerage_per_order * 4
                + (sell + buy) * costs.exchange_txn_pct
                + (sell + buy) * costs.sebi_turnover_pct
            )
            * costs.gst_pct
        )

    y0 = net_for_credit(0.0)
    y1 = net_for_credit(1.0)
    slope = y1 - y0
    if slope <= 0:
        raise ValueError("Could not solve credit requirement")
    return float((float(target_rupees) - y0) / slope)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Research-only NIFTY entry-timing / DTE / SD / exit optimization. "
            "Writes results only to the research branch output directory."
        )
    )
    ap.add_argument("--options-dir", required=True)
    ap.add_argument("--spot", required=True)
    ap.add_argument("--config", default="config_dte6_frozen_75_25.yaml")
    ap.add_argument("--out-dir", default="results/entry_timing_profit_target")
    ap.add_argument("--dtes", nargs="+", default=[str(x) for x in DEFAULT_DTES])
    ap.add_argument("--sds", nargs="+", default=[str(x) for x in DEFAULT_SDS])
    ap.add_argument(
        "--profit-captures",
        nargs="+",
        default=[str(x) for x in DEFAULT_PROFIT_CAPTURES],
    )
    ap.add_argument(
        "--stop-multiples",
        nargs="+",
        default=[str(x) for x in DEFAULT_STOP_MULTIPLES],
    )
    ap.add_argument("--lots", type=int, default=5)
    ap.add_argument("--reference-lot-size", type=int, default=65)
    ap.add_argument("--target-low", type=float, default=6000.0)
    ap.add_argument("--target-high", type=float, default=8000.0)
    ap.add_argument(
        "--require-targetable-band",
        action="store_true",
        help="Keep only entries whose ex-ante current-size target P&L is within the requested rupee band.",
    )
    ap.add_argument("--min-train-trades", type=int, default=25)
    ap.add_argument("--min-full-sample-trades", type=int, default=40)
    ap.add_argument("--max-excluded-weekdays", type=int, default=2)
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--rebalance-months", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg, costs, _ = load_config(args.config)
    cfg = replace(cfg, lots=1)
    spot = load_spot(Path(args.spot))

    dtes = _ints(args.dtes)
    sds = _floats(args.sds)
    captures = _floats(args.profit_captures)
    stops = _floats(args.stop_multiples)

    print(
        "Building candidate trade table: "
        f"DTE={dtes}, SD={sds}, capture={captures}, stop={stops}"
    )
    candidates = build_candidate_trades(
        options_dir=Path(args.options_dir),
        spot=spot,
        base_cfg=cfg,
        costs=costs,
        dtes=dtes,
        sds=sds,
        profit_captures=captures,
        stop_multiples=stops,
    )
    candidates = add_lot_metrics(
        candidates,
        costs,
        lots=args.lots,
        target_low=args.target_low,
        target_high=args.target_high,
        reference_lot_size=args.reference_lot_size,
    )
    print(
        "Candidate table built: "
        f"{len(candidates):,} rows; "
        f"current-size targetable rows="
        f"{int(candidates['target_band_hit_5lot_current'].sum()):,}"
    )
    candidates.to_csv(out / "candidate_trades.csv", index=False)

    analysis_candidates = (
        candidates[candidates["target_band_hit_5lot_current"]].copy()
        if args.require_targetable_band
        else candidates
    )

    # Stage A: timing diagnostics with the currently frozen 2-SD / 75% / 2.5x mechanics.
    timing = analysis_candidates[
        analysis_candidates["sd"].eq(2.0)
        & analysis_candidates["profit_capture"].eq(0.75)
        & analysis_candidates["stop_multiple"].eq(2.5)
    ].copy()
    timing_policies = evaluate_policies(
        timing,
        min_trades=args.min_full_sample_trades,
        max_excluded_days=args.max_excluded_weekdays,
        pnl_column="net_pnl_5lot_current",
        target_column="target_band_hit_5lot_current",
    )
    timing_policies.sort_values(
        ["expectancy", "profit_factor"],
        ascending=[False, False],
    ).to_csv(out / "stage_a_timing_policies.csv", index=False)

    # Stage B: joint but walk-forward-controlled tuning.
    full_policies = evaluate_policies(
        analysis_candidates,
        min_trades=args.min_full_sample_trades,
        max_excluded_days=args.max_excluded_weekdays,
        pnl_column="net_pnl_5lot_current",
        target_column="target_band_hit_5lot_current",
    )
    full_policies.sort_values(
        ["expectancy", "profit_factor"],
        ascending=[False, False],
    ).to_csv(out / "full_sample_policies.csv", index=False)

    selected, oos = walk_forward_select(
        candidates,
        train_months=args.train_months,
        test_months=args.test_months,
        rebalance_months=args.rebalance_months,
        min_train_trades=args.min_train_trades,
        target_low=args.target_low,
        target_high=args.target_high,
        max_excluded_days=args.max_excluded_weekdays,
        pnl_column="net_pnl_5lot_current",
        target_column="target_band_hit_5lot_current",
        require_target_band=args.require_targetable_band,
    )
    selected.to_csv(out / "walk_forward_selection.csv", index=False)
    oos.to_csv(out / "walk_forward_oos_trades.csv", index=False)

    if not oos.empty:
        oos_report = report(
            oos,
            pnl_column="net_pnl_5lot_current",
            target_column="target_band_hit_5lot_current",
        )
        oos_report["positive_test_folds"] = (
            selected.get("test_trades", pd.Series(dtype=float))
            .gt(0).sum()
            if not selected.empty
            else 0
        )
        oos_report["oos_start"] = str(oos["entry_timestamp"].min().date())
        oos_report["oos_end"] = str(oos["entry_timestamp"].max().date())
        pd.Series(oos_report).to_csv(out / "walk_forward_oos_report.csv")

    # Target-credit map for the actual 5-lot deployment objective.
    rows = []
    for capture in captures:
        for target in [args.target_low, (args.target_low + args.target_high) / 2.0, args.target_high]:
            credit = _required_credit_points(
                target_rupees=target,
                lots=args.lots,
                lot_size=args.reference_lot_size,
                capture=float(capture),
                costs=costs,
            )
            rows.append(
                {
                    "lots": args.lots,
                    "capture": float(capture),
                    "target_net_rupees": float(target),
                    "required_post_slippage_credit_points": credit,
                    "required_raw_bid_credit_approx_points": credit + 2.0 * costs.slippage_points_per_leg,
                }
            )
    pd.DataFrame(rows).to_csv(out / "target_credit_requirements.csv", index=False)

    print("\nStage A timing policy snapshot:")
    if timing_policies.empty:
        print("No timing policies met the minimum trade count.")
    else:
        print(timing_policies.head(12).to_string(index=False))

    print("\nWalk-forward selections:")
    print(selected.to_string(index=False) if not selected.empty else "No qualifying walk-forward selections.")

    if not oos.empty:
        print("\nWalk-forward OOS report:")
        print(pd.Series(oos_report).to_string())


if __name__ == "__main__":
    main()
