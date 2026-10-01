from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.adaptive_robustness import (
    DEFAULT_STRESS_SLIPPAGES,
    DEFAULT_WINNER_REMOVALS,
    build_frozen_control_trades,
    grouped_oos_stats,
    pair_adaptive_with_control,
    pnl_stats,
    run_slippage_stress,
    winner_concentration,
)
from src.adaptive_target_optimizer import (
    build_adaptive_target_trades,
    evaluate_adaptive_policies,
    summarize_oos,
    walk_forward_adaptive,
)
from src.entry_timing_optimizer import load_spot
from src.strangle_backtest import load_config


def _floats(values: list[str]) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def _ints(values: list[str]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Research-only robustness battery for the adaptive NIFTY short-strangle: "
            "slippage stress, same-date control pairing, winner concentration, "
            "and OOS breakdowns."
        )
    )
    ap.add_argument("--options-dir", required=True)
    ap.add_argument("--spot", required=True)
    ap.add_argument("--config", default="config_dte6_frozen_75_25.yaml")
    ap.add_argument("--out-dir", default="results/adaptive_robustness")
    ap.add_argument("--session-offsets", nargs="+", default=["2", "3", "4", "5", "6"])
    ap.add_argument("--sds", nargs="+", default=["1.5", "1.75", "2.0", "2.25"])
    ap.add_argument("--targets", nargs="+", default=["6000", "7000", "8000"])
    ap.add_argument("--stop-multiples", nargs="+", default=["2.0", "2.5"])
    ap.add_argument(
        "--stress-slippages",
        nargs="+",
        default=[str(x) for x in DEFAULT_STRESS_SLIPPAGES],
        help="Exact end-to-end rebuild levels, in option points per leg.",
    )
    ap.add_argument(
        "--winner-removals",
        nargs="+",
        default=[str(x) for x in DEFAULT_WINNER_REMOVALS],
        help="Top winners to remove for concentration analysis.",
    )
    ap.add_argument("--lots", type=int, default=5)
    ap.add_argument("--reference-lot-size", type=int, default=65)
    ap.add_argument("--min-train-trades", type=int, default=20)
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--rebalance-months", type=int, default=3)
    args = ap.parse_args()

    if args.lots <= 0 or args.reference_lot_size <= 0:
        raise ValueError("--lots and --reference-lot-size must be > 0")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg, costs, _ = load_config(args.config)
    cfg = replace(cfg, lots=1)
    spot = load_spot(Path(args.spot))

    session_offsets = _ints(args.session_offsets)
    sds = _floats(args.sds)
    targets = _floats(args.targets)
    stops = _floats(args.stop_multiples)
    stress_slippages = _floats(args.stress_slippages)
    winner_removals = _ints(args.winner_removals)

    print(
        "Building baseline adaptive OOS: "
        f"session_offsets={session_offsets}, SD={sds}, targets={targets}, "
        f"stops={stops}, lots={args.lots} x {args.reference_lot_size}"
    )
    baseline_candidates = build_adaptive_target_trades(
        options_dir=Path(args.options_dir),
        spot=spot,
        base_cfg=cfg,
        costs=costs,
        session_offsets=session_offsets,
        sds=sds,
        target_rupees=targets,
        stop_multiples=stops,
        reference_lot_size=args.reference_lot_size,
        lots=args.lots,
    )
    baseline_selected, baseline_oos = walk_forward_adaptive(
        baseline_candidates,
        train_months=args.train_months,
        test_months=args.test_months,
        rebalance_months=args.rebalance_months,
        min_train_trades=args.min_train_trades,
    )
    baseline_summary = summarize_oos(baseline_oos) if not baseline_oos.empty else {"trades": 0}

    baseline_candidates.to_csv(out / "baseline_candidate_trades.csv", index=False)
    baseline_selected.to_csv(out / "baseline_walk_forward_selection.csv", index=False)
    baseline_oos.to_csv(out / "baseline_walk_forward_oos_trades.csv", index=False)
    pd.Series(baseline_summary).to_csv(out / "baseline_walk_forward_oos_report.csv")

    concentration = winner_concentration(
        baseline_oos,
        removals=winner_removals,
    )
    concentration.to_csv(out / "winner_concentration.csv", index=False)

    by_target = grouped_oos_stats(baseline_oos, ["target_rupees"])
    by_target.to_csv(out / "oos_by_target.csv", index=False)

    by_offset = grouped_oos_stats(baseline_oos, ["session_offset"])
    by_offset.to_csv(out / "oos_by_session_offset.csv", index=False)

    by_target_offset = grouped_oos_stats(
        baseline_oos,
        ["target_rupees", "session_offset"],
    )
    by_target_offset.to_csv(out / "oos_by_target_and_session_offset.csv", index=False)

    print("\nBaseline OOS:")
    print(pd.Series(baseline_summary).to_string())

    print("\nWinner concentration:")
    print(concentration.to_string(index=False))

    print("\nOOS by target:")
    print(by_target.to_string(index=False) if not by_target.empty else "No OOS trades.")

    print("\nOOS by session offset:")
    print(by_offset.to_string(index=False) if not by_offset.empty else "No OOS trades.")

    print("\nBuilding same-date/same-expiry frozen 2SD/75%/2.5x control for paired comparison...")
    control = build_frozen_control_trades(
        options_dir=Path(args.options_dir),
        spot=spot,
        adaptive_oos=baseline_oos,
        base_cfg=cfg,
        costs=costs,
        reference_lot_size=args.reference_lot_size,
        lots=args.lots,
    )
    control.to_csv(out / "frozen_control_trades.csv", index=False)
    paired, paired_summary = pair_adaptive_with_control(baseline_oos, control)
    paired.to_csv(out / "paired_adaptive_vs_frozen_control.csv", index=False)
    pd.Series(paired_summary).to_csv(out / "paired_adaptive_vs_frozen_control_report.csv")

    print("\nPaired adaptive vs frozen control:")
    print(pd.Series(paired_summary).to_string())

    print("\nRunning slippage stress:")
    stress, frozen_stress = run_slippage_stress(
        options_dir=Path(args.options_dir),
        spot=spot,
        base_cfg=cfg,
        costs=costs,
        session_offsets=session_offsets,
        sds=sds,
        targets=targets,
        stop_multiples=stops,
        reference_lot_size=args.reference_lot_size,
        lots=args.lots,
        baseline_selected=baseline_selected,
        baseline_oos=baseline_oos,
        stress_slippages=stress_slippages,
        train_months=args.train_months,
        test_months=args.test_months,
        rebalance_months=args.rebalance_months,
        min_train_trades=args.min_train_trades,
    )
    stress.to_csv(out / "slippage_stress.csv", index=False)

    for slippage, trades in frozen_stress.items():
        label = f"{slippage:g}".replace(".", "_")
        trades.to_csv(
            out / f"slippage_{label}_baseline_selection_oos.csv",
            index=False,
        )

    print(stress.to_string(index=False))


if __name__ == "__main__":
    main()
