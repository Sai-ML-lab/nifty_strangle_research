from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.adaptive_target_optimizer import (
    DEFAULT_SDS,
    DEFAULT_STOPS,
    DEFAULT_TARGETS,
    SESSION_OFFSETS,
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
            "Research-only NIFTY short-strangle optimization using trading-session "
            "entry offsets and fixed rupee profit targets."
        )
    )
    ap.add_argument("--options-dir", required=True)
    ap.add_argument("--spot", required=True)
    ap.add_argument("--config", default="config_dte6_frozen_75_25.yaml")
    ap.add_argument(
        "--out-dir",
        default="results/adaptive_rupee_target",
    )
    ap.add_argument(
        "--session-offsets",
        nargs="+",
        default=[str(x) for x in SESSION_OFFSETS],
    )
    ap.add_argument("--sds", nargs="+", default=[str(x) for x in DEFAULT_SDS])
    ap.add_argument(
        "--targets",
        nargs="+",
        default=[str(x) for x in DEFAULT_TARGETS],
        help="Fixed net rupee targets per trade for 5 current NIFTY lots.",
    )
    ap.add_argument(
        "--stop-multiples",
        nargs="+",
        default=[str(x) for x in DEFAULT_STOPS],
    )
    ap.add_argument("--lots", type=int, default=5)
    ap.add_argument("--reference-lot-size", type=int, default=65)
    ap.add_argument("--min-train-trades", type=int, default=20)
    ap.add_argument("--min-train-half-trades", type=int, default=10)
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--rebalance-months", type=int, default=3)
    args = ap.parse_args()

    if args.lots <= 0:
        raise ValueError("--lots must be > 0")
    if args.reference_lot_size <= 0:
        raise ValueError("--reference-lot-size must be > 0")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg, costs, _ = load_config(args.config)
    cfg = replace(cfg, lots=1)
    spot = load_spot(Path(args.spot))

    session_offsets = _ints(args.session_offsets)
    sds = _floats(args.sds)
    targets = _floats(args.targets)
    stops = _floats(args.stop_multiples)

    print(
        "Building adaptive candidate table: "
        f"session_offsets={session_offsets}, SD={sds}, "
        f"targets={targets}, stops={stops}, "
        f"lots={args.lots} x {args.reference_lot_size}"
    )
    candidates = build_adaptive_target_trades(
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
    candidates.to_csv(out / "adaptive_candidate_trades.csv", index=False)
    print(f"Candidate trades retained: {len(candidates):,}")

    policies = evaluate_adaptive_policies(
        candidates,
        min_trades=args.min_train_trades,
        min_half_trades=args.min_train_half_trades,
    )
    policies.sort_values(
        ["expectancy", "profit_factor", "target_hit_rate"],
        ascending=[False, False, False],
    ).to_csv(out / "adaptive_full_sample_policies.csv", index=False)

    selected, oos = walk_forward_adaptive(
        candidates,
        train_months=args.train_months,
        test_months=args.test_months,
        rebalance_months=args.rebalance_months,
        min_train_trades=args.min_train_trades,
    )
    selected.to_csv(out / "adaptive_walk_forward_selection.csv", index=False)
    oos.to_csv(out / "adaptive_walk_forward_oos_trades.csv", index=False)

    if not oos.empty:
        summary = summarize_oos(oos)
    else:
        summary = {"trades": 0, "positive_test_folds": 0, "total_test_folds": 0}

    pd.Series(summary).to_csv(out / "adaptive_walk_forward_oos_report.csv")

    by_target = []
    for target in targets:
        subset = oos[oos["target_rupees"].eq(float(target))] if not oos.empty else pd.DataFrame()
        row = {"target_rupees": float(target), **summarize_oos(subset)}
        by_target.append(row)
    pd.DataFrame(by_target).to_csv(out / "adaptive_oos_by_target.csv", index=False)

    print("\nFull-sample policy snapshot:")
    print(
        policies.head(15).to_string(index=False)
        if not policies.empty
        else "No policy reached the minimum sample size."
    )

    print("\nWalk-forward selections:")
    print(
        selected.to_string(index=False)
        if not selected.empty
        else "No walk-forward folds produced a selection."
    )

    print("\nWalk-forward OOS report:")
    print(pd.Series(summary).to_string())

    if not selected.empty:
        print("\nSelection status:")
        print(selected["selection_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
