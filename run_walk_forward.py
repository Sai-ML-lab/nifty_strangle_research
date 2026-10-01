from pathlib import Path
import argparse

from src.strangle_backtest import load_config, prepare_data
from src.walk_forward import walk_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="results/walk_forward.csv")
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--rebalance-months", type=int, default=3)
    args = ap.parse_args()

    cfg, costs, _ = load_config(args.config)
    data = prepare_data(args.data, cfg, slippage_points=costs.slippage_points_per_leg)
    wf = walk_forward(
        data, cfg, costs,
        train_months=args.train_months,
        test_months=args.test_months,
        rebalance_months=args.rebalance_months,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wf.to_csv(out, index=False)
    print(wf.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
