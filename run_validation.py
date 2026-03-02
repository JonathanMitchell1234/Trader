"""
Validation runner for benchmark market windows.

Runs two backtests used for strategy regression checks:
1) Bear window: verifies bear-market adaptation improves over baseline.
2) Bull window: verifies bull-market performance is preserved.
"""

from __future__ import annotations

import argparse
import datetime as dt

import config
from backtest import Backtester


BASELINE_BEAR_RETURN_PCT = -1.84
BASELINE_BULL_RETURN_PCT = 202.33


def _run_window(
    label: str,
    start: dt.date,
    end: dt.date,
    symbols: list[str],
    capital: float,
) -> dict:
    print("\n" + "=" * 72)
    print(f"{label}: {start} -> {end}")
    print("=" * 72)

    bt = Backtester(
        symbols=symbols,
        start_date=start,
        end_date=end,
        initial_capital=capital,
    )
    stats = bt.run()
    if not stats:
        raise RuntimeError(f"No stats returned for {label}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bear/bull benchmark backtests and check regression guardrails.",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=300,
        help="Initial capital for each benchmark run (default: 300)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Override symbols. If omitted, uses full WATCHLIST.",
    )
    parser.add_argument(
        "--bull-tolerance",
        type=float,
        default=0.05,
        help="Allowed bull return drop vs baseline in pct points (default: 0.05).",
    )
    args = parser.parse_args()

    symbols = args.symbols if args.symbols else list(config.WATCHLIST)

    bear_stats = _run_window(
        label="Bear Validation",
        start=dt.date(2020, 1, 1),
        end=dt.date(2022, 2, 1),
        symbols=symbols,
        capital=args.capital,
    )
    bull_stats = _run_window(
        label="Bull Validation",
        start=dt.date(2023, 1, 1),
        end=dt.date(2026, 2, 15),
        symbols=symbols,
        capital=args.capital,
    )

    bear_return = float(bear_stats.get("total_return_pct", 0.0))
    bull_return = float(bull_stats.get("total_return_pct", 0.0))

    bear_pass = bear_return > BASELINE_BEAR_RETURN_PCT
    bull_floor = BASELINE_BULL_RETURN_PCT - args.bull_tolerance
    bull_pass = bull_return >= bull_floor

    print("\n" + "=" * 72)
    print("VALIDATION SUMMARY")
    print("=" * 72)
    print(
        f"Bear return : {bear_return:+.2f}%  (baseline {BASELINE_BEAR_RETURN_PCT:+.2f}%)"
    )
    print(
        f"Bull return : {bull_return:+.2f}%  "
        f"(baseline {BASELINE_BULL_RETURN_PCT:+.2f}%, floor {bull_floor:+.2f}%)"
    )
    print(f"Bear improved: {'PASS' if bear_pass else 'FAIL'}")
    print(f"Bull preserved: {'PASS' if bull_pass else 'FAIL'}")

    if not (bear_pass and bull_pass):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
