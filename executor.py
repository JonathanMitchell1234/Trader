"""
Trade Executor – the glue between strategy signals and the broker.
Handles the full lifecycle: scan → signal → size → order → track.
"""

from __future__ import annotations

import datetime as dt

import config
from broker import AlpacaBroker
from indicators import compute_all
from pdt_guard import PDTGuard
from risk_manager import RiskManager
from screener import Screener
from strategy import check_entry, check_exit
from logger import get_logger

log = get_logger("executor")


class TradeExecutor:
    """Orchestrates a single scan-and-act cycle."""

    def __init__(self) -> None:
        self.broker = AlpacaBroker()
        self.pdt = PDTGuard()
        self.screener = Screener(self.broker)
        self._init_risk_manager()

    def _init_risk_manager(self) -> None:
        equity = self.broker.get_equity()
        n_positions = len(self.broker.get_positions())
        self.risk = RiskManager(equity, n_positions)
        log.info(
            "Account: $%.2f  (%s mode, max %d positions, %.0f%% per position)",
            equity,
            "small" if equity < config.SMALL_ACCOUNT_THRESHOLD else "normal",
            config.get_max_positions(equity),
            config.get_position_pct(equity) * 100,
        )

    def refresh(self) -> None:
        """Refresh equity / position count before each cycle."""
        self._init_risk_manager()

    # ─────────────────────────────────────────────────────────
    # EXIT SCAN – check existing positions for exit signals
    # ─────────────────────────────────────────────────────────
    def scan_exits(self) -> int:
        """
        Iterate over open positions, check exit signals, and sell
        where appropriate (respecting PDT guard).
        Returns the number of positions closed.
        """
        positions = self.broker.get_positions()
        active_symbols = {p.symbol for p in positions}
        self.pdt.cleanup_stale(active_symbols)

        closed = 0
        for pos in positions:
            symbol = pos.symbol
            qty = int(pos.qty)

            # PDT check – can we sell today?
            if not self.pdt.can_sell_today(symbol):
                days = self.pdt.days_held(symbol)
                log.info(
                    "Skipping exit check for %s – held %s day(s), PDT blocked",
                    symbol,
                    days,
                )
                continue

            # Fetch fresh data and compute indicators
            try:
                df = self.broker.get_bars(symbol)
                if df is None or len(df) < config.EMA_TREND + 5:
                    continue
                df = compute_all(df)
            except Exception as exc:
                log.warning("Data error for %s: %s", symbol, exc)
                continue

            entry_price = float(pos.avg_entry_price)
            signal = check_exit(df, entry_price)

            if signal is not None:
                log.info(
                    "EXIT  %s  qty=%d  entry=%.2f  now=%.2f  reasons=%s",
                    symbol,
                    qty,
                    entry_price,
                    signal["price"],
                    signal["reasons"],
                )
                try:
                    # Cancel any existing orders for the symbol first
                    for order in self.broker.get_open_orders(symbol):
                        self.broker.api.cancel_order(order.id)

                    self.broker.submit_market_sell(symbol, qty)
                    self.pdt.record_sell(symbol)
                    closed += 1
                except Exception as exc:
                    log.error("Sell order failed for %s: %s", symbol, exc)
            else:
                # If position is profitable, consider adding trailing stop
                current_price = float(pos.current_price)
                unrealised_pct = (current_price - entry_price) / entry_price

                # Determine which trailing stop level applies
                if unrealised_pct >= config.TRAILING_STOP_TIGHT_ACTIVATE:
                    trail_pct = config.TRAILING_STOP_TIGHT_PCT
                elif unrealised_pct >= config.TRAILING_STOP_ACTIVATE_PCT:
                    trail_pct = config.TRAILING_STOP_PCT
                else:
                    trail_pct = None

                if trail_pct is not None:
                    # Check if trailing stop already exists
                    open_orders = self.broker.get_open_orders(symbol)
                    has_trailing = any(
                        o.type == "trailing_stop" for o in open_orders
                    )
                    if not has_trailing:
                        log.info(
                            "Adding trailing stop for %s (%.1f%% profit, trail=%.1f%%)",
                            symbol,
                            unrealised_pct * 100,
                            trail_pct * 100,
                        )
                        try:
                            self.broker.submit_trailing_stop(
                                symbol, qty, trail_pct
                            )
                        except Exception as exc:
                            log.warning("Trailing stop failed for %s: %s", symbol, exc)

        log.info("Exit scan complete – closed %d position(s)", closed)
        return closed

    # ─────────────────────────────────────────────────────────
    # ENTRY SCAN – look for new swing-trade setups
    # ─────────────────────────────────────────────────────────
    def scan_entries(self) -> int:
        """
        Screen the watchlist, evaluate entry signals, size positions,
        and submit buy orders.
        Returns the number of new positions opened.
        """
        self.refresh()

        if not self.risk.can_open_new_position():
            log.info("Max positions reached – skipping entry scan")
            return 0

        # Market regime filter: don't open new longs in a bear market
        if config.MARKET_REGIME_ENABLED:
            try:
                spy_df = self.broker.get_bars(config.MARKET_REGIME_SYMBOL, limit=250)
                if spy_df is not None and len(spy_df) >= 200:
                    from indicators import compute_all as _compute
                    spy_df = _compute(spy_df)
                    spy_row = spy_df.iloc[-1]
                    spy_ema200 = spy_row.get("ema_200", None)
                    if spy_ema200 is not None and spy_row["close"] < spy_ema200:
                        log.info("BEAR MARKET – %s below EMA-200, skipping entries",
                                 config.MARKET_REGIME_SYMBOL)
                        return 0
            except Exception as exc:
                log.warning("Market regime check failed: %s", exc)

        # Symbols we already hold – skip them
        held = {p.symbol for p in self.broker.get_positions()}
        # Symbols with pending buy orders – skip them too
        pending = {o.symbol for o in self.broker.get_open_orders() if o.side == "buy"}

        candidates = self.screener.screen()
        opened = 0

        for c in candidates:
            symbol = c["symbol"]
            if symbol in held or symbol in pending:
                continue

            if not self.risk.can_open_new_position():
                break

            if not self.pdt.can_buy_today(symbol):
                continue

            signal = check_entry(c["df"])
            if signal is None:
                continue

            entry_price = signal["price"]
            atr = signal["atr"]
            stop_loss = self.risk.compute_stop_loss(entry_price, atr)
            take_profit = self.risk.compute_take_profit(entry_price, atr)

            qty = self.risk.calculate_position_size(
                entry_price=entry_price,
                stop_price=stop_loss,
                buying_power=self.broker.get_buying_power(),
            )
            if qty == 0:
                log.info("Position size = 0 for %s - skipping", symbol)
                continue

            log.info(
                "ENTRY %s  qty=%.3f  price=%.2f  SL=%.2f  TP=%.2f  [%s]",
                symbol,
                qty,
                entry_price,
                stop_loss,
                take_profit,
                signal["reason"],
            )

            try:
                self.broker.submit_market_buy(
                    symbol, qty, stop_loss=stop_loss, take_profit=take_profit
                )
                self.pdt.record_buy(symbol)
                opened += 1
                # Update counter so risk manager knows
                self.risk.open_positions += 1
            except Exception as exc:
                log.error("Buy order failed for %s: %s", symbol, exc)

        log.info("Entry scan complete – opened %d position(s)", opened)
        return opened

    # ─────────────────────────────────────────────────────────
    # FULL CYCLE
    # ─────────────────────────────────────────────────────────
    def run_cycle(self) -> None:
        """Execute one full scan cycle: exits first, then entries."""
        if not self.broker.is_market_open():
            log.info("Market is closed – skipping cycle")
            return

        log.info("=" * 60)
        log.info("CYCLE START  equity=$%.2f  positions=%d",
                 self.broker.get_equity(),
                 len(self.broker.get_positions()))
        log.info("=" * 60)

        self.scan_exits()
        self.scan_entries()

        # Summary
        positions = self.broker.get_positions()
        equity = self.broker.get_equity()
        log.info(
            "CYCLE END  equity=$%.2f  positions=%d  symbols=%s",
            equity,
            len(positions),
            [p.symbol for p in positions],
        )
