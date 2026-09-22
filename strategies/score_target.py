from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.trading.strategy import Strategy


def symbol_scores(path: str | Path) -> dict[str, float]:
    """Load ``{SYMBOL: score}`` from a scores JSON file.

    Accepts the notebook export (``{"asof": ..., "scores": {...}}``) or a
    flat ``{symbol: score}`` mapping. Keys are upper-cased.
    """
    payload = json.loads(Path(path).expanduser().read_text())
    if isinstance(payload, dict) and "series" in payload:
        raise ValueError(f"{path} holds a score series, not a snapshot")
    if isinstance(payload, dict) and isinstance(payload.get("scores"), dict):
        raw = payload["scores"]
    elif isinstance(payload, dict) and payload:
        raw = payload
    else:
        raise ValueError(f"unexpected scores payload in {path}")
    return {str(k).upper(): float(v) for k, v in raw.items()}


def score_for_symbol(scores: dict[str, float], symbol: str) -> float:
    """Case-insensitive lookup; unknown symbols score 0 (stay flat)."""
    return float(scores.get(str(symbol).upper(), 0.0))


def _parse_ts(key: object) -> int:
    """Parse a series timestamp key to nanoseconds (ISO-8601 or unix seconds)."""
    if isinstance(key, bool):
        raise ValueError(f"bad series timestamp: {key!r}")
    if isinstance(key, (int, float)):
        val = float(key)
        return int(val * 1e9) if val < 1e12 else int(val)
    text = str(key)
    try:
        val = float(text)
        return int(val * 1e9) if val < 1e12 else int(val)
    except ValueError:
        pass
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)


def score_history(path: str | Path) -> Optional[list[tuple[int, dict[str, float]]]]:
    """Load an optional ``{"series": {"<ts>": {SYMBOL: score}}}`` payload.

    Returns ``(ts_ns, {UPPER_SYMBOL: score})`` entries sorted by time, or
    ``None`` when the file is a snapshot (see :func:`symbol_scores`).
    """
    payload = json.loads(Path(path).expanduser().read_text())
    if not isinstance(payload, dict) or "series" not in payload:
        return None
    raw = payload["series"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"empty or invalid series in {path}")
    history = [
        (_parse_ts(key), {str(k).upper(): float(v) for k, v in scores.items()})
        for key, scores in raw.items()
    ]
    history.sort(key=lambda entry: entry[0])
    return history


def series_score(
    history: list[tuple[int, dict[str, float]]], symbol: str, ts_ns: int
) -> float:
    """Score for ``symbol`` at ``ts_ns``: latest series point at/before it."""
    score = 0.0
    for entry_ts, scores in history:
        if entry_ts > ts_ns:
            break
        score = score_for_symbol(scores, symbol)
    return score


class ScoreTargetConfig(StrategyConfig):
    """Configuration for ScoreTargetStrategy."""

    instrument_id: InstrumentId
    bar_type: BarType
    # JSON produced by notebook 02 (scores_to_json) — read at on_start when
    # ``score`` is None. May hold a snapshot (v0.5/v0.6) or a time series
    # (v0.7.0, ``{"series": {...}}``). NOTE: the backtest factory always
    # builds a BTCUSDT instrument id, so file lookup by symbol only matches
    # keys like ``BTCUSDT``; Portfolio mode injects the real per-leg score.
    scores_path: str = "notebooks/scores_qlib.json"
    # Per-leg score injected by portfolio_runner (from scores_qlib.json via
    # the app). None → fall back to the scores_path lookup.
    score: Optional[float] = None
    # Buy only when score > min_score; default gates non-positive scores out.
    # With a series file the strategy also exits when the score falls back
    # to/below this threshold (v0.7.0 rebalancing).
    min_score: float = 0.0
    # Minimum milliseconds between series re-reads (0 = every bar). Snapshot
    # and injected scores are constant, so this has no effect on them.
    rebalance_interval_ms: int = 0
    # Sized like the other example strategies; in Portfolio mode the runner
    # scales this by the leg's capital share (see portfolio_runner).
    trade_size: Decimal = Decimal("0.1")


class ScoreTargetStrategy(Strategy):
    """Take a long position while the model score for this asset clears ``min_score``.

    Score sources (in order): the injected ``score`` param (set per leg by
    ``portfolio_runner`` from ``scores_qlib.json``), or a file lookup in
    ``scores_path`` — either a snapshot (constant) or, as of v0.7.0, a time
    series (``{"series": {<ts>: {symbol: score}}}``). Snapshot/injected
    scores buy once and hold (v0.6.0 behavior). A series re-gates every bar
    (throttled by ``rebalance_interval_ms``): the strategy exits when the
    score falls back to ``min_score`` and re-enters when it recovers, so
    positions track the model over the backtest window. Assets scoring
    at/below ``min_score`` stay flat — their allocation sits in cash.
    """

    def __init__(self, config: ScoreTargetConfig) -> None:
        super().__init__(config)
        self.instrument: Optional[Instrument] = None
        self._bought: bool = False
        self._score: float = 0.0
        self._history: Optional[list[tuple[int, dict[str, float]]]] = None
        self._last_refresh_ns: Optional[int] = None

    # ───────────────────── lifecycle ─────────────────────
    def on_start(self) -> None:
        """Resolve instrument, read this symbol's score, subscribe to bars."""
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"{self.config.instrument_id} not found – stopping strategy")
            self.stop()
            return

        symbol = str(self.config.instrument_id.symbol)
        if self.config.score is not None:
            self._score = float(self.config.score)
            self.log.info(
                f"injected score for leg: {self._score} (min_score={self.config.min_score})"
            )
        else:
            try:
                history = score_history(self.config.scores_path)
                if history is not None:
                    self._history = history
                    self._score = 0.0  # resolved against the first bar
                    self.log.info(
                        f"loaded {len(history)} score points for {symbol} "
                        f"(rebalance_interval_ms={self.config.rebalance_interval_ms})"
                    )
                else:
                    scores = symbol_scores(self.config.scores_path)
                    self._score = score_for_symbol(scores, symbol)
                    self.log.info(
                        f"score for {symbol}: {self._score} (min_score={self.config.min_score})"
                    )
            except FileNotFoundError:
                self.log.error(
                    f"scores file not found: {self.config.scores_path} – stopping strategy"
                )
                self.stop()
                return
            except (ValueError, json.JSONDecodeError, TypeError) as exc:
                self.log.error(
                    f"bad scores file {self.config.scores_path}: {exc} – stopping"
                )
                self.stop()
                return

        try:
            self.request_bars(self.config.bar_type)
        except TypeError:
            pass  # request_bars API differs across builds; backtest bars arrive via subscribe
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        """Re-gate the position against the current score (series refresh + thresholds)."""
        if bar.bar_type != self.config.bar_type:
            return
        if self._history is not None:
            interval_ns = max(0, self.config.rebalance_interval_ms) * 1_000_000
            if self._last_refresh_ns is None or (
                bar.ts_event - self._last_refresh_ns
            ) >= interval_ns:
                self._score = series_score(
                    self._history,
                    str(self.config.instrument_id.symbol),
                    bar.ts_event,
                )
                self._last_refresh_ns = bar.ts_event
        if self._score > self.config.min_score:
            if not self._bought and self.portfolio.is_flat(self.config.instrument_id):
                self._buy()
        elif self._bought and not self.portfolio.is_flat(self.config.instrument_id):
            self._sell()

    # ───────────────────── helpers ──────────────────────
    def _buy(self) -> None:
        if self.instrument is None:
            return
        qty = self.instrument.make_qty(self.config.trade_size)  # type: ignore
        self.submit_order(
            self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.BUY,
                quantity=qty,
            )
        )
        self._bought = True

    def _sell(self) -> None:
        """Close the long opened by :meth:`_buy` (same size → flat)."""
        if self.instrument is None:
            return
        qty = self.instrument.make_qty(self.config.trade_size)  # type: ignore
        self.submit_order(
            self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.SELL,
                quantity=qty,
            )
        )
        self._bought = False
