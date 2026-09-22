from __future__ import annotations

import json
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


class ScoreTargetConfig(StrategyConfig):
    """Configuration for ScoreTargetStrategy."""

    instrument_id: InstrumentId
    bar_type: BarType
    # JSON produced by notebook 02 (scores_to_json) — read at on_start when
    # ``score`` is None. NOTE: the backtest factory always builds a BTCUSDT
    # instrument id, so file lookup by symbol only matches keys like
    # ``BTCUSDT``; Portfolio mode injects the real per-leg score instead.
    scores_path: str = "notebooks/scores_qlib.json"
    # Per-leg score injected by portfolio_runner (from scores_qlib.json via
    # the app). None → fall back to the scores_path lookup.
    score: Optional[float] = None
    # Buy only when score > min_score; default gates non-positive scores out.
    min_score: float = 0.0
    # Sized like the other example strategies; in Portfolio mode the runner
    # scales this by the leg's capital share (see portfolio_runner).
    trade_size: Decimal = Decimal("0.1")


class ScoreTargetStrategy(Strategy):
    """Buy once when the model score for this asset clears ``min_score``; stay flat otherwise.

    The score comes from (in order): the injected ``score`` param (set per
    leg by ``portfolio_runner`` from ``scores_qlib.json``), or a file lookup
    in ``scores_path`` keyed by instrument symbol. Assets scoring at/below
    ``min_score`` never order — their allocation sits in cash. This closes
    the loop from notebook 02's ``scores_qlib.json`` to position taking.
    """

    def __init__(self, config: ScoreTargetConfig) -> None:
        super().__init__(config)
        self.instrument: Optional[Instrument] = None
        self._bought: bool = False
        self._score: float = 0.0

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
                scores = symbol_scores(self.config.scores_path)
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
            self._score = score_for_symbol(scores, symbol)
            self.log.info(
                f"score for {symbol}: {self._score} (min_score={self.config.min_score})"
            )

        try:
            self.request_bars(self.config.bar_type)
        except TypeError:
            pass  # request_bars API differs across builds; backtest bars arrive via subscribe
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        """Buy on the first bar if the score clears the threshold."""
        if bar.bar_type != self.config.bar_type:
            return
        if self._bought:
            return
        if self._score > self.config.min_score and self.portfolio.is_flat(
            self.config.instrument_id
        ):
            self._buy()

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
