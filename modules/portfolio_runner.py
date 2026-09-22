# portfolio_runner.py
# -*- coding: utf-8 -*-
"""Multi-asset portfolio backtests (v0.3.0, custom weights since v0.4.x).

Runs N independent single-asset backtests with either an equal capital
split or caller-supplied per-asset weights (e.g. skfolio output from
``notebooks/01_skfolio_portfolio_optimization.ipynb``), aligns the
per-asset equity curves into a portfolio summary equity, and computes
per-asset contribution analysis.

Each leg reuses the existing single-asset runner unchanged (including its
BTCUSDT instrument factory); legs run on separate engines, so only the
equity/PnL aggregation is portfolio-level — cross-asset position netting
and shared-margin accounting are out of scope.
"""

from __future__ import annotations

import inspect
import logging
import numbers
from decimal import Decimal
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy

from .backtest_runner import run_backtest

_logger = logging.getLogger(__name__)

__all__ = ["run_portfolio_backtest"]


def _cfg_field_default(cfg_cls: Type[StrategyConfig], field: str) -> Any:
    """Best-effort read of a config field default.

    Nautilus configs are msgspec structs (no ``model_fields``), so the
    constructor signature is the reliable source; pydantic/dataclass are
    fallbacks for other config styles.
    """
    import inspect

    try:
        param = inspect.signature(cfg_cls).parameters.get(field)
        if param is not None and param.default is not inspect.Parameter.empty:
            return param.default
    except (TypeError, ValueError):
        pass
    mf = getattr(cfg_cls, "model_fields", None)
    if isinstance(mf, dict) and field in mf:
        default = mf[field].default
        if default is not None:
            return default
    dfields = getattr(cfg_cls, "__dataclass_fields__", {})
    if field in dfields:
        f = dfields[field]
        if f.default is not None:
            return f.default
    return None


def _scale_trade_size(
    params: Dict[str, Any],
    cfg_cls: Type[StrategyConfig],
    ratio: float,
) -> Dict[str, Any]:
    """Scale the strategy's ``trade_size`` by a capital ratio.

    Strategies size orders with a fixed-unit ``trade_size`` (they do not
    read the account balance), so per-leg capital weights would otherwise
    change only idle cash — not exposure. Scaling ``trade_size`` by
    ``allocation / equal_allocation`` makes PnL proportional to the
    allocated capital. ``ratio == 1`` returns ``params`` unchanged.
    """
    if ratio == 1.0 or not np.isfinite(ratio) or ratio <= 0:
        return params
    scaled = dict(params)
    base = scaled.get("trade_size", _cfg_field_default(cfg_cls, "trade_size"))
    if base is None or isinstance(base, bool):
        return scaled
    if isinstance(base, Decimal):
        scaled["trade_size"] = base * Decimal(str(ratio))
    elif isinstance(base, int):
        scaled["trade_size"] = max(1, int(round(base * ratio)))
    elif isinstance(base, float):
        scaled["trade_size"] = base * ratio
    else:
        return scaled
    return scaled


def _tz_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    return pd.DatetimeIndex(idx)


def _align_equity(curves: Dict[str, pd.Series], flat_value: float | Dict[str, float]) -> pd.DataFrame:
    """Align per-asset equity curves onto a shared tz-naive index.

    Before an asset's first observation the curve is held at its starting
    allocation; after the last observation the final value is forward-filled.

    ``flat_value`` is either a scalar (equal split) or a per-label dict of
    starting allocations (custom weights).
    """
    indices = [_tz_naive(s.index) for s in curves.values() if not s.empty]
    if not indices:
        return pd.DataFrame(columns=list(curves))
    idx = indices[0]
    for extra in indices[1:]:
        idx = idx.union(extra)
    idx = pd.DatetimeIndex(idx).sort_values()

    out: Dict[str, pd.Series] = {}
    for label, s in curves.items():
        fv = (
            float(flat_value.get(label, 0.0))
            if isinstance(flat_value, dict)
            else float(flat_value)
        )
        if s.empty:
            out[label] = pd.Series(fv, index=idx, dtype="float64")
            continue
        ser = pd.Series(s.to_numpy(dtype="float64"), index=_tz_naive(s.index))
        ser = ser[~ser.index.duplicated(keep="last")].sort_index()
        ser = ser.reindex(idx).ffill()
        ser = ser.fillna(fv)
        out[label] = ser
    return pd.DataFrame(out, index=idx)


def _sharpe(returns: pd.Series) -> float:
    if returns.empty or returns.std(ddof=0) == 0:
        return float("nan")
    return float((returns.mean() / returns.std(ddof=0)) * np.sqrt(252))


def _sortino(returns: pd.Series) -> float:
    neg = returns[returns < 0]
    if returns.empty or neg.empty or neg.std(ddof=0) == 0:
        return float("nan")
    return float((returns.mean() / neg.std(ddof=0)) * np.sqrt(252))


def _max_dd_abs(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float((series.cummax() - series).max())


def _max_dd_pct(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    peak = series.cummax().replace(0, np.nan)
    dd = ((peak - series) / peak).dropna()
    return float(dd.max()) if not dd.empty else float("nan")


def run_portfolio_backtest(
    strat_cls: Type[Strategy],
    cfg_cls: Type[StrategyConfig],
    params: Dict[str, Any],
    assets: List[Tuple[str, pd.DataFrame]],
    actor_cls: type,
    start_balance: float = 10_000.0,
    weights: Dict[str, float] | None = None,
    scores: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Run a portfolio of independent single-asset backtests.

    Parameters
    ----------
    strat_cls, cfg_cls, params, actor_cls
        Same contract as :func:`modules.backtest_runner.run_backtest`.
    assets : list of (label, DataFrame)
        One OHLCV DataFrame per asset (layout produced by ``load_ohlcv_csv``).
    start_balance : float
        Total portfolio starting capital.
    weights : dict, optional
        Per-asset relative weights (e.g. from ``notebooks/weights_skfolio.json``).
        Keys are matched against asset labels (case-insensitive) and
        renormalized over the selected assets. Missing / non-positive entries
        count as 0; custom legs allocating less than $1 are skipped. The
        strategy's fixed-unit ``trade_size`` is scaled by each leg's capital
        ratio so exposure follows the weights. When omitted (or when no
        positive weight matches), capital is split equally — the v0.3.0
        default, with ``trade_size`` untouched.
    scores : dict, optional
        Per-asset model scores (e.g. from ``notebooks/scores_qlib.json``).
        Matched against asset labels (case-insensitive; absent → 0.0) and
        injected as the ``score`` config param per leg — used by
        ``ScoreTargetStrategy`` to gate entries. Requires the strategy
        config to expose a ``score`` field; otherwise ignored.

    Returns
    -------
    dict
        ``assets``          – label → single-asset ``run_backtest`` result
        ``failed``          – label → error message for skipped legs
        ``equity_components`` – aligned per-asset equity DataFrame
        ``equity_df``       – summary DataFrame ``{"equity": total}``
        ``contributions``   – per-asset contribution rows (DataFrame)
        ``trades_df``       – concatenated trades of all legs
        ``metrics``         – aggregate portfolio metrics
        ``start_balance`` / ``allocation`` / ``allocations`` /
        ``weights_source`` / ``n_assets``
    """
    if not assets:
        raise ValueError("assets must contain at least one (label, dataframe) pair")
    if start_balance <= 0:
        raise ValueError("start_balance must be positive")

    n = len(assets)
    equal_allocation = start_balance / n

    # Per-leg score injection (only if the strategy config accepts it).
    scores_lookup: Dict[str, float] | None = None
    if scores is not None and "score" in inspect.signature(cfg_cls).parameters:
        scores_lookup = {}
        for sk, sv in scores.items():
            try:
                fv = float(sv)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                scores_lookup[str(sk).upper()] = fv

    # ── resolve per-leg allocations ────────────────────────────────────
    weights_source = "equal"
    allocs: Dict[str, float] = {label: equal_allocation for label, _ in assets}
    if weights:
        lookup = {
            str(k).upper(): float(v)
            for k, v in weights.items()
            if isinstance(v, numbers.Real) and np.isfinite(float(v)) and float(v) > 0
        }
        raw = {label: lookup.get(str(label).upper(), 0.0) for label, _ in assets}
        total = sum(raw.values())
        if total > 0:
            allocs = {label: start_balance * w / total for label, w in raw.items()}
            weights_source = "custom"
        else:
            _logger.warning(
                "weights provided but none match the selected assets — using equal split"
            )

    results: Dict[str, Any] = {}
    failed: Dict[str, str] = {}
    curves: Dict[str, pd.Series] = {}
    contrib_rows: List[Dict[str, Any]] = []

    for label, df in assets:
        allocation = allocs[label]
        if df is None or df.empty:
            failed[label] = "empty dataframe"
            _logger.warning("Portfolio leg %s skipped: empty dataframe", label)
            continue
        if weights_source == "custom" and allocation < 1.0:
            failed[label] = f"custom weight too small (allocation ${allocation:,.2f} < $1)"
            _logger.warning("Portfolio leg %s skipped: %s", label, failed[label])
            continue
        leg_params = params
        if weights_source == "custom":
            leg_params = _scale_trade_size(
                params, cfg_cls, allocation / equal_allocation
            )
        if scores_lookup is not None:
            leg_params = {
                **leg_params,
                "score": scores_lookup.get(str(label).upper(), 0.0),
            }
        try:
            res = run_backtest(
                strat_cls,
                cfg_cls,
                leg_params,
                df,
                actor_cls=actor_cls,
                starting_balance=allocation,
            )
        except Exception as exc:  # keep remaining legs running
            failed[label] = str(exc)
            _logger.exception("Portfolio leg %s failed", label)
            continue

        results[label] = res

        eq = res.get("equity_df")
        if eq is None or eq.empty or "equity" not in eq.columns:
            series = pd.Series(allocation, index=df.index[:1], dtype="float64")
        else:
            series = eq["equity"].astype("float64")
        if getattr(series.index, "tz", None) is not None:
            series.index = series.index.tz_convert(None)
        curves[label] = series

        m = res.get("metrics") or {}
        first = float(series.iloc[0]) if not series.empty else allocation
        last = float(series.iloc[-1]) if not series.empty else allocation
        pnl = last - first
        contrib_rows.append(
            {
                "asset": label,
                "start_equity": first,
                "final_equity": last,
                "pnl": pnl,
                "pnl_pct": (pnl / first * 100.0) if first else float("nan"),
                "trades": int(m.get("num_trades", 0) or 0),
                "win_rate": float(m.get("win_rate", float("nan"))),
                "profit_factor": float(m.get("profit_factor", float("nan"))),
                "max_drawdown": float(m.get("max_drawdown", float("nan"))),
            }
        )

    if not results:
        raise RuntimeError(f"All portfolio legs failed: {failed}")

    components = _align_equity(curves, allocs)
    summary = components.sum(axis=1).to_frame(name="equity")
    summary.sort_index(inplace=True)

    contrib = pd.DataFrame(contrib_rows)
    if not contrib.empty:
        total_pnl = float(contrib["pnl"].sum())
        contrib["share_pct"] = (
            contrib["pnl"] / total_pnl * 100.0 if total_pnl else float("nan")
        )
        contrib = contrib.sort_values("pnl", ascending=False).reset_index(drop=True)
        for col in (
            "start_equity",
            "final_equity",
            "pnl",
            "pnl_pct",
            "share_pct",
            "win_rate",
            "profit_factor",
            "max_drawdown",
        ):
            if col in contrib.columns:
                contrib[col] = contrib[col].round(2)

    trade_frames: List[pd.DataFrame] = []
    for label, res in results.items():
        t = res.get("trades_df")
        if t is not None and not t.empty:
            t = t.copy()
            t["asset"] = label
            trade_frames.append(t)
    all_trades = (
        pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    )

    eq_series = summary["equity"]
    total_start = float(eq_series.iloc[0])
    total_final = float(eq_series.iloc[-1])
    total_profit = total_final - total_start
    returns = eq_series.pct_change().dropna()
    sharpe = _sharpe(returns)
    sortino = _sortino(returns)
    max_dd = _max_dd_abs(eq_series)
    max_dd_pct = _max_dd_pct(eq_series)

    if all_trades.empty:
        gains = losses = 0.0
        num_trades = 0
        win_rate = 0.0
    else:
        gains = float(all_trades.loc[all_trades["profit"] > 0, "profit"].sum())
        losses = float(all_trades.loc[all_trades["profit"] < 0, "profit"].sum())
        num_trades = int(len(all_trades))
        win_rate = float((all_trades["profit"] > 0).mean() * 100.0)

    if losses == 0:
        profit_factor = np.inf if gains > 0 else np.nan
    else:
        profit_factor = gains / abs(losses)

    metrics = {
        "total_start": round(total_start, 2),
        "total_final": round(total_final, 2),
        "total_profit": round(total_profit, 2),
        "total_return_pct": (
            round(total_profit / total_start * 100.0, 2) if total_start else float("nan")
        ),
        "sharpe": round(sharpe, 2) if not np.isnan(sharpe) else float("nan"),
        "sortino": round(sortino, 2) if not np.isnan(sortino) else float("nan"),
        "max_drawdown": round(max_dd, 2) if not np.isnan(max_dd) else float("nan"),
        "max_drawdown_pct": (
            round(max_dd_pct * 100.0, 2) if not np.isnan(max_dd_pct) else float("nan")
        ),
        "num_trades": num_trades,
        "win_rate": round(win_rate, 2),
        "profit_factor": (
            round(float(profit_factor), 2) if np.isfinite(profit_factor) else float(profit_factor)
        ),
    }

    return {
        "assets": results,
        "failed": failed,
        "equity_components": components,
        "equity_df": summary,
        "contributions": contrib,
        "trades_df": all_trades,
        "metrics": metrics,
        "start_balance": float(start_balance),
        "allocation": float(equal_allocation),
        "allocations": {label: float(allocs[label]) for label, _ in assets},
        "weights_source": weights_source,
        "n_assets": len(results),
    }
