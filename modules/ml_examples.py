# ml_examples.py
# -*- coding: utf-8 -*-
"""Shared helpers for the v0.4.0 ML example notebooks (Qlib + skfolio).

Everything here works off the repo's OHLCV CSVs via :class:`DataConnector`,
so the notebooks demonstrate the same data path the Streamlit app uses.

Heavy libraries (``skfolio``, ``qlib``) are imported lazily inside the
functions that need them — importing this module stays cheap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_connector import DataConnector

__all__ = [
    "load_close_panel",
    "load_returns_panel",
    "load_ohlcv_frames",
    "make_features",
    "train_test_split",
    "skfolio_baselines",
    "summarize_portfolios",
    "weights_to_json",
    "dump_to_qlib",
    "run_qlib_lgbm",
]


# ─────────────────────────────── data ────────────────────────────────────────
def load_close_panel(
    csv_dir: str | Path = ".",
    symbols: Optional[Iterable[str]] = None,
    timeframe: str = "15min",
    exchange: str = "BINANCE",
) -> pd.DataFrame:
    """Return a close-price panel (index: UTC-naive DatetimeIndex, cols: symbols)."""
    conn = DataConnector(csv_dir=csv_dir)
    datasets = [
        d
        for d in conn.list_csv_datasets()
        if d["timeframe"] == timeframe and d["exchange"] == exchange
    ]
    if not datasets:
        raise FileNotFoundError(
            f"No CSV datasets for {exchange} @ {timeframe} in {csv_dir}"
        )
    wanted = list(symbols) if symbols else sorted({d["symbol"] for d in datasets})
    closes: Dict[str, pd.Series] = {}
    for sym in wanted:
        path = conn.get_csv_path(exchange, sym, timeframe)
        df = conn.load("CSV", path)
        s = df["close"].astype("float64")
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_convert(None)
        closes[sym] = s
    panel = pd.DataFrame(closes).sort_index().dropna(how="all")
    return panel


def load_returns_panel(
    csv_dir: str | Path = ".",
    symbols: Optional[Iterable[str]] = None,
    timeframe: str = "15min",
    exchange: str = "BINANCE",
) -> pd.DataFrame:
    """Simple-return panel derived from :func:`load_close_panel`."""
    closes = load_close_panel(csv_dir, symbols, timeframe, exchange)
    return closes.pct_change().dropna(how="all")


def make_features(closes: pd.DataFrame) -> pd.DataFrame:
    """Lightweight cross-sectional features (momentum / volatility / trend).

    Returns a MultiIndex-column DataFrame: (feature, symbol).
    Suitable as a starting point for ML models in either notebook.
    """
    rets = closes.pct_change()
    feats = pd.DataFrame(index=closes.index)
    out: Dict[str, pd.Series] = {}
    for sym in closes.columns:
        c = closes[sym]
        r = rets[sym]
        out[("ret_1", sym)] = r
        out[("ret_5", sym)] = c.pct_change(5)
        out[("ret_20", sym)] = c.pct_change(20)
        out[("mom_50", sym)] = c / c.rolling(50).mean() - 1.0
        out[("vol_20", sym)] = r.rolling(20).std()
        out[("rsi_14", sym)] = _rsi(c, 14)
    feats = pd.DataFrame(out, index=closes.index)
    feats.columns = pd.MultiIndex.from_tuples(feats.columns, names=["feature", "symbol"])
    return feats.dropna(how="all")


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(period).mean()
    loss = (-delta.clip(upper=0.0)).rolling(period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def train_test_split(
    returns: pd.DataFrame, test_frac: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological (no-shuffle) split of a returns panel."""
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be in (0, 1)")
    cut = int(len(returns) * (1.0 - test_frac))
    return returns.iloc[:cut].copy(), returns.iloc[cut:].copy()


# ─────────────────────────────── skfolio ─────────────────────────────────────
def skfolio_baselines(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Dict[str, object]:
    """Fit simple skfolio optimizers on train returns, predict on test.

    Returns a dict ``{name: out_of_sample_portfolio}`` for
    EqualWeighted, MeanRisk (mean–variance) and HierarchicalRiskParity.
    """
    from skfolio.optimization import EqualWeighted, HierarchicalRiskParity, MeanRisk

    models = {
        "EqualWeighted": EqualWeighted(),
        "MeanRisk": MeanRisk(),
        "HRP": HierarchicalRiskParity(),
    }
    fitted: Dict[str, object] = {}
    for name, model in models.items():
        model.fit(X_train)
        fitted[name] = model.predict(X_test)
    return fitted


def summarize_portfolios(portfolios: Dict[str, object]) -> pd.DataFrame:
    """Build a compact comparison table of out-of-sample portfolio metrics."""
    rows = {}
    for name, pf in portfolios.items():
        rows[name] = {
            "ann_return": pf.annualized_mean,
            "ann_vol": pf.annualized_standard_deviation,
            "sharpe": pf.sharpe_ratio,
            "sortino": pf.sortino_ratio,
            "max_drawdown": pf.max_drawdown,
            "calmar": pf.calmar_ratio,
        }
    table = pd.DataFrame(rows).T
    table.index.name = "portfolio"
    return table.round(4)


def weights_to_json(portfolios: Dict[str, object], path: str | Path) -> Path:
    """Persist out-of-sample weights to JSON (for use by the Streamlit app)."""
    payload: Dict[str, Dict[str, float]] = {}
    for name, pf in portfolios.items():
        weights = np.asarray(pf.weights, dtype="float64")
        assets = list(pf.assets)
        payload[name] = {str(a): float(w) for a, w in zip(assets, weights)}
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2))
    return path


# ─────────────────────────────── qlib ────────────────────────────────────────
def load_ohlcv_frames(
    csv_dir: str | Path = ".",
    symbols: Optional[Iterable[str]] = None,
    timeframe: str = "15min",
    exchange: str = "BINANCE",
) -> Dict[str, pd.DataFrame]:
    """Return ``{symbol: OHLCV DataFrame}`` (volume filled with NaN if absent)."""
    conn = DataConnector(csv_dir=csv_dir)
    datasets = [
        d
        for d in conn.list_csv_datasets()
        if d["timeframe"] == timeframe and d["exchange"] == exchange
    ]
    if not datasets:
        raise FileNotFoundError(
            f"No CSV datasets for {exchange} @ {timeframe} in {csv_dir}"
        )
    wanted = list(symbols) if symbols else sorted({d["symbol"] for d in datasets})
    frames: Dict[str, pd.DataFrame] = {}
    for sym in wanted:
        path = conn.get_csv_path(exchange, sym, timeframe)
        df = conn.load("CSV", path).copy()
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert(None)
        if "volume" not in df.columns:
            df["volume"] = np.nan
        frames[sym] = df[["open", "high", "low", "close", "volume"]].astype("float64")
    return frames


def dump_to_qlib(
    qlib_dir: str | Path,
    frames: Dict[str, pd.DataFrame],
    freq: str = "15min",
    market: str = "crypto",
) -> Path:
    """Dump OHLCV frames into the qlib binary data layout.

    Writes ``calendars/{freq}.txt``, ``instruments/{market}.txt`` and
    ``features/{symbol}/{field}.{freq}.bin`` following the format used by
    qlib's official ``scripts/dump_bin.py`` (float32: first value = calendar
    start index, remaining values = feature samples).

    Returns the resolved qlib data directory (pass as ``provider_uri``).
    """
    qlib_dir = Path(qlib_dir).expanduser().resolve()
    high_freq = not (freq == "day")
    dt_fmt = "%Y-%m-%d %H:%M:%S" if high_freq else "%Y-%m-%d"

    # ── union calendar ────────────────────────────────────────────────
    all_idx = None
    for df in frames.values():
        idx = pd.DatetimeIndex(df.index)
        all_idx = idx if all_idx is None else all_idx.union(idx)
    if all_idx is None or len(all_idx) == 0:
        raise ValueError("no data to dump")
    all_idx = all_idx.sort_values().unique()
    cal_index = {ts: i for i, ts in enumerate(all_idx)}

    cal_dir = qlib_dir / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)
    cal_path = cal_dir / f"{freq}.txt".lower()
    cal_path.write_text("\n".join(ts.strftime(dt_fmt) for ts in all_idx) + "\n")

    # ── instruments ───────────────────────────────────────────────────
    inst_dir = qlib_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for sym, df in frames.items():
        code = sym.upper()
        start = pd.Timestamp(df.index.min()).strftime(dt_fmt)
        end = pd.Timestamp(df.index.max()).strftime(dt_fmt)
        lines.append(f"{code}\t{start}\t{end}")
    (inst_dir / f"{market}.txt").write_text("\n".join(lines) + "\n")
    (inst_dir / "all.txt").write_text("\n".join(lines) + "\n")

    # ── features ──────────────────────────────────────────────────────
    for sym, df in frames.items():
        fdir = qlib_dir / "features" / sym.lower()
        fdir.mkdir(parents=True, exist_ok=True)
        # align onto the union calendar, restricted to this symbol's span
        sub = df.reindex(all_idx)
        start_idx = cal_index[pd.Timestamp(df.index.min())]
        end_idx = cal_index[pd.Timestamp(df.index.max())]
        aligned = sub.iloc[start_idx : end_idx + 1]
        for field in df.columns:
            values = aligned[field].to_numpy(dtype="float64")
            bin_arr = np.hstack([start_idx, values]).astype("<f")
            bin_arr.tofile(str((fdir / f"{field.lower()}.{freq.lower()}.bin").resolve()))
        # Alpha158 references $vwap; fall back to close when absent
        if "vwap" not in df.columns:
            values = aligned["close"].to_numpy(dtype="float64")
            bin_arr = np.hstack([start_idx, values]).astype("<f")
            bin_arr.tofile(str((fdir / f"vwap.{freq.lower()}.bin").resolve()))

    return qlib_dir


def run_qlib_lgbm(
    provider_uri: str | Path,
    freq: str = "15min",
    market: str = "crypto",
    instruments: Optional[List[str]] = None,
    train_span=("2024-09-01", "2025-01-31"),
    valid_span=("2025-02-01", "2025-02-28"),
    test_span=("2025-03-01", "2025-04-08"),
) -> Tuple[pd.DataFrame, object]:
    """Train a qlib LightGBM model on dumped data; return (predictions, model).

    Predictions: DataFrame indexed by (datetime, instrument) with column
    ``score`` — ready for IC / quantile evaluation in the notebook.
    """
    import os
    import tempfile

    # mlflow ≥3 rejects the legacy file store unless opted in.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    import qlib
    from qlib.utils import init_instance_by_config

    provider_uri = str(Path(provider_uri).expanduser().resolve())
    qlib.init(provider_uri=provider_uri, region="us")

    # qlib defaults its mlflow file-store to ./mlruns (cwd) — redirect it to a
    # temp dir so notebook runs don't litter the repo with mlruns/.
    from qlib.workflow import R

    mlruns_dir = Path(tempfile.gettempdir()) / "nautilus_qlib_mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    R.set_uri(f"file:{mlruns_dir}")

    from qlib.data import D

    # sanity: features readable
    probe = D.features(
        instruments or ["BTCUSD"],
        ["$close"],
        start_time=train_span[0],
        end_time=train_span[1],
        freq=freq,
    )
    if probe is None or probe.empty:
        raise RuntimeError("qlib returned no feature data — dump failed?")

    data_handler_config = {
        "start_time": train_span[0],
        "end_time": test_span[1],
        "instruments": market,
        "freq": freq,
        "infer_processors": [],
        # NOTE: do NOT use CSRankNorm here — cross-sectional rank over only 3
        # assets collapses features/label to ~3 discrete levels and the model
        # degenerates to a constant predictor. LightGBM needs no feature scaling;
        # a plain train-fitted z-score keeps the label continuous.
        "learn_processors": [
            "DropnaLabel",
            {"class": "ZScoreNorm", "kwargs": {"fields_group": "label"}},
        ],
        "fit_start_time": train_span[0],
        "fit_end_time": train_span[1],
    }
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": train_span,
                "valid": valid_span,
                "test": test_span,
            },
        },
    }
    model_config = {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 4,
        },
    }

    dataset = init_instance_by_config(dataset_config)
    model = init_instance_by_config(model_config)
    model.fit(dataset)
    pred = model.predict(dataset)
    # pred: MultiIndex (datetime, instrument) → Series; normalise to frame
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    elif "score" not in getattr(pred, "columns", []):
        pred = pred.rename(columns={pred.columns[0]: "score"})
    return pred, model
