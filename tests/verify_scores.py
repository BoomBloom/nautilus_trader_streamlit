# verify_scores.py — v0.5.0: Qlib scores → capital weights
from __future__ import annotations

import json
import os
import sys
import pathlib
import tempfile
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from modules.ml_examples import scores_to_json, scores_to_weights
from modules.data_connector import DataConnector
from modules.portfolio_runner import run_portfolio_backtest
from modules.strategy_loader import discover_strategies
from modules.dashboard_actor import DashboardPublisher


def main() -> int:
    # ── [1] scores_to_weights unit checks ──
    w = scores_to_weights({"BTCUSD": 0.02, "ETHUSD": -0.01, "SOLUSD": 0.06})
    assert set(w) == {"BTCUSD", "ETHUSD", "SOLUSD"}
    assert w["ETHUSD"] == 0.0, w
    assert abs(sum(w.values()) - 1.0) < 1e-12, w
    assert abs(w["BTCUSD"] - 0.25) < 1e-12 and abs(w["SOLUSD"] - 0.75) < 1e-12, w
    eq = scores_to_weights({"A": -1.0, "B": -2.0})
    assert abs(eq["A"] - 0.5) < 1e-12 and abs(eq["B"] - 0.5) < 1e-12, eq
    assert scores_to_weights({}) == {}
    assert scores_to_weights({"A": float("nan")}) == {} or scores_to_weights({"A": float("nan")})["A"] > 0
    print("[1] scores_to_weights OK —", w, eq)

    # ── [2] scores_to_json roundtrip (synthetic MultiIndex pred) ──
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2025-03-01", periods=5, freq="15min"), ["BTCUSD", "ETHUSD", "SOLUSD"]],
        names=["datetime", "instrument"],
    )
    rng = np.random.default_rng(7)
    pred = pd.DataFrame({"score": rng.normal(0, 1, len(idx))}, index=idx)
    with tempfile.TemporaryDirectory() as td:
        out = scores_to_json(pred, pathlib.Path(td) / "scores.json", tail=2)
        payload = json.loads(out.read_text())
    assert set(payload) == {"asof", "scores"}, payload
    assert set(payload["scores"]) == {"BTCUSD", "ETHUSD", "SOLUSD"}
    # tail=2 → mean of last 2 timestamps per instrument
    want = pred["score"].groupby(level="instrument").tail(2).groupby(level="instrument").mean()
    for k, v in payload["scores"].items():
        assert abs(v - float(want[k])) < 1e-9, (k, v, want[k])
    assert payload["asof"] == str(pred.index.get_level_values("datetime").max())
    print("[2] scores_to_json OK — asof", payload["asof"])

    # ── [3] committed artifact exists & converts ──
    art = ROOT / "notebooks" / "scores_qlib.json"
    assert art.exists(), f"missing {art} — run notebook 02 first"
    art_payload = json.loads(art.read_text())
    art_scores = art_payload.get("scores", art_payload)
    art_w = scores_to_weights({k: float(v) for k, v in art_scores.items()})
    assert abs(sum(art_w.values()) - 1.0) < 1e-9, art_w
    print("[3] scores_qlib.json OK —", art_payload.get("asof"), art_w)

    # ── [4] runner with score-derived weights ──
    conn = DataConnector(csv_dir=ROOT)
    assets = [
        (s, conn.load("CSV", conn.get_csv_path("BINANCE", s, "15min")))
        for s in ["BTCUSD", "ETHUSD", "SOLUSD"]
    ]
    info = discover_strategies()["BuyAndHoldStrategy"]
    eq = run_portfolio_backtest(
        info.strategy_cls, info.cfg_cls, {}, assets,
        actor_cls=DashboardPublisher, start_balance=10_000.0,
    )
    cw = run_portfolio_backtest(
        info.strategy_cls, info.cfg_cls, {}, assets,
        actor_cls=DashboardPublisher, start_balance=10_000.0,
        weights=art_w,
    )
    assert eq["weights_source"] == "equal", eq["weights_source"]
    assert cw["weights_source"] == "custom", cw["weights_source"]
    for sym, wv in art_w.items():
        want = 10_000.0 * wv
        got = cw["allocations"].get(sym, 0.0)
        if wv <= 0.0 or want < 1.0:
            # zero or sub-$1 legs must be skipped by the runner
            assert sym in cw.get("failed", {}) or abs(got) < 1.0, (sym, got, want)
        else:
            assert abs(got - want) < 0.01, (sym, got, want)
    print(f"[4] runner score-weights OK — allocs={cw['allocations']} failed={cw.get('failed', {})}")

    print("\nSCORES VERIFY PASSED ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
