# test_score_strategy.py — v0.6.0: score-driven position taking
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

from strategies.score_target import score_for_symbol, symbol_scores
from modules.data_connector import DataConnector
from modules.portfolio_runner import run_portfolio_backtest
from modules.strategy_loader import discover_strategies
from modules.dashboard_actor import DashboardPublisher

SYMS = ["BTCUSD", "ETHUSD", "SOLUSD"]
START = "2025-03-01T00:00:00Z"
END = "2025-03-08T00:00:00Z"


def load_assets(symbols):
    import pandas as pd

    conn = DataConnector(csv_dir=ROOT)
    start_dt = pd.to_datetime(START, utc=True)
    end_dt = pd.to_datetime(END, utc=True)
    return [
        (sym, conn.load("CSV", conn.get_csv_path("BINANCE", sym, "15"),
                        start=start_dt, end=end_dt))
        for sym in symbols
    ]


def main() -> int:
    # ── [1] symbol_scores: wrapped payload + flat payload ──
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "s.json"
        p.write_text(json.dumps(
            {"asof": "2025-04-08 00:00:00",
             "scores": {"BTCUSD": 0.5, "ethusd": -0.1, "SOLUSD": 0.0}}))
        sc = symbol_scores(p)
        p.write_text(json.dumps({"BTCUSD": 0.25}))
        sc2 = symbol_scores(p)
    assert sc == {"BTCUSD": 0.5, "ETHUSD": -0.1, "SOLUSD": 0.0}, sc
    assert sc2 == {"BTCUSD": 0.25}, sc2
    assert score_for_symbol(sc, "btcusd") == 0.5          # case-insensitive
    assert score_for_symbol(sc, "DOGEUSD") == 0.0         # unknown → flat
    print("[1] symbol_scores + score_for_symbol OK")

    # ── [2] strategy class discovered ──
    infos = discover_strategies()
    assert "ScoreTargetStrategy" in infos, sorted(infos)
    info = infos["ScoreTargetStrategy"]
    print("[2] ScoreTargetStrategy discovered OK")

    # ── [3] portfolio run: BTC buys, ETH/SOL stay flat (score injection) ──
    assets = load_assets(SYMS)
    assert all(len(d) > 0 for _, d in assets), "no data"
    res = run_portfolio_backtest(
        info.strategy_cls, info.cfg_cls,
        {},
        assets,
        actor_cls=DashboardPublisher,
        start_balance=10_000.0,
        scores={"BTCUSD": 0.5, "ETHUSD": -0.1, "SOLUSD": 0.0},
    )
    assert res["n_assets"] == 3, res["n_assets"]          # all legs run
    assert res["weights_source"] == "equal", res["weights_source"]
    contrib = res["contributions"]
    pnl = dict(zip(contrib["asset"], contrib["pnl"]))
    assert abs(pnl["BTCUSD"]) > 1e-9, pnl                 # traded
    assert abs(pnl["ETHUSD"]) == 0.0, pnl                 # flat (score ≤ 0)
    assert abs(pnl["SOLUSD"]) == 0.0, pnl                 # flat (score == 0)
    assert res["metrics"]["num_trades"] > 0, res["metrics"]
    print(f"[3] score injection OK — pnl={pnl} trades={res['metrics']['num_trades']}")

    # ── [3b] file fallback (no score param) keyed by instrument symbol ──
    # The backtest factory always builds a BTCUSDT instrument, so a file
    # keyed BTCUSDT matches every leg.
    with tempfile.TemporaryDirectory() as td:
        spath = str(pathlib.Path(td) / "scores.json")
        pathlib.Path(spath).write_text(json.dumps({"BTCUSDT": 0.5}))
        res_fb = run_portfolio_backtest(
            info.strategy_cls, info.cfg_cls,
            {"scores_path": spath},
            assets,
            actor_cls=DashboardPublisher,
            start_balance=10_000.0,
        )
    assert res_fb["metrics"]["num_trades"] > 0, res_fb["metrics"]
    print(f"[3b] file fallback OK — trades={res_fb['metrics']['num_trades']}")

    # ── [4] min_score raises the bar → BTC also flat ──
    res2 = run_portfolio_backtest(
        info.strategy_cls, info.cfg_cls,
        {"min_score": 0.99},                               # 0.99 > 0.5
        assets,
        actor_cls=DashboardPublisher,
        start_balance=10_000.0,
        scores={"BTCUSD": 0.5, "ETHUSD": -0.1, "SOLUSD": 0.0},
    )
    assert res2["metrics"]["num_trades"] == 0, res2["metrics"]
    print("[4] min_score gate OK — all flat, 0 trades")

    # ── [5] committed default artifact loads (repo scores_qlib.json) ──
    default_path = ROOT / "notebooks" / "scores_qlib.json"
    assert default_path.exists(), default_path
    dsc = symbol_scores(default_path)
    assert set(dsc) == {"BTCUSD", "ETHUSD", "SOLUSD"}, dsc
    print(f"[5] default scores artifact OK — {dsc}")

    print("\nSCORE STRATEGY VERIFY PASSED ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
