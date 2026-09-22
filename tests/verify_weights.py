# verify_weights.py — v0.4.x custom-weights portfolio runner
from __future__ import annotations

import json
import os
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

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
    out = []
    for sym in symbols:
        path = conn.get_csv_path("BINANCE", sym, "15")
        df = conn.load("CSV", path, start=start_dt, end=end_dt)
        out.append((sym, df))
    return out


def main() -> int:
    info = discover_strategies()["BuyAndHoldStrategy"]
    assets = load_assets(SYMS)
    for s, d in assets:
        print(f"  {s}: {len(d)} rows")
    assert all(len(d) > 0 for _, d in assets), "no data"

    # ── 1. equal split (v0.3.0 regression) ─────────────────────────────
    eq = run_portfolio_backtest(
        info.strategy_cls, info.cfg_cls, {}, assets,
        actor_cls=DashboardPublisher, start_balance=10_000.0,
    )
    assert eq["weights_source"] == "equal", eq["weights_source"]
    allocs = eq["allocations"]
    assert all(abs(v - 10_000 / 3) < 1e-6 for v in allocs.values()), allocs
    first = float(eq["equity_df"]["equity"].iloc[0])
    assert abs(first - 10_000) <= 0.02 * 10_000, f"first equity {first}"
    print(f"[1] equal split OK — first_equity={first:.2f} allocs={allocs}")

    # ── 2. custom weights 50/30/20 ─────────────────────────────────────
    w = {"BTCUSD": 0.5, "ETHUSD": 0.3, "SOLUSD": 0.2}
    cw = run_portfolio_backtest(
        info.strategy_cls, info.cfg_cls, {}, assets,
        actor_cls=DashboardPublisher, start_balance=10_000.0, weights=w,
    )
    assert cw["weights_source"] == "custom", cw["weights_source"]
    ca = cw["allocations"]
    assert abs(ca["BTCUSD"] - 5_000) < 1e-6, ca
    assert abs(ca["ETHUSD"] - 3_000) < 1e-6, ca
    assert abs(ca["SOLUSD"] - 2_000) < 1e-6, ca
    first2 = float(cw["equity_df"]["equity"].iloc[0])
    assert abs(first2 - 10_000) <= 0.05 * 10_000, f"first equity {first2}"
    contrib = cw["contributions"]
    starts = dict(zip(contrib["asset"], contrib["start_equity"]))
    for sym, want in ca.items():
        got = starts.get(sym)
        # start_equity is rounded in contributions; allow $1
        assert got is not None and abs(got - want) <= 1.0, (sym, got, want)
    total_pnl = float(contrib["pnl"].sum())
    assert abs(total_pnl - cw["metrics"]["total_profit"]) <= 1.0, (
        total_pnl,
        cw["metrics"]["total_profit"],
    )
    print(f"[2] custom 50/30/20 OK — first={first2:.2f} pnl={total_pnl:.2f}")

    # ── 3. subset renormalize (BTC+ETH only from 0.5/0.3/0.2) ─────────
    sub = [(s, d) for s, d in assets if s in ("BTCUSD", "ETHUSD")]
    rs = run_portfolio_backtest(
        info.strategy_cls, info.cfg_cls, {}, sub,
        actor_cls=DashboardPublisher, start_balance=10_000.0, weights=w,
    )
    sa = rs["allocations"]
    # 0.5/(0.5+0.3)=0.625 → 6250 ; 0.375 → 3750
    assert abs(sa["BTCUSD"] - 6_250) < 1e-6, sa
    assert abs(sa["ETHUSD"] - 3_750) < 1e-6, sa
    assert rs["weights_source"] == "custom"
    print(f"[3] subset renormalize OK — {sa}")

    # ── 4. zero-weight legs skipped ────────────────────────────────────
    only_btc = {"BTCUSD": 1.0, "ETHUSD": 0.0, "SOLUSD": 0.0}
    zw = run_portfolio_backtest(
        info.strategy_cls, info.cfg_cls, {}, assets,
        actor_cls=DashboardPublisher, start_balance=10_000.0, weights=only_btc,
    )
    assert zw["n_assets"] == 1, zw["n_assets"]
    assert "ETHUSD" in zw["failed"] and "SOLUSD" in zw["failed"], zw["failed"]
    assert "too small" in zw["failed"]["ETHUSD"], zw["failed"]
    assert abs(zw["allocations"]["BTCUSD"] - 10_000) < 1e-6
    print(f"[4] zero-weight skip OK — failed={zw['failed']}")

    # ── 5. real notebook JSON loads & runs ─────────────────────────────
    wfile = ROOT / "notebooks" / "weights_skfolio.json"
    payload = json.loads(wfile.read_text())
    hrp = payload["HRP"]
    jr = run_portfolio_backtest(
        info.strategy_cls, info.cfg_cls, {}, assets,
        actor_cls=DashboardPublisher, start_balance=10_000.0, weights=hrp,
    )
    assert jr["weights_source"] == "custom"
    ssum = sum(jr["allocations"].values())
    assert abs(ssum - 10_000) < 1e-6, ssum
    print(f"[5] HRP json OK — allocs={jr['allocations']}")

    # ── 6. no matching weights → equal fallback ────────────────────────
    fb = run_portfolio_backtest(
        info.strategy_cls, info.cfg_cls, {}, assets,
        actor_cls=DashboardPublisher, start_balance=10_000.0,
        weights={"DOGEUSD": 1.0},
    )
    assert fb["weights_source"] == "equal", fb["weights_source"]
    print("[6] unmatched-weights fallback to equal OK")

    print("\nWEIGHTS VERIFY PASSED ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
