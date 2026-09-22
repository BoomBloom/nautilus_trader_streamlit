# app_test_weights.py — E2E: Portfolio mode + skfolio weights toggle
from __future__ import annotations

import json
import os
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # weights path is relative: notebooks/weights_skfolio.json

from streamlit.testing.v1 import AppTest


def main() -> int:
    at = AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=60)
    at.run()
    if at.exception:
        print("EXCEPTION initial:", [e.value for e in at.exception])
        return 1
    print("[1] initial run OK")

    # pin the strategy (discover order is env-dependent otherwise; ScoreTarget
    # stays flat without scores and would make finals identical across modes)
    at.selectbox(key="strat_name").set_value("MeanReversionStrategy")
    at.run()
    if at.exception:
        print("EXCEPTION strategy set:", [e.value for e in at.exception])
        return 1

    # switch to Portfolio mode
    at.radio(key="bt_mode").set_value("Portfolio (multi-asset)")
    at.run()
    if at.exception:
        print("EXCEPTION mode switch:", [e.value for e in at.exception])
        return 1
    print("[2] portfolio sidebar OK")

    # allocation radio exists, default Equal split
    radios = {w.key: w.value for w in at.radio}
    print("    radios:", radios)
    assert radios.get("pf_weights_src") == "Equal split", radios

    # switch to skfolio weights + pick HRP (distinct from 1/N)
    at.radio(key="pf_weights_src").set_value("skfolio weights (JSON)")
    at.run()
    if at.exception:
        print("EXCEPTION skfolio select:", [e.value for e in at.exception])
        return 1
    boxes = {w.key: w.value for w in at.selectbox}
    print("    selectboxes:", boxes)
    assert "pf_weights_model" in boxes, "model selectbox missing"
    at.selectbox(key="pf_weights_model").set_value("HRP")
    at.run()
    if at.exception:
        print("EXCEPTION model set:", [e.value for e in at.exception])
        return 1

    # preview caption rendered? (st.caption → at.caption in AppTest)
    caps = " || ".join(str(c.value) for c in at.caption)
    assert "JSON weights" in caps, caps[:400]
    print("[3] skfolio widgets + preview caption OK")

    # run the portfolio backtest (3 real legs)
    at.button(key="run_bt").click()
    at.run(timeout=180)
    if at.exception:
        print("EXCEPTION run:", [e.value for e in at.exception])
        return 1
    pf = at.session_state["pf_result"]
    assert pf is not None, "pf_result is None"
    assert pf["weights_source"] == "custom", pf["weights_source"]
    allocs = pf["allocations"]
    hrp = json.loads((ROOT / "notebooks" / "weights_skfolio.json").read_text())["HRP"]
    for sym in ("BTCUSD", "ETHUSD", "SOLUSD"):
        want = 10_000.0 * hrp[sym]
        assert abs(allocs[sym] - want) < 0.01, (sym, allocs[sym], want)
    assert pf["n_assets"] == 3, pf
    print(f"[4] custom HRP run OK — allocs={allocs} metrics={pf['metrics']}")

    # dashboard shows custom caption
    cap2 = " || ".join(str(c.value) for c in at.caption)
    assert "Custom weights" in cap2, cap2[:400]
    print("[5] dashboard custom-weights caption OK")

    # regression: back to Equal split
    at.radio(key="pf_weights_src").set_value("Equal split")
    at.run()
    at.button(key="run_bt").click()
    at.run(timeout=180)
    if at.exception:
        print("EXCEPTION equal run:", [e.value for e in at.exception])
        return 1
    pf2 = at.session_state["pf_result"]
    assert pf2["weights_source"] == "equal", pf2["weights_source"]
    assert all(abs(v - 10_000 / 3) < 1e-6 for v in pf2["allocations"].values())
    assert pf2["n_assets"] == 3
    # equal vs custom must now differ (trade_size scales with capital)
    assert pf["metrics"]["total_final"] != pf2["metrics"]["total_final"], (
        pf["metrics"]["total_final"], pf2["metrics"]["total_final"])
    cap3 = " || ".join(str(c.value) for c in at.caption)
    assert "equal split" in cap3, cap3[:400]
    print(f"[6] equal-split regression OK — custom_final={pf['metrics']['total_final']} equal_final={pf2['metrics']['total_final']}")

    # regression: Qlib scores (JSON) path (v0.5.0)
    at.radio(key="pf_weights_src").set_value("Qlib scores (JSON)")
    at.run()
    if at.exception:
        print("EXCEPTION qlib select:", [e.value for e in at.exception])
        return 1
    caps4 = " || ".join(str(c.value) for c in at.caption)
    assert "Qlib scores" in caps4, caps4[:400]
    print("[7] qlib scores sidebar + preview caption OK")
    at.button(key="run_bt").click()
    at.run(timeout=180)
    if at.exception:
        print("EXCEPTION qlib run:", [e.value for e in at.exception])
        return 1
    pf3 = at.session_state["pf_result"]
    assert pf3["weights_source"] == "custom", pf3["weights_source"]
    from modules.ml_examples import scores_to_weights
    _sp = json.loads((ROOT / "notebooks" / "scores_qlib.json").read_text())
    _sw = scores_to_weights(_sp.get("scores", _sp))
    for sym in ("BTCUSD", "ETHUSD", "SOLUSD"):
        want = 10_000.0 * _sw.get(sym, 0.0)
        got = pf3["allocations"].get(sym, 0.0)
        if _sw.get(sym, 0.0) <= 0.0 or want < 1.0:
            assert sym in pf3.get("failed", {}) or abs(got) < 1.0, (sym, got, want)
        else:
            assert abs(got - want) < 0.01, (sym, got, want)
    print(f"[8] qlib scores run OK — allocs={pf3['allocations']} failed={pf3.get('failed', {})}")

    print("\nAPP TEST (weights) PASSED ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
