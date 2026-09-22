# NautilusTrader Streamlit

![DEMO](https://github.com/Sergey-1221/nautilus_trader_streamlit/raw/main/image/demo.png)
[DEMO](https://nautilustrader.streamlit.app/)

## 🎯 Project Goal

**Create a simple and convenient Streamlit-based extension for the NautilusTrader platform, enabling developers to quickly visualize, analyze, and convincingly demonstrate the performance of trading strategies to investors and teams — without wasting time on complex frontend development.**

NautilusTrader is a rapidly evolving open-source platform for algorithmic trading. This project emerged from the need to simplify and accelerate the visualization of strategy data created with it.

---

## 🌟 Why is it convenient?

* ✅ **Minimal setup**: Quickly visualize your locally run backtests by simply connecting strategy outputs (CSV/Parquet).
* ✅ **Fully interactive**: Instantly get dynamic charts and metrics.
* ✅ **No frontend code**: Work entirely in Python, no harder than Jupyter Notebook.
* ✅ **Great for presentations**: Beautiful and clear visualizations for investors and team members.
* ✅ **Time-saving**: Quickly identify bugs and strategy issues.
* ✅ **Improved visuals**: Themed widgets, icons, and styled data tables.
* ✅ **Price chart enhancements**: Trade markers, optional volume bars, and cumulative PnL overlay rendered with **TradingView Lightweight Charts** (chosen for smooth rendering of thousands of bars).
* ✅ **Detailed trade tooltips**: Hover markers to see entry, exit, and PnL info.
* ✅ **Chart options**: Choose line or candlesticks and overlay SMA/EMA lines.
* ✅ **Structured metrics**: Grouped performance KPIs with a progress bar showing edge over buy‑and‑hold.

---

## 🛠️ Quick Start

Install the dependencies and run the app:

```bash
pip install -r requirements.txt
streamlit run app/main.py
```
--- 

## 📌 Roadmap

| Version    | Status         | Features                                                                                   |
| ---------- | -------------- | ------------------------------------------------------------------------------------------ |
| **v0.1.5** | ✅ Done         | Basic single-asset strategy visualization.                                                 |
| **v0.2.0** | 🚧 In progress | Single-asset dashboard is fully usable and intuitive, featuring clear equity curves, drawdown analysis, trade markers, and essential risk metrics (VaR, Sharpe ratio).                |
| **v0.3.0** | ✅ Done         | Multi-asset portfolio backtests: summary equity, asset contribution analysis.              |
| **v0.4.0** | ✅ Done         | Integration of ML libraries (Qlib, skfolio) demonstrating example ML strategies based on Jupyter Notebook, showcasing integration methods and standard ML algorithms. |

> ⚠️ ClickHouse integration is provided only as an example and is not guaranteed to be stable.

---

## 🤖 ML Notebooks (v0.4.0)

Two executed example notebooks live in [`notebooks/`](./notebooks), sharing the same
`DataConnector` CSV path the Streamlit app uses (helpers in `modules/ml_examples.py`):

| Notebook | Library | What it demonstrates |
| -------- | ------- | -------------------- |
| [`01_skfolio_portfolio_optimization.ipynb`](./notebooks/01_skfolio_portfolio_optimization.ipynb) | **skfolio** | EDA → chronological split → `EqualWeighted` / `MeanRisk` / `HierarchicalRiskParity` → out-of-sample metrics, equity curves, weights bar charts → exports `notebooks/weights_skfolio.json` |
| [`02_qlib_ml_strategy.ipynb`](./notebooks/02_qlib_ml_strategy.ipynb) | **pyqlib (Qlib)** | CSV → qlib `.bin` dump → `qlib.init` → **Alpha158** features + **LightGBM** (leak-free train/valid/test segments) → cross-sectional IC & score-quantile evaluation |

```bash
pip install -r requirements.txt -r requirements-ml.txt
jupyter notebook notebooks/

# Headless re-execution (all outputs verified):
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=<your-kernel> \
  notebooks/01_skfolio_portfolio_optimization.ipynb \
  notebooks/02_qlib_ml_strategy.ipynb
```

> 📝 The sample CSVs are synthetic GBM series — IC ≈ 0 is the *correct* result;
> the notebooks showcase a reproducible pipeline. Point `load_ohlcv_frames` at
> real market data to chase actual signal. Install **`pyqlib`**, not the
> unrelated squatter package `qlib`, on PyPI.
---

## 🚫 Out-of-scope

* Live trading dashboards and order management functionality.
* Stable ClickHouse integration or support for remote databases (provided as experimental examples only).
* Complex trading strategies
---

## 🤝 How to Contribute

* ⭐ **Star the repository**.
* 🐞 **Open Issues** if you find bugs.
* 🚀 **Submit Pull Requests** if you’d like to add new features.

## License
This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.
