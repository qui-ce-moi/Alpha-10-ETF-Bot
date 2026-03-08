<p align="center">
  <h1 align="center">📈 Hierarchical MARL — Sector ETF Trading Bot</h1>
  <p align="center">
    A multi-agent reinforcement learning system that combines <strong>macroeconomic analysis</strong> with <strong>technical indicators</strong> to generate intelligent BUY / SELL / HOLD signals across 10 S&P 500 sector ETFs.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/RL-PPO-orange?logo=openai&logoColor=white" alt="PPO">
  <img src="https://img.shields.io/badge/broker-Interactive%20Brokers-red" alt="IBKR">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## 🧠 What Is This?

Most trading bots rely on price charts alone. **This bot is different.** It uses a **two-layer intelligence system**:

| Layer | What It Does | Data Source |
|---|---|---|
| **Macro Layer** | Reads the "health" of the economy (GDP, inflation, employment, yield curve) | [FRED API](https://fred.stlouisfed.org/) |
| **Technical Layer** | Reads price momentum and trend signals (SMA-50/200, RSI-14, 21-day volatility) | [Yahoo Finance](https://finance.yahoo.com/) |

A **PPO (Proximal Policy Optimization)** agent is trained *separately* for each sector. Then a **Strategic Agent Module (SAM)** acts as a portfolio manager — it polls every sector expert and produces a unified allocation signal. Think of it as a _board of directors_ where each member is a domain specialist.

---

## 🏗️ Architecture

```
                      ┌─────────────────────────────┐
                      │   FRED API (Macro Data)      │
                      │   GDP · CPI · Yield Curve    │
                      │   Non-Farm Payrolls · PCE     │
                      └──────────────┬────────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │      data_fetcher.py         │
                      │  Merge + Feature Engineering  │
                      │  SMA · RSI · Volatility       │
                      │  Min-Max Normalization         │
                      └──────────────┬────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
     ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
     │  PPO Expert  │      │  PPO Expert  │ ...  │  PPO Expert  │
     │    (XLK)     │      │    (XLF)     │      │    (XLB)     │
     └──────┬───────┘      └──────┬───────┘      └──────┬───────┘
            │                     │                     │
            └─────────────────────┼─────────────────────┘
                                  ▼
                      ┌─────────────────────────────┐
                      │   SAM (Strategic Agent)      │
                      │   Aggregates expert signals  │
                      │   → BUY / SELL / HOLD        │
                      └──────────────┬────────────────┘
                                     │
                      ┌──────────────┴────────────────┐
                      ▼                               ▼
             ┌──────────────┐               ┌──────────────┐
             │  Backtester  │               │   Execution  │
             │  (Simulated) │               │   (IBKR)     │
             └──────────────┘               └──────────────┘
```

---

## 🎯 Target Assets — 10 SPDR Sector ETFs

| Ticker | Sector | Ticker | Sector |
|--------|--------|--------|--------|
| `XLK` | Technology | `XLC` | Communication Services |
| `XLF` | Financials | `XLU` | Utilities |
| `XLV` | Health Care | `XLP` | Consumer Staples |
| `XLE` | Energy | `XLB` | Materials |
| `XLY` | Consumer Discretionary | `XLI` | Industrials |

---

## 📂 Project Structure

```text
etf_bot/
├── main.py             # Orchestrator — CLI entry point
├── config.py           # All settings, API keys, hyperparameters
├── data_fetcher.py     # FRED macro data + Yahoo Finance ETF data + feature engineering
├── environment.py      # Gymnasium RL environment (SectorTradingEnv)
├── model_trainer.py    # PPO training loop + SAM (Strategic Agent Module)
├── backtester.py       # Historical simulation vs S&P 500 benchmark
├── execution.py        # Live order execution via Interactive Brokers (IBKR)
├── requirements.txt    # Python dependencies
├── .env.example        # Template for secrets (FRED key, IBKR config)
├── .gitignore          # Ignored files (models, data, secrets)
└── LICENSE             # MIT License
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/qui-ce-moi/etf_bot.git
cd etf_bot

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in your credentials:

| Variable | Description | Where to Get It |
|---|---|---|
| `FRED_API_KEY` | Federal Reserve Economic Data API key | [fred.stlouisfed.org/docs/api](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `IB_HOST` | IBKR TWS/Gateway host (default `127.0.0.1`) | Your local machine |
| `IB_PORT` | IBKR TWS/Gateway port (default `7497`) | TWS → API Settings |
| `IB_CLIENT_ID` | IBKR client ID (default `1`) | TWS → API Settings |

> [!WARNING]
> **Never commit your `.env` file to version control.** It contains sensitive API keys. The `.gitignore` is already configured to exclude it.

### 3. Run the Bot

```bash
# Full pipeline: fetch data → train models → backtest (default)
python main.py

# Train sector expert models only
python main.py --train

# Run backtest only (requires pre-trained models)
python main.py --backtest

# Generate live signals + execute via IBKR
python main.py --live

# Explicit full pipeline
python main.py --all
```

---

## 🔬 How It Works — Step by Step

### Step 1 · Macro Data Ingestion
`data_fetcher.py` pulls 8 macroeconomic time series from FRED (starting from 2010):

| Series | FRED Code | Description |
|---|---|---|
| GDP Growth | `A191RL1Q225SBEA` | Real GDP quarterly growth rate |
| Core Inflation | `PCEPILFE` | Core PCE price index |
| Yield Curve | `T10Y2Y` | 10-Year minus 2-Year Treasury spread |
| Non-Farm Payrolls | `PAYEMS` | Total non-farm employment |
| Real GDP | `GDP` | Gross Domestic Product (nominal) |
| Core PCE | `PCE` | Personal Consumption Expenditures |
| 10Y Treasury Yield | `DGS10` | 10-Year Treasury constant maturity rate |
| CPI (Inflation) | `CPIAUCSL` | Consumer Price Index for All Urban Consumers |

### Step 2 · Technical Feature Engineering
For each of the 10 sector ETFs, the system computes:
- **SMA-50 & SMA-200** — Short and long-term moving averages for trend detection
- **RSI-14** — Relative Strength Index to identify overbought/oversold conditions
- **21-Day Volatility** — Rolling standard deviation of daily returns

### Step 3 · State Space Normalization
All features are Min-Max scaled to \[0, 1\] so the RL agent treats every feature equally. A placeholder sentiment score is also injected (to be replaced with FinBERT in the future).

### Step 4 · Model Training
Each sector gets its own **PPO agent** trained inside a custom `Gymnasium` environment (`SectorTradingEnv`). The agent observes the combined macro + technical state and learns to choose:
- `0` → **SELL** (short exposure)
- `1` → **HOLD** (no action)
- `2` → **BUY** (long exposure)

Reward = daily return × direction of the agent's bet.

### Step 5 · Strategic Allocation (SAM)
The `StrategicAgentModule` loads all trained sector experts, feeds each one the latest market data, and collects their signals into a unified portfolio allocation.

### Step 6 · Backtest or Live Execution
- **Backtest**: Simulates the strategy over the last 100 trading days against an S&P 500 (SPY) benchmark. Generates a performance chart (`backtest_result.png`) with alpha calculation.
- **Live**: Connects to IBKR and (optionally) submits real market orders.

---

## 🔒 Safety & Risk

| Safeguard | Details |
|---|---|
| **Live trading is disabled by default** | Order execution lines in `execution.py` are commented out behind a safety lock. You must manually uncomment them to enable real orders. |
| **API keys are never hardcoded** | All secrets are loaded from `.env` via `python-dotenv`. |
| **Graceful IBKR fallback** | If the broker connection fails, signals are still printed to the console — no crash, no accidental orders. |

> [!CAUTION]
> **This bot is for educational and research purposes.** Past backtest performance does not guarantee future results. Always paper-trade extensively before risking real capital. The authors assume no liability for financial losses.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Reinforcement Learning** | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO) · [Gymnasium](https://gymnasium.farama.org/) |
| **Deep Learning Backend** | [PyTorch](https://pytorch.org/) |
| **Market Data** | [yfinance](https://github.com/ranaroussi/yfinance) |
| **Macro Data** | [fredapi](https://github.com/mortada/fredapi) |
| **Live Execution** | [ib_insync](https://ib-insync.readthedocs.io/) (Interactive Brokers) |
| **Data Processing** | pandas · NumPy |
| **Visualization** | Matplotlib |
| **Config Management** | python-dotenv |

---

## 🗺️ Roadmap

- [ ] Replace placeholder sentiment with **FinBERT** NLP model for real market sentiment analysis
- [ ] Add **Sharpe Ratio**, **Max Drawdown**, and **Sortino Ratio** to backtest metrics
- [ ] Implement a **meta-learner** on top of SAM for dynamic sector weighting
- [ ] Add **position sizing** logic (Kelly Criterion or risk-parity)
- [ ] Support **paper trading** mode via IBKR Paper account
- [ ] Add **scheduler** for daily automated runs (cron / Windows Task Scheduler)
- [ ] Dashboard UI with **Streamlit** for real-time signal monitoring

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <sub>Built with ☕ and curiosity by <a href="https://github.com/qui-ce-moi">qui-ce-moi</a></sub>
</p>
