# MT5 Bridge

Components to stream FXPro (MT5) quotes into the bot.

## Layout

```
mt5_bridge/
├── FXProBridge.mq5   # Expert Advisor for MetaTrader 5 (publishes candles via ZeroMQ)
├── mt5_bridge.py     # Python listener that writes JSON caches for the bot
└── README.md         # This guide
```

## Prerequisites

- MetaTrader 5 terminal logged into FXPro account.
- Ability to add a custom Expert Advisor (Algo Trading enabled).
- Python environment for the bot (`pip install -r requirements.txt` will now pull `pyzmq`).

## Installation Steps

1. **ZeroMQ runtime for MT5**
   - Copy a 64-bit `libzmq.dll` (ZeroMQ runtime) into `MQL5/Libraries/`.
   - If you don’t have one, download the latest ZeroMQ Windows build (e.g. from https://github.com/zeromq/libzmq/releases) and extract `libzmq.dll`.

2. **Copy the EA**
   - Put `FXProBridge.mq5` into `MQL5/Experts/` (open MT5 → `File → Open Data Folder`).
   - Compile it in MetaEditor.

3. **Attach to a chart**
   - Open MT5, enable “Algo Trading”.
   - For each required symbol ensure the symbol is visible (Market Watch → Show All if needed).
   - Drag `FXProBridge` onto any chart (one instance can handle multiple symbols/TFs).
   - Configure inputs:
     - `Symbols`: e.g. `EURUSD,GBPUSD,USDCHF,USDCAD,XAUUSD`
     - `Timeframes`: `H4,H1,M15,M5`
     - `Endpoint`: `tcp://127.0.0.1:7777`
   - Click OK; the EA starts publishing JSON candles via ZeroMQ PUB socket.

4. **Run the Python bridge**
   ```bash
   cd /path/to/Forex_TradingBot
   python mt5_bridge/mt5_bridge.py --host 127.0.0.1 --port 7777 --limit 500
   ```
   - The script subscribes to the EA and writes JSON files to `ai_data/mt5_cache/` (one per symbol+TF).
   - Logs go to stdout and `ai_data/mt5_bridge.log`.

5. **Switch the bot to MT5 data**
   - In `config.py`, set `DATA_SOURCE = "MT5"` (value is read from env var `DATA_SOURCE` if preferred).
   - Restart the bot (`python main.py`), it will use the cached candles instead of TradingView when the flag is set.

## Operational Notes

- EA resends the latest `InpBars` candles each timer tick; the Python bridge deduplicates and keeps the last `limit` entries per instrument/TF.
- Both EA and Python listener must run while the bot is live; if either stops, data refresh pauses.
- To troubleshoot, check MT5 Experts log and `ai_data/mt5_bridge.log`.

Further adjustments (e.g., incremental updates only, command channel, risk of resending too often) can be added later if needed.
