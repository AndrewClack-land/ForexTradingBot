# MT5 Native Bridge

This helper replaces the old ZeroMQ/EA flow and pulls candles straight from the
MetaTrader5 terminal via the official `MetaTrader5` Python package.

## Requirements

1. MT5 terminal must be running on the same machine and logged into your FxPro
   (demo or live) account.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (this now includes the `MetaTrader5` package)
3. Provide credentials via environment variables or CLI flags:
   - `MT5_LOGIN`
   - `MT5_PASSWORD`
   - `MT5_SERVER` (e.g. `FxPro-MT5 Demo`)

## Running

```bash
cd E:\Forex_TradingBot\mt5_bridge
python mt5_native_bridge.py \
  --server "FxPro-MT5 Demo" \
  --login 591216595 \
  --password "YOUR_PASSWORD" \
  --symbols "EURUSD,GBPUSD,USDCAD,GOLD:XAUUSD" \
  --timeframes "15m,1h,4h,1d" \
  --lookback-days 15 \
  --interval 60
```

- Symbols support mapping `MT5_SYMBOL:BOT_SYMBOL`. Example `GOLD:XAUUSD`
  keeps MT5 instrument `GOLD` but writes cache as `XAUUSD` for the bot.
- Timeframes can be any combination of `15m, 1h, 4h, 1d`.
- `--lookback-days` defines how many calendar days of history are written each
  refresh. The file always contains the full window; no incremental merge is
  required.
- `--interval` (seconds) controls how often the sync runs. Use `--once` to
  generate caches a single time and exit.

## Output

JSON files are written to `ai_data/mt5_cache/` as `<SYMBOL>_<tf>.json`, matching
what `core.data_feed.DataFeed` expects when `DATA_SOURCE=MT5` (or `AUTO` with an
existing cache).

Example file names:

- `EURUSD_15m.json`
- `GBPUSD_1h.json`
- `USDCAD_4h.json`
- `XAUUSD_1d.json`

Each entry contains `symbol`, `tf`, `timestamp` (UTC ISO8601) and OHLCV values.
