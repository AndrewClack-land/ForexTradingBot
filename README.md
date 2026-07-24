# ForexTradingBot

Автономный торговый бот для MetaTrader 5 (FxPro): сам находит сетапы по SMC/ICT-логике,
исполняет сделки с разбивкой на частичные тейки и публикует сигналы в Telegram-канал.
Работает 24/5 на Linux VPS (терминал MT5 под Wine), торгует GOLD, EURUSD, GBPUSD, USDCAD.

## На чём основан

- **Python 3.11** + официальный пакет **MetaTrader5** — котировки и исполнение напрямую через терминал;
- **Smart Money Concepts / ICT**: premium/discount, ордерблоки, rejection-блоки,
  FVG, liquidity sweep (turtle soup), фракталы Вильямса;
- **python-telegram-bot** — сигналы и команды (`/status`, `/open`, `/report`, `/universe`);
- **SQLite + CSV/Parquet** — журнал сделок и статистика для AI-фильтра;
- сигналы считаются **только по закрытым свечам** (без перерисовки).

## Как бот принимает направление (HTF bias)

Направление определяется ансамблем независимых моделей (ensemble learning) по нескольким
таймфреймам (`core/strategy_narrative.py: calc_narrative`):

| Модель | Вклад |
| --- | --- |
| **Premium/Discount 1H** — последний закрытый 15M close внутри диапазона последней закрытой H1-свечи | +2 |
| **Ложный пробой фрактала 4H** — свеча проколола уровень, но закрылась внутри (разворот) | +2 |
| **Истинный пробой фрактала 15M** — закрытие за уровнем (продолжение) | +1 |
| **Ордерблок (OB)** на 1H — сторона последнего активного блока | +1 |
| **Rejection Block (RB)** на 1H — последний валидный неповреждённый блок | +1 |

Bias принимается при перевесе голосов ≥ `HTF_SCORE_MARGIN` (по умолчанию 2).
Строгость регулирует **FVG-режим 1H** (LuxAlgo Instantaneous Mitigation): против
направления режима порог ужесточается на +1. SMA/EMA в принятии решения не участвуют;
при смешанном счёте bias = NEUTRAL (входа нет).

Прежний 5-дневный dealing range удалён полностью. Его заменяет недирекционная
**оценка волатильности**: `R(t)` не добавляет баллы LONG/SHORT, но режим PANIC
блокирует вход, а Expected Move проверяет достижимость TP1.

## Фильтры перед входом

1. **Торговая сессия** — только London / New York;
2. **Пятница после 21:00 МСК** — новые входы блокируются, открытые позиции закрываются перед выходными;
3. **Частота входов**: кулдаун 60 мин после выбитого стопа (`POST_SL_COOLDOWN_MIN`),
   максимум 3 сетапа на символ в день (`MAX_SETUPS_PER_SYMBOL_PER_DAY`),
   одна и та же зона/уровень триггера не торгуется повторно в течение дня;
4. **Дневной стоп бота** — при просадке −3% от баланса на начало дня (`DAILY_MAX_LOSS_PCT`)
   новые входы блокируются до следующего дня;
5. **Режим волатильности** — вход блокируется при `R(t) >= VOL_REGIME_MAX_R`;
   TP1 должен быть не дальше `EM_TP_MAX_RATIO × EM1D`.

## Триггеры входа

Проверяются по очереди, первый сработавший даёт сигнал ENTER:

1. **15M Rejection Block** — отклонение от блока с длинной тенью по направлению bias;
2. **15M Turtle Soup** — ложный пробой локального экстремума (liquidity sweep) с возвратом;
3. **H1 Pivot Reclaim на 15M** — возврат цены за пивот-уровень 1H;
4. **Касание ордерблока 1H** — вход от валидного OB (порог касания в долях ATR).

## Риск и сопровождение

- **Стоп** — за фрактал Вильямса на 1H с ATR-буфером, риск ограничен коридором min/max ATR
  (min 0.75×ATR — микро-стопы не раздувают объём);
- **Жёсткий лимит SL-риска** — суммарный номинальный убыток до брокерского Stop Loss
  для одного сетапа не превышает 1% от зафиксированного стартового капитала.
  При `MT5_INITIAL_CAPITAL=0` баланс фиксируется один раз в
  `ai_data/risk_capital.json` и не меняется после сделок или перезапуска VPS;
  явное значение `MT5_INITIAL_CAPITAL` переопределяет этот снимок;
- **Сайзинг** — денежный убыток рассчитывается через MT5 `order_calc_profit` в
  валюте счёта после применения брокерской минимальной дистанции SL, с учётом
  комиссии и резерва ожидаемого проскальзывания. Объём всегда округляется вниз
  по `volume_step`; если минимальный лот временно не помещается в бюджет, бот
  сохраняет технический SL, рассчитывает риск-совместимую цену и ждёт её вместо
  открытия завышенного риска;
- потолок объёма `MT5_MAX_VOLUME` — по умолчанию 10 лотов на сетап;
- **4 тейк-профита** по уровням RR (1.0 / rr_min / 2.0 / 3.0);
- **Split-TP** — каждая цель открывается отдельной позицией со своим брокерским TP,
  но все ноги делят единый 1%-й риск-бюджет; перед каждой отправкой проверяется
  оставшийся суммарный риск;
- **Безубыток** — после TP1 стопы всех оставшихся ног переносятся на цену входа;
- **Position Adding** (`core/position_adding.py`, по умолчанию выключено,
  `POSITION_ADDING_ENABLED=1`) — поэтапный набор объёма в одну идею: вход 1 с
  риском 1%, каждое новое подтверждение того же направления добавляет ещё 1%, а
  совокупный риск идеи никогда не превышает `IDEA_MAX_RISK_PCT` (2%). Перед
  входом, который не помещается в лимит, самые ранние входы переводятся в
  безубыток (методика: перед третьим входом первый уходит в БУ, второй остаётся
  со своим стопом). Добавка требует подтверждения ценой (`ADDON_MIN_PROGRESS_R`,
  0.5R от последнего входа) и запрещена после `ADDON_MAX_PROGRESS_PCT` (50%)
  пути до финального TP — правило пирамидинга Kolachi. Максимум
  `IDEA_MAX_ENTRIES` входов (3), каждый со своими split-ногами, БУ и TP;
  неудачная идея стоит ≈2%, а не сумму входов;
- **AI-фильтр** (`core/m1/`) — статистическая оценка p(TP) по истории символа, может отклонить вход.

## Структура

```text
main.py                    # ядро: тикер, сигналы, сопровождение позиций
core/strategy_narrative.py # стратегия: bias, фильтры, триггеры, стоп/тейки
core/trade_journal.py      # журнал сделок (SQLite → CSV/Parquet)
core/mt5_guard.py          # глобальный лок на все вызовы mt5.*
executors/mt5_executor.py  # исполнение: split-entry, BE, закрытия, фолбэк fill-mode
bot/telegram_bot.py        # сигналы в канал + команды
mt5_bridge/                # кэш котировок (фолбэк для DataFeed)
deploy/                    # systemd-юниты + запуск на VPS (Wine 10 + Xvfb)
```

## Запуск

- **Windows**: `start_bot.ps1` (GUI-лаунчер, секреты в `.env`);
- **VPS (Ubuntu)**: `forexbot.service` → `deploy/run_bot.sh` — терминал MT5 запускается
  самим `mt5.initialize()` под Wine, виртуальный дисплей даёт `xvfb99.service`.

Конфигурация — через `.env` (см. `.env.example`): доступы MT5, токен Telegram,
риск-параметры, режим частичных тейков (`PARTIAL_TP_MODE=split|monitor`).

## Agentic stack (разработка и эксплуатация)

Проект разрабатывается и сопровождается в паре с **Claude Code** (Anthropic):
код-ревью, деплой на VPS по SSH, headless-отладка MT5 под Wine/Xvfb,
персистентная память проекта между сессиями. Внутри бота LLM нет —
только статистический p(TP)-фильтр по собственному журналу сделок.

## Дисклеймер

Проект для исследовательских целей. Торговля на рынке Forex сопряжена с высоким риском —
используйте демо-счёт; ответственность за реальные сделки лежит на пользователе.

---

## ForexTradingBot (English)

Autonomous trading bot for MetaTrader 5 (FxPro): finds setups using SMC/ICT logic,
executes trades with split partial take-profits and posts signals to a Telegram channel.
Runs 24/5 on a Linux VPS (MT5 terminal under Wine), trades GOLD, EURUSD, GBPUSD, USDCAD.

### Built on

- **Python 3.11** + the official **MetaTrader5** package — quotes and execution directly through the terminal;
- **Smart Money Concepts / ICT**: premium/discount, order blocks, rejection blocks,
  FVG, liquidity sweeps (turtle soup), Williams fractals;
- **python-telegram-bot** — signals and commands (`/status`, `/open`, `/report`, `/universe`);
- **SQLite + CSV/Parquet** — trade journal and statistics for the AI filter;
- signals are computed on **closed candles only** (no repainting).

### How the bot picks direction (HTF bias)

Direction comes from an ensemble of independent models on selected
timeframes (`core/strategy_narrative.py: calc_narrative`):

| Model | Weight |
| --- | --- |
| **1H Premium/Discount** — latest closed 15M close inside the latest closed H1 candle range | +2 |
| **4H fractal false breakout** — wick pierced the level, close back inside (reversal) | +2 |
| **15M fractal true breakout** — close beyond the level (continuation) | +1 |
| **Order Block (OB)** on 1H — side of the most recent active block | +1 |
| **Rejection Block (RB)** on 1H — most recent valid, unbroken block | +1 |

Bias is accepted once the vote margin reaches `HTF_SCORE_MARGIN` (default 2).
Strictness is regulated by the **1H FVG regime** (LuxAlgo Instantaneous Mitigation):
against the regime direction the required margin tightens by +1. SMA/EMA take no part
in the decision; on a mixed score the bias is NEUTRAL (no entry).

The former 5-day dealing range has been removed completely. It is replaced by
the non-directional **volatility assessment**: `R(t)` does not add LONG/SHORT
votes, but PANIC blocks entry and Expected Move checks whether TP1 is reachable.

### Entry filters

1. **Trading session** — London / New York only;
2. **Friday after 21:00 MSK** — new entries blocked, open positions force-closed before the weekend;
3. **Entry frequency**: 60-min cooldown after a stop-out (`POST_SL_COOLDOWN_MIN`),
   max 3 setups per symbol per day (`MAX_SETUPS_PER_SYMBOL_PER_DAY`),
   the same trigger zone/level is never re-traded within the day;
4. **Bot-wide daily stop** — at −3% from the day's starting balance (`DAILY_MAX_LOSS_PCT`)
   new entries are blocked until the next day;
5. **Volatility regime** — entry is blocked at `R(t) >= VOL_REGIME_MAX_R`;
   TP1 must be within `EM_TP_MAX_RATIO × EM1D`.

### Entry triggers

Checked in order; the first one that fires produces an ENTER signal:

1. **15M Rejection Block** — rejection from a block with a long wick in the bias direction;
2. **15M Turtle Soup** — false break of a local extreme (liquidity sweep) with reclaim;
3. **H1 Pivot Reclaim on 15M** — price reclaiming an H1 pivot level;
4. **1H Order Block touch** — entry off a valid OB (touch threshold in ATR fractions).

### Risk & trade management

- **Stop** — behind a 1H Williams fractal with an ATR buffer; risk clamped to a min/max ATR corridor
  (min 0.75×ATR — micro-stops cannot balloon the volume);
- **Hard SL-risk cap** — the aggregate nominal loss to the broker-side Stop Loss
  for one setup cannot exceed 1% of fixed starting capital. With
  `MT5_INITIAL_CAPITAL=0`, balance is captured once in
  `ai_data/risk_capital.json` and survives trades and VPS restarts; an explicit
  `MT5_INITIAL_CAPITAL` overrides that snapshot;
- **Sizing** — monetary loss is calculated with MT5 `order_calc_profit` in the
  account currency after applying the broker minimum SL distance, including
  commission and an expected-slippage reserve. Volume is always rounded down
  to `volume_step`; when `volume_min` temporarily exceeds budget, the bot keeps
  the technical SL, calculates a risk-compatible entry, and waits for that price;
- volume is additionally capped by `MT5_MAX_VOLUME` (default 10 lots per setup);
- **4 take-profits** at RR levels (1.0 / rr_min / 2.0 / 3.0);
- **Split-TP** — each target is opened as a separate position with its own broker-side TP,
  but all legs share one 1% risk budget and the remaining aggregate risk is
  checked before every order submission;
- **Break-even** — after TP1 the stops of all remaining legs move to the entry price;
- **Position Adding** (`core/position_adding.py`, off by default, enable with
  `POSITION_ADDING_ENABLED=1`) — staged entries into one idea: entry 1 risks 1%,
  every later confirmation of the same direction adds another 1%, and the idea's
  aggregate risk never exceeds `IDEA_MAX_RISK_PCT` (2%). Before an entry that
  would not fit, the oldest entries are moved to break-even (per the
  methodology: the third entry puts entry 1 at BE while entry 2 keeps its stop).
  An add-on needs price confirmation (`ADDON_MIN_PROGRESS_R`, 0.5R from the last
  entry) and is refused past `ADDON_MAX_PROGRESS_PCT` (50%) of the way to the
  final TP — the Kolachi pyramiding rule. At most `IDEA_MAX_ENTRIES` entries (3),
  each with its own split legs, break-even and TPs; a failed idea costs ≈2%
  rather than the sum of its entries;
- **AI filter** (`core/m1/`) — statistical p(TP) estimate from the symbol's history; may reject an entry.

### Project layout

```text
main.py                    # core: ticker, signals, position management
core/strategy_narrative.py # strategy: bias, filters, triggers, stop/TPs
core/trade_journal.py      # trade journal (SQLite → CSV/Parquet)
core/mt5_guard.py          # global lock around all mt5.* calls
executors/mt5_executor.py  # execution: split entry, BE, closes, fill-mode fallback
bot/telegram_bot.py        # channel signals + commands
mt5_bridge/                # quotes cache (fallback for DataFeed)
deploy/                    # systemd units + VPS launcher (Wine 10 + Xvfb)
```

### Running

- **Windows**: `start_bot.ps1` (GUI launcher, secrets in `.env`);
- **VPS (Ubuntu)**: `forexbot.service` → `deploy/run_bot.sh` — the MT5 terminal is launched
  by `mt5.initialize()` itself under Wine, with the virtual display provided by `xvfb99.service`.

Configuration lives in `.env` (see `.env.example`): MT5 credentials, Telegram token,
risk parameters, partial-TP mode (`PARTIAL_TP_MODE=split|monitor`).

### Agentic stack (development workflow)

The project is developed and operated in tandem with **Claude Code** (Anthropic):
code review, SSH deployments to the VPS, headless MT5 debugging under Wine/Xvfb,
persistent project memory across sessions. There is no LLM inside the bot itself —
only a statistical p(TP) filter built on its own trade journal.

### LSE historical snapshots

London Strategic Edge is integrated only as an offline research/backfill
source. It is deliberately isolated from the live MT5 `DataFeed`; LSE candles
must not be treated as FxPro Bid/Ask execution quotes.

The importer has its own non-login OS account, Python 3.13 virtual environment,
application tree, secret file and private state directory. Do not put
`LSE_API_KEY` in the project's `.env` or `.env.example`.

#### One-time VPS installation

Create the unprivileged account and root-owned application directories:

```bash
getent passwd forexbot-backtest >/dev/null || \
  sudo useradd --system --user-group \
    --home-dir /var/lib/forexbot-backtest --no-create-home \
    --shell /usr/sbin/nologin forexbot-backtest
sudo install -d -o root -g root -m 0755 /opt/forexbot-backtest/app
sudo install -d -o forexbot-backtest -g forexbot-backtest -m 0700 \
  /var/lib/forexbot-backtest
sudo install -d -o root -g forexbot-backtest -m 0550 \
  /srv/forexbot-backtest/snapshots
```

From the repository checkout, install only files tracked by the selected Git
commit. This deliberately excludes ignored or untracked `.env` files:

```bash
release_commit="$(git rev-parse --verify HEAD)"
git archive --format=tar "$release_commit" \
  backtest requirements-backtest.txt | \
  sudo tar --extract --file=- --directory=/opt/forexbot-backtest/app
printf '%s\n' "$release_commit" | \
  sudo tee /opt/forexbot-backtest/RELEASE_COMMIT >/dev/null
sudo chown -R root:root /opt/forexbot-backtest/app
sudo chmod -R go-w /opt/forexbot-backtest/app
if sudo find /opt/forexbot-backtest/app -name '.env*' -print -quit | grep -q .; then
  echo "ERROR: an environment file reached the isolated app tree"
  exit 1
fi
```

Build an isolated Python 3.13 environment. Direct dependencies are version
pinned; `installed.freeze.txt` records the complete resolved environment for
the deployment, but is not a cryptographic hash lock:

```bash
sudo python3.13 -m venv /opt/forexbot-backtest/venv
sudo /opt/forexbot-backtest/venv/bin/python -m pip install \
  --only-binary=:all: \
  -r /opt/forexbot-backtest/app/requirements-backtest.txt
sudo /opt/forexbot-backtest/venv/bin/python -m pip check
sudo sh -c '/opt/forexbot-backtest/venv/bin/python -m pip freeze --all \
  > /opt/forexbot-backtest/installed.freeze.txt'
sudo chown -R root:root /opt/forexbot-backtest
sudo chmod -R go-w /opt/forexbot-backtest
```

Keep the API key only in the existing root-only secret file. `sudoedit` should
leave exactly one `LSE_API_KEY=...` line; do not print or source that file:

```bash
sudo install -d -o root -g root -m 0700 /etc/forexbot
if ! sudo test -e /etc/forexbot/lse.env; then
  sudo install -o root -g root -m 0600 /dev/null /etc/forexbot/lse.env
fi
sudo chown root:root /etc/forexbot/lse.env
sudo chmod 0600 /etc/forexbot/lse.env
sudoedit /etc/forexbot/lse.env
sudo stat -c '%U:%G %a %n' /etc/forexbot/lse.env
sudo grep -Eq '^LSE_API_KEY=.+$' /etc/forexbot/lse.env
```

Install the non-secret job configuration and hardened oneshot unit:

```bash
sudo install -o root -g root -m 0644 deploy/lse-import.env.example \
  /etc/forexbot/lse-import.env
sudoedit /etc/forexbot/lse-import.env
sudo install -m 0644 deploy/forexbot-lse-import.service \
  /etc/systemd/system/forexbot-lse-import.service
sudo systemctl daemon-reload
sudo systemd-analyze verify /etc/systemd/system/forexbot-lse-import.service
```

The default example uses paced REST for the long 2020–2026, three-symbol
backfill and builds `1m`, `5m`, `15m`, `1h`, `4h` and `1d`. REST requests use
only provider-supported `YYYY-MM-DD` bounds: each request covers up to three
complete UTC dates, and the importer filters the first/last partial dates back
to the exact requested timestamps during normalization. Every ascending window
is checked with a descending one-row tail probe. If a plan cap silently
truncates the ascending result, the timestamps disagree and the run fails
closed with a recommendation to use bulk export. Empty weekend windows are
skipped without ending the backfill. This means two paced HTTP requests per
UTC date window (plus any retry attempts), which is designed to keep the normal
long run inside the unit's three-hour timeout at the default interval; repeated
rate-limit retries can still extend it. Leave
`LSE_DATASET=` empty to resolve and validate every symbol through the provider
catalog (for example, Forex is `fx`, while gold is `commodity`). The
`LSE_OUTPUT_DIR` target must not exist before the run; use a new versioned path
for every snapshot.

For a deliberately short bulk export, use settings such as these instead:

```ini
LSE_OUTPUT_DIR=/var/lib/forexbot-backtest/incoming/eurusd-2025-v1
LSE_SYMBOLS=EURUSD
LSE_START=2025-01-01
LSE_END=2026-06-01
LSE_TRANSPORT=export
LSE_DATASET=fx
```

That interval is shorter than the importer's 600-day export chunk and needs
one job. The importer preflights a maximum of five export jobs per run to stay
within the currently observed hourly account limit. For long or multi-symbol
ranges, keep the paced REST configuration.

#### Remove the key from the live bot

If the temporary live-service drop-in was used while testing the credential,
inspect it, remove that exact drop-in, and restart the live service:

```bash
sudo systemctl cat forexbot.service
sudo rm -f -- /etc/systemd/system/forexbot.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl restart forexbot.service
sudo systemctl is-active --quiet forexbot.service
sudo bash -c 'pid="$(systemctl show -p MainPID --value forexbot.service)"; \
  if grep -zq "^LSE_API_KEY=" "/proc/$pid/environ"; then \
    echo "ERROR: live bot still has LSE_API_KEY"; exit 1; \
  else echo "OK: live bot has no LSE_API_KEY"; fi'
```

After this verification, only `forexbot-lse-import.service` receives the key;
the live Wine/MT5 process does not import the network-facing SDK.

#### Create and seal a snapshot

Start each import intentionally and inspect its result:

```bash
sudo systemctl start forexbot-lse-import.service
sudo systemctl show forexbot-lse-import.service \
  -p Result -p ExecMainStatus
sudo journalctl -u forexbot-lse-import.service --no-pager -n 200
```

The importer downloads M1, validates UTC/OHLC/duplicates, and rebuilds
5M/M15/H1/H4/D1. It rejects truncated range edges and internal source gaps
larger than the configured holiday/weekend tolerance, and drops derived bars
with less than 95% of their expected M1 source rows. H4 and D1 use
`Europe/Athens` wall-clock boundaries by default, matching an EET/EEST broker
session without hard-coding a summer `+3` offset. Every Parquet candle stores
its explicit UTC close time, so 23/25-hour DST days remain causal in
`HistoricalDataset.frame_asof`.

The production example intentionally starts with the three FX pairs. Before
adding `GOLD=XAU/USD`, verify both the catalog's available date range and the
provider's daily maintenance calendar: a scheduled break can require a
symbol-aware H4 completeness rule. Do not lower the global quality threshold
just to force incomplete gold bars through validation.

After a successful run, move the exact new snapshot from the writable incoming
area to a root-owned archive. The archive directory itself is not writable by
the importer, so the importer cannot rename or delete a sealed snapshot:

```bash
incoming=/var/lib/forexbot-backtest/incoming/fx-2020-2026-v1
snapshot=/srv/forexbot-backtest/snapshots/fx-2020-2026-v1
sudo test -d "$incoming"
sudo test ! -e "$snapshot"
sudo install -d -o root -g forexbot-backtest -m 0550 \
  /srv/forexbot-backtest/snapshots
sudo chown -R root:forexbot-backtest "$incoming"
sudo find "$incoming" -type d -exec chmod 0550 {} +
sudo find "$incoming" -type f -exec chmod 0440 {} +
sudo mv -- "$incoming" "$snapshot"
sudo -u forexbot-backtest env PYTHONPATH=/opt/forexbot-backtest/app \
  /opt/forexbot-backtest/venv/bin/python \
  -m backtest verify --data "$snapshot"
sudo -u forexbot-backtest test ! -w "$snapshot"
sudo -u forexbot-backtest test ! -w /srv/forexbot-backtest/snapshots
```

This is operational immutability against the importer account; `root` can
still deliberately administer the files. `source_manifest.json` records
provider/version, the requested half-open interval, quality counters and data
file hashes; it never contains `LSE_API_KEY`. Its `runtime_environment` also
binds the snapshot to the exact deployed Git commit and the SHA-256 of
`installed.freeze.txt`; a copy is embedded as `environment.freeze.txt`.
The final `manifest.json` hashes both that lock and the provenance file, and
subsequent loads fail if the stored manifest no longer matches the snapshot.

No timer is installed or recommended: each run consumes provider quota and
must be started deliberately.

### Disclaimer

For research purposes. Forex trading carries high risk — use a demo account;
responsibility for live trades rests with the user.
