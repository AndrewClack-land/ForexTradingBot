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
sudo install -d -o root -g root -m 0755 \
  /opt/forexbot-backtest/releases
sudo install -d -o forexbot-backtest -g forexbot-backtest -m 0700 \
  /var/lib/forexbot-backtest
sudo install -d -o root -g forexbot-backtest -m 0550 \
  /srv/forexbot-backtest/snapshots
```

From the repository checkout, build a fresh versioned release using only files
tracked by the selected Git commit. The release is never extracted over a
running tree. Its manifest hashes every shipped file, and the `current`
symlink is changed only after the venv and lock file are complete. Ignored or
untracked `.env` files cannot enter this archive:

```bash
release_commit="$(git rev-parse --verify HEAD)"
release_root="/opt/forexbot-backtest/releases/$release_commit"
release_stage="/opt/forexbot-backtest/releases/.stage-$release_commit-$$"
sudo test ! -e "$release_root" || {
  echo "ERROR: release already exists: $release_root"
  exit 1
}
sudo install -d -o root -g root -m 0755 "$release_stage/app"
git archive --format=tar "$release_commit" \
  backtest \
  core/__init__.py \
  core/strategy_narrative.py core/htf_context.py \
  core/narrative_scoring.py core/pivot_trigger.py core/vol_regime.py \
  deploy/build_backtest_release_manifest.py \
  requirements-backtest.txt | \
  sudo tar --extract --file=- --directory="$release_stage/app"
sudo python3 \
  "$release_stage/app/deploy/build_backtest_release_manifest.py" \
  --root "$release_stage/app" \
  --commit "$release_commit" \
  --output "$release_stage/RELEASE_MANIFEST.json"
printf '%s\n' "$release_commit" | \
  sudo tee "$release_stage/RELEASE_COMMIT" >/dev/null
sudo chmod 0444 \
  "$release_stage/RELEASE_COMMIT" \
  "$release_stage/RELEASE_MANIFEST.json"
sudo chown -R root:root "$release_stage"
sudo chmod -R go-w "$release_stage"
if sudo find "$release_stage/app" -name '.env*' -print -quit | grep -q .; then
  echo "ERROR: an environment file reached the isolated app tree"
  exit 1
fi
sudo mv "$release_stage" "$release_root"
```

Build an isolated Python 3.13 environment. Direct dependencies are version
pinned; `installed.freeze.txt` records the complete resolved environment for
the deployment and its SHA-256 is embedded in every strategy report:

```bash
sudo python3.13 -m venv "$release_root/venv"
sudo "$release_root/venv/bin/python" -m pip install \
  --only-binary=:all: \
  -r "$release_root/app/requirements-backtest.txt"
sudo "$release_root/venv/bin/python" -m pip check
sudo sh -c "\"$release_root/venv/bin/python\" -m pip freeze --all \
  > \"$release_root/installed.freeze.txt\""
sudo chmod 0444 "$release_root/installed.freeze.txt"
sudo chown -R root:root "$release_root"
sudo chmod -R go-w "$release_root"
sudo ln -s "$release_root" /opt/forexbot-backtest/current.next
sudo mv -Tf /opt/forexbot-backtest/current.next \
  /opt/forexbot-backtest/current
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
UTC date window (plus any retry attempts). The production import can take
several hours: network latency, provider throttling and local resampling all
affect elapsed time. The unit therefore has a finite 12-hour start timeout and
a five-minute stop grace period; these are safety limits, not an ETA. Leave
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
sudo systemctl start --no-block forexbot-lse-import.service
sudo systemctl show forexbot-lse-import.service \
  -p ActiveState -p SubState -p MainPID -p Result -p ExecMainStatus
sudo journalctl -u forexbot-lse-import.service --no-pager -n 200
```

While `ActiveState=activating`, `Result` and `ExecMainStatus` describe the last
completed invocation and are not the current run's final result. The normal
successful terminal state is `ActiveState=inactive`, `Result=success`,
`ExecMainStatus=0` and `MainPID=0`. The importer publishes atomically, so the
configured final incoming path
`/var/lib/forexbot-backtest/incoming/fx-2020-2026-v1` appears only after all
validation succeeds.

Current releases write bounded progress messages to the journal, including the
phase, symbol, completed/total UTC windows, request count, resampling timeframe,
and finalization/publish phases:

```bash
sudo journalctl -fu forexbot-lse-import.service
```

For REST, download progress is emitted at roughly five-percent intervals rather
than for every request, so the journal remains useful without being flooded.

For the default three symbols and six timeframes, progress is visible as up to
18 Parquet files in the hidden staging directory. Six files mean one symbol
has been resampled, 12 mean two symbols, and 18 mean all three have been
resampled; manifest generation and final validation still remain after file
18. This is a progress indicator, not proof of a valid snapshot:

```bash
watch -n 30 "sudo find /var/lib/forexbot-backtest/incoming \
  -maxdepth 2 -type f -name '*.parquet' -printf '%p\n' | sort"
```

Do not add `Restart=` to this oneshot unit. An automatic retry can consume
provider quota, create another partial staging directory, and obscure the
original failure. Start every retry manually after diagnosis.

If systemd reports `Result=timeout` and `ExecMainStatus=15`, the start timeout
sent `SIGTERM`; inspect the timestamps, unit limits, journal and retained
staging files before retrying:

```bash
sudo systemctl show forexbot-lse-import.service \
  -p Result -p ExecMainStatus -p ActiveEnterTimestamp \
  -p InactiveEnterTimestamp -p TimeoutStartUSec -p TimeoutStopUSec
sudo journalctl -u forexbot-lse-import.service \
  --since "12 hours ago" --no-pager
sudo find /var/lib/forexbot-backtest/incoming -maxdepth 2 \
  -type f -printf '%p\n' | sort
```

Install the current unit before the next attempt so the 12-hour timeout,
five-minute stop grace period and updated memory limits are active:

```bash
sudo install -o root -g root -m 0644 \
  deploy/forexbot-lse-import.service \
  /etc/systemd/system/forexbot-lse-import.service
sudo systemctl daemon-reload
sudo systemd-analyze verify /etc/systemd/system/forexbot-lse-import.service
sudo systemctl reset-failed forexbot-lse-import.service
```

There is one narrow manual recovery path for the current interrupted run. Use
it only while the unit is inactive, the final target does not exist, exactly
the expected 18 Parquet files are present in one partial directory, and
`source_manifest.json` exists and validates every declared file size and
SHA-256. The following guard performs those checks, builds the final manifest,
and exits without moving anything if any condition fails:

```bash
sudo systemctl is-active --quiet forexbot-lse-import.service && {
  echo "ERROR: importer is still active"
  exit 1
}
target=/var/lib/forexbot-backtest/incoming/fx-2020-2026-v1
sudo test ! -e "$target" || {
  echo "ERROR: final target already exists"
  exit 1
}
mapfile -t partials < <(sudo find /var/lib/forexbot-backtest/incoming \
  -mindepth 1 -maxdepth 1 -type d \
  -name '.fx-2020-2026-v1.partial-*' -print)
test "${#partials[@]}" -eq 1 || {
  echo "ERROR: expected exactly one partial directory"
  exit 1
}
partial="${partials[0]}"
sudo -u forexbot-backtest env \
  PYTHONPATH=/opt/forexbot-backtest/current/app PARTIAL="$partial" \
  /opt/forexbot-backtest/current/venv/bin/python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

from backtest.data import HistoricalDataset

root = Path(os.environ["PARTIAL"])
symbols = ("EURUSD", "GBPUSD", "USDCAD")
timeframes = ("1m", "5m", "15m", "1h", "4h", "1d")
expected = {
    f"{symbol}_{timeframe}.parquet"
    for symbol in symbols
    for timeframe in timeframes
}
actual = {path.name for path in root.glob("*.parquet") if path.is_file()}
if actual != expected:
    raise SystemExit(
        f"expected exact 18 Parquet files; mismatch={expected ^ actual}"
    )
source_path = root / "source_manifest.json"
source = json.loads(source_path.read_text(encoding="utf-8"))
if source.get("provider") != "london_strategic_edge":
    raise SystemExit("unexpected source_manifest provider")
entries = source.get("files")
if (
    not isinstance(entries, list)
    or len(entries) != 18
    or {item.get("path") for item in entries} != expected
):
    raise SystemExit("source_manifest does not declare the exact 18 files")
declared_symbols = {
    item.get("bot_symbol")
    for item in source.get("symbols", [])
    if isinstance(item, dict)
}
if declared_symbols != set(symbols):
    raise SystemExit(
        f"source_manifest symbol mismatch: {declared_symbols ^ set(symbols)}"
    )
for item in entries:
    path = root / item["path"]
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"invalid snapshot file: {path.name}")
    if path.stat().st_size != int(item["bytes"]):
        raise SystemExit(f"size mismatch: {path.name}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != item["sha256"]:
        raise SystemExit(f"SHA-256 mismatch: {path.name}")
dataset = HistoricalDataset.load(root)
readiness = dataset.audit_readiness(symbols=symbols)
if not readiness["wfo_ready"]:
    raise SystemExit(f"WFO BLOCKED: {readiness['reasons']}")
dataset.write_manifest(root / "manifest.json")
print(f"WFO READY; validated manifest {dataset.manifest_sha256}")
PY
sudo test ! -e "$target"
sudo mv -T -- "$partial" "$target"
sudo -u forexbot-backtest env \
  PYTHONPATH=/opt/forexbot-backtest/current/app \
  /opt/forexbot-backtest/current/venv/bin/python \
  -m backtest verify --data "$target"
```

If there are fewer than 18 exact files, no `source_manifest.json`, a hash
mismatch, or more than one ambiguous partial directory, do not publish it.
Keep it for diagnosis, choose a fresh versioned `LSE_OUTPUT_DIR`, and manually
start a new import with the updated unit.

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
sudo -u forexbot-backtest env \
  PYTHONPATH=/opt/forexbot-backtest/current/app \
  /opt/forexbot-backtest/current/venv/bin/python \
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

#### Run the narrative strategy offline

`python -m backtest run` connects the immutable snapshot to the exact
`NarrativeStrategy`, the volatility gate and the three-leg outcome simulator.
It evaluates the strategy once per completed M15 candle and supplies at most
299 already-closed D/4H/1H/15M candles. It must not call
`Core._closed_bars_view`: the snapshot loader has already excluded forming
candles.

The default execution model is deliberately conservative and reproducible:

- the signal becomes known at the M15 close;
- it can fill only at a later M1 open inside the strategy entry range;
- the range expires after 15 minutes;
- TP weights are 50/30/20 percent and TP1 moves the remaining legs to
  break-even;
- an M1 candle touching both an adverse and favorable level uses
  `stop-first`;
- remaining exposure gets a causal time exit at Friday close, the configured
  maximum holding period, or an OOS fold boundary;
- each setup risks a fixed one percent of the explicitly supplied starting
  capital, never a compounded balance.

Use the same no-network systemd sandbox for both smoke and full runs. Paste the
following function once in the current shell. It resolves `current` to an exact
versioned, root-owned release and removes all live/LSE credentials from the
job:

```bash
release_root="$(readlink -f /opt/forexbot-backtest/current)"
snapshot=/srv/forexbot-backtest/snapshots/fx-2020-2026-v1
runs=/var/lib/forexbot-backtest/runs
sudo install -d -o forexbot-backtest -g forexbot-backtest -m 0700 "$runs"

backtest_sandbox() {
  unit="$1"
  shift
  sudo systemd-run --unit="$unit" --collect --no-block \
    --property=Type=oneshot \
    --property=User=forexbot-backtest \
    --property=Group=forexbot-backtest \
    --property="WorkingDirectory=$release_root/app" \
    --property=Environment=PYTHONDONTWRITEBYTECODE=1 \
    --property=Environment=PYTHONUNBUFFERED=1 \
    --property=Environment=PYTHONNOUSERSITE=1 \
    --property=Environment=PYTHONSAFEPATH=1 \
    --property="Environment=PYTHONPATH=$release_root/app" \
    --property=Environment=MT5_EXECUTION=0 \
    --property='UnsetEnvironment=LSE_API_KEY MT5_LOGIN MT5_PASSWORD MT5_SERVER TELEGRAM_TOKEN TELEGRAM_CHANNEL_ID' \
    --property=UMask=0077 \
    --property=PrivateNetwork=yes \
    --property=RestrictAddressFamilies=AF_UNIX \
    --property=NoNewPrivileges=yes \
    --property=RestrictSUIDSGID=yes \
    --property=CapabilityBoundingSet= \
    --property=AmbientCapabilities= \
    --property=PrivateDevices=yes \
    --property=PrivateTmp=yes \
    --property=ProtectSystem=strict \
    --property=ProtectHome=yes \
    --property=InaccessiblePaths=/etc/forexbot \
    --property="ReadOnlyPaths=$release_root" \
    --property="ReadOnlyPaths=$snapshot" \
    --property="ReadWritePaths=$runs" \
    --property=ProtectProc=invisible \
    --property=ProcSubset=pid \
    --property=ProtectKernelTunables=yes \
    --property=ProtectKernelModules=yes \
    --property=ProtectControlGroups=yes \
    --property=ProtectClock=yes \
    --property=ProtectHostname=yes \
    --property=LockPersonality=yes \
    --property=RestrictRealtime=yes \
    --property=RestrictNamespaces=yes \
    --property=KeyringMode=private \
    --property=LimitCORE=0 \
    --property=TasksMax=128 \
    --property=MemoryHigh=4G \
    --property=MemoryMax=6G \
    --property=Nice=10 \
    --property=CPUWeight=20 \
    --property=IOWeight=20 \
    --property=TimeoutStartSec=24h \
    --property=TimeoutStopSec=5min \
    "$@"
}
```

Run a short one-symbol diagnostic before the full walk-forward job. The output
directory must not already exist:

```bash
backtest_sandbox forexbot-backtest-smoke-eurusd \
  "$release_root/venv/bin/python" \
  -m backtest run \
  --data "$snapshot" \
  --symbols EURUSD \
  --start 2025-01-01 --end 2025-04-01 \
  --initial-capital REPLACE_WITH_STARTING_CAPITAL \
  --risk-fraction 0.01 \
  --profile production-deterministic \
  --intrabar-policy stop-first \
  --release-commit-file "$release_root/RELEASE_COMMIT" \
  --release-manifest-file "$release_root/RELEASE_MANIFEST.json" \
  --environment-lock-file "$release_root/installed.freeze.txt" \
  --output "$runs/eurusd-smoke-2025q1"
sudo journalctl -fu forexbot-backtest-smoke-eurusd.service
```

For a disconnect-safe full run, start a transient systemd service with no
network and no LSE/live-bot environment. The fixed current strategy has no
parameter optimizer yet: the two-year train interval supplies the rolling
walk-forward boundary and historical context; metrics are reported only for
the following non-overlapping six-month OOS intervals.

```bash
backtest_sandbox forexbot-backtest-run-fx-v1 \
  "$release_root/venv/bin/python" -m backtest run \
  --data "$snapshot" \
  --symbols EURUSD GBPUSD USDCAD \
  --initial-capital REPLACE_WITH_STARTING_CAPITAL \
  --risk-fraction 0.01 \
  --profile production-deterministic \
  --train 730D --test 180D --step 180D \
  --intrabar-policy stop-first \
  --release-commit-file "$release_root/RELEASE_COMMIT" \
  --release-manifest-file "$release_root/RELEASE_MANIFEST.json" \
  --environment-lock-file "$release_root/installed.freeze.txt" \
  --output "$runs/fx-wfo-v1"
sudo journalctl -fu forexbot-backtest-run-fx-v1.service
```

The report is atomically published and contains `summary.json`,
`candidates.csv`, one auditable candidate/policy decision in `executions.csv`,
`setups.csv`, `legs.csv`, `folds.csv`, `config.json` and a SHA-256
`manifest.json`. It also freezes the five narrative votes in
`candidate_factors.csv`, joins outcomes in
`setup_factor_attribution.csv`, and aggregates them by symbol, side, fold,
year, quarter, trigger, volatility regime, and FVG side in
`factor_summary.csv`. Candidate rows start only at raw `ENTER` (a baseline
directional bias plus a detected trigger); outcome rows are the narrower
executed-and-filled population. Empty CSV reports still contain stable
headers. A second run with `--intrabar-policy tp-first` (or `both`) measures
sensitivity to unknowable M1 intrabar ordering.

The same run creates a conditional, leakage-aware shadow score:
`shadow_models.json`, `shadow_scores.csv`, `shadow_coefficients.csv`, and
`shadow_score_metrics.csv`. The rolling ridge model uses only earlier
`stop-first` setups whose labels have matured before the next fold, with a
purge equal to entry TTL plus maximum holding time. It never changes candidate
IDs, entries, fills, SL/TP, execution dispositions, setup count, or risk.
Its target is `net_R` conditional on a production-filled setup. Coefficients
describe prior OOS raw `ENTER` signals that also survived execution gates and
filled; they are not unbiased replacement weights or a fill-probability
model.

A real replacement-weight optimizer needs a separate v2 research pass. That
pass must freeze factor vectors on every completed M15 decision, generate
technical LONG and SHORT opportunities independently of the baseline bias,
fit weights only inside each train interval, freeze the model, and replay the
following OOS interval. Without that counterfactual population, directly
optimizing the five weights would inherit selection bias from the old weights.

The report is a gross strategy diagnostic, not a broker-accurate PnL
statement. LSE OHLCV does not contain historical Bid/Ask spread, FxPro
commission, swap, slippage or tick ordering. The deterministic profile
includes sessions, Friday close, the volatility/expected-move gate, daily
setup cap, duplicate-trigger guard and post-loss cooldown. It intentionally
does not reuse today's AI SQLite state in the past (that would leak future
outcomes). It also excludes pyramiding, the bot-wide daily equity-loss brake,
anti-hedge/portfolio-correlation rules and broker lot constraints; those
require a later portfolio-level simulator.

### Disclaimer

For research purposes. Forex trading carries high risk — use a demo account;
responsibility for live trades rests with the user.
