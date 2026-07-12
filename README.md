# ForexTradingBot

Автономный торговый бот для MetaTrader 5 (FxPro): сам находит сетапы по SMC/ICT-логике,
исполняет сделки с разбивкой на частичные тейки и публикует сигналы в Telegram-канал.
Работает 24/5 на Linux VPS (терминал MT5 под Wine), торгует GOLD, EURUSD, GBPUSD, USDCAD.

## На чём основан

- **Python 3.11** + официальный пакет **MetaTrader5** — котировки и исполнение напрямую через терминал;
- **Smart Money Concepts / ICT**: premium/discount, дилинг-ренджи, ордерблоки, rejection-блоки,
  FVG, liquidity sweep (turtle soup), фракталы Вильямса;
- **python-telegram-bot** — сигналы и команды (`/status`, `/open`, `/report`, `/universe`);
- **SQLite + CSV/Parquet** — журнал сделок и статистика для AI-фильтра;
- сигналы считаются **только по закрытым свечам** (без перерисовки).

## Как бот принимает направление (HTF bias)

Направление определяется ансамблем независимых моделей (ensemble learning) по старшим
таймфреймам (`core/strategy_narrative.py: calc_narrative`):

| Модель | Вклад |
| --- | --- |
| **Daily Premium/Discount** — текущая цена относительно PDH/PDL вчерашнего дня | +2 |
| **5-дневный дилинг-рендж** — доминирующая сторона строки и её вероятность (≥55%) | +1 |
| **Ложный пробой фрактала D1** — свеча проколола уровень, но закрылась внутри (разворот) | +2 |
| **Истинный пробой фрактала D1** — закрытие за уровнем (продолжение) | +1 |
| **Ордерблок (OB)** на 4H — сторона ближайшего валидного блока | +1 |
| **Rejection Block (RB)** на 4H — валидный неповреждённый блок | +1 |

Bias принимается при перевесе голосов ≥ `HTF_SCORE_MARGIN` (по умолчанию 2).
Строгость регулирует **FVG-режим 1H** (LuxAlgo Instantaneous Mitigation): против
направления режима порог ужесточается на +1. SMA/EMA в принятии решения не участвуют;
при смешанном счёте bias = NEUTRAL (входа нет).

## Фильтры перед входом

1. **Торговая сессия** — только London / New York;
2. **Пятница после 21:00 МСК** — новые входы блокируются, открытые позиции закрываются перед выходными;
3. **Частота входов**: кулдаун 60 мин после выбитого стопа (`POST_SL_COOLDOWN_MIN`),
   максимум 3 сетапа на символ в день (`MAX_SETUPS_PER_SYMBOL_PER_DAY`),
   одна и та же зона/уровень триггера не торгуется повторно в течение дня;
4. **Дневной стоп бота** — при просадке −3% от баланса на начало дня (`DAILY_MAX_LOSS_PCT`)
   новые входы блокируются до следующего дня.

## Триггеры входа

Проверяются по очереди, первый сработавший даёт сигнал ENTER:

1. **15M Rejection Block** — отклонение от блока с длинной тенью по направлению bias;
2. **15M Turtle Soup** — ложный пробой локального экстремума (liquidity sweep) с возвратом;
3. **H1 Pivot Reclaim на 15M** — возврат цены за пивот-уровень 1H;
4. **Касание ордерблока 1H** — вход от валидного OB (порог касания в долях ATR).

## Риск и сопровождение

- **Стоп** — за фрактал Вильямса на 1H с ATR-буфером, риск ограничен коридором min/max ATR
  (min 0.75×ATR — микро-стопы не раздувают объём);
- **Сайзинг** — от риска на сделку с учётом комиссии и ожидаемого проскальзывания,
  потолок объёма `MT5_MAX_VOLUME` (по умолчанию 10 лотов на сетап);
- **4 тейк-профита** по уровням RR (1.0 / rr_min / 2.0 / 3.0);
- **Split-TP** — каждая цель открывается отдельной позицией со своим брокерским TP,
  промежуточные тейки исполняет сам брокер точно по цене;
- **Безубыток** — после TP1 стопы всех оставшихся ног переносятся на цену входа;
- объём считается от риска на сделку (`MT5_RISK_PCT`, по умолчанию 1% на сделку);
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
- **Smart Money Concepts / ICT**: premium/discount, dealing ranges, order blocks, rejection blocks,
  FVG, liquidity sweeps (turtle soup), Williams fractals;
- **python-telegram-bot** — signals and commands (`/status`, `/open`, `/report`, `/universe`);
- **SQLite + CSV/Parquet** — trade journal and statistics for the AI filter;
- signals are computed on **closed candles only** (no repainting).

### How the bot picks direction (HTF bias)

Direction comes from an ensemble of independent models on the higher
timeframes (`core/strategy_narrative.py: calc_narrative`):

| Model | Weight |
| --- | --- |
| **Daily Premium/Discount** — current price vs yesterday's PDH/PDL | +2 |
| **D1 fractal false breakout** — wick pierced the level, close back inside (reversal) | +2 |
| **D1 fractal true breakout** — close beyond the level (continuation) | +1 |
| **5-day dealing range** — dominant side of the current row and its probability (≥55%) | +1 |
| **Order Block (OB)** on 4H — side of the nearest valid block | +1 |
| **Rejection Block (RB)** on 4H — valid, unbroken block | +1 |

Bias is accepted once the vote margin reaches `HTF_SCORE_MARGIN` (default 2).
Strictness is regulated by the **1H FVG regime** (LuxAlgo Instantaneous Mitigation):
against the regime direction the required margin tightens by +1. SMA/EMA take no part
in the decision; on a mixed score the bias is NEUTRAL (no entry).

### Entry filters

1. **Trading session** — London / New York only;
2. **Friday after 21:00 MSK** — new entries blocked, open positions force-closed before the weekend;
3. **Entry frequency**: 60-min cooldown after a stop-out (`POST_SL_COOLDOWN_MIN`),
   max 3 setups per symbol per day (`MAX_SETUPS_PER_SYMBOL_PER_DAY`),
   the same trigger zone/level is never re-traded within the day;
4. **Bot-wide daily stop** — at −3% from the day's starting balance (`DAILY_MAX_LOSS_PCT`)
   new entries are blocked until the next day.

### Entry triggers

Checked in order; the first one that fires produces an ENTER signal:

1. **15M Rejection Block** — rejection from a block with a long wick in the bias direction;
2. **15M Turtle Soup** — false break of a local extreme (liquidity sweep) with reclaim;
3. **H1 Pivot Reclaim on 15M** — price reclaiming an H1 pivot level;
4. **1H Order Block touch** — entry off a valid OB (touch threshold in ATR fractions).

### Risk & trade management

- **Stop** — behind a 1H Williams fractal with an ATR buffer; risk clamped to a min/max ATR corridor
  (min 0.75×ATR — micro-stops cannot balloon the volume);
- **Sizing** — from per-trade risk including commission and expected slippage,
  volume capped by `MT5_MAX_VOLUME` (default 10 lots per setup);
- **4 take-profits** at RR levels (1.0 / rr_min / 2.0 / 3.0);
- **Split-TP** — each target is opened as a separate position with its own broker-side TP,
  so intermediate TPs are filled by the broker exactly at price;
- **Break-even** — after TP1 the stops of all remaining legs move to the entry price;
- volume is sized from per-trade risk (`MT5_RISK_PCT`, default 1% per trade);
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

### Disclaimer

For research purposes. Forex trading carries high risk — use a demo account;
responsibility for live trades rests with the user.
