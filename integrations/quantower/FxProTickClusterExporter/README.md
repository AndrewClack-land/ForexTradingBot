# FxPro Tick Cluster Exporter (diagnostic)

Отдельный индикатор Quantower выгружает один **закрытый UTC-день** M15-кластеров:

- OHLC, `Volume` и `Ticks` каждой свечи;
- `VolumeAnalysisData.Total`: `Trades`, `Volume`, `Buy/Sell Volume`,
  `Buy/Sell Trades`, `Delta`;
- все записи `PriceLevels`, отсортированные и нормализованные в целые
  `price_ticks`;
- фактические типы `HistoryType`, `VolumeType`, `DeltaCalculationType` и
  realtime-возможности FxPro symbol;
- отдельные однодневные probes `Last` и `BidAsk`;
- coverage, causality metadata и SHA-256 содержимого.

Это **raw diagnostic schema v1**, а не готовый production-sidecar. Для него
принудительно закреплены:

```text
classification_status = UNVERIFIED
exchange_executions_proven = false
aggressor_polarity_proven = false
eligible_for_sidecar_conversion = false
historical_available_at = null
```

Установленный FxPro connector сообщает `VolumeType=Ticks`,
`DeltaCalculationType=TickDirection`, разрешает realtime ticks, но не realtime
volume/trades. Поэтому Quantower `Trades` здесь трактуется как число
классифицированных tick-events. Оно не называется биржевой лентой сделок.
Поле `connector_vendor=FxPro` является операторской меткой, а не извлечённой
учётной записью/connection identity; это явно записывается в manifest.

## Совместимость и сборка

Проверенная конфигурация:

```text
Quantower 1.146.14
TradingPlatform.BusinessLayer 1.146.14.0
Target framework net10.0-windows
```

С установленным .NET 10 SDK:

```powershell
cd E:\Forex_TradingBot\integrations\quantower\FxProTickClusterExporter
.\build.ps1 `
  -QuantowerBin "E:\Quantower\TradingPlatform\v1.146.14\bin" `
  -DotnetPath "dotnet"
```

С переносным SDK замените `-DotnetPath` на полный путь к `dotnet.exe`.
Проект не использует NuGet-пакеты и ссылается только на установленный
`TradingPlatform.BusinessLayer.dll`.

## Установка в Quantower

Закройте Quantower перед заменой существующей версии DLL. Для первой установки:

```powershell
$src = "E:\Forex_TradingBot\integrations\quantower\FxProTickClusterExporter\bin\Release\net10.0-windows"
$dst = "E:\Quantower\Settings\Scripts\Indicators\FxPro Tick Absorption Exporter"
New-Item -ItemType Directory -Path $dst -Force
Copy-Item "$src\FxProTickClusterExporter.dll" $dst
Copy-Item "$src\FxProTickClusterExporter.pdb" $dst
```

После копирования запустите/перезапустите Quantower.

## Выгрузка одного дня

1. Подключите FxPro и откройте M15 chart нужного FX-symbol.
2. Ограничьте загруженную chart-history минимальным диапазоном, который включает
   нужный день. `IsRequirePriceLevelsCalculation=true` заставляет Quantower
   рассчитывать уровни для всей загруженной истории, а не только для дня,
   который затем отфильтрует exporter.
3. Добавьте индикатор **FxPro Tick Cluster Exporter (diagnostic)**.
4. Укажите `UTC days ago=1` для предыдущего UTC-дня, `Strategy symbol` и
   отдельный `Output directory`.
5. Оставьте `Connector label=FxPro` и
   `Run Last/BidAsk history probes=true`. Успешные оба probes обязательны для
   статуса `VALID_DIAGNOSTIC`; `SKIPPED`/`ERROR` сохраняются для разбора, но
   валидатор их не пропускает.
6. Дождитесь в Quantower Logs строки `diagnostic snapshot completed`.

Экспорт создаётся атомарно:

```text
fxpro-tick-cluster-EURUSD-YYYY-MM-DD-<id>/
  bars.json
  manifest.json
```

`manifest.json` записывается последним. Каталог `.partial-*` не считается
готовым экспортом. Существующий финальный каталог никогда не перезаписывается.

По умолчанию `DateTimeKind.Unspecified` отклоняется. Параметр
`Assume unspecified bar times are UTC` разрешён только для диагностики и
фиксируется в manifest; он не доказывает causality. Такой raw-export можно
изучить вручную, но `VALID_DIAGNOSTIC` требует значение `false`.

## Проверка результата

Из корня репозитория:

```powershell
python tools\validate_quantower_cluster_diagnostic.py `
  "D:\QuantowerDiagnostics\fxpro-tick-cluster-EURUSD-YYYY-MM-DD-<id>"
```

Успешный результат начинается с `VALID_DIAGNOSTIC`. Валидатор проверяет хэши,
M15/UTC, OHLC, уровни, суммы Total, FxPro/Ticks/TickDirection metadata,
Last/BidAsk probes, symbol mapping, coverage и консервативные provenance-флаги.

Не передавайте этот каталог в:

```text
python -m backtest ... --orderflow-data <diagnostic-export>
```

Текущий `backtest/orderflow_data.py` принимает только sealed schema с доказанной
семантикой `buy_volume=aggressor_at_ask; sell_volume=aggressor_at_bid` и causal
`available_at`. Конвертер diagnostic -> sidecar появится лишь после ручной
проверки одного дня и отдельного правила исторической задержки; до этого live
Absorption остаётся выключенным.
