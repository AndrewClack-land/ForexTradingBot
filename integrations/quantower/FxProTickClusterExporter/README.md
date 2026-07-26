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

Это **raw diagnostic schema v2**, а не готовый production-sidecar. Для него
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

Проверенный connector имеет default `HistoryType=Bid`, но отдельный `BidAsk`
probe возвращает распознанные tick-items, а `Last` probe — ноль записей.
Классификация `quantower_tick_reconstructed_bidask_history_available` означает
только доступность реконструируемой Bid/Ask tick-history. Она не доказывает
реальные исполнения или aggressor polarity. Если Quantower возвращает
`MinVolumeAnalysisTickSize` как неопределённое нечисловое значение, manifest
честно сохраняет `null`, а не придумывает шаг. Конечное отрицательное значение
считается ошибкой источника и останавливает экспорт.

## Границы M15

В проверенной связке FxPro/Quantower `HistoryItemBar.TimeRight` является
включительной правой границей. Например, для бара `10:30` Quantower сообщает
`10:44:59.9999999`, то есть `TimeLeft + 15 минут - 100 нс`. Экспортер сохраняет
исходный span в `source_bar_span_ticks`, но нормализует запись в канонический
полуоткрытый UTC-интервал `[10:30, 10:45)` и записывает `bar_close=10:45`.

Точная исходная семантика фиксируется в
`manifest.calculation.bar_right_boundary_semantics`. Допустимы только точный
exclusive period end и наблюдавшийся inclusive period end минус 100 нс.
Неизвестные, приблизительные и смешанные варианты отклоняются.
Все timestamp-поля обязаны иметь точный канонический формат экспортера
`YYYY-MM-DDTHH:MM:SS.fffZ`: дополнительные дробные знаки не округляются и не
обрезаются, а приводят к отказу валидатора.

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

1. Подключите FxPro и откройте chart нужного FX-symbol строго на `M15`, не
   `M30` и не другой timeframe.
2. Ограничьте загруженную chart-history минимальным диапазоном, который включает
   нужный день. `IsRequirePriceLevelsCalculation=true` заставляет Quantower
   рассчитывать уровни для всей загруженной истории, а не только для дня,
   который затем отфильтрует exporter.
3. Добавьте индикатор **FxPro Tick Cluster Exporter (diagnostic)**.
4. Укажите `UTC days ago=1` для предыдущего календарного UTC-дня,
   `Strategy symbol` и отдельный `Output directory`. Это календарные дни, а не
   торговые сессии: выбранный день должен быть закрыт и загружен; выходной
   обычно завершится `NO_BARS_FOR_TARGET_UTC_DAY`, а неполный день останется
   с coverage warning.
5. Оставьте `Connector label=FxPro` и
   `Run Last/BidAsk history probes=true`. Успешные оба probes обязательны для
   статуса `VALID_DIAGNOSTIC`; `SKIPPED`/`ERROR` сохраняются для разбора, но
   валидатор их не пропускает.
6. После изменения timeframe или `UTC days ago` нажмите `Apply/OK`, затем
   удалите и снова добавьте индикатор либо перезапустите Quantower. Экспорт
   одноразовый на инициализацию и запускается после
   `VolumeAnalysisData_Loaded`.
7. Дождитесь в Quantower Logs строки `diagnostic snapshot completed`. До неё
   валидатор не запускайте.

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
  "D:\QuantowerDiagnostics\fxpro-tick-cluster-EURUSD-YYYY-MM-DD-<id>" `
  --exporter-dll `
  "E:\Quantower\Settings\Scripts\Indicators\FxPro Tick Absorption Exporter\FxProTickClusterExporter.dll"
```

Передавайте конкретный финальный каталог с `bars.json` и `manifest.json`, а не
общий output root и не `.partial-*`. Quantower может загружать DLL из памяти,
поэтому self-hash в manifest иногда недоступен. В этом случае
`--exporter-dll` обязателен: валидатор сверяет CLR MVID DLL с manifest и
вычисляет её SHA-256.

Успешный результат имеет статус `VALID_DIAGNOSTIC`. Валидатор проверяет хэши и
идентичность DLL, каноническое `bar_close - bar_open = 15 минут`, точный
`source_bar_span_ticks` относительно `bar_right_boundary_semantics`, UTC, OHLC,
уровни, суммы Total, FxPro/Ticks/TickDirection metadata, Last/BidAsk probes,
symbol mapping, coverage и консервативные provenance-флаги.

Не передавайте этот каталог в:

```text
python -m backtest ... --orderflow-data <diagnostic-export>
```

Текущий `backtest/orderflow_data.py` принимает только sealed schema с доказанной
семантикой `buy_volume=aggressor_at_ask; sell_volume=aggressor_at_bid` и causal
`available_at`. Конвертер diagnostic -> sidecar появится лишь после ручной
проверки одного дня и отдельного правила исторической задержки; до этого live
Absorption остаётся выключенным.
