using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using TradingPlatform.BusinessLayer;

namespace ForexBot.Quantower;

/// <summary>
/// Exports one closed UTC day of native Quantower M15 Volume Analysis data.
///
/// This is deliberately a provenance diagnostic. It does not assert that the
/// FxPro tick reconstruction represents exchange executions or true aggressor
/// polarity, and its output is not a production AbsorptionEventDataset.
/// </summary>
public sealed class FxProTickClusterExporter : Indicator, IVolumeAnalysisIndicator
{
    private const string DiagnosticSchema = "forexbot.quantower-cluster-diagnostic";
    private const string BarsSchema = "forexbot.quantower-cluster-diagnostic-bars";
    private const int SchemaVersion = 1;
    private const string ExporterVersion = "0.1.0";
    private const string Timeframe = "15m";
    private const int ExpectedSlots = 96;

    private static readonly TimeSpan M15 = TimeSpan.FromMinutes(15);
    private static readonly UTF8Encoding Utf8NoBom = new(false);
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.Never,
    };

    private int generation;
    private int exportStarted;
    private DateTime historyRequestStartedAtUtc;

    [InputParameter("UTC days ago", 10, 1, 30, 1, 0)]
    public int DaysAgo = 1;

    [InputParameter("Output directory", 20)]
    public string OutputDirectory = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
        "ForexBot",
        "QuantowerDiagnostics"
    );

    [InputParameter("Strategy symbol (optional)", 30)]
    public string StrategySymbol = string.Empty;

    [InputParameter("Connector label", 40)]
    public string ConnectorLabel = "FxPro";

    [InputParameter("Run Last/BidAsk history probes", 50)]
    public bool RunHistoryProbes = true;

    [InputParameter("Assume unspecified bar times are UTC", 60)]
    public bool AssumeUnspecifiedTimesAreUtc = false;

    public FxProTickClusterExporter()
    {
        Name = "FxPro Tick Cluster Exporter (diagnostic)";
        Description =
            "Exports one closed UTC day of M15 Quantower Volume Analysis totals "
            + "and PriceLevels with conservative provenance metadata.";
        SeparateWindow = false;
    }

    public bool IsRequirePriceLevelsCalculation => true;

    protected override void OnInit()
    {
        Interlocked.Increment(ref generation);
        Interlocked.Exchange(ref exportStarted, 0);
        historyRequestStartedAtUtc = DateTime.UtcNow;

        if (DaysAgo < 1)
        {
            Core.Loggers.Log(
                "FxPro Tick Cluster Exporter: UTC days ago must be at least 1.",
                LoggingLevel.Error
            );
        }

        if (Symbol is null || HistoricalData is null)
        {
            Core.Loggers.Log(
                "FxPro Tick Cluster Exporter: symbol/history is unavailable.",
                LoggingLevel.Error
            );
        }
    }

    protected override void OnUpdate(UpdateArgs args)
    {
        // Export is triggered only by VolumeAnalysisData_Loaded(), after Quantower
        // has completed the requested price-level calculation.
    }

    protected override void OnClear()
    {
        Interlocked.Increment(ref generation);
        Interlocked.Exchange(ref exportStarted, 0);
    }

    public void VolumeAnalysisData_Loaded()
    {
        if (Interlocked.CompareExchange(ref exportStarted, 1, 0) != 0)
        {
            return;
        }

        int activeGeneration = Volatile.Read(ref generation);

        try
        {
            ExportSnapshot snapshot = CaptureSnapshot();

            _ = Task.Run(() =>
            {
                try
                {
                    if (activeGeneration != Volatile.Read(ref generation))
                    {
                        return;
                    }

                    string finalDirectory = WriteSnapshot(snapshot, activeGeneration);
                    Core.Loggers.Log(
                        "FxPro Tick Cluster Exporter: diagnostic snapshot completed: "
                            + Path.GetFileName(finalDirectory),
                        LoggingLevel.System
                    );
                }
                catch (Exception ex)
                {
                    Core.Loggers.Log(
                        "FxPro Tick Cluster Exporter: export failed ("
                            + SafeErrorCode(ex)
                            + ").",
                        LoggingLevel.Error
                    );
                }
            });
        }
        catch (Exception ex)
        {
            Core.Loggers.Log(
                "FxPro Tick Cluster Exporter: snapshot failed ("
                    + SafeErrorCode(ex)
                    + ").",
                LoggingLevel.Error
            );
        }
    }

    private ExportSnapshot CaptureSnapshot()
    {
        if (DaysAgo < 1)
        {
            throw new InvalidOperationException("CURRENT_OR_FUTURE_DAY_REJECTED");
        }

        if (Symbol is null || HistoricalData is null)
        {
            throw new InvalidOperationException("SYMBOL_OR_HISTORY_UNAVAILABLE");
        }

        if (string.IsNullOrWhiteSpace(OutputDirectory))
        {
            throw new InvalidOperationException("OUTPUT_DIRECTORY_REQUIRED");
        }

        var progress = HistoricalData.VolumeAnalysisCalculationProgress;
        if (progress is null || progress.State != VolumeAnalysisCalculationState.Finished)
        {
            throw new InvalidOperationException("VOLUME_ANALYSIS_NOT_FINISHED");
        }

        DateTime capturedAtUtc = DateTime.UtcNow;
        DateTime dayStartUtc = capturedAtUtc.Date.AddDays(-DaysAgo);
        DateTime dayEndUtc = dayStartUtc.AddDays(1);
        if (dayEndUtc > capturedAtUtc)
        {
            throw new InvalidOperationException("EXPORT_DAY_NOT_CLOSED");
        }

        string sourceSymbol = RequireSafeMetadata(Symbol.Name, "source symbol");
        string strategySymbol = string.IsNullOrWhiteSpace(StrategySymbol)
            ? sourceSymbol
            : RequireSafeMetadata(StrategySymbol.Trim(), "strategy symbol");
        string connectorLabel = RequireSafeMetadata(ConnectorLabel.Trim(), "connector label");
        if (!string.Equals(connectorLabel, "FxPro", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("CONNECTOR_LABEL_MUST_BE_FXPRO");
        }
        connectorLabel = "FxPro";
        string outputRoot = Path.GetFullPath(OutputDirectory.Trim());

        double tickSize = Symbol.TickSize;
        RequireFinitePositive(tickSize, "symbol tick size");

        var bars = new List<DiagnosticBar>();
        var seenBarOpens = new HashSet<long>();
        int barsWithPriceLevels = 0;
        bool allLevelSumsMatch = true;
        int nullPriceLevelBars = 0;
        int emptyPriceLevelBars = 0;

        for (int index = 0; index < HistoricalData.Count; index++)
        {
            if (HistoricalData[index, SeekOriginHistory.Begin] is not HistoryItemBar bar)
            {
                continue;
            }

            DateTime barOpenUtc = NormalizeUtc(bar.TimeLeft, "bar_open");
            DateTime barCloseUtc = NormalizeUtc(bar.TimeRight, "bar_close");
            if (barOpenUtc < dayStartUtc || barOpenUtc >= dayEndUtc)
            {
                continue;
            }

            if (barCloseUtc - barOpenUtc != M15)
            {
                throw new InvalidOperationException("CHART_IS_NOT_STRICT_M15");
            }

            RequireM15Boundary(barOpenUtc);
            if (barCloseUtc > dayEndUtc || barCloseUtc > capturedAtUtc)
            {
                throw new InvalidOperationException("UNFINALIZED_BAR_REJECTED");
            }

            if (!seenBarOpens.Add(barOpenUtc.Ticks))
            {
                throw new InvalidOperationException("DUPLICATE_BAR_OPEN");
            }

            DiagnosticBar snapshot = CaptureBar(
                bar,
                sourceSymbol,
                strategySymbol,
                tickSize,
                barOpenUtc,
                barCloseUtc,
                capturedAtUtc
            );
            bars.Add(snapshot);

            if (snapshot.PriceLevels.Count > 0)
            {
                barsWithPriceLevels++;
            }

            if (snapshot.Validation.PriceLevelsStatus == "NULL")
            {
                nullPriceLevelBars++;
            }
            else if (snapshot.Validation.PriceLevelsStatus == "EMPTY")
            {
                emptyPriceLevelBars++;
            }

            allLevelSumsMatch &= snapshot.Validation.SumLevelsMatchesTotal;
        }

        bars.Sort((left, right) =>
            string.CompareOrdinal(left.BarOpen, right.BarOpen)
        );

        if (bars.Count == 0)
        {
            throw new InvalidOperationException("NO_BARS_FOR_TARGET_UTC_DAY");
        }

        List<string> missingBarOpens = BuildMissingSlots(dayStartUtc, seenBarOpens);
        ProbeEvidence evidence = RunHistoryProbes
            ? CaptureHistoryProbes(dayStartUtc, dayEndUtc)
            : ProbeEvidence.Skipped();

        string classification = ClassifySemantics(evidence);
        string validationStatus =
            missingBarOpens.Count == 0
            && barsWithPriceLevels == bars.Count
            && allLevelSumsMatch
                ? "PASS"
                : "WARN";

        var exportId = Guid.NewGuid().ToString("D");
        return new ExportSnapshot
        {
            ExportId = exportId,
            OutputRoot = outputRoot,
            CreatedAt = FormatUtc(capturedAtUtc),
            DayUtc = dayStartUtc.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
            FromInclusive = FormatUtc(dayStartUtc),
            ToExclusive = FormatUtc(dayEndUtc),
            SourceSymbol = sourceSymbol,
            StrategySymbol = strategySymbol,
            ConnectorLabel = connectorLabel,
            TickSize = DecimalString(tickSize),
            MinVolumeAnalysisTickSize = DecimalString(Symbol.MinVolumeAnalysisTickSize),
            SymbolHistoryType = EnumSnapshot.From(Symbol.HistoryType),
            SymbolVolumeType = EnumSnapshot.From(Symbol.VolumeType),
            DeltaCalculationType = EnumSnapshot.From(Symbol.DeltaCalculationType),
            ChartAggregation = HistoricalData.Aggregation?.ToString() ?? "UNKNOWN",
            AllowCalculateRealtimeTicks = Symbol.AllowCalculateRealtimeTicks,
            AllowCalculateRealtimeVolume = Symbol.AllowCalculateRealtimeVolume,
            AllowCalculateRealtimeTrades = Symbol.AllowCalculateRealtimeTrades,
            UnspecifiedTimesAssumedUtc = AssumeUnspecifiedTimesAreUtc,
            Classification = classification,
            Evidence = evidence,
            HistoryRequestStartedAt = FormatUtc(historyRequestStartedAtUtc),
            VolumeAnalysisFinishedAt = FormatUtc(capturedAtUtc),
            SnapshotFinishedAt = FormatUtc(DateTime.UtcNow),
            ExpectedUtcSlots = ExpectedSlots,
            BarsWithPriceLevels = barsWithPriceLevels,
            MissingBarOpens = missingBarOpens,
            NullPriceLevelBars = nullPriceLevelBars,
            EmptyPriceLevelBars = emptyPriceLevelBars,
            ValidationStatus = validationStatus,
            Bars = bars,
        };
    }

    private DiagnosticBar CaptureBar(
        HistoryItemBar bar,
        string sourceSymbol,
        string strategySymbol,
        double tickSize,
        DateTime barOpenUtc,
        DateTime barCloseUtc,
        DateTime capturedAtUtc
    )
    {
        RequireOhlc(bar.Open, bar.High, bar.Low, bar.Close);
        RequireFiniteNonNegative(bar.Volume, "bar volume");
        if (bar.Ticks < 0)
        {
            throw new InvalidOperationException("NEGATIVE_BAR_TICKS");
        }

        VolumeAnalysisData? volumeData = bar.VolumeAnalysisData;
        if (volumeData?.Total is null)
        {
            throw new InvalidOperationException("VOLUME_ANALYSIS_TOTAL_UNAVAILABLE");
        }

        VolumeFields total = CaptureVolumeFields(volumeData.Total);
        var levels = new List<PriceLevel>();
        string priceLevelsStatus;

        if (volumeData.PriceLevels is null)
        {
            priceLevelsStatus = "NULL";
        }
        else if (volumeData.PriceLevels.Count == 0)
        {
            priceLevelsStatus = "EMPTY";
        }
        else
        {
            priceLevelsStatus = "AVAILABLE";
            var seenTicks = new HashSet<long>();
            foreach (KeyValuePair<double, VolumeAnalysisItem> pair in volumeData.PriceLevels)
            {
                double price = pair.Key;
                RequireFinitePositive(price, "price level");
                if (pair.Value is null)
                {
                    throw new InvalidOperationException("NULL_PRICE_LEVEL_VALUE");
                }

                long priceTicks = CheckedPriceTicks(price, tickSize);
                if (!seenTicks.Add(priceTicks))
                {
                    throw new InvalidOperationException("DUPLICATE_NORMALIZED_PRICE_LEVEL");
                }

                if (price < bar.Low - tickSize || price > bar.High + tickSize)
                {
                    throw new InvalidOperationException("PRICE_LEVEL_OUTSIDE_BAR_RANGE");
                }

                VolumeFields fields = CaptureVolumeFields(pair.Value);
                levels.Add(
                    new PriceLevel
                    {
                        Price = DecimalString(price),
                        PriceTicks = priceTicks,
                        Volume = fields.Volume,
                        Trades = fields.Trades,
                        BuyVolume = fields.BuyVolume,
                        SellVolume = fields.SellVolume,
                        BuyTrades = fields.BuyTrades,
                        SellTrades = fields.SellTrades,
                        Delta = fields.Delta,
                    }
                );
            }

            levels.Sort((left, right) => left.PriceTicks.CompareTo(right.PriceTicks));
        }

        VolumeSums levelSums = SumLevels(levels);
        bool sumsMatch =
            ApproximatelyEqual(levelSums.Volume, ParseDecimalString(total.Volume))
            && levelSums.Trades == total.Trades
            && ApproximatelyEqual(levelSums.BuyVolume, ParseDecimalString(total.BuyVolume))
            && ApproximatelyEqual(levelSums.SellVolume, ParseDecimalString(total.SellVolume))
            && levelSums.BuyTrades == total.BuyTrades
            && levelSums.SellTrades == total.SellTrades;

        double unclassifiedVolume =
            ParseDecimalString(total.Volume)
            - ParseDecimalString(total.BuyVolume)
            - ParseDecimalString(total.SellVolume);
        int unclassifiedTrades = total.Trades - total.BuyTrades - total.SellTrades;

        return new DiagnosticBar
        {
            SchemaVersion = SchemaVersion,
            Revision = 0,
            SourceSymbol = sourceSymbol,
            Symbol = strategySymbol,
            Timeframe = Timeframe,
            BarOpen = FormatUtc(barOpenUtc),
            BarClose = FormatUtc(barCloseUtc),
            CapturedAt = FormatUtc(capturedAtUtc),
            HistoricalAvailableAt = null,
            AvailabilitySemantics = "unknown_historical_latency",
            Finalized = true,
            FinalizationBasis = "bar_close_before_closed_export_day",
            Ohlc = new Ohlc
            {
                Open = DecimalString(bar.Open),
                High = DecimalString(bar.High),
                Low = DecimalString(bar.Low),
                Close = DecimalString(bar.Close),
            },
            BarVolume = DecimalString(bar.Volume),
            BarTicks = bar.Ticks,
            Total = total,
            PriceLevels = levels,
            Validation = new BarValidation
            {
                PriceLevelsStatus = priceLevelsStatus,
                LevelCount = levels.Count,
                SumLevelsMatchesTotal = sumsMatch,
                UnclassifiedVolume = DecimalString(unclassifiedVolume),
                UnclassifiedTrades = unclassifiedTrades,
                LevelSums = levelSums.ToFields(),
            },
        };
    }

    private ProbeEvidence CaptureHistoryProbes(DateTime fromUtc, DateTime toUtc)
    {
        return new ProbeEvidence
        {
            LastProbe = CaptureLastProbe(fromUtc, toUtc),
            BidAskProbe = CaptureBidAskProbe(fromUtc, toUtc),
        };
    }

    private HistoryProbe CaptureLastProbe(DateTime fromUtc, DateTime toUtc)
    {
        var result = HistoryProbe.For(HistoryType.Last);
        try
        {
            using HistoricalData history = Symbol.GetTickHistory(
                HistoryType.Last,
                fromUtc,
                toUtc
            );
            result.Status = "OK";
            result.Items = history.Count;

            for (int index = 0; index < history.Count; index++)
            {
                if (history[index, SeekOriginHistory.Begin] is not HistoryItemLast item)
                {
                    continue;
                }

                result.TypedItems++;
                if (item.Volume > 0 && double.IsFinite(item.Volume))
                {
                    result.PositiveSizeItems++;
                }

                string aggressor = item.AggressorFlag.ToString();
                if (aggressor.Contains("Buy", StringComparison.OrdinalIgnoreCase))
                {
                    result.AggressorBuy++;
                }
                else if (aggressor.Contains("Sell", StringComparison.OrdinalIgnoreCase))
                {
                    result.AggressorSell++;
                }
                else
                {
                    result.AggressorUnknown++;
                }
            }
        }
        catch (Exception ex)
        {
            result.Status = "ERROR";
            result.ErrorType = ex.GetType().Name;
        }

        return result;
    }

    private HistoryProbe CaptureBidAskProbe(DateTime fromUtc, DateTime toUtc)
    {
        var result = HistoryProbe.For(HistoryType.BidAsk);
        try
        {
            using HistoricalData history = Symbol.GetTickHistory(
                HistoryType.BidAsk,
                fromUtc,
                toUtc
            );
            result.Status = "OK";
            result.Items = history.Count;

            for (int index = 0; index < history.Count; index++)
            {
                if (history[index, SeekOriginHistory.Begin] is not HistoryItemTick item)
                {
                    continue;
                }

                result.TypedItems++;
                if (
                    (item.BidSize > 0 && double.IsFinite(item.BidSize))
                    || (item.AskSize > 0 && double.IsFinite(item.AskSize))
                )
                {
                    result.PositiveSizeItems++;
                }
            }
        }
        catch (Exception ex)
        {
            result.Status = "ERROR";
            result.ErrorType = ex.GetType().Name;
        }

        return result;
    }

    private string ClassifySemantics(ProbeEvidence evidence)
    {
        if (
            evidence.LastProbe.Status == "OK"
            && evidence.LastProbe.Items == 0
            && evidence.BidAskProbe.Status == "OK"
            && evidence.BidAskProbe.Items > 0
            && evidence.BidAskProbe.TypedItems > 0
            && Symbol.HistoryType == HistoryType.BidAsk
            && Symbol.VolumeType == SymbolVolumeType.Ticks
        )
        {
            return "quantower_bidask_tick_reconstructed";
        }

        if (
            evidence.LastProbe.Status == "OK"
            && evidence.LastProbe.PositiveSizeItems > 0
            && evidence.LastProbe.AggressorBuy + evidence.LastProbe.AggressorSell > 0
        )
        {
            return "trade_capable_source_unverified_calculation";
        }

        return "unknown";
    }

    private string WriteSnapshot(ExportSnapshot snapshot, int activeGeneration)
    {
        string root = snapshot.OutputRoot;
        Directory.CreateDirectory(root);

        string safeSymbol = SafeFileComponent(snapshot.StrategySymbol);
        string finalName =
            "fxpro-tick-cluster-"
            + safeSymbol
            + "-"
            + snapshot.DayUtc
            + "-"
            + snapshot.ExportId[..8];
        string finalDirectory = Path.Combine(root, finalName);
        string partialDirectory = Path.Combine(
            root,
            "." + finalName + ".partial-" + Guid.NewGuid().ToString("N")
        );

        if (Directory.Exists(finalDirectory) || File.Exists(finalDirectory))
        {
            throw new IOException("FINAL_EXPORT_ALREADY_EXISTS");
        }

        Directory.CreateDirectory(partialDirectory);
        try
        {
            var barsDocument = new BarsDocument
            {
                Schema = BarsSchema,
                SchemaVersion = SchemaVersion,
                ExportId = snapshot.ExportId,
                Bars = snapshot.Bars,
            };
            byte[] barsBytes = JsonSerializer.SerializeToUtf8Bytes(
                barsDocument,
                JsonOptions
            );
            string barsPath = Path.Combine(partialDirectory, "bars.json");
            WriteFileAtomically(barsPath, barsBytes);
            string barsSha256 = Sha256Hex(barsBytes);

            var manifest = BuildManifest(snapshot, barsBytes.Length, barsSha256);
            byte[] manifestBytes = JsonSerializer.SerializeToUtf8Bytes(
                manifest,
                JsonOptions
            );
            string manifestPath = Path.Combine(partialDirectory, "manifest.json");
            WriteFileAtomically(manifestPath, manifestBytes);

            if (activeGeneration != Volatile.Read(ref generation))
            {
                throw new OperationCanceledException("INDICATOR_GENERATION_CHANGED");
            }

            Directory.Move(partialDirectory, finalDirectory);
            return finalDirectory;
        }
        catch
        {
            if (Directory.Exists(partialDirectory))
            {
                Directory.Delete(partialDirectory, recursive: true);
            }

            throw;
        }
    }

    private static DiagnosticManifest BuildManifest(
        ExportSnapshot snapshot,
        long barsSize,
        string barsSha256
    )
    {
        return new DiagnosticManifest
        {
            Schema = DiagnosticSchema,
            SchemaVersion = SchemaVersion,
            ExportId = snapshot.ExportId,
            CreatedAt = snapshot.CreatedAt,
            DayUtc = snapshot.DayUtc,
            Interval = new ExportInterval
            {
                FromInclusive = snapshot.FromInclusive,
                ToExclusive = snapshot.ToExclusive,
            },
            Timeframe = Timeframe,
            Source = new SourceMetadata
            {
                Platform = "Quantower",
                PlatformVersion =
                    typeof(Indicator).Assembly.GetName().Version?.ToString() ?? "UNKNOWN",
                ConnectorVendor = snapshot.ConnectorLabel,
                ConnectorVendorSemantics = "operator_label_not_connection_identity",
                ExporterVersion = ExporterVersion,
                ExporterBinarySha256 = TryAssemblySha256(),
            },
            Instrument = new InstrumentMetadata
            {
                SourceSymbol = snapshot.SourceSymbol,
                StrategySymbol = snapshot.StrategySymbol,
                TickSize = snapshot.TickSize,
                MinVolumeAnalysisTickSize = snapshot.MinVolumeAnalysisTickSize,
                SymbolHistoryType = snapshot.SymbolHistoryType,
                SymbolVolumeType = snapshot.SymbolVolumeType,
                SymbolDeltaCalculationType = snapshot.DeltaCalculationType,
                AllowCalculateRealtimeTicks = snapshot.AllowCalculateRealtimeTicks,
                AllowCalculateRealtimeVolume = snapshot.AllowCalculateRealtimeVolume,
                AllowCalculateRealtimeTrades = snapshot.AllowCalculateRealtimeTrades,
            },
            Calculation = new CalculationMetadata
            {
                Period = Timeframe,
                ChartAggregation = snapshot.ChartAggregation,
                PriceLevelsRequested = true,
                VolumeBasis = "native_quantower_volume_analysis",
                Timezone = "UTC",
                UnspecifiedTimesAssumedUtc = snapshot.UnspecifiedTimesAssumedUtc,
            },
            Semantics = new SemanticsMetadata
            {
                FactorName = "FxPro Tick Absorption 15M",
                Classification = snapshot.Classification,
                ClassificationStatus = "UNVERIFIED",
                BuyVolume = "native Quantower VolumeAnalysisItem.BuyVolume",
                SellVolume = "native Quantower VolumeAnalysisItem.SellVolume",
                Trades =
                    "native Quantower VolumeAnalysisItem.Trades; classified tick-events, "
                    + "not assumed exchange executions",
                ExchangeExecutionsProven = false,
                AggressorPolarityProven = false,
                Evidence = snapshot.Evidence,
            },
            Causality = new CausalityMetadata
            {
                HistoryRequestStartedAt = snapshot.HistoryRequestStartedAt,
                HistoryRequestStartSemantics = "indicator_on_init_lower_bound",
                VolumeAnalysisFinishedAt = snapshot.VolumeAnalysisFinishedAt,
                SnapshotFinishedAt = snapshot.SnapshotFinishedAt,
                HistoricalAvailabilitySemantics =
                    "snapshot_observed_at_not_historical_feed_latency",
            },
            Coverage = new CoverageMetadata
            {
                ExpectedUtcSlots = snapshot.ExpectedUtcSlots,
                BarsReturned = snapshot.Bars.Count,
                BarsWithPriceLevels = snapshot.BarsWithPriceLevels,
                MissingBarOpens = snapshot.MissingBarOpens,
                NullPriceLevelBars = snapshot.NullPriceLevelBars,
                EmptyPriceLevelBars = snapshot.EmptyPriceLevelBars,
                ValidationStatus = snapshot.ValidationStatus,
                EligibleForSidecarConversion = false,
            },
            Files =
            [
                new FileMetadata
                {
                    Path = "bars.json",
                    Size = barsSize,
                    Sha256 = barsSha256,
                    Records = snapshot.Bars.Count,
                },
            ],
            ContentSha256 = barsSha256,
        };
    }

    private static void WriteFileAtomically(string finalPath, byte[] bytes)
    {
        string temporaryPath = finalPath + ".tmp";
        using (
            var stream = new FileStream(
                temporaryPath,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None
            )
        )
        {
            stream.Write(bytes, 0, bytes.Length);
            stream.Flush(flushToDisk: true);
        }

        File.Move(temporaryPath, finalPath);
    }

    private static VolumeFields CaptureVolumeFields(VolumeAnalysisItem item)
    {
        RequireFiniteNonNegative(item.Volume, "volume");
        RequireFiniteNonNegative(item.BuyVolume, "buy volume");
        RequireFiniteNonNegative(item.SellVolume, "sell volume");
        RequireFinite(item.Delta, "delta");
        if (item.Trades < 0 || item.BuyTrades < 0 || item.SellTrades < 0)
        {
            throw new InvalidOperationException("NEGATIVE_TRADE_COUNT");
        }

        return new VolumeFields
        {
            Volume = DecimalString(item.Volume),
            Trades = item.Trades,
            BuyVolume = DecimalString(item.BuyVolume),
            SellVolume = DecimalString(item.SellVolume),
            BuyTrades = item.BuyTrades,
            SellTrades = item.SellTrades,
            Delta = DecimalString(item.Delta),
        };
    }

    private static VolumeSums SumLevels(IEnumerable<PriceLevel> levels)
    {
        var sums = new VolumeSums();
        foreach (PriceLevel level in levels)
        {
            sums.Volume += ParseDecimalString(level.Volume);
            sums.Trades += level.Trades;
            sums.BuyVolume += ParseDecimalString(level.BuyVolume);
            sums.SellVolume += ParseDecimalString(level.SellVolume);
            sums.BuyTrades += level.BuyTrades;
            sums.SellTrades += level.SellTrades;
            sums.Delta += ParseDecimalString(level.Delta);
        }

        return sums;
    }

    private static List<string> BuildMissingSlots(
        DateTime dayStartUtc,
        HashSet<long> seenBarOpens
    )
    {
        var missing = new List<string>();
        for (int slot = 0; slot < ExpectedSlots; slot++)
        {
            DateTime expected = dayStartUtc.AddMinutes(slot * 15);
            if (!seenBarOpens.Contains(expected.Ticks))
            {
                missing.Add(FormatUtc(expected));
            }
        }

        return missing;
    }

    private DateTime NormalizeUtc(DateTime value, string field)
    {
        if (value.Kind == DateTimeKind.Utc)
        {
            return value;
        }

        if (value.Kind == DateTimeKind.Local)
        {
            return value.ToUniversalTime();
        }

        if (AssumeUnspecifiedTimesAreUtc)
        {
            return DateTime.SpecifyKind(value, DateTimeKind.Utc);
        }

        throw new InvalidOperationException(
            field.ToUpperInvariant() + "_HAS_UNSPECIFIED_TIMEZONE"
        );
    }

    private static void RequireM15Boundary(DateTime value)
    {
        if (
            value.Kind != DateTimeKind.Utc
            || value.Minute % 15 != 0
            || value.Second != 0
            || value.Millisecond != 0
        )
        {
            throw new InvalidOperationException("BAR_OPEN_NOT_ON_M15_UTC_BOUNDARY");
        }
    }

    private static void RequireOhlc(double open, double high, double low, double close)
    {
        RequireFinite(open, "open");
        RequireFinite(high, "high");
        RequireFinite(low, "low");
        RequireFinite(close, "close");
        if (low > high || open < low || open > high || close < low || close > high)
        {
            throw new InvalidOperationException("INCONSISTENT_OHLC");
        }
    }

    private static void RequireFinite(double value, string field)
    {
        if (!double.IsFinite(value))
        {
            throw new InvalidOperationException(
                field.ToUpperInvariant().Replace(' ', '_') + "_NOT_FINITE"
            );
        }
    }

    private static void RequireFinitePositive(double value, string field)
    {
        RequireFinite(value, field);
        if (value <= 0)
        {
            throw new InvalidOperationException(
                field.ToUpperInvariant().Replace(' ', '_') + "_NOT_POSITIVE"
            );
        }
    }

    private static void RequireFiniteNonNegative(double value, string field)
    {
        RequireFinite(value, field);
        if (value < 0)
        {
            throw new InvalidOperationException(
                field.ToUpperInvariant().Replace(' ', '_') + "_NEGATIVE"
            );
        }
    }

    private static long CheckedPriceTicks(double price, double tickSize)
    {
        double rawTicks = price / tickSize;
        if (!double.IsFinite(rawTicks) || rawTicks > long.MaxValue || rawTicks < long.MinValue)
        {
            throw new InvalidOperationException("PRICE_TICKS_OUT_OF_RANGE");
        }

        long ticks = checked((long)Math.Round(rawTicks, MidpointRounding.AwayFromZero));
        double reconstructed = ticks * tickSize;
        double tolerance = Math.Max(1e-12, Math.Abs(tickSize) * 1e-6);
        if (Math.Abs(reconstructed - price) > tolerance)
        {
            throw new InvalidOperationException("PRICE_NOT_ALIGNED_TO_SYMBOL_TICK");
        }

        return ticks;
    }

    private static bool ApproximatelyEqual(double left, double right)
    {
        double tolerance = Math.Max(1e-8, Math.Max(Math.Abs(left), Math.Abs(right)) * 1e-9);
        return Math.Abs(left - right) <= tolerance;
    }

    private static string DecimalString(double value)
    {
        RequireFinite(value, "decimal value");
        if (value == 0d)
        {
            return "0";
        }

        return value.ToString("G17", CultureInfo.InvariantCulture);
    }

    private static double ParseDecimalString(string value)
    {
        return double.Parse(value, NumberStyles.Float, CultureInfo.InvariantCulture);
    }

    private static string FormatUtc(DateTime value)
    {
        if (value.Kind != DateTimeKind.Utc)
        {
            throw new InvalidOperationException("NON_UTC_TIMESTAMP");
        }

        return value.ToString("yyyy-MM-dd'T'HH:mm:ss.fff'Z'", CultureInfo.InvariantCulture);
    }

    private static string RequireSafeMetadata(string value, string field)
    {
        if (
            string.IsNullOrWhiteSpace(value)
            || value.Length > 128
            || value.IndexOfAny(['\r', '\n', '\0']) >= 0
        )
        {
            throw new InvalidOperationException(
                field.ToUpperInvariant().Replace(' ', '_') + "_INVALID"
            );
        }

        return value.Trim();
    }

    private static string SafeFileComponent(string value)
    {
        char[] invalid = Path.GetInvalidFileNameChars();
        var builder = new StringBuilder(value.Length);
        foreach (char character in value)
        {
            if (
                invalid.Contains(character)
                || char.IsWhiteSpace(character)
                || character == '/'
                || character == '\\'
            )
            {
                builder.Append('_');
            }
            else
            {
                builder.Append(char.ToUpperInvariant(character));
            }
        }

        string result = builder.ToString().Trim('_', '.');
        return string.IsNullOrEmpty(result) ? "UNKNOWN" : result;
    }

    private static string Sha256Hex(byte[] bytes)
    {
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }

    private static string SafeErrorCode(Exception exception)
    {
        string message = exception.Message;
        if (
            (exception is InvalidOperationException
                || exception is OperationCanceledException)
            && message.Length is > 0 and <= 96
            && message.All(character =>
                character == '_'
                || char.IsAsciiLetterUpper(character)
                || char.IsDigit(character)
            )
        )
        {
            return message;
        }

        return exception.GetType().Name;
    }

    private static string? TryAssemblySha256()
    {
        try
        {
            string location = Assembly.GetExecutingAssembly().Location;
            if (string.IsNullOrWhiteSpace(location) || !File.Exists(location))
            {
                return null;
            }

            using FileStream stream = File.OpenRead(location);
            return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
        }
        catch
        {
            return null;
        }
    }

    private sealed class ExportSnapshot
    {
        public required string ExportId { get; init; }
        public required string OutputRoot { get; init; }
        public required string CreatedAt { get; init; }
        public required string DayUtc { get; init; }
        public required string FromInclusive { get; init; }
        public required string ToExclusive { get; init; }
        public required string SourceSymbol { get; init; }
        public required string StrategySymbol { get; init; }
        public required string ConnectorLabel { get; init; }
        public required string TickSize { get; init; }
        public required string MinVolumeAnalysisTickSize { get; init; }
        public required EnumSnapshot SymbolHistoryType { get; init; }
        public required EnumSnapshot SymbolVolumeType { get; init; }
        public required EnumSnapshot DeltaCalculationType { get; init; }
        public required string ChartAggregation { get; init; }
        public required bool AllowCalculateRealtimeTicks { get; init; }
        public required bool AllowCalculateRealtimeVolume { get; init; }
        public required bool AllowCalculateRealtimeTrades { get; init; }
        public required bool UnspecifiedTimesAssumedUtc { get; init; }
        public required string Classification { get; init; }
        public required ProbeEvidence Evidence { get; init; }
        public required string HistoryRequestStartedAt { get; init; }
        public required string VolumeAnalysisFinishedAt { get; init; }
        public required string SnapshotFinishedAt { get; init; }
        public required int ExpectedUtcSlots { get; init; }
        public required int BarsWithPriceLevels { get; init; }
        public required List<string> MissingBarOpens { get; init; }
        public required int NullPriceLevelBars { get; init; }
        public required int EmptyPriceLevelBars { get; init; }
        public required string ValidationStatus { get; init; }
        public required List<DiagnosticBar> Bars { get; init; }
    }

    private sealed class BarsDocument
    {
        public required string Schema { get; init; }
        public required int SchemaVersion { get; init; }
        public required string ExportId { get; init; }
        public required List<DiagnosticBar> Bars { get; init; }
    }

    private sealed class DiagnosticManifest
    {
        public required string Schema { get; init; }
        public required int SchemaVersion { get; init; }
        public required string ExportId { get; init; }
        public required string CreatedAt { get; init; }
        public required string DayUtc { get; init; }
        public required ExportInterval Interval { get; init; }
        public required string Timeframe { get; init; }
        public required SourceMetadata Source { get; init; }
        public required InstrumentMetadata Instrument { get; init; }
        public required CalculationMetadata Calculation { get; init; }
        public required SemanticsMetadata Semantics { get; init; }
        public required CausalityMetadata Causality { get; init; }
        public required CoverageMetadata Coverage { get; init; }
        public required List<FileMetadata> Files { get; init; }
        public required string ContentSha256 { get; init; }
    }

    private sealed class ExportInterval
    {
        public required string FromInclusive { get; init; }
        public required string ToExclusive { get; init; }
    }

    private sealed class SourceMetadata
    {
        public required string Platform { get; init; }
        public required string PlatformVersion { get; init; }
        public required string ConnectorVendor { get; init; }
        public required string ConnectorVendorSemantics { get; init; }
        public required string ExporterVersion { get; init; }
        public string? ExporterBinarySha256 { get; init; }
    }

    private sealed class InstrumentMetadata
    {
        public required string SourceSymbol { get; init; }
        public required string StrategySymbol { get; init; }
        public required string TickSize { get; init; }
        public required string MinVolumeAnalysisTickSize { get; init; }
        public required EnumSnapshot SymbolHistoryType { get; init; }
        public required EnumSnapshot SymbolVolumeType { get; init; }
        public required EnumSnapshot SymbolDeltaCalculationType { get; init; }
        public required bool AllowCalculateRealtimeTicks { get; init; }
        public required bool AllowCalculateRealtimeVolume { get; init; }
        public required bool AllowCalculateRealtimeTrades { get; init; }
    }

    private sealed class CalculationMetadata
    {
        public required string Period { get; init; }
        public required string ChartAggregation { get; init; }
        public required bool PriceLevelsRequested { get; init; }
        public required string VolumeBasis { get; init; }
        public required string Timezone { get; init; }
        public required bool UnspecifiedTimesAssumedUtc { get; init; }
    }

    private sealed class SemanticsMetadata
    {
        public required string FactorName { get; init; }
        public required string Classification { get; init; }
        public required string ClassificationStatus { get; init; }
        public required string BuyVolume { get; init; }
        public required string SellVolume { get; init; }
        public required string Trades { get; init; }
        public required bool ExchangeExecutionsProven { get; init; }
        public required bool AggressorPolarityProven { get; init; }
        public required ProbeEvidence Evidence { get; init; }
    }

    private sealed class CausalityMetadata
    {
        public required string HistoryRequestStartedAt { get; init; }
        public required string HistoryRequestStartSemantics { get; init; }
        public required string VolumeAnalysisFinishedAt { get; init; }
        public required string SnapshotFinishedAt { get; init; }
        public required string HistoricalAvailabilitySemantics { get; init; }
    }

    private sealed class CoverageMetadata
    {
        public required int ExpectedUtcSlots { get; init; }
        public required int BarsReturned { get; init; }
        public required int BarsWithPriceLevels { get; init; }
        public required List<string> MissingBarOpens { get; init; }
        public required int NullPriceLevelBars { get; init; }
        public required int EmptyPriceLevelBars { get; init; }
        public required string ValidationStatus { get; init; }
        public required bool EligibleForSidecarConversion { get; init; }
    }

    private sealed class FileMetadata
    {
        public required string Path { get; init; }
        public required long Size { get; init; }
        public required string Sha256 { get; init; }
        public required int Records { get; init; }
    }

    private sealed class DiagnosticBar
    {
        public required int SchemaVersion { get; init; }
        public required int Revision { get; init; }
        public required string SourceSymbol { get; init; }
        public required string Symbol { get; init; }
        public required string Timeframe { get; init; }
        public required string BarOpen { get; init; }
        public required string BarClose { get; init; }
        public required string CapturedAt { get; init; }
        public string? HistoricalAvailableAt { get; init; }
        public required string AvailabilitySemantics { get; init; }
        public required bool Finalized { get; init; }
        public required string FinalizationBasis { get; init; }
        public required Ohlc Ohlc { get; init; }
        public required string BarVolume { get; init; }
        public required long BarTicks { get; init; }
        public required VolumeFields Total { get; init; }
        public required List<PriceLevel> PriceLevels { get; init; }
        public required BarValidation Validation { get; init; }
    }

    private sealed class Ohlc
    {
        public required string Open { get; init; }
        public required string High { get; init; }
        public required string Low { get; init; }
        public required string Close { get; init; }
    }

    private class VolumeFields
    {
        public required string Volume { get; init; }
        public required int Trades { get; init; }
        public required string BuyVolume { get; init; }
        public required string SellVolume { get; init; }
        public required int BuyTrades { get; init; }
        public required int SellTrades { get; init; }
        public required string Delta { get; init; }
    }

    private sealed class PriceLevel : VolumeFields
    {
        public required string Price { get; init; }
        public required long PriceTicks { get; init; }
    }

    private sealed class BarValidation
    {
        public required string PriceLevelsStatus { get; init; }
        public required int LevelCount { get; init; }
        public required bool SumLevelsMatchesTotal { get; init; }
        public required string UnclassifiedVolume { get; init; }
        public required int UnclassifiedTrades { get; init; }
        public required VolumeFields LevelSums { get; init; }
    }

    private sealed class VolumeSums
    {
        public double Volume { get; set; }
        public int Trades { get; set; }
        public double BuyVolume { get; set; }
        public double SellVolume { get; set; }
        public int BuyTrades { get; set; }
        public int SellTrades { get; set; }
        public double Delta { get; set; }

        public VolumeFields ToFields()
        {
            return new VolumeFields
            {
                Volume = DecimalString(Volume),
                Trades = Trades,
                BuyVolume = DecimalString(BuyVolume),
                SellVolume = DecimalString(SellVolume),
                BuyTrades = BuyTrades,
                SellTrades = SellTrades,
                Delta = DecimalString(Delta),
            };
        }
    }

    private sealed class EnumSnapshot
    {
        public required string Name { get; init; }
        public required int NumericValue { get; init; }

        public static EnumSnapshot From<T>(T value)
            where T : struct, Enum
        {
            return new EnumSnapshot
            {
                Name = value.ToString(),
                NumericValue = Convert.ToInt32(value, CultureInfo.InvariantCulture),
            };
        }
    }

    private sealed class ProbeEvidence
    {
        public required HistoryProbe LastProbe { get; init; }
        public required HistoryProbe BidAskProbe { get; init; }

        public static ProbeEvidence Skipped()
        {
            return new ProbeEvidence
            {
                LastProbe = HistoryProbe.Skipped(HistoryType.Last),
                BidAskProbe = HistoryProbe.Skipped(HistoryType.BidAsk),
            };
        }
    }

    private sealed class HistoryProbe
    {
        public required string Status { get; set; }
        public required EnumSnapshot RequestedHistoryType { get; init; }
        public int Items { get; set; }
        public int TypedItems { get; set; }
        public int PositiveSizeItems { get; set; }
        public int AggressorBuy { get; set; }
        public int AggressorSell { get; set; }
        public int AggressorUnknown { get; set; }
        public string? ErrorType { get; set; }

        public static HistoryProbe For(HistoryType historyType)
        {
            return new HistoryProbe
            {
                Status = "PENDING",
                RequestedHistoryType = EnumSnapshot.From(historyType),
            };
        }

        public static HistoryProbe Skipped(HistoryType historyType)
        {
            return new HistoryProbe
            {
                Status = "SKIPPED",
                RequestedHistoryType = EnumSnapshot.From(historyType),
            };
        }
    }
}
