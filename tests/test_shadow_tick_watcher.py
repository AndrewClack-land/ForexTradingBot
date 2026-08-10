from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from core.shadow_tick_watcher import (
    MT5ReadOnlyFacade,
    QuoteTick,
    ShadowPlanSpec,
    ShadowTickWatcher,
    classify_quote,
    stable_plan_id,
)


UTC = timezone.utc
BASE = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)


def at(offset_msc: int) -> datetime:
    return BASE + timedelta(milliseconds=offset_msc)


def msc(offset_msc: int) -> int:
    return int(at(offset_msc).timestamp() * 1000)


class FakeFeed:
    def __init__(
        self,
        ticks: list[QuoteTick] | None = None,
        *,
        activation_quote: QuoteTick | None = None,
        history_error: Exception | None = None,
    ) -> None:
        self.ticks = list(ticks or [])
        self.activation_quote = activation_quote or QuoteTick(msc(0), 101.0, 101.1)
        self.history_error = history_error
        self.calls: list[tuple[str, int, int, int]] = []

    @staticmethod
    def _datetime_msc(value: datetime) -> int:
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return int(value.timestamp() * 1000)

    def symbol_info_tick(self, symbol: str) -> QuoteTick:
        del symbol
        return self.activation_quote

    def copy_ticks_range(
        self, symbol: str, start: datetime, end: datetime, flags: int
    ) -> list[QuoteTick]:
        start_msc = self._datetime_msc(start)
        end_msc = self._datetime_msc(end)
        self.calls.append((symbol, start_msc, end_msc, flags))
        if self.history_error is not None:
            raise self.history_error
        return [
            tick
            for tick in self.ticks
            if start_msc <= tick.time_msc <= end_msc
        ]


class FailFromFeed(FakeFeed):
    def __init__(self, fail_from_msc: int) -> None:
        super().__init__()
        self.fail_from_msc = int(fail_from_msc)

    def copy_ticks_range(
        self, symbol: str, start: datetime, end: datetime, flags: int
    ) -> list[QuoteTick]:
        start_msc = self._datetime_msc(start)
        end_msc = self._datetime_msc(end)
        if start_msc >= self.fail_from_msc:
            self.calls.append((symbol, start_msc, end_msc, flags))
            raise RuntimeError('later history range unavailable')
        return super().copy_ticks_range(symbol, start, end, flags)


def facade(feed: FakeFeed) -> MT5ReadOnlyFacade:
    return MT5ReadOnlyFacade(
        symbol_info_tick=feed.symbol_info_tick,
        copy_ticks_range=feed.copy_ticks_range,
        copy_ticks_info=17,
    )


def plan(
    plan_id: str = 'plan-1',
    *,
    side: str = 'LONG',
    activated_msc: int = 0,
    expires_msc: int = 5_000,
    source_symbol: str = 'EURUSD',
    auto_invalidate: bool = True,
) -> ShadowPlanSpec:
    is_long = side == 'LONG'
    return ShadowPlanSpec(
        plan_id=plan_id,
        symbol='EURUSD',
        source_symbol=source_symbol,
        side=side,
        entry_min=100.0,
        entry_max=101.0,
        planned_entry=101.0 if is_long else 100.0,
        stop_price=99.0 if is_long else 102.0,
        point=0.1,
        activated_at_utc=at(activated_msc),
        activation_tick_msc=msc(activated_msc),
        expires_at_utc=at(expires_msc),
        candidate_signature='H1_RB|bar',
        trigger_kind='RB_H1',
        setup_tf='H1',
        strategy_version='test',
        deployment_id='unit',
        metadata={'shadow_only': True},
        auto_invalidate=auto_invalidate,
    )


def test_quote_semantics_strict_tolerant_and_broker_stop() -> None:
    long_plan = plan()
    ask_outside = classify_quote(
        long_plan, QuoteTick(msc(10), bid=100.5, ask=101.2)
    )
    assert ask_outside.valid
    assert not ask_outside.strict_touch

    tolerant = classify_quote(
        long_plan, QuoteTick(msc(20), bid=99.7, ask=99.9)
    )
    assert not tolerant.strict_touch
    assert tolerant.tolerant_touch
    assert tolerant.tolerance == pytest.approx(0.2)

    broker_long = classify_quote(
        long_plan, QuoteTick(msc(30), bid=98.9, ask=99.1)
    )
    assert broker_long.broker_invalidated
    assert not broker_long.legacy_invalidated

    short_plan = plan('short', side='SHORT')
    bid_outside = classify_quote(
        short_plan, QuoteTick(msc(40), bid=99.8, ask=100.5)
    )
    assert not bid_outside.strict_touch
    broker_short = classify_quote(
        short_plan, QuoteTick(msc(50), bid=101.9, ask=102.1)
    )
    assert broker_short.broker_invalidated
    assert not broker_short.legacy_invalidated


def test_limit_gap_is_distinct_from_strict_zone() -> None:
    classification = classify_quote(
        plan(),
        QuoteTick(msc(100), bid=99.3, ask=99.5),
        previous_executable_price=102.0,
    )
    assert not classification.strict_touch
    assert classification.tolerant_touch is False
    assert classification.limit_threshold
    assert classification.limit_cross
    assert classification.gap_beyond_zone
    assert classification.zone_gap_cross


def test_causal_touch_missed_by_60s_and_expiry_is_exclusive(tmp_path) -> None:
    feed = FakeFeed(
        [
            QuoteTick(msc(-1), 100.3, 100.5),
            QuoteTick(msc(1_000), 100.3, 100.5),
            QuoteTick(msc(5_000), 100.4, 100.6),
        ]
    )
    watcher = ShadowTickWatcher(mt5=facade(feed), db_path=tmp_path / 'shadow.db')
    try:
        assert watcher.register_plan(plan())
        assert watcher.record_scan(
            'plan-1',
            QuoteTick(msc(4_000), 101.2, 101.4),
            observed_at_utc=at(4_000),
            candidate_matches=True,
            policy_executable=True,
            disposition='outside_entry_range',
        )
        watcher.poll_once(now_utc=at(6_000))

        row = watcher.get_plan('plan-1')
        assert row is not None
        assert row['finalized'] == 1
        assert row['status'] == 'EXPIRED'
        assert row['first_strict_touch_msc'] == msc(1_000)
        assert row['tick_count'] == 1
        assert row['missed_by_cadence'] == 1
        assert row['missed_by_policy'] == 1
        assert row['result_class'] == 'MISSED_BY_CADENCE'
        assert feed.calls[0][1] == msc(0)
        assert feed.calls[-1][2] == msc(5_000) - 1

        metrics = watcher.summary()
        assert metrics['plans'] == 1
        assert metrics['tick_touched'] == 1
        assert metrics['missed_by_cadence'] == 1
        assert metrics['miss_rate'] == pytest.approx(1.0)
    finally:
        watcher.close()


def test_scan_can_see_touch_while_policy_rejects_it(tmp_path) -> None:
    quote = QuoteTick(msc(1_000), 100.3, 100.5)
    feed = FakeFeed([quote])
    watcher = ShadowTickWatcher(mt5=facade(feed), db_path=tmp_path / 'shadow.db')
    try:
        assert watcher.register_plan(plan())
        assert watcher.record_scan(
            'plan-1',
            quote,
            observed_at_utc=at(1_000),
            candidate_matches=False,
            policy_executable=False,
            disposition='candidate_changed',
        )
        watcher.poll_once(now_utc=at(6_000))

        row = watcher.get_plan('plan-1')
        assert row is not None
        assert row['first_scan_touch_msc'] == msc(1_000)
        assert row['first_policy_touch_msc'] is None
        assert row['missed_by_cadence'] == 0
        assert row['missed_by_policy'] == 1
        assert row['result_class'] == 'SEEN_BY_60S'
    finally:
        watcher.close()


def test_raw_scan_is_promoted_to_eligible_on_the_same_minute_scan(tmp_path) -> None:
    quote = QuoteTick(msc(1_000), 100.3, 100.5)
    db_path = tmp_path / 'shadow.db'
    watcher = ShadowTickWatcher(mt5=facade(FakeFeed([quote])), db_path=db_path)
    try:
        assert watcher.register_plan(plan())
        assert watcher.record_scan(
            'plan-1',
            quote,
            observed_at_utc=at(1_000),
            candidate_matches=False,
            policy_executable=False,
            disposition='RAW_60S_SCAN',
        )
        assert watcher.record_scan(
            'plan-1',
            quote,
            observed_at_utc=at(1_000),
            candidate_matches=True,
            policy_executable=True,
            disposition='ELIGIBLE_60S_SCAN',
        )

        with sqlite3.connect(db_path) as connection:
            rows = connection.execute(
                """
                SELECT candidate_matches, policy_executable, disposition
                FROM scan_observations WHERE plan_id = ?
                """,
                ('plan-1',),
            ).fetchall()
        assert rows == [(1, 1, 'ELIGIBLE_60S_SCAN')]

        watcher.poll_once(now_utc=at(6_000))
        row = watcher.get_plan('plan-1')
        assert row is not None
        assert row['first_scan_touch_msc'] == msc(1_000)
        assert row['first_policy_touch_msc'] == msc(1_000)
        assert row['missed_by_cadence'] == 0
        assert row['missed_by_policy'] == 0
    finally:
        watcher.close()


def test_gap_and_stop_on_same_tick_is_not_limit_recoverable(tmp_path) -> None:
    quote = QuoteTick(msc(1_000), bid=98.5, ask=98.7)
    watcher = ShadowTickWatcher(
        mt5=facade(FakeFeed([quote])), db_path=tmp_path / 'shadow.db'
    )
    try:
        assert watcher.register_plan(plan())
        watcher.poll_once(now_utc=at(2_000))

        row = watcher.get_plan('plan-1')
        assert row is not None
        assert row['finalized'] == 1
        assert row['status'] == 'INVALIDATED'
        assert row['first_gap_msc'] == msc(1_000)
        assert row['first_limit_touch_msc'] == msc(1_000)
        assert row['first_broker_invalidated_msc'] == msc(1_000)
        assert row['recoverable_by_limit'] == 0
    finally:
        watcher.close()


def test_cancel_horizon_is_exclusive(tmp_path) -> None:
    terminal_tick = QuoteTick(msc(2_000), 100.3, 100.5)
    watcher = ShadowTickWatcher(
        mt5=facade(FakeFeed([terminal_tick])), db_path=tmp_path / 'shadow.db'
    )
    try:
        assert watcher.register_plan(plan())
        assert watcher.record_terminal(
            'plan-1',
            'CANCELLED',
            at_utc=at(2_000),
            tick_msc=msc(2_000),
            reason='live candidate disappeared',
            inclusive=False,
        )
        watcher.poll_once(now_utc=at(3_000))

        row = watcher.get_plan('plan-1')
        assert row is not None
        assert row['finalized'] == 1
        assert row['covered_through_msc'] == msc(2_000) - 1
        assert row['first_strict_touch_msc'] is None
    finally:
        watcher.close()


def test_restart_backfill_is_idempotent_and_preserves_distinct_same_ms_ticks(
    tmp_path,
) -> None:
    ticks = [
        QuoteTick(msc(1_000), 100.3, 100.5, 2),
        QuoteTick(msc(1_000), 100.2, 100.4, 1),
        QuoteTick(msc(1_000), 100.3, 100.5, 2),
    ]
    db_path = tmp_path / 'shadow.db'
    feed = FakeFeed(ticks)
    first = ShadowTickWatcher(mt5=facade(feed), db_path=db_path)
    try:
        assert first.register_plan(plan())
        first.poll_once(now_utc=at(2_000))
        row = first.get_plan('plan-1')
        assert row is not None
        assert row['tick_count'] == 2
        assert row['strict_touch_ticks'] == 2
    finally:
        first.close()

    second = ShadowTickWatcher(mt5=facade(feed), db_path=db_path)
    try:
        second.poll_once(now_utc=at(6_000))
        row = second.get_plan('plan-1')
        assert row is not None
        assert row['finalized'] == 1
        assert row['tick_count'] == 2
        assert row['strict_touch_ticks'] == 2
    finally:
        second.close()


def test_duplicate_registration_does_not_extend_ttl_and_db_uses_wal(tmp_path) -> None:
    db_path = tmp_path / 'shadow.db'
    watcher = ShadowTickWatcher(mt5=facade(FakeFeed()), db_path=db_path)
    try:
        assert watcher.register_plan(plan(expires_msc=5_000))
        assert not watcher.register_plan(plan(expires_msc=50_000))
        row = watcher.get_plan('plan-1')
        assert row is not None
        assert row['expires_tick_msc'] == msc(5_000)
        with sqlite3.connect(db_path) as connection:
            assert connection.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
    finally:
        watcher.close()


def test_history_is_grouped_per_symbol_and_idle_poll_does_no_mt5_reads(
    tmp_path,
) -> None:
    feed = FakeFeed()
    watcher = ShadowTickWatcher(mt5=facade(feed), db_path=tmp_path / 'group.db')
    try:
        assert watcher.register_plan(plan('a'))
        assert watcher.register_plan(plan('b'))
        watcher.poll_once(now_utc=at(1_000))
        assert len(feed.calls) == 1
        assert feed.calls[0][0] == 'EURUSD'
    finally:
        watcher.close()

    idle_feed = FakeFeed()
    idle = ShadowTickWatcher(mt5=facade(idle_feed), db_path=tmp_path / 'idle.db')
    try:
        idle.poll_once(now_utc=at(10_000))
        assert idle_feed.calls == []
    finally:
        idle.close()


def test_history_failure_finalizes_as_unknown_coverage(tmp_path) -> None:
    feed = FakeFeed(history_error=RuntimeError('history unavailable'))
    watcher = ShadowTickWatcher(mt5=facade(feed), db_path=tmp_path / 'shadow.db')
    try:
        assert watcher.register_plan(plan())
        watcher.poll_once(now_utc=at(6_000))

        row = watcher.get_plan('plan-1')
        assert row is not None
        assert row['finalized'] == 1
        assert row['data_complete'] == 0
        assert row['gap_count'] == 1
        assert row['result_class'] == 'UNKNOWN_COVERAGE'
        assert row['missed_by_cadence'] is None
        assert watcher.summary()['unknown_coverage'] == 1
    finally:
        watcher.close()


def test_fail_open_and_facade_has_no_order_capability(tmp_path) -> None:
    class DangerousModule:
        COPY_TICKS_INFO = 1

        @staticmethod
        def symbol_info_tick(symbol: str) -> QuoteTick:
            del symbol
            return QuoteTick(msc(0), 100.0, 100.1)

        @staticmethod
        def copy_ticks_range(
            symbol: str, start: datetime, end: datetime, flags: int
        ) -> list[QuoteTick]:
            del symbol, start, end, flags
            return []

        @staticmethod
        def order_send(*args: object) -> None:
            raise AssertionError(f'order_send must be unreachable: {args}')

    read_only = MT5ReadOnlyFacade.from_module(DangerousModule)
    assert not hasattr(read_only, 'order_send')
    assert not hasattr(read_only, '__dict__')

    def broken_connection(*args: object, **kwargs: object) -> sqlite3.Connection:
        del args, kwargs
        raise OSError('disk unavailable')

    watcher = ShadowTickWatcher(
        mt5=read_only,
        db_path=tmp_path / 'cannot-open.db',
        connection_factory=broken_connection,
    )
    assert not watcher.enabled
    assert watcher.disabled_reason is not None
    assert not watcher.register_plan(plan())
    watcher.poll_once(now_utc=at(6_000))
    assert watcher.summary()['plans'] == 0


def test_terminal_horizon_never_moves_later_and_earlier_fill_wins(
    tmp_path,
) -> None:
    watcher = ShadowTickWatcher(
        mt5=facade(FakeFeed()),
        db_path=tmp_path / 'terminal-ordering.db',
    )
    try:
        assert watcher.register_plan(plan(expires_msc=5_000))
        assert watcher.record_terminal(
            'plan-1',
            'EXPIRED',
            at_utc=at(5_000),
            tick_msc=msc(5_000),
            inclusive=False,
        )
        assert not watcher.record_terminal(
            'plan-1',
            'CANCELLED',
            at_utc=at(6_000),
            tick_msc=msc(6_000),
            inclusive=False,
        )
        row = watcher.get_plan('plan-1')
        assert row is not None
        assert row['status'] == 'EXPIRED'
        assert row['terminal_tick_msc'] == msc(5_000)

        assert watcher.record_terminal(
            'plan-1',
            'FILLED',
            at_utc=at(4_000),
            tick_msc=msc(4_000),
            inclusive=True,
        )
        row = watcher.get_plan('plan-1')
        assert row is not None
        assert row['status'] == 'FILLED'
        assert row['terminal_tick_msc'] == msc(4_000)
    finally:
        watcher.close()


@pytest.mark.parametrize(
    ('old_status', 'old_inclusive'),
    [
        ('EXPIRED', False),
        ('INVALIDATED', True),
        ('CANCELLED', False),
    ],
)
def test_late_earlier_fill_corrects_finalized_inferred_terminal_without_replay(
    tmp_path,
    old_status: str,
    old_inclusive: bool,
) -> None:
    quote_after_fill = QuoteTick(msc(3_500), 100.3, 100.5)
    db_path = tmp_path / f'late-fill-{old_status.lower()}.db'
    watcher = ShadowTickWatcher(
        mt5=facade(FakeFeed([quote_after_fill])),
        db_path=db_path,
    )
    try:
        assert watcher.register_plan(plan(expires_msc=5_000))
        assert watcher.record_scan(
            'plan-1',
            quote_after_fill,
            observed_at_utc=at(3_500),
            candidate_matches=True,
            policy_executable=True,
            disposition='ELIGIBLE_60S_SCAN',
        )
        assert watcher.record_terminal(
            'plan-1',
            old_status,
            at_utc=at(4_000),
            tick_msc=msc(4_000),
            inclusive=old_inclusive,
        )
        watcher.poll_once(now_utc=at(6_000))

        before = watcher.get_plan('plan-1')
        assert before is not None
        assert before['finalized'] == 1
        assert before['status'] == old_status
        assert before['first_strict_touch_msc'] == msc(3_500)
        assert before['first_scan_touch_msc'] == msc(3_500)
        assert before['first_policy_touch_msc'] == msc(3_500)

        with sqlite3.connect(db_path) as connection:
            batches_before = connection.execute(
                'SELECT COUNT(*) FROM tick_batches'
            ).fetchone()[0]
            strict_events_before = connection.execute(
                """
                SELECT COUNT(*) FROM events
                WHERE event_type = 'TOUCH_STRICT'
                """
            ).fetchone()[0]
            finalized_events_before = connection.execute(
                """
                SELECT COUNT(*) FROM events
                WHERE event_type = 'PLAN_FINALIZED'
                """
            ).fetchone()[0]
        assert batches_before == 1
        assert strict_events_before == 1
        assert finalized_events_before == 1

        assert not watcher.record_terminal(
            'plan-1',
            'FILLED',
            at_utc=at(4_000),
            tick_msc=msc(4_000),
            inclusive=True,
        )
        assert not watcher.record_terminal(
            'plan-1',
            'FILLED',
            at_utc=at(4_500),
            tick_msc=msc(4_500),
            inclusive=True,
        )
        assert watcher.record_terminal(
            'plan-1',
            'FILLED',
            at_utc=at(2_000),
            tick_msc=msc(2_000),
            reason='late authoritative broker deal',
            inclusive=True,
        )

        corrected = watcher.get_plan('plan-1')
        assert corrected is not None
        assert corrected['finalized'] == 1
        assert corrected['status'] == 'FILLED'
        assert corrected['terminal_tick_msc'] == msc(2_000)
        assert corrected['first_strict_touch_msc'] is None
        assert corrected['first_tolerant_touch_msc'] is None
        assert corrected['first_limit_touch_msc'] is None
        assert corrected['first_scan_touch_msc'] is None
        assert corrected['first_policy_touch_msc'] is None
        assert corrected['missed_by_cadence'] == 0
        assert corrected['missed_by_policy'] == 0
        assert corrected['recoverable_by_limit'] == 0
        assert corrected['result_class'] == 'NO_STRICT_TICK_TOUCH'

        with sqlite3.connect(db_path) as connection:
            batches_after = connection.execute(
                'SELECT COUNT(*) FROM tick_batches'
            ).fetchone()[0]
            strict_events_after = connection.execute(
                """
                SELECT COUNT(*) FROM events
                WHERE event_type = 'TOUCH_STRICT'
                """
            ).fetchone()[0]
            finalized_events_after = connection.execute(
                """
                SELECT COUNT(*) FROM events
                WHERE event_type = 'PLAN_FINALIZED'
                """
            ).fetchone()[0]
            correction_events = connection.execute(
                """
                SELECT COUNT(*) FROM events
                WHERE event_type = 'PLAN_CORRECTED_EARLIER_FILL'
                """
            ).fetchone()[0]
        assert batches_after == batches_before
        assert strict_events_after == strict_events_before
        assert finalized_events_after == 1
        assert correction_events == 1
        assert watcher.summary()['tick_touched'] == 0
    finally:
        watcher.close()


def test_earlier_fill_discards_only_coverage_gaps_after_corrected_horizon(
    tmp_path,
) -> None:
    feed = FailFromFeed(msc(2_000))
    watcher = ShadowTickWatcher(
        mt5=facade(feed),
        db_path=tmp_path / 'late-fill-gap.db',
        max_query_span_seconds=2.0,
    )
    try:
        assert watcher.register_plan(plan(expires_msc=5_000))
        watcher.poll_once(now_utc=at(6_000))
        before = watcher.get_plan('plan-1')
        assert before is not None
        assert before['finalized'] == 1
        assert before['status'] == 'EXPIRED'
        assert before['data_complete'] == 0
        assert before['result_class'] == 'UNKNOWN_COVERAGE'

        assert watcher.record_terminal(
            'plan-1',
            'FILLED',
            at_utc=at(1_500),
            tick_msc=msc(1_500),
            reason='broker deal predates failed history range',
            inclusive=True,
        )
        corrected = watcher.get_plan('plan-1')
        assert corrected is not None
        assert corrected['finalized'] == 1
        assert corrected['status'] == 'FILLED'
        assert corrected['data_complete'] == 1
        assert corrected['gap_count'] == 0
        assert corrected['result_class'] == 'NO_STRICT_TICK_TOUCH'
        assert watcher.summary()['unknown_coverage'] == 0

        with sqlite3.connect(watcher.db_path) as connection:
            assert connection.execute(
                'SELECT COUNT(*) FROM tick_batches'
            ).fetchone()[0] == 1
            assert connection.execute(
                """
                SELECT COUNT(*) FROM events
                WHERE event_type = 'DATA_GAP'
                """
            ).fetchone()[0] == 1
    finally:
        watcher.close()


def test_stable_plan_id_is_repeatable_and_bar_scoped() -> None:
    common = {
        'deployment_id': 'prod-abc',
        'symbol': 'eurusd',
        'candidate_signature': 'RB_H1|LONG|1.1000',
    }
    first = stable_plan_id(**common, decision_bar_close=at(0))
    assert first == stable_plan_id(**common, decision_bar_close=at(0))
    assert first != stable_plan_id(**common, decision_bar_close=at(3_600_000))
