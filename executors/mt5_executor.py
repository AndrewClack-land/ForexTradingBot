from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

import MetaTrader5 as mt5


def _send_request(request: Dict[str, Any]):
    """order_send() compatible with both MetaTrader5 package generations.

    The 5.0.5xxx rewrite rejects the classic positional-dict call with
    (-2, 'Unnamed arguments not allowed') and wants named arguments; it also
    rejects None values and numpy integer scalars outright. Older packages
    (the local Windows install) only accept the positional dict. Sanitize the
    request, try positional first, fall back to named arguments on -2.
    """
    req = {}
    for key, value in request.items():
        if value is None:
            continue
        if type(value).__module__ == "numpy":  # np.float64 / np.int64 → native
            value = value.item()
        req[key] = value
    result = mt5.order_send(req)
    if result is None and mt5.last_error()[0] == -2:
        result = mt5.order_send(**req)
    return result


def _calc_profit_request(
    action: int,
    symbol: str,
    volume: float,
    price_open: float,
    price_close: float,
):
    """Call order_calc_profit across positional and named-only MT5 builds."""
    named_retry = False
    try:
        result = mt5.order_calc_profit(
            action,
            symbol,
            volume,
            price_open,
            price_close,
        )
    except TypeError:
        result = None
        named_retry = True
    if result is None:
        try:
            error = mt5.last_error()
        except Exception:
            error = None
        if named_retry or (error and error[0] == -2):
            result = mt5.order_calc_profit(
                action=action,
                symbol=symbol,
                volume=volume,
                price_open=price_open,
                price_close=price_close,
            )
    return result


@dataclass
class MT5Settings:
    login: int
    password: str
    server: str
    risk_pct: float = 0.01
    # Fixed risk base. When omitted/zero, it is captured once from account
    # balance and persisted at risk_state_path across bot/VPS restarts.
    initial_capital: float = 0.0
    risk_state_path: Optional[str] = None
    magic: int = 20260318
    slippage: int = 20
    retry_sec: float = 0.3
    # Sizing guards: hard cap on total lots per setup and round-turn
    # commission per lot included in the risk-per-lot calculation.
    max_volume: float = 10.0
    commission_per_lot: float = 7.0


@dataclass(frozen=True)
class _RiskLimit:
    capital_base: float
    budget_amount: float
    fraction: float
    margin_free: float
    account_currency: str


@dataclass(frozen=True)
class _RiskSizing:
    volume: float
    risk_amount: float
    risk_per_lot: float
    limit: _RiskLimit


class RiskLimitError(RuntimeError):
    """Entry rejected because its broker-side SL risk cannot fit the hard cap."""


class RiskCapacityError(RiskLimitError):
    """The signal remains valid but current entry geometry cannot fit min lot."""

    def __init__(
        self,
        message: str,
        *,
        target_entry: Optional[float] = None,
        required_capital: Optional[float] = None,
        minimum_volume: Optional[float] = None,
    ):
        super().__init__(message)
        self.target_entry = target_entry
        self.required_capital = required_capital
        self.minimum_volume = minimum_volume

    def to_payload(self) -> Dict[str, float]:
        payload: Dict[str, float] = {}
        if self.target_entry is not None:
            payload["target_entry"] = float(self.target_entry)
        if self.required_capital is not None:
            payload["required_initial_capital"] = float(self.required_capital)
        if self.minimum_volume is not None:
            payload["broker_min_volume"] = float(self.minimum_volume)
        return payload


# This is a code-level ceiling, independent of .env. A lower configured value
# remains valid, but no configuration can make a new setup risk more than 1%.
_HARD_MAX_STOP_RISK_PCT = 0.01


# Absolute floor (in price units) for the stop distance used in SIZING, and the
# expected exit slippage added on top. A 4-pip stop on EURUSD sized 67 lots on
# 2026-07-10; 3 pips of SL slippage then nearly doubled the planned loss.
# Volume is derived from max(actual stop, floor) + slippage — the real stop
# order is not modified.
_MIN_SIZING_STOP: Dict[str, float] = {
    "EURUSD": 0.0008,
    "GBPUSD": 0.0008,
    "USDCAD": 0.0008,
    "GOLD":   3.0,
    "XAUUSD": 3.0,
}
_EXPECTED_SLIPPAGE: Dict[str, float] = {
    "EURUSD": 0.0002,
    "GBPUSD": 0.0002,
    "USDCAD": 0.0002,
    "GOLD":   0.30,
    "XAUUSD": 0.30,
}


class MT5Executor:
    """Wraps MetaTrader5 order operations + position sizing."""

    def __init__(self, settings: MT5Settings):
        self.settings = settings
        self.logger = logging.getLogger("MT5Executor")
        self._fill_mode_cache: Dict[str, int] = {}  # symbol → last working fill mode
        self._initial_capital: Optional[float] = None
        self._connect()
        self._initial_capital = self._resolve_initial_capital()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @property
    def initial_capital(self) -> float:
        return self._resolve_initial_capital()

    def execute_entry(self, symbol: str, *, side: str, entry_price: float, stop_price: float,
                      tp_price: Optional[float], comment: Optional[str] = None,
                      entry_min: Optional[float] = None,
                      entry_max: Optional[float] = None) -> Dict[str, Any]:
        # Use current tick price for volume calculation — signal price can diverge from real fill
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick data for {symbol} before entry")
        actual_entry = float(tick.ask if side.upper() == "LONG" else tick.bid)
        volume = self._calc_volume(symbol, actual_entry, stop_price, side=side)

        order_result = self._send_order(
            symbol,
            side,
            volume,
            entry_price,
            stop_price,
            tp_price,
            comment=comment,
            entry_min=entry_min,
            entry_max=entry_max,
        )
        fill_price = order_result["price"]
        deal_ticket = order_result["deal"]

        # In MT5 hedging mode, order ticket ≠ position ticket. Resolve position_id from deal.
        position_id = self._find_position_id_from_deal(deal_ticket)
        if position_id is None:
            # Fallback: find any matching position for this symbol+magic
            pos = self._find_position(symbol, None)
            position_id = getattr(pos, "ticket", None)

        return {
            "ticket": order_result["ticket"],
            "position_id": position_id,
            "volume": order_result["volume"],
            "price": fill_price,   # actual broker fill price — not signal price
            "stop_price": order_result["stop_price"],
            "risk_amount": order_result["risk_amount"],
            "risk_budget_amount": order_result["risk_budget_amount"],
            "risk_capital_base": order_result["risk_capital_base"],
            "risk_pct": order_result["risk_pct"],
            "comment": comment or "Signal",
        }

    def execute_split_entry(
        self,
        symbol: str,
        *,
        side: str,
        entry_price: float,
        stop_price: float,
        tp_prices: List[float],
        volumes_per_tp: List[float],
        comment: Optional[str] = None,
        entry_min: Optional[float] = None,
        entry_max: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """MK-style partial TP: open N sub-positions, each with its own broker TP.

        Returns a list of order results (one per TP level).  The caller stores
        all position_ids so that MOVE_BE / full-close can iterate over them.

        Why:
          - Each sub-position closes automatically at its TP level — no bot polling needed.
          - Intermediate TPs hit at the exact price, not at the next bot tick.
          - Replicates Magic Keys EA behaviour from within Python.
        """
        if len(tp_prices) != len(volumes_per_tp):
            raise ValueError("tp_prices and volumes_per_tp must have the same length")

        info = mt5.symbol_info(symbol)
        vol_min = float(getattr(info, "volume_min", 0.0) or 0.0) if info is not None else 0.0
        if vol_min <= 0:
            vol_min = 0.01
        step = float(getattr(info, "volume_step", 0.0) or 0.0) if info is not None else 0.0
        if step <= 0:
            step = 0.01

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick data for {symbol} before split entry")
        actual_entry = float(tick.ask if side.upper() == "LONG" else tick.bid)
        risk_limit = self._risk_limit()

        # Convert the requested allocation to broker-step units and, if price
        # drift reduced the safe total since the caller sized it, scale the
        # whole setup down before any leg reaches order_send().
        raw_volumes = [max(0.0, float(value)) for value in volumes_per_tp]
        requested_total = sum(raw_volumes)
        if requested_total <= 0:
            return []
        preflight = self._size_volume_for_risk(
            symbol,
            side,
            actual_entry,
            stop_price,
            requested_volume=requested_total,
            risk_limit=risk_limit,
        )
        target_units = max(0, int(math.floor(preflight.volume / step + 1e-12)))
        exact_units = [
            target_units * value / requested_total
            for value in raw_volumes
        ]
        units = [int(math.floor(value + 1e-12)) for value in exact_units]
        remainder_units = target_units - sum(units)
        allocation_order = sorted(
            range(len(units)),
            key=lambda idx: (exact_units[idx] - units[idx], -idx),
            reverse=True,
        )
        for idx in allocation_order[:remainder_units]:
            units[idx] += 1
        normalized_volumes = [round(value * step, 8) for value in units]

        # Merge sub-minimum allocations toward the *nearest* target. Carrying
        # them forward would leave tiny setups with only TP3, which lowers the
        # chance of realizing any win. Reverse merging yields TP1-only when the
        # broker minimum permits just one leg.
        for idx in range(len(normalized_volumes) - 1, 0, -1):
            if 0 < normalized_volumes[idx] < vol_min:
                normalized_volumes[idx - 1] += normalized_volumes[idx]
                normalized_volumes[idx] = 0.0
        normalized_volumes = [round(value, 8) for value in normalized_volumes]

        results: List[Dict[str, Any]] = []

        def _rollback_opened_legs() -> None:
            for opened in results:
                pid = opened.get("position_id")
                try:
                    closed = self.close_trade(symbol, position_id=pid, volume=None)
                    if not closed:
                        raise RuntimeError("broker did not confirm close")
                    self.logger.warning(
                        "Split entry rollback: closed leg position_id=%s", pid,
                    )
                except Exception as rollback_exc:
                    self.logger.error(
                        "Split entry rollback FAILED for position %s: %s — orphaned leg!",
                        pid, rollback_exc,
                    )

        carry = 0.0  # volume from legs too small to open, merged into the next leg
        risk_used_amount = 0.0
        for i, (vol, tp) in enumerate(zip(normalized_volumes, tp_prices)):
            vol = round(vol + carry, 8)
            carry = 0.0
            if vol <= 0:
                self.logger.warning(
                    "Split entry: skipping leg %d/%d — volume %.4f ≤ 0 (rounding remainder)",
                    i + 1, len(tp_prices), vol,
                )
                continue
            if vol < vol_min:
                self.logger.warning(
                    "Split entry: leg %d/%d volume %.4f < volume_min %.2f — merging into next leg",
                    i + 1, len(tp_prices), vol, vol_min,
                )
                carry = vol
                continue
            leg_comment = f"{(comment or 'Bot')[:20]} TP{i + 1}"
            try:
                order_result = self._send_order(
                    symbol,
                    side,
                    vol,
                    entry_price,
                    stop_price,
                    tp,
                    comment=leg_comment,
                    entry_min=entry_min,
                    entry_max=entry_max,
                    risk_limit=risk_limit,
                    risk_used_amount=risk_used_amount,
                )
            except RiskLimitError:
                # The final fresh-tick guard can leave too little budget for one
                # more broker-minimum leg. Previously opened legs remain a valid,
                # tracked setup below the cap; stop adding exposure.
                if results:
                    self.logger.warning(
                        "Split entry stopped after %d/%d legs: remaining 1%% risk "
                        "budget cannot fit another minimum leg",
                        len(results), len(tp_prices),
                    )
                    break
                raise
            except Exception:
                # Roll back the legs opened so far — a half-opened split entry is not
                # registered in active_trades and would be orphaned in the market.
                _rollback_opened_legs()
                raise
            deal_ticket = order_result["deal"]
            position_id: Optional[int] = None
            # Deal/position visibility can lag order_send under Wine. Poll for a
            # bounded period, but never substitute the order ticket: in hedging
            # mode it is not a position id and would orphan lifecycle management.
            import time as _t
            for _ in range(10):
                position_id = self._find_position_id_from_deal(deal_ticket)
                if position_id is not None:
                    break
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    own = [p for p in positions if p.magic == self.settings.magic]
                    already_known = {r["position_id"] for r in results if r.get("position_id")}
                    fresh = [p for p in own if p.ticket not in already_known]
                    if fresh:
                        position_id = int(fresh[-1].ticket)
                        break
                _t.sleep(0.1)
            if position_id is None:
                _rollback_opened_legs()
                raise RuntimeError(
                    f"Split entry leg TP{i + 1} filled (deal={deal_ticket}) but exact "
                    "MT5 position_id was not published; refusing an untrackable setup"
                )
            results.append({
                "ticket": order_result["ticket"],
                "position_id": position_id,
                "volume": order_result["volume"],
                "price": order_result["price"],
                "stop_price": order_result["stop_price"],
                "risk_amount": order_result["risk_amount"],
                "risk_budget_amount": order_result["risk_budget_amount"],
                "risk_capital_base": order_result["risk_capital_base"],
                "risk_pct": order_result["risk_pct"],
                "tp": tp,
                "tp_index": i + 1,
                "comment": leg_comment,
            })
            risk_used_amount += float(order_result["risk_amount"])
            self.logger.info(
                "Split entry leg %d/%d: %s %s vol=%.4f tp=%.5f "
                "position_id=%s setup_risk=%.2f/%.2f %s",
                i + 1, len(tp_prices), symbol, side, order_result["volume"], tp,
                position_id, risk_used_amount, risk_limit.budget_amount,
                risk_limit.account_currency,
            )
        if carry > 0:
            self.logger.warning(
                "Split entry: %.4f lots left unallocated (below volume_min on the last leg)", carry,
            )
        return results

    def move_stop_all(self, symbol: str, *, position_ids: List[int], new_stop: float) -> int:
        """Move SL to new_stop on every position in position_ids. Returns count updated."""
        updated = 0
        for pid in position_ids:
            try:
                ok = self.move_stop(symbol, position_id=pid, new_stop=new_stop)
                if ok:
                    updated += 1
            except Exception as exc:
                self.logger.warning("move_stop_all: failed for position %s: %s", pid, exc)
        return updated

    def close_trade(self, symbol: str, *, position_id: Optional[int], volume: Optional[float]) -> bool:
        # A vanished split leg must never fall back to another position for the
        # same symbol: that could close TP2 while the caller intended the already
        # closed TP1 ticket. Symbol fallback is retained only for legacy calls
        # that do not supply a position id.
        position = self._find_position(symbol, position_id, strict=bool(position_id))
        if position is None:
            self.logger.warning("No MT5 position found for %s (ticket=%s)", symbol, position_id)
            return False
        # Cap volume at the actual position size to avoid "Invalid volume"
        vol = min(volume or position.volume, position.volume)
        if vol <= 0:
            self.logger.warning("Invalid volume when closing %s", symbol)
            return False
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick data for {symbol}")
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "position": position.ticket,
            "volume": vol,
            "type": order_type,
            "price": price,
            "deviation": self.settings.slippage,
            "magic": self.settings.magic,
            "comment": "Close by bot",
        }
        fill_modes = self._resolve_fill_modes(symbol)
        unsupported_code = getattr(mt5, "TRADE_RETCODE_INVALID_FILLING", 10030)
        last_result = None
        for fill_mode in fill_modes:
            request["type_filling"] = fill_mode
            result = _send_request(request)
            # order_send returns None (no result object) when the terminal is not
            # ready to take requests yet — e.g. right after a cold start.
            if result is None:
                raise RuntimeError(f"order_send close returned None: {mt5.last_error()}")
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self._fill_mode_cache[symbol] = fill_mode
                return True
            last_result = result
            if result.retcode != unsupported_code:
                break
        raise RuntimeError(f"order_send close failed: {last_result}")

    def move_stop(self, symbol: str, *, position_id: Optional[int], new_stop: float) -> bool:
        # strict=True: if this specific leg is already closed (TP hit), return False silently.
        position = self._find_position(symbol, position_id, strict=bool(position_id))
        if position is None:
            self.logger.info(
                "move_stop: position %s not found for %s (likely closed by TP — skipping)",
                position_id, symbol,
            )
            return False

        # Never weaken an existing stop. This also makes retries idempotent when
        # one leg was updated before another leg failed.
        current_stop = float(getattr(position, "sl", 0.0) or 0.0)
        is_buy = position.type == mt5.POSITION_TYPE_BUY

        # Require the requested BE level itself to be broker-valid. Clamping it
        # after a retrace would silently turn break-even into a loss and prevent
        # future retries once Core sets moved_to_be=True.
        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        if tick is not None and info is not None:
            point = float(getattr(info, "point", 0.0) or getattr(info, "tick_size", 0.0) or 0.0)
            if point > 0:
                epsilon = point * 0.5
                if current_stop > 0:
                    already_protected = (
                        current_stop >= new_stop - epsilon
                        if is_buy
                        else current_stop <= new_stop + epsilon
                    )
                    if already_protected:
                        return True
                stops_level = int(getattr(info, "trade_stops_level", 0) or 0)
                spread_pts = int(getattr(info, "spread", 0) or 0)
                freeze_level = int(getattr(info, "trade_freeze_level", 0) or 0)
                min_gap = float(max(stops_level, spread_pts, freeze_level, 1) * point)
                if stops_level == 0 and spread_pts > 0:
                    min_gap = max(min_gap, float(spread_pts * 2 * point))
                ref_price = float(tick.bid if is_buy else tick.ask)
                invalid_for_market = (
                    new_stop > ref_price - min_gap
                    if is_buy
                    else new_stop < ref_price + min_gap
                )
                if invalid_for_market:
                    self.logger.info(
                        "move_stop: requested BE %.5f is inside broker gap for %s "
                        "(reference %.5f, min_gap %.5f) — retrying later",
                        new_stop, symbol, ref_price, min_gap,
                    )
                    return False

        if current_stop > 0:
            would_weaken = current_stop >= new_stop if is_buy else current_stop <= new_stop
            if would_weaken:
                return True

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": position.ticket,
            "sl": new_stop,
            "tp": position.tp,
            "magic": self.settings.magic,
        }
        result = _send_request(request)
        if result is None:
            raise RuntimeError(
                f"Failed to update SL for {symbol}: order_send returned None ({mt5.last_error()})"
            )
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Failed to update SL for {symbol}: {result}")
        return True

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        # MT5_TERMINAL_PATH: explicit terminal64.exe location. Required where the
        # MetaTrader5 package cannot discover the terminal itself (e.g. Wine on a
        # Linux VPS); optional on a normal Windows install.
        term_path = os.getenv("MT5_TERMINAL_PATH", "").strip()
        kwargs = dict(
            login=self.settings.login,
            password=self.settings.password,
            server=self.settings.server,
            # A cold terminal start (fresh VPS boot) can exceed the default 60s IPC window
            timeout=120000,
        )
        ok = mt5.initialize(term_path, **kwargs) if term_path else mt5.initialize(**kwargs)
        if not ok:
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        self.logger.info("MT5 executor connected as %s@%s", self.settings.login, self.settings.server)

    def connection_alive(self) -> bool:
        """True when the terminal is reachable and the account is logged in.

        Used to distinguish "positions really closed" from "API call failed
        because the terminal link dropped" — positions_get() returns None in
        the latter case and must never be read as an empty position list.
        """
        try:
            return mt5.account_info() is not None
        except Exception:
            return False

    def reconnect(self) -> bool:
        """Re-initialize a dropped terminal connection. Returns True on success."""
        try:
            mt5.shutdown()
        except Exception:
            pass
        try:
            self._connect()
            return True
        except Exception as exc:
            self.logger.warning("MT5 reconnect failed: %s", exc)
            return False

    def get_position_close_info(self, position_id: Optional[int]) -> Optional[Dict[str, Any]]:
        """Return the latest broker exit deal for a position, or ``None``.

        Position disappearance and deal-history publication are not atomic in
        MT5. Callers therefore distinguish ``None`` (history not visible yet)
        from an exit whose reason is not TP/SL. The returned mapping contains
        only JSON-safe primitives so it can be copied into persisted trade state.
        """
        if not position_id:
            return None
        try:
            deals = mt5.history_deals_get(position=int(position_id))
        except Exception:
            return None
        if not deals:
            return None
        reason_map = {}
        for constant, label in (
            ("DEAL_REASON_CLIENT", "MANUAL"),
            ("DEAL_REASON_MOBILE", "MOBILE"),
            ("DEAL_REASON_WEB", "WEB"),
            ("DEAL_REASON_EXPERT", "EXPERT"),
            ("DEAL_REASON_SL", "SL"),
            ("DEAL_REASON_TP", "TP"),
            ("DEAL_REASON_SO", "STOP_OUT"),
            ("DEAL_REASON_ROLLOVER", "ROLLOVER"),
            ("DEAL_REASON_VMARGIN", "VMARGIN"),
            ("DEAL_REASON_SPLIT", "SPLIT"),
            ("DEAL_REASON_CORPORATE_ACTION", "CORPORATE_ACTION"),
        ):
            value = getattr(mt5, constant, None)
            if value is not None:
                reason_map[int(value)] = label

        exit_entries = {int(getattr(mt5, "DEAL_ENTRY_OUT", 1))}
        for constant in ("DEAL_ENTRY_OUT_BY", "DEAL_ENTRY_INOUT"):
            value = getattr(mt5, constant, None)
            if value is not None:
                exit_entries.add(int(value))

        exit_deals = [
            deal for deal in deals
            if int(getattr(deal, "entry", -1)) in exit_entries
        ]
        if not exit_deals:
            return None
        deal = max(
            exit_deals,
            key=lambda d: (
                int(getattr(d, "time_msc", 0) or 0),
                int(getattr(d, "time", 0) or 0),
                int(getattr(d, "ticket", 0) or 0),
            ),
        )
        reason_code = int(getattr(deal, "reason", -1))
        total_volume = sum(float(getattr(item, "volume", 0.0) or 0.0) for item in exit_deals)
        weighted_price = sum(
            float(getattr(item, "price", 0.0) or 0.0)
            * float(getattr(item, "volume", 0.0) or 0.0)
            for item in exit_deals
        )
        # Monetary result belongs to the whole position lifecycle. Brokers often
        # charge entry commission on DEAL_ENTRY_IN and exit commission/fee on
        # DEAL_ENTRY_OUT; summing only exit deals overstates every setup.
        profit = sum(float(getattr(item, "profit", 0.0) or 0.0) for item in deals)
        commission = sum(float(getattr(item, "commission", 0.0) or 0.0) for item in deals)
        swap = sum(float(getattr(item, "swap", 0.0) or 0.0) for item in deals)
        fee = sum(float(getattr(item, "fee", 0.0) or 0.0) for item in deals)
        return {
            "reason": reason_map.get(reason_code, "OTHER"),
            "reason_code": reason_code,
            "deal_ticket": int(getattr(deal, "ticket", 0) or 0),
            "price": (
                weighted_price / total_volume
                if total_volume > 0
                else float(getattr(deal, "price", 0.0) or 0.0)
            ),
            "volume": total_volume,
            "profit": profit,
            "commission": commission,
            "swap": swap,
            "fee": fee,
            "net": profit + commission + swap + fee,
            "time": int(getattr(deal, "time", 0) or 0),
            "time_msc": int(getattr(deal, "time_msc", 0) or 0),
        }

    def get_position_close_reason(self, position_id: Optional[int]) -> Optional[str]:
        """Backward-compatible TP/SL-only view of ``get_position_close_info``."""
        info = self.get_position_close_info(position_id)
        if info and info.get("reason") in {"TP", "SL"}:
            return str(info["reason"])
        return None

    def get_open_position_ids(self, symbol: str) -> Optional[set[int]]:
        """Return this strategy's open tickets, or ``None`` on an MT5 query failure."""
        try:
            positions = mt5.positions_get(symbol=symbol)
        except Exception:
            return None
        if positions is None:
            return None
        return {
            int(getattr(pos, "ticket", 0) or 0)
            for pos in positions
            if int(getattr(pos, "ticket", 0) or 0) > 0
            and getattr(pos, "magic", None) == self.settings.magic
        }

    def _resolve_initial_capital(self, account=None) -> float:
        cached = getattr(self, "_initial_capital", None)
        if cached is not None:
            cached = float(cached)
            if math.isfinite(cached) and cached > 0:
                return cached

        configured_raw = getattr(self.settings, "initial_capital", 0.0)
        try:
            configured = float(configured_raw or 0.0)
        except (TypeError, ValueError) as exc:
            raise RiskLimitError(
                f"Invalid MT5_INITIAL_CAPITAL {configured_raw!r}"
            ) from exc
        if not math.isfinite(configured) or configured < 0:
            raise RiskLimitError(
                f"Invalid MT5_INITIAL_CAPITAL {configured!r}"
            )
        if configured > 0:
            self._initial_capital = configured
            return configured

        account = account or mt5.account_info()
        if account is None:
            raise RiskLimitError(f"MT5 account_info unavailable: {mt5.last_error()}")
        try:
            default_login = int(getattr(self.settings, "login", 0) or 0)
            account_login = int(getattr(account, "login", default_login) or default_login)
            account_server = str(
                getattr(account, "server", None)
                or getattr(self.settings, "server", "")
                or ""
            )
            current_balance = float(account.balance)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RiskLimitError(f"Invalid MT5 account data: {exc}") from exc
        if not math.isfinite(current_balance) or current_balance <= 0:
            raise RiskLimitError(
                f"Cannot capture initial capital from balance={current_balance!r}"
            )

        state_path_raw = getattr(self.settings, "risk_state_path", None)
        state_path = Path(state_path_raw) if state_path_raw else None
        if state_path is not None and state_path.exists():
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
                stored_login = int(payload["account_login"])
                stored_server = str(payload.get("account_server") or "")
                stored_capital = float(payload["initial_capital"])
            except Exception as exc:
                raise RiskLimitError(
                    f"Invalid persisted initial-capital state at {state_path}: {exc}"
                ) from exc
            if stored_login == account_login and stored_server == account_server:
                if not math.isfinite(stored_capital) or stored_capital <= 0:
                    raise RiskLimitError(
                        f"Invalid persisted initial capital {stored_capital!r}"
                    )
                self._initial_capital = stored_capital
                return stored_capital
            self.logger.warning(
                "Risk-capital state belongs to %s@%s; recapturing for %s@%s",
                stored_login, stored_server, account_login, account_server,
            )

        captured = current_balance
        if state_path is not None:
            payload = {
                "account_login": account_login,
                "account_server": account_server,
                "account_currency": str(getattr(account, "currency", "") or ""),
                "initial_capital": captured,
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            try:
                state_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = state_path.with_name(f"{state_path.name}.tmp")
                tmp_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tmp_path.replace(state_path)
            except Exception as exc:
                raise RiskLimitError(
                    f"Cannot persist initial capital at {state_path}: {exc}"
                ) from exc

        self._initial_capital = captured
        self.logger.warning(
            "Initial risk capital fixed at %.2f %s",
            captured,
            str(getattr(account, "currency", "") or ""),
        )
        return captured

    def _risk_limit(self) -> _RiskLimit:
        account = mt5.account_info()
        if account is None:
            raise RiskLimitError(f"MT5 account_info unavailable: {mt5.last_error()}")

        try:
            balance = float(account.balance)
            equity = float(account.equity)
            margin_free = float(account.margin_free)
            configured_pct = float(self.settings.risk_pct)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RiskLimitError(f"Invalid MT5 account/risk data: {exc}") from exc

        values = (balance, equity, margin_free, configured_pct)
        if not all(math.isfinite(value) for value in values):
            raise RiskLimitError("Non-finite MT5 account/risk data")
        if balance <= 0 or equity <= 0:
            raise RiskLimitError(
                f"Cannot size risk with balance={balance:.2f}, equity={equity:.2f}"
            )
        if configured_pct <= 0:
            raise RiskLimitError(
                f"MT5_RISK_PCT must be positive, got {configured_pct!r}"
            )

        effective_pct = min(configured_pct, _HARD_MAX_STOP_RISK_PCT)
        # The risk base is fixed once at strategy start and persisted. Later
        # deposits, withdrawals, P/L, floating equity, and VPS restarts do not
        # silently resize the monetary risk budget.
        capital_base = self._resolve_initial_capital(account)
        budget_amount = capital_base * effective_pct
        if not math.isfinite(budget_amount) or budget_amount <= 0:
            raise RiskLimitError(f"Invalid stop-risk budget {budget_amount!r}")

        return _RiskLimit(
            capital_base=capital_base,
            budget_amount=budget_amount,
            fraction=effective_pct,
            margin_free=max(0.0, margin_free),
            account_currency=str(getattr(account, "currency", "") or ""),
        )

    def _get_symbol_info(self, symbol: str):
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Unknown symbol {symbol} in MT5")
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                raise RuntimeError(f"Cannot select symbol {symbol}")
            info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Cannot select symbol {symbol}")
        return info

    @staticmethod
    def _floor_volume(volume: float, step: float) -> float:
        if not math.isfinite(volume) or not math.isfinite(step) or step <= 0:
            return 0.0
        units = max(0, int(math.floor(volume / step + 1e-12)))
        return round(units * step, 8)

    @staticmethod
    def _broker_min_gap(info) -> float:
        point = float(
            getattr(info, "point", 0.0)
            or getattr(info, "trade_tick_size", 0.0)
            or getattr(info, "tick_size", 0.0)
            or 0.0
        )
        if not math.isfinite(point) or point <= 0:
            return 0.0
        stops_level = int(getattr(info, "trade_stops_level", 0) or 0)
        spread_points = int(getattr(info, "spread", 0) or 0)
        freeze_level = int(getattr(info, "trade_freeze_level", 0) or 0)
        min_gap = float(max(stops_level, spread_points, freeze_level, 1) * point)
        if stops_level == 0 and spread_points > 0:
            min_gap = max(min_gap, float(spread_points * 2 * point))
        return min_gap

    def _risk_per_lot(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_price: float,
        *,
        info=None,
    ) -> float:
        info = info or self._get_symbol_info(symbol)
        side_key = str(side).upper()
        if side_key not in {"LONG", "SHORT"}:
            raise RiskLimitError(f"Unsupported entry side {side!r}")
        try:
            entry = float(entry_price)
            stop = float(stop_price)
        except (TypeError, ValueError) as exc:
            raise RiskLimitError(f"Invalid entry/SL for {symbol}: {exc}") from exc
        if not math.isfinite(entry) or not math.isfinite(stop) or entry <= 0 or stop <= 0:
            raise RiskLimitError(
                f"Non-finite or non-positive entry/SL for {symbol}: {entry!r}/{stop!r}"
            )
        if side_key == "LONG" and stop >= entry:
            raise RiskLimitError(
                f"LONG SL must be below entry for {symbol}: entry={entry}, sl={stop}"
            )
        if side_key == "SHORT" and stop <= entry:
            raise RiskLimitError(
                f"SHORT SL must be above entry for {symbol}: entry={entry}, sl={stop}"
            )

        point = float(
            getattr(info, "point", 0.0)
            or getattr(info, "trade_tick_size", 0.0)
            or getattr(info, "tick_size", 0.0)
            or 0.0
        )
        if not math.isfinite(point) or point <= 0:
            raise RiskLimitError(f"Invalid point/tick size for {symbol}: {point!r}")

        # Size against the actual broker-side SL, an absolute micro-stop floor,
        # and expected adverse execution slippage. order_calc_profit converts
        # the loss to the account currency for FX, metals, and CFDs.
        stop_distance = abs(entry - stop)
        sym_key = symbol.upper()
        sizing_distance = max(
            stop_distance,
            point,
            _MIN_SIZING_STOP.get(sym_key, 0.0),
        )
        expected_stop_slippage = _EXPECTED_SLIPPAGE.get(sym_key, 0.0)
        try:
            entry_deviation_points = max(float(self.settings.slippage), 0.0)
        except (TypeError, ValueError) as exc:
            raise RiskLimitError(
                f"Invalid MT5 slippage/deviation {self.settings.slippage!r}"
            ) from exc
        if not math.isfinite(entry_deviation_points):
            raise RiskLimitError(
                f"Invalid MT5 slippage/deviation {entry_deviation_points!r}"
            )
        # Reserve both the maximum accepted adverse entry deviation and the
        # expected stop execution slippage. This keeps a normal within-deviation
        # fill from consuming the reserve intended for the eventual SL exit.
        sizing_distance += expected_stop_slippage + entry_deviation_points * point
        adverse_stop = (
            entry - sizing_distance
            if side_key == "LONG"
            else entry + sizing_distance
        )
        order_type = mt5.ORDER_TYPE_BUY if side_key == "LONG" else mt5.ORDER_TYPE_SELL
        estimated_profit = _calc_profit_request(
            order_type,
            symbol,
            1.0,
            entry,
            adverse_stop,
        )
        if estimated_profit is None:
            raise RiskLimitError(
                f"MT5 order_calc_profit failed for {symbol}: {mt5.last_error()}"
            )
        try:
            estimated_profit = float(estimated_profit)
        except (TypeError, ValueError) as exc:
            raise RiskLimitError(
                f"Invalid order_calc_profit result for {symbol}: {estimated_profit!r}"
            ) from exc
        if not math.isfinite(estimated_profit) or estimated_profit >= 0:
            raise RiskLimitError(
                f"Invalid stop-loss estimate for {symbol}: {estimated_profit!r}"
            )

        commission = max(float(self.settings.commission_per_lot), 0.0)
        if not math.isfinite(commission):
            raise RiskLimitError(f"Invalid commission_per_lot {commission!r}")
        risk_per_lot = -estimated_profit + commission
        if not math.isfinite(risk_per_lot) or risk_per_lot <= 0:
            raise RiskLimitError(
                f"Invalid risk per lot for {symbol}: {risk_per_lot!r}"
            )
        return risk_per_lot

    def _risk_compatible_entry(
        self,
        symbol: str,
        side: str,
        current_entry: float,
        stop_price: float,
        *,
        minimum_volume: float,
        available_risk: float,
        info,
    ) -> Optional[float]:
        """Best price at which broker minimum volume can fit the risk budget.

        The technical SL is never tightened. LONG waits for a lower entry closer
        to SL; SHORT waits for a higher entry closer to SL.
        """
        side_key = str(side).upper()
        entry = float(current_entry)
        stop = float(stop_price)
        point = float(
            getattr(info, "point", 0.0)
            or getattr(info, "trade_tick_size", 0.0)
            or getattr(info, "tick_size", 0.0)
            or 0.0
        )
        price_step = float(
            getattr(info, "trade_tick_size", 0.0)
            or getattr(info, "tick_size", 0.0)
            or point
        )
        if (
            side_key not in {"LONG", "SHORT"}
            or not all(
                math.isfinite(value)
                for value in (
                    entry,
                    stop,
                    point,
                    price_step,
                    minimum_volume,
                    available_risk,
                )
            )
            or point <= 0
            or price_step <= 0
            or minimum_volume <= 0
            or available_risk <= 0
        ):
            return None

        current_distance = abs(entry - stop)
        minimum_distance = max(price_step, self._broker_min_gap(info))
        if current_distance <= minimum_distance:
            return None

        def _entry_at(distance: float) -> float:
            return stop + distance if side_key == "LONG" else stop - distance

        closest_entry = _entry_at(minimum_distance)
        if closest_entry <= 0:
            return None
        closest_risk = (
            self._risk_per_lot(
                symbol,
                side_key,
                closest_entry,
                stop,
                info=info,
            )
            * minimum_volume
        )
        if closest_risk > available_risk + max(1e-8, available_risk * 1e-10):
            return None

        low = minimum_distance
        high = current_distance
        best = low
        for _ in range(48):
            middle = (low + high) / 2.0
            candidate_entry = _entry_at(middle)
            candidate_risk = (
                self._risk_per_lot(
                    symbol,
                    side_key,
                    candidate_entry,
                    stop,
                    info=info,
                )
                * minimum_volume
            )
            if candidate_risk <= available_risk:
                best = middle
                low = middle
            else:
                high = middle

        digits = int(getattr(info, "digits", 0) or 0)
        if digits <= 0:
            point_text = f"{price_step:.10f}".rstrip("0")
            digits = (
                len(point_text.split(".", 1)[1])
                if "." in point_text
                else 0
            )
        raw_target = _entry_at(best)
        if side_key == "LONG":
            aligned_target = (
                math.floor(raw_target / price_step + 1e-12) * price_step
            )
        else:
            aligned_target = (
                math.ceil(raw_target / price_step - 1e-12) * price_step
            )
        return round(aligned_target, digits)

    def _size_volume_for_risk(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_price: float,
        *,
        requested_volume: Optional[float] = None,
        risk_limit: Optional[_RiskLimit] = None,
        risk_used_amount: float = 0.0,
    ) -> _RiskSizing:
        info = self._get_symbol_info(symbol)
        limit = risk_limit or self._risk_limit()
        try:
            used = float(risk_used_amount)
        except (TypeError, ValueError) as exc:
            raise RiskLimitError(f"Invalid used risk {risk_used_amount!r}") from exc
        if not math.isfinite(used) or used < 0:
            raise RiskLimitError(f"Invalid used risk {used!r}")

        available_risk = limit.budget_amount - used
        tolerance = max(1e-8, limit.budget_amount * 1e-10)
        if available_risk <= tolerance:
            raise RiskLimitError(
                f"Stop-risk budget exhausted for {symbol}: "
                f"{used:.2f}/{limit.budget_amount:.2f} {limit.account_currency}"
            )

        risk_per_lot = self._risk_per_lot(
            symbol,
            side,
            entry_price,
            stop_price,
            info=info,
        )
        step = float(getattr(info, "volume_step", 0.0) or 0.0)
        vol_min = float(getattr(info, "volume_min", 0.0) or 0.0)
        vol_max = float(getattr(info, "volume_max", 0.0) or 0.0)
        if not all(math.isfinite(value) for value in (step, vol_min, vol_max)):
            raise RiskLimitError(f"Non-finite broker volume rules for {symbol}")
        if step <= 0 or vol_min <= 0 or vol_max <= 0 or vol_max < vol_min:
            raise RiskLimitError(
                f"Invalid broker volume rules for {symbol}: "
                f"min={vol_min}, max={vol_max}, step={step}"
            )
        try:
            configured_max_volume = float(self.settings.max_volume)
        except (TypeError, ValueError) as exc:
            raise RiskLimitError(
                f"Invalid configured max volume {self.settings.max_volume!r}"
            ) from exc
        if not math.isfinite(configured_max_volume):
            raise RiskLimitError(
                f"Invalid configured max volume {configured_max_volume!r}"
            )
        if configured_max_volume > 0:
            vol_max = min(vol_max, configured_max_volume)
        if vol_max < vol_min:
            raise RiskLimitError(
                f"Configured max volume {vol_max} is below broker minimum {vol_min}"
            )

        risk_capacity_volume = min(available_risk / risk_per_lot, vol_max)
        if risk_capacity_volume + 1e-12 < vol_min:
            min_lot_risk = vol_min * risk_per_lot
            target_entry = self._risk_compatible_entry(
                symbol,
                side,
                entry_price,
                stop_price,
                minimum_volume=vol_min,
                available_risk=available_risk,
                info=info,
            )
            required_capital = (
                (used + min_lot_risk) / limit.fraction
                if limit.fraction > 0
                else None
            )
            target_text = (
                f"; wait for entry near {target_entry:g}"
                if target_entry is not None
                else "; no broker-valid entry fits the current minimum lot"
            )
            raise RiskCapacityError(
                f"Risk-optimized entry pending for {symbol}: broker minimum "
                f"{vol_min:g} lot risks {min_lot_risk:.2f} "
                f"{limit.account_currency}, available {available_risk:.2f}"
                f"{target_text}",
                target_entry=target_entry,
                required_capital=required_capital,
                minimum_volume=vol_min,
            )

        volume = risk_capacity_volume
        if requested_volume is not None:
            try:
                requested = float(requested_volume)
            except (TypeError, ValueError) as exc:
                raise RiskLimitError(
                    f"Invalid requested volume {requested_volume!r}"
                ) from exc
            if not math.isfinite(requested) or requested <= 0:
                raise RiskLimitError(f"Invalid requested volume {requested!r}")
            volume = min(volume, requested)

        # Cap by available free margin (never use more than 80% on one setup).
        try:
            order_type = (
                mt5.ORDER_TYPE_BUY
                if str(side).upper() == "LONG"
                else mt5.ORDER_TYPE_SELL
            )
            margin_per_lot = mt5.order_calc_margin(order_type, symbol, 1.0, entry_price)
            if margin_per_lot is not None:
                margin_per_lot = float(margin_per_lot)
            if margin_per_lot and math.isfinite(margin_per_lot) and margin_per_lot > 0:
                max_vol_by_margin = self._floor_volume(
                    (limit.margin_free * 0.8) / margin_per_lot,
                    step,
                )
                if volume > max_vol_by_margin:
                    self.logger.warning(
                        "Volume capped by margin: %.8f lots → %.8f "
                        "(free_margin=%.2f, margin_per_lot=%.2f)",
                        volume, max_vol_by_margin, limit.margin_free, margin_per_lot,
                    )
                    volume = max_vol_by_margin
        except Exception:
            # Margin calculation is a separate broker check; the SL-risk guard
            # remains authoritative and order_send will reject insufficient margin.
            self.logger.warning("MT5 order_calc_margin failed for %s", symbol)

        volume = self._floor_volume(volume, step)
        if volume + 1e-12 < vol_min:
            raise RiskLimitError(
                f"Broker minimum {vol_min:g} lot for {symbol} cannot fit "
                "the requested allocation or available free margin"
            )

        risk_amount = volume * risk_per_lot
        if risk_amount > available_risk + tolerance:
            # Defensive loop for unusual floating-point/step combinations.
            volume = self._floor_volume(volume - step, step)
            risk_amount = volume * risk_per_lot
        if volume + 1e-12 < vol_min or risk_amount > available_risk + tolerance:
            raise RiskLimitError(
                f"Cannot fit broker-valid volume for {symbol} under "
                f"{available_risk:.2f} {limit.account_currency} stop-risk budget"
            )

        return _RiskSizing(
            volume=volume,
            risk_amount=risk_amount,
            risk_per_lot=risk_per_lot,
            limit=limit,
        )

    def _calc_volume(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        *,
        side: str = "LONG",
    ) -> float:
        sizing = self._size_volume_for_risk(
            symbol,
            side,
            entry_price,
            stop_price,
        )
        self.logger.info(
            "Risk sizing: %s %s volume=%.4f stop_risk=%.2f/%.2f %s (%.4f%%)",
            symbol,
            side,
            sizing.volume,
            sizing.risk_amount,
            sizing.limit.budget_amount,
            sizing.limit.account_currency,
            sizing.limit.fraction * 100.0,
        )
        return sizing.volume

    def _send_order(self, symbol: str, side: str, volume: float,
                    planned_entry: float,
                    stop_price: Optional[float], tp_price: Optional[float], comment: Optional[str],
                    *, entry_min: Optional[float] = None,
                    entry_max: Optional[float] = None,
                    risk_limit: Optional[_RiskLimit] = None,
                    risk_used_amount: float = 0.0) -> Dict[str, Any]:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick data for {symbol}")
        info = mt5.symbol_info(symbol)
        point = 0.0
        if info is not None:
            point = float(getattr(info, "point", 0.0) or getattr(info, "tick_size", 0.0) or 0.0)
        is_buy = side.upper() == "LONG"
        order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        price = tick.ask if is_buy else tick.bid
        current_spread = abs(float(tick.ask) - float(tick.bid))

        # The strategy calculates SL/TP from the worst edge of its entry zone.
        # Enforce that zone at execution time so a delayed market order cannot
        # silently turn a validated setup into a materially different trade.
        if entry_min is not None or entry_max is not None:
            lower = float(entry_min) if entry_min is not None else float("-inf")
            upper = float(entry_max) if entry_max is not None else float("inf")
            if lower > upper:
                raise RuntimeError(
                    f"Invalid entry range for {symbol}: {lower:.5f} > {upper:.5f}"
                )
            entry_tolerance = max(current_spread, point)
            if price < lower - entry_tolerance or price > upper + entry_tolerance:
                raise RuntimeError(
                    f"Stale signal rejected for {symbol}: execution price {price:.5f} "
                    f"outside entry range [{lower:.5f}, {upper:.5f}] "
                    f"(spread tolerance {entry_tolerance:.5f})"
                )
        # TP levels from signals are absolute technical levels — do NOT shift them
        # relative to the actual fill price (that was the root cause of TP drift).
        stops_level = 0
        spread_points = 0
        freeze_level = 0
        if info is not None:
            stops_level = int(getattr(info, "trade_stops_level", 0) or 0)
            spread_points = int(getattr(info, "spread", 0) or 0)
            freeze_level = int(getattr(info, "trade_freeze_level", 0) or 0)
        min_gap = float(max(stops_level, spread_points, freeze_level, 1) * point) if point > 0 else 0.0
        # When broker reports stops_level=0 but the actual minimum is the spread-based distance,
        # use at least 2× the current spread as a safety margin to avoid [Invalid stops].
        if point > 0 and stops_level == 0 and spread_points > 0:
            spread_gap = float(spread_points * 2 * point)
            min_gap = max(min_gap, spread_gap)

        # Save original signal SL before broker-distance clamping.
        # The stale check must use the unmodified strategy SL, not the clamped one,
        # otherwise `original_sl_distance` shrinks to min_gap and the ratio becomes
        # meaningless (always triggers).
        original_stop_price = stop_price

        if stop_price is not None:
            if is_buy:
                stop_price = min(stop_price, price - min_gap)
            else:
                stop_price = max(stop_price, price + min_gap)

        if tp_price is not None:
            if is_buy:
                if tp_price <= price:
                    raise RuntimeError(
                        f"Stale signal rejected for {symbol}: LONG target {tp_price:.5f} "
                        f"already reached by execution price {price:.5f}"
                    )
                tp_price = max(tp_price, price + min_gap)
            else:
                if tp_price >= price:
                    raise RuntimeError(
                        f"Stale signal rejected for {symbol}: SHORT target {tp_price:.5f} "
                        f"already reached by execution price {price:.5f}"
                    )
                tp_price = min(tp_price, price - min_gap)

        # Reject stale signals where price has moved too close to SL or already through it.
        # Use the ORIGINAL signal SL (pre-clamp) so the distance ratio reflects the true setup.
        if original_stop_price is not None and planned_entry is not None:
            # Hard reject: price has already crossed through the signal's SL level.
            # Entering here would result in an instant stop-out.
            if is_buy and price <= original_stop_price:
                raise RuntimeError(
                    f"Stale signal rejected for {symbol}: price {price:.5f} already at/below "
                    f"signal SL {original_stop_price:.5f} — setup invalidated"
                )
            if not is_buy and price >= original_stop_price:
                raise RuntimeError(
                    f"Stale signal rejected for {symbol}: price {price:.5f} already at/above "
                    f"signal SL {original_stop_price:.5f} — setup invalidated"
                )
            # Soft reject: price still above SL but remaining room < 30% of planned.
            original_sl_distance = abs(planned_entry - original_stop_price)
            current_sl_distance = abs(price - original_stop_price)
            if original_sl_distance > 0 and current_sl_distance < original_sl_distance * 0.3:
                raise RuntimeError(
                    f"Stale signal rejected for {symbol}: price moved to {price:.5f}, "
                    f"SL at {original_stop_price:.5f} (remaining dist={current_sl_distance:.5f} "
                    f"< 30% of planned {original_sl_distance:.5f})"
                )

        if stop_price is None:
            raise RiskLimitError(f"Entry rejected for {symbol}: broker-side SL is required")
        sizing = self._size_volume_for_risk(
            symbol,
            side,
            float(price),
            float(stop_price),
            requested_volume=volume,
            risk_limit=risk_limit,
            risk_used_amount=risk_used_amount,
        )
        if sizing.volume + 1e-12 < float(volume):
            self.logger.warning(
                "Final 1%% stop-risk guard reduced %s %s volume %.8f → %.8f",
                symbol, side, volume, sizing.volume,
            )
        volume = sizing.volume

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": stop_price,
            "tp": tp_price,
            "deviation": self.settings.slippage,
            "magic": self.settings.magic,
            "comment": comment or "Bot",
        }
        fill_modes = self._resolve_fill_modes(symbol)
        last_result = None
        unsupported_code = getattr(mt5, "TRADE_RETCODE_INVALID_FILLING", 10030)
        success_codes = {
            mt5.TRADE_RETCODE_DONE,
            getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010),
        }
        for fill_mode in fill_modes:
            request["type_filling"] = fill_mode
            result = _send_request(request)
            # order_send returns None (no result object) when the terminal is not
            # ready to take requests yet — e.g. right after a cold start.
            if result is None:
                raise RuntimeError(f"MT5 order_send returned None: {mt5.last_error()}")
            if result.retcode in success_codes:
                self._fill_mode_cache[symbol] = fill_mode
                filled_volume = float(getattr(result, "volume", 0.0) or volume)
                if not math.isfinite(filled_volume) or filled_volume <= 0:
                    filled_volume = volume
                if filled_volume > volume + 1e-8:
                    # A broker must not fill more than requested. Do not hide the
                    # anomaly: return it for tracking and surface it in logs.
                    self.logger.error(
                        "Broker overfill anomaly for %s: requested %.8f, filled %.8f",
                        symbol, volume, filled_volume,
                    )
                filled_risk = sizing.risk_per_lot * filled_volume
                return {
                    "ticket": int(result.order),
                    "price": float(result.price),   # actual broker fill price
                    "deal": int(result.deal),
                    "volume": filled_volume,
                    "stop_price": float(stop_price),
                    "risk_amount": filled_risk,
                    "risk_budget_amount": sizing.limit.budget_amount,
                    "risk_capital_base": sizing.limit.capital_base,
                    "risk_pct": sizing.limit.fraction,
                    "partial_fill": result.retcode != mt5.TRADE_RETCODE_DONE,
                }
            last_result = result
            if result.retcode != unsupported_code:
                break
        raise RuntimeError(f"MT5 order_send failed: {last_result or result}")

    def _resolve_fill_modes(self, symbol: str) -> List[int]:
        # If we already know the working mode for this symbol, try it first.
        cached = self._fill_mode_cache.get(symbol)

        info = mt5.symbol_info(symbol)
        modes: List[int] = []
        if info is not None:
            # fillings is a bitmask: bit 0 = FOK, bit 1 = IOC, bit 2 = RETURN
            fillings = int(getattr(info, "fillings", 0) or 0)
            for mode in (mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK):
                flag = 1 << int(mode)
                if fillings & flag:
                    modes.append(int(mode))
            # NOTE: filling_mode is also a bitmask — do NOT insert it directly as a mode value
        if not modes:
            # fallback: try all modes in preferred order (IOC before RETURN — broader broker support)
            modes = [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]
        else:
            # append remaining modes as fallbacks in case broker bitmask is wrong
            for fallback in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
                if fallback not in modes:
                    modes.append(fallback)

        # Move cached mode to front so the proven mode is tried first, avoiding
        # a guaranteed [Unsupported filling mode] error on every order.
        if cached is not None and cached in modes:
            modes = [cached] + [m for m in modes if m != cached]

        return modes

    def _find_position_id_from_deal(self, deal_ticket: int) -> Optional[int]:
        """Resolve position_id from a deal ticket (required in MT5 hedging mode)."""
        from datetime import datetime, timedelta
        # Deal timestamps are in BROKER SERVER time (often UTC+2/+3), so a narrow
        # UTC-based window misses freshly created deals. ±1 day covers any offset.
        now = datetime.now()
        try:
            deals = mt5.history_deals_get(now - timedelta(days=1), now + timedelta(days=1))
            if deals:
                for deal in deals:
                    if int(deal.ticket) == deal_ticket:
                        pos_id = int(getattr(deal, "position_id", 0) or 0)
                        if pos_id:
                            return pos_id
        except Exception:
            pass
        return None

    def get_current_price(self, symbol: str, side: Optional[str]) -> Optional[float]:
        """Return the price MT5 itself uses to evaluate the position's SL/TP.

        LONG (BUY) positions are closed by a SELL at BID → broker triggers on BID.
        SHORT (SELL) positions are closed by a BUY at ASK → broker triggers on ASK.
        Monitoring on the same side of the spread as the broker keeps the bot's
        SL/TP checks aligned with the actual fills (a BID-only check detects a
        SHORT stop-out one spread late and a SHORT TP one spread early).
        Pass side=None (no open trade) to get BID.
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        if side is not None and str(side).upper() == "SHORT":
            return float(tick.ask)
        return float(tick.bid)

    def get_position(self, symbol: str, position_id: Optional[int]) -> Optional[Any]:
        return self._find_position(symbol, position_id)

    def list_positions(self) -> List[Dict[str, Any]]:
        positions = mt5.positions_get()
        if not positions:
            return []
        out: List[Dict[str, Any]] = []
        for pos in positions:
            try:
                if getattr(pos, "magic", None) != self.settings.magic:
                    continue
                out.append(
                    {
                        "symbol": str(getattr(pos, "symbol", "")).upper(),
                        "ticket": int(getattr(pos, "ticket", 0) or 0),
                        "side": "LONG" if getattr(pos, "type", 0) == mt5.POSITION_TYPE_BUY else "SHORT",
                        "entry_price": float(getattr(pos, "price_open", 0.0) or 0.0),
                        "stop": float(getattr(pos, "sl", 0.0) or 0.0),
                        "tp": float(getattr(pos, "tp", 0.0) or 0.0),
                        "volume": float(getattr(pos, "volume", 0.0) or 0.0),
                        "comment": str(getattr(pos, "comment", "") or ""),
                        "time": float(getattr(pos, "time", 0) or 0),
                    }
                )
            except Exception:
                continue
        return out

    def _find_position(self, symbol: str, ticket: Optional[int], strict: bool = False) -> Optional[Any]:
        """Find a position by ticket or by symbol+magic.

        strict=True: when a ticket is supplied and not found (e.g. already closed by TP),
        return None immediately — do NOT fall back to the symbol search.
        This prevents accidentally operating on a different open leg.
        """
        if ticket:
            positions = mt5.positions_get(ticket=ticket)
            if positions:
                return positions[0]
            if strict:
                return None  # position closed — don't touch another leg
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return None
        for pos in positions:
            if pos.magic == self.settings.magic:
                return pos
        return positions[0]
