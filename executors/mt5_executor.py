from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import MetaTrader5 as mt5


@dataclass
class MT5Settings:
    login: int
    password: str
    server: str
    risk_pct: float = 0.01
    magic: int = 20260318
    slippage: int = 20
    retry_sec: float = 0.3


class MT5Executor:
    """Wraps MetaTrader5 order operations + position sizing."""

    def __init__(self, settings: MT5Settings):
        self.settings = settings
        self.logger = logging.getLogger("MT5Executor")
        self._fill_mode_cache: Dict[str, int] = {}  # symbol → last working fill mode
        self._connect()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def execute_entry(self, symbol: str, *, side: str, entry_price: float, stop_price: float,
                      tp_price: Optional[float], comment: Optional[str] = None) -> Dict[str, Any]:
        # Use current tick price for volume calculation — signal price can diverge from real fill
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick data for {symbol} before entry")
        actual_entry = float(tick.ask if side.upper() == "LONG" else tick.bid)
        volume = self._calc_volume(symbol, actual_entry, stop_price)

        order_result = self._send_order(symbol, side, volume, entry_price, stop_price, tp_price, comment=comment)
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
            "volume": volume,
            "price": fill_price,   # actual broker fill price — not signal price
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

        results: List[Dict[str, Any]] = []
        for i, (vol, tp) in enumerate(zip(volumes_per_tp, tp_prices)):
            if vol <= 0:
                self.logger.warning(
                    "Split entry: skipping leg %d/%d — volume %.4f ≤ 0 (rounding remainder)",
                    i + 1, len(tp_prices), vol,
                )
                continue
            leg_comment = f"{(comment or 'Bot')[:20]} TP{i + 1}"
            order_result = self._send_order(
                symbol, side, vol, entry_price, stop_price, tp, comment=leg_comment
            )
            deal_ticket = order_result["deal"]
            position_id = self._find_position_id_from_deal(deal_ticket)
            if position_id is None:
                # In hedging mode we may have multiple positions — pick the newest one
                import time as _t
                _t.sleep(0.05)
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    own = [p for p in positions if p.magic == self.settings.magic]
                    already_known = {r["position_id"] for r in results if r.get("position_id")}
                    fresh = [p for p in own if p.ticket not in already_known]
                    if fresh:
                        position_id = fresh[-1].ticket
            results.append({
                "ticket": order_result["ticket"],
                "position_id": position_id,
                "volume": vol,
                "price": order_result["price"],
                "tp": tp,
                "tp_index": i + 1,
                "comment": leg_comment,
            })
            self.logger.info(
                "Split entry leg %d/%d: %s %s vol=%.2f tp=%.5f position_id=%s",
                i + 1, len(tp_prices), symbol, side, vol, tp, position_id,
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
        position = self._find_position(symbol, position_id)
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
            result = mt5.order_send(request)
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

        # Enforce broker minimum stop distance so we never send "Invalid stops".
        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        if tick is not None and info is not None:
            point = float(getattr(info, "point", 0.0) or getattr(info, "tick_size", 0.0) or 0.0)
            if point > 0:
                stops_level = int(getattr(info, "trade_stops_level", 0) or 0)
                spread_pts = int(getattr(info, "spread", 0) or 0)
                freeze_level = int(getattr(info, "trade_freeze_level", 0) or 0)
                min_gap = float(max(stops_level, spread_pts, freeze_level, 1) * point)
                if stops_level == 0 and spread_pts > 0:
                    min_gap = max(min_gap, float(spread_pts * 2 * point))
                is_buy = position.type == mt5.POSITION_TYPE_BUY
                ref_price = float(tick.bid if is_buy else tick.ask)
                if is_buy:
                    # SL must be below bid by at least min_gap
                    new_stop = min(new_stop, ref_price - min_gap)
                else:
                    # SL must be above ask by at least min_gap
                    new_stop = max(new_stop, ref_price + min_gap)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": position.ticket,
            "sl": new_stop,
            "tp": position.tp,
            "magic": self.settings.magic,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Failed to update SL for {symbol}: {result}")
        return True

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        if not mt5.initialize(login=self.settings.login,
                              password=self.settings.password,
                              server=self.settings.server):
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        self.logger.info("MT5 executor connected as %s@%s", self.settings.login, self.settings.server)

    def _calc_volume(self, symbol: str, entry_price: float, stop_price: float) -> float:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Unknown symbol {symbol} in MT5")
        if not info.visible:
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Cannot select symbol {symbol}")

        point = info.point or info.tick_size
        if point <= 0:
            raise RuntimeError(f"Bad point for {symbol}")
        stop_distance = abs(entry_price - stop_price)
        if stop_distance < point:
            stop_distance = point

        ticks = stop_distance / point
        tick_value = getattr(info, "tick_value", None)
        if not tick_value:
            contract_size = getattr(info, "trade_contract_size", 1.0)
            tick_value = contract_size * point
        tick_value = max(tick_value, 1e-9)
        risk_per_lot = max(ticks * tick_value, 1e-6)

        account = mt5.account_info()
        if account is None:
            raise RuntimeError("MT5 account_info unavailable")
        risk_amount = max(account.equity * max(self.settings.risk_pct, 0.0001), 1.0)

        volume = risk_amount / risk_per_lot
        # info.volume_max can be None or 0 on some brokers — guard against an
        # unbounded result that causes "Invalid volume" (e.g. 184467M lots).
        vol_max = float(info.volume_max) if (getattr(info, "volume_max", None) and info.volume_max > 0) else 500.0
        volume = max(info.volume_min or 0.01, min(volume, vol_max))
        step = info.volume_step or 0.01
        volume = math.floor(volume / step) * step
        if volume < info.volume_min:
            volume = info.volume_min

        # Cap by available free margin (never use more than 80% of free margin on one trade)
        try:
            order_type = mt5.ORDER_TYPE_BUY  # margin is direction-agnostic for most brokers
            margin_per_lot = mt5.order_calc_margin(order_type, symbol, 1.0, entry_price)
            if margin_per_lot and margin_per_lot > 0:
                max_vol_by_margin = math.floor((account.margin_free * 0.8) / margin_per_lot / step) * step
                if max_vol_by_margin > 0 and volume > max_vol_by_margin:
                    self.logger.warning(
                        "Volume capped by margin: %s lots → %s (free_margin=%.2f, margin_per_lot=%.2f)",
                        volume, max_vol_by_margin, account.margin_free, margin_per_lot,
                    )
                    volume = max_vol_by_margin
        except Exception:
            pass

        volume = max(info.volume_min or 0.01, volume)
        return round(volume, 2)

    def _send_order(self, symbol: str, side: str, volume: float,
                    planned_entry: float,
                    stop_price: Optional[float], tp_price: Optional[float], comment: Optional[str]) -> Dict[str, Any]:
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
                    # TP below current ask — mirror SL distance on the other side
                    risk = (price - stop_price) if stop_price else min_gap * 10
                    tp_price = price + max(risk, min_gap)
                else:
                    tp_price = max(tp_price, price + min_gap)
            else:
                if tp_price >= price:
                    risk = (stop_price - price) if stop_price else min_gap * 10
                    tp_price = price - max(risk, min_gap)
                else:
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
        for fill_mode in fill_modes:
            request["type_filling"] = fill_mode
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self._fill_mode_cache[symbol] = fill_mode
                return {
                    "ticket": int(result.order),
                    "price": float(result.price),   # actual broker fill price
                    "deal": int(result.deal),
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
        import time as _time
        from_ts = int(_time.time()) - 60
        to_ts = int(_time.time()) + 5
        try:
            deals = mt5.history_deals_get(from_ts, to_ts)
            if deals:
                for deal in deals:
                    if int(deal.ticket) == deal_ticket:
                        pos_id = int(getattr(deal, "position_id", 0) or 0)
                        if pos_id:
                            return pos_id
        except Exception:
            pass
        return None

    def get_current_price(self, symbol: str, side: str) -> Optional[float]:
        """Return current BID price for SL/TP monitoring.

        Always returns BID regardless of side — using ASK for SHORT monitoring causes
        false exits because the 30-pt GOLD spread makes ASK >> signal SL level.
        MT5 itself uses BID to evaluate SHORT SL orders, so BID is the correct reference.
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
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
