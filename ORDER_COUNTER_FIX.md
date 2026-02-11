# Order Counter Not Updating - Fix

**Date**: 2026-01-02 22:25
**Issue**: Order counter in GUI not incrementing when orders are placed

## Problem Analysis

### Root Cause
The GUI order counter (`order_count`) was only incrementing when ORDER_FILLED events were received. However:

1. **Removed immediate status check**: In an earlier fix, we removed the immediate order status check after placing orders because the API was unreliable for newly placed orders
2. **Delayed FILLED events**: ORDER_FILLED events only occur when the periodic account update detects filled orders (every few seconds)
3. **Missing ORDER_PLACED event**: There was no event published when an order was initially placed

**Result**: Orders were being placed successfully, but the GUI counter wouldn't update until much later (or at all if the periodic check didn't catch it).

### Evidence
- Orders appearing on Blofin exchange ✅
- Log messages showing "Order placed successfully" ✅
- GUI order counter staying at 0 ❌
- No immediate feedback to user ❌

## Solution Applied

### 1. Added ORDER_PLACED Event Type
**File**: `crypto_trading/core/interfaces.py:247-252`

```python
class EventType(Enum):
    ORDER_PLACED = "order_placed"      # NEW
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    SIGNAL_GENERATED = "signal_generated"
    ERROR_OCCURRED = "error_occurred"
```

### 2. Publish ORDER_PLACED Event After Order Execution
**File**: `crypto_trading/core/trading_engine.py:336-343`

```python
# Track the order
self.account_state_manager.add_active_order(placed_order)

# Publish ORDER_PLACED event so GUI can update immediately
await self.event_bus.publish(
    Event(
        type=EventType.ORDER_PLACED,
        data={"order": placed_order},
        timestamp=datetime.now()
    )
)
```

**When it fires**: Immediately after `execution_strategy.execute()` returns successfully

### 3. Subscribe to ORDER_PLACED in GUI
**File**: `crypto_trading/gui/trading_gui.py:117`

```python
# Subscribe to order events
self.event_bus.subscribe(EventType.ORDER_PLACED, self._on_order_placed)
self.event_bus.subscribe(EventType.ORDER_FILLED, self._on_order_filled)
self.event_bus.subscribe(EventType.ORDER_CANCELLED, self._on_order_cancelled)
```

### 4. Handle ORDER_PLACED Event
**File**: `crypto_trading/gui/trading_gui.py:170-203`

```python
async def _on_order_placed(self, event: Event):
    """Handle order placed event."""
    try:
        order = event.data.get('order')
        self.order_count += 1  # Increment counter immediately

        if order:
            action = order.side.value.upper()
            symbol = order.symbol
            amount = order.amount
            price = order.price if order.price else "MARKET"

            # Track order to agent mapping
            agent_name = order.metadata.get('agent_name', 'Unknown')
            if agent_name != 'Unknown':
                self.order_to_agent[order.id] = agent_name
                if agent_name not in self.agent_orders:
                    self.agent_orders[agent_name] = []
                self.agent_orders[agent_name].append(order.id)

            message = f"✅ ORDER #{self.order_count} PLACED: {action} {amount} {symbol} @ ${price}"
        else:
            message = f"✅ ORDER #{self.order_count} PLACED"

        # Update GUI on main thread
        if self.root:
            self.root.after(0, lambda: self._log_message(message, "order"))
            self.root.after(0, lambda: self.order_counter_label.config(text=str(self.order_count)))
            self.root.after(0, lambda: self._update_agent_status_display())

        logger.info(f"Order placed: {message}")
    except Exception as e:
        logger.error(f"Error handling order placed event: {e}")
```

### 5. Updated ORDER_FILLED Handler
**File**: `crypto_trading/gui/trading_gui.py:205-231`

**Changes**:
- Removed `self.order_count += 1` (already incremented in PLACED)
- Changed message from "ORDER #N" to "ORDER FILLED"
- Kept position updates (still important for filled orders)

```python
async def _on_order_filled(self, event: Event):
    """Handle order filled event."""
    try:
        order = event.data.get('order')
        # Don't increment counter - already done in _on_order_placed

        if order:
            action = order.side.value.upper()
            symbol = order.symbol
            amount = order.filled_amount if hasattr(order, 'filled_amount') else order.amount
            price = order.average_price if hasattr(order, 'average_price') and order.average_price else order.price

            message = f"✅ ORDER FILLED: {action} {amount} {symbol} @ ${price}"
        else:
            message = f"✅ ORDER FILLED"

        # Update GUI on main thread
        if self.root:
            self.root.after(0, lambda: self._log_message(message, "order"))
            # Update positions display since a new order was filled
            self.root.after(0, lambda: self._update_positions_display())
            self.root.after(0, lambda: self._update_agent_status_display())
```

## Expected Behavior After Fix

### Order Lifecycle Messages

**1. Signal Generated**
```
🔔 SIGNAL #1: RSI Agent → BUY BTC-USDT (confidence: 72.3%)
```

**2. Order Placed (IMMEDIATE)**
```
✅ ORDER #1 PLACED: BUY 0.025 BTC-USDT @ $89500
```
- Counter increments immediately
- User sees instant feedback
- Order ID tracked to agent

**3. Order Filled (LATER)**
```
✅ ORDER FILLED: BUY 0.025 BTC-USDT @ $89475
```
- Positions window updates
- Agent status updates
- No counter increment (already counted)

### GUI Updates

| Event | Counter | Activity Log | Position Window |
|-------|---------|-------------|-----------------|
| Signal Generated | No change | ✅ Shows signal | No change |
| Order Placed | +1 immediately | ✅ Shows "PLACED" | No change |
| Order Filled | No change | ✅ Shows "FILLED" | ✅ Updates |

## Benefits

1. ✅ **Immediate feedback**: User sees counter update as soon as order is placed
2. ✅ **Accurate count**: Counts all placed orders, not just filled ones
3. ✅ **Better UX**: No confusion about whether orders are being placed
4. ✅ **Dual notifications**: User sees both PLACED and FILLED messages
5. ✅ **Agent tracking**: Order-to-agent mapping happens immediately

## Testing

**Test 1: Normal order placement**
1. Start trading
2. Wait for signal
3. ✅ Counter should increment when "ORDER PLACED" appears
4. ✅ Counter should NOT increment again when "ORDER FILLED" appears

**Test 2: Multiple orders**
1. Run for 5 minutes
2. Let multiple orders place
3. ✅ Counter should increment for each order placed
4. ✅ Log should show "ORDER #1 PLACED", "ORDER #2 PLACED", etc.

**Test 3: Order rejection**
1. Let exposure hit limit
2. Signal generated but order rejected
3. ✅ Counter should NOT increment (no ORDER_PLACED event)
4. ✅ Error message shown instead

## What Changed

| Component | Before | After |
|-----------|--------|-------|
| EventType enum | 4 event types | 5 event types (added ORDER_PLACED) |
| Trading engine | Only tracked orders | Tracks + publishes ORDER_PLACED |
| GUI subscriptions | 2 order events | 3 order events (added ORDER_PLACED) |
| Order counter | Increments on FILLED | Increments on PLACED |
| User feedback | Delayed | Immediate |

## Backward Compatibility

✅ **Fully compatible**:
- ORDER_FILLED events still work normally
- Positions still update on FILLED
- Agent tracking works for both PLACED and FILLED
- No breaking changes to existing code

## Success Criteria

After this fix:
- ✅ Order counter increments immediately when order is placed
- ✅ User sees "ORDER #N PLACED" message instantly
- ✅ Counter doesn't double-increment when order fills
- ✅ All orders are counted (not just filled ones)
- ✅ System provides clear visual feedback
