# Blofin API Documentation Findings & Critical Fix

**Date**: 2026-01-02 23:45
**Issue**: System not using `reduceOnly` parameter when closing positions - **CRITICAL SAFETY ISSUE**

## API Documentation Research

### Sources Consulted
- [BloFin API guide](https://docs.blofin.com/index.html)
- [GitHub blofin-api-docs](https://github.com/blofin/blofin-api-docs)
- [Raw API documentation (index.md)](https://raw.githubusercontent.com/blofin/blofin-api-docs/main/index.md)

## Key Findings

### 1. No Dedicated Close-Position Endpoint

**Searched for**: `POST /api/v1/trade/close-position`
**Result**: **Does NOT exist** in the official API documentation

The API docs reference a "Close Positions" section in the table of contents, but the actual implementation uses the **regular order placement endpoint** with special parameters.

### 2. How to Properly Close Positions

According to the official Blofin API documentation:

**Endpoint**: `POST /api/v1/trade/order` (same as placing regular orders)

**Special Parameter**: `"reduceOnly": "true"`

**Documentation Quote**:
> *"When activated, orders can only reduce in position size. If the opposite order exceeds your position, the position will be fully closed, and no new position will be opened."*

### 3. Supported Order Types

The API supports these order types:
1. **`market`** - Execute immediately at current market price
2. **`limit`** - Execute at specified price or better
3. **`post_only`** - Limit orders that don't match existing orders
4. **`fok`** - Fill-or-Kill order (execute completely or cancel)
5. **`ioc`** - Immediate-or-Cancel order (fill available, cancel remainder)

**Market orders ARE supported** and do not require a price parameter.

### 4. The reduceOnly Parameter

**Purpose**: Safety mechanism to prevent accidentally opening new positions when closing

**Behavior**:
- ✅ If `reduceOnly: "true"` and order size = position size → Close position completely
- ✅ If `reduceOnly: "true"` and order size > position size → Close position completely, **cancel the excess**
- ❌ If `reduceOnly: "false"` (or not set) and order size > position size → Close position AND open new position in opposite direction

**Format**: String `"true"` (not boolean `true`)

## Critical Safety Issue Found

### The Problem

Our code was **NOT using `reduceOnly`** when closing positions!

**Old code** (crypto_trading/core/trading_engine.py):
```python
close_order = Order(
    id="",
    symbol=position.symbol,
    side=order_side,
    type=OrderType.MARKET,
    amount=abs(position.amount),
    price=None,
    status=OrderStatus.PENDING,
    timestamp=datetime.now()
    # ❌ NO reduce_only parameter!
)
```

### What Could Go Wrong

**Scenario**: You have a 0.069 BTC long position

**Closing order placed**: SELL 0.070 BTC (slightly larger due to rounding or calculation error)

**Without `reduceOnly`**:
1. Sells 0.069 BTC → Closes long position ✅
2. Sells additional 0.001 BTC → **Opens 0.001 BTC SHORT position** ❌

**With `reduceOnly`**:
1. Sells 0.069 BTC → Closes long position ✅
2. Excess 0.001 BTC → **Automatically cancelled** ✅
3. No new position opened ✅

### Real Risk

This could have resulted in:
- Accidentally opening opposite positions when trying to close
- Unexpected margin usage
- Unintended exposure
- Positions in the wrong direction
- Difficult-to-track trades

**This is a CRITICAL safety issue** that needed immediate fixing!

## Fixes Applied

### 1. Added `reduce_only` Field to Order Model
**File**: `crypto_trading/core/models.py:86-102`

```python
@dataclass
class Order:
    """Represents a trading order."""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    amount: Decimal
    price: Optional[Decimal]
    status: OrderStatus
    timestamp: datetime
    filled_amount: Decimal = Decimal('0')
    average_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    fees: Decimal = Decimal('0')
    reduce_only: bool = False  # NEW: If True, order can only reduce position size
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 2. Updated Blofin Client to Send reduceOnly
**File**: `crypto_trading/exchange/blofin_client.py:182-184`

```python
# Add reduceOnly parameter if set (for closing positions safely)
if order.reduce_only:
    order_data["reduceOnly"] = "true"  # Blofin expects string "true", not boolean
```

### 3. Set reduce_only=True in close_all_positions
**File**: `crypto_trading/core/trading_engine.py:477-489`

```python
# Create order to close position
# IMPORTANT: Set reduce_only=True to ensure we only close positions, not open new ones
close_order = Order(
    id="",
    symbol=position.symbol,
    side=order_side,
    type=OrderType.LIMIT if close_price else OrderType.MARKET,
    amount=abs(position.amount),
    price=close_price,
    status=OrderStatus.PENDING,
    timestamp=datetime.now(),
    reduce_only=True  # Only reduces position, won't open new position if size exceeds
)
```

## Expected API Request

**Before** (UNSAFE):
```json
{
  "instId": "BTC-USDT",
  "marginMode": "isolated",
  "positionSide": "net",
  "side": "sell",
  "orderType": "market",
  "size": "69.1"
}
```
**Risk**: If position is 69.0 contracts, this opens 0.1 contract SHORT!

**After** (SAFE):
```json
{
  "instId": "BTC-USDT",
  "marginMode": "isolated",
  "positionSide": "net",
  "side": "sell",
  "orderType": "market",
  "size": "69.1",
  "reduceOnly": "true"
}
```
**Safe**: Closes 69.0 contract position, cancels excess 0.1 contracts ✅

## Additional Findings

### Market Orders Work Fine

Despite earlier suspicions, **market orders ARE supported** by Blofin and should work correctly.

The previous issues with market orders were likely due to:
- Not using `reduceOnly` causing unexpected behavior
- Not checking fill status properly
- API timing issues (order placed but not filled immediately)

### Why Limit Orders Are Still Better

While market orders work, **limit orders are still preferred** for closing positions:

1. **Price control**: Know the worst price you'll get
2. **Slippage protection**: Won't get terrible fills in low liquidity
3. **Verification**: Can check if order filled vs pending
4. **Safety**: 0.1% away from market still fills quickly

**Current implementation**: Uses limit orders at market price ± 0.1% with `reduceOnly=true`

## Testing Recommendations

### Test Case 1: Exact Position Size
1. Open position: 0.05 BTC
2. Close with order size: 0.05 BTC
3. ✅ Should close completely with no new position

### Test Case 2: Slightly Larger Order Size
1. Open position: 0.05 BTC
2. Close with order size: 0.051 BTC (1% larger)
3. ✅ Should close 0.05 BTC, cancel 0.001 BTC excess
4. ✅ Should NOT open 0.001 BTC short position

### Test Case 3: Much Larger Order Size
1. Open position: 0.05 BTC
2. Close with order size: 0.10 BTC (2x larger)
3. ✅ Should close 0.05 BTC, cancel 0.05 BTC excess
4. ✅ Should NOT open 0.05 BTC short position

### Test Case 4: Multiple Positions
1. Open positions: BTC 0.05, ETH 0.10
2. Close all
3. ✅ Both should close completely
4. ✅ No new opposite positions

## Verification

### Check Blofin Exchange After Close

**Positions Tab**:
- ✅ Position should be gone
- ❌ Should NOT see new position in opposite direction

**Orders Tab**:
- ✅ Close order should show "Filled" or "Partially Filled"
- ✅ If order size was too large, excess should be cancelled
- ❌ Should NOT see new order opening opposite position

**Order Details**:
- Look for `"reduceOnly": "true"` in the order (if Blofin shows this)
- Check filled amount vs ordered amount
- If filled < ordered, the difference was cancelled (correct behavior)

## Impact Assessment

### Before Fix (HIGH RISK)
- ❌ Could accidentally open opposite positions
- ❌ No protection against over-selling/over-buying
- ❌ Unpredictable behavior with rounding errors
- ❌ Difficult to debug position issues
- ❌ Potential for significant unintended exposure

### After Fix (SAFE)
- ✅ Guaranteed to only close positions
- ✅ Protected against rounding/calculation errors
- ✅ Predictable and safe behavior
- ✅ Follows Blofin best practices
- ✅ No risk of accidental opposite positions

## Recommendations

### 1. Always Use reduceOnly for Closing

Whenever closing a position, **always set `reduce_only=True`**:

```python
close_order = Order(
    ...
    reduce_only=True  # CRITICAL for safety!
)
```

### 2. Consider reduceOnly for Risk Management

Even for regular trading, `reduceOnly` can be useful:
- Taking profits (reduce position size)
- Scaling out of positions
- Partial closes
- Stop-loss orders (only close, don't reverse)

### 3. Future Enhancement: Position-Aware Validation

Add validation to ensure close order size doesn't exceed position:

```python
if close_order.reduce_only:
    position = get_position(symbol)
    if close_order.amount > position.amount:
        logger.warning(f"Close order size ({close_order.amount}) exceeds position ({position.amount})")
        # Either:
        # A) Adjust order size to match position exactly
        close_order.amount = position.amount
        # B) Warn user but rely on reduceOnly to handle it
```

## Summary

### Critical Finding
The system was **NOT using `reduceOnly`** parameter when closing positions, creating a serious safety risk of accidentally opening opposite positions.

### Resolution
- ✅ Added `reduce_only` field to Order model
- ✅ Updated Blofin client to send `"reduceOnly": "true"` parameter
- ✅ Set `reduce_only=True` in close_all_positions function
- ✅ Now following Blofin API best practices
- ✅ Eliminated risk of accidental opposite positions

### Testing Required
**IMPORTANT**: Test the close positions functionality to verify:
1. Positions close completely
2. No opposite positions are opened
3. Order shows `reduceOnly: true` in Blofin
4. Behavior is safe and predictable

This was a **critical safety fix** that protects against unintended trading positions! 🛡️
