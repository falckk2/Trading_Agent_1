# Risk Manager Debug Logging Enhancement

## Issue
Orders are being rejected by the risk manager with generic error:
```
WARNING | Signal execution skipped: Order rejected by risk manager
ERROR | Trading error: ⚠️ ERROR from order_execution: Order rejected by risk manager
```

The original logging didn't specify WHICH validation check was failing.

## Solution: Enhanced Logging

Added detailed logging to `crypto_trading/core/risk_manager.py` to identify the exact validation failure.

### Changes Made

#### 1. Main Validation Function (lines 59-94)
Added comprehensive logging at each validation step:
```python
logger.info(f"Validating order: {order.symbol} {order.side.value} {order.amount} @ ${order.price if order.price else 'MARKET'}")

# Each check now logs specific failure:
logger.warning("❌ VALIDATION FAILED: Order basics check")
logger.warning("❌ VALIDATION FAILED: Daily loss limit exceeded")
logger.warning(f"❌ VALIDATION FAILED: Position limits exceeded (current positions: {len(positions)})")
logger.warning("❌ VALIDATION FAILED: Exposure limits exceeded")
logger.warning("❌ VALIDATION FAILED: Order size limits")

# Success case:
logger.info(f"✅ Order validation PASSED for {order.symbol}")
```

#### 2. Basic Validation (lines 96-110)
Added parameter values to error messages:
```python
logger.warning(f"Order rejected: Invalid amount ({order.amount})")
logger.warning(f"Order rejected: Invalid price ({order.price})")
```

#### 3. Order Size Validation (lines 173-192)
Added detailed comparison of actual vs required values:
```python
logger.warning(
    f"Order size check failed: "
    f"Order value=${float(order_value):.2f} (min=${float(min_order_size):.2f}), "
    f"Order amount={float(order.amount):.6f} (min={float(min_order_amount):.6f})"
)
```

### Risk Manager Validation Checks

The risk manager validates orders through 5 checks:

1. **Order Basics** (`_validate_order_basics`)
   - Amount must be > 0
   - Price must be > 0 (if provided)
   - Symbol must exist

2. **Daily Loss Limit** (`_check_daily_loss_limit`)
   - Checks if today's losses < max_daily_loss_pct of portfolio_value
   - Config: `max_daily_loss_pct: 0.5` (50%)
   - Config: `portfolio_value: 50000.0`
   - Max daily loss: $25,000

3. **Position Limits** (`_check_position_limits`)
   - Max positions per symbol: `max_positions_per_symbol: 10`
   - Max total positions: `max_total_positions: 50`

4. **Exposure Limits** (`_check_exposure_limits`)
   - Total exposure < max_total_exposure_pct of portfolio_value
   - Config: `max_total_exposure_pct: 0.9` (90%)
   - Config: `portfolio_value: 50000.0`
   - Max exposure: $45,000
   - **This check already had detailed logging** (shows current/order/new/max exposure)

5. **Order Size Limits** (`_check_order_size_limits`)
   - Order value must be >= `min_order_size: 5.0` USD
   - Order amount must be >= `min_order_amount: 0.0001` contracts
   - **Now has detailed logging showing actual vs required**

## Common Rejection Reasons

### Most Likely Issues:

1. **Order Size Too Small**
   - Signal generates very small position size
   - Order value < $5.00 or amount < 0.0001 contracts
   - **Expected log**: "Order size check failed: Order value=$X.XX (min=$5.00), Order amount=0.XXXXXX (min=0.000100)"

2. **Exposure Limit Exceeded**
   - Already have open position(s) using up the exposure limit
   - New order would push total exposure > $45,000 (90% of $50,000)
   - **Expected log**: "Exposure limit exceeded: Current=$X, Order=$Y, New Total=$Z, Max Allowed=$45,000"

3. **Position Limits Exceeded**
   - Already have max positions open
   - **Expected log**: "❌ VALIDATION FAILED: Position limits exceeded (current positions: X)"

## Testing Instructions

1. **Clear any existing positions** (click "Close All Positions")
2. **Enable trading** (click "Start Trading")
3. **Wait for signal generation**
4. **Check the logs** for detailed validation output

### Expected Log Output (Success):
```
INFO | Validating order: BTC-USDT buy 0.005 @ $MARKET
INFO | ✅ Order validation PASSED for BTC-USDT
INFO | Order placed: [order-id] - BTC-USDT buy 0.005
```

### Expected Log Output (Failure):
```
INFO | Validating order: BTC-USDT buy 0.00005 @ $MARKET
WARNING | Order size check failed: Order value=$4.50 (min=$5.00), Order amount=0.000050 (min=0.000100)
WARNING | ❌ VALIDATION FAILED: Order size limits
WARNING | Signal execution skipped: Order rejected by risk manager
```

## Files Modified
- `crypto_trading/core/risk_manager.py` - Enhanced logging in validate_order(), _validate_order_basics(), _check_order_size_limits()

## Next Steps

After this logging enhancement, the next time an order is rejected, the logs will clearly show:
- The full order details (symbol, side, amount, price)
- Which specific validation check failed
- The actual values vs required values (for size checks)
- Current system state (for position/exposure checks)

This will allow us to quickly identify and fix the root cause.
