# Close Positions AttributeError Fix

**Date**: 2026-01-02 23:30
**Issue**: Close position failed with `'MarketData' object has no attribute 'last_price'`

## Error from Logs

```
2026-01-02 23:27:20 | WARNING | Could not get market price, using market order: 'MarketData' object has no attribute 'last_price'
2026-01-02 23:27:21 | INFO | Order placed successfully with ID: 1000118400444
2026-01-02 23:27:21 | INFO | Close order placed: 1000118400444 - BTC-USDT sell 0.0691 (MARKET)
2026-01-02 23:27:22 | INFO | Position closing complete: 0 closed, 0 failed
```

## Root Causes

### Issue 1: Wrong Attribute Name
**File**: `crypto_trading/core/trading_engine.py:465`

**Before**:
```python
market_data = await self.exchange_client.get_market_data(position.symbol)
market_price = market_data.last_price  # ❌ AttributeError!
```

**Problem**: The `MarketData` dataclass doesn't have a `last_price` attribute.

**Actual MarketData structure** (from `crypto_trading/core/models.py:63-74`):
```python
@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal        # ✅ This is what we need!
    volume: Decimal
    bid: Optional[Decimal] = None    # ✅ Best for SELL orders
    ask: Optional[Decimal] = None    # ✅ Best for BUY orders
    spread: Optional[Decimal] = None
```

### Issue 2: Misleading Final Summary
**File**: `crypto_trading/core/trading_engine.py:526`

**Before**:
```python
logger.info(f"Position closing complete: {closed_count} closed, {failed_count} failed")
# Shows: "0 closed, 0 failed" - but doesn't mention pending orders!
```

**Problem**: Log doesn't explain that order was placed but not filled yet.

## Fixes Applied

### Fix 1: Use Correct Attributes
**File**: `crypto_trading/core/trading_engine.py:460-475`

**After**:
```python
try:
    market_data = await self.exchange_client.get_market_data(position.symbol)
    # Use bid/ask if available, otherwise use close price
    if order_side == OrderSide.SELL:
        # For sell: use bid price if available, otherwise close
        market_price = market_data.bid if market_data.bid else market_data.close
        # Sell at 0.1% below to ensure fill
        close_price = market_price * Decimal("0.999")
    else:
        # For buy: use ask price if available, otherwise close
        market_price = market_data.ask if market_data.ask else market_data.close
        # Buy at 0.1% above to ensure fill
        close_price = market_price * Decimal("1.001")
except Exception as e:
    logger.warning(f"Could not get market price, using market order: {e}")
    close_price = None
```

**Improvements**:
- ✅ Uses `market_data.close` (the closing price of the candle)
- ✅ Uses `market_data.bid` for SELL orders (best price to sell at)
- ✅ Uses `market_data.ask` for BUY orders (best price to buy at)
- ✅ Falls back to `close` if bid/ask not available
- ✅ No more AttributeError

### Fix 2: Better Summary Messages
**File**: `crypto_trading/core/trading_engine.py:519-536`

**After**:
```python
# Calculate pending orders (total - closed - failed)
pending_count = len(positions) - closed_count - failed_count

result = {
    'success': failed_count == 0,
    'closed_count': closed_count,
    'failed_count': failed_count,
    'pending_count': pending_count,  # NEW!
    'details': details
}

# Log comprehensive summary
if pending_count > 0:
    logger.info(f"Position closing complete: {closed_count} filled, {pending_count} pending, {failed_count} failed")
    logger.warning(f"⚠️  {pending_count} close order(s) placed but not filled yet - check Blofin exchange for order status")
    logger.warning("Pending orders may fill in the next few seconds/minutes. Check the 'details' for order IDs.")
else:
    logger.info(f"Position closing complete: {closed_count} closed, {failed_count} failed")
```

**Improvements**:
- ✅ Calculates and reports pending orders
- ✅ Explains that orders are placed but not filled
- ✅ Tells user to check Blofin exchange
- ✅ Adds `pending_count` to result dict
- ✅ Clear warning messages

## Expected Behavior After Fix

### Scenario 1: Order Fills Immediately

```
[23:30:15] INFO | Closing 1 open positions...
[23:30:15] INFO | Close order placed: 1000118400450 - BTC-USDT sell 0.0691 (LIMIT @ $90000.50)
[23:30:16] INFO | ✅ Position closed (filled): BTC-USDT - sell 0.0691
[23:30:16] INFO | Position closing complete: 1 closed, 0 failed
```

### Scenario 2: Order Placed But Not Filled (Your Case)

```
[23:30:15] INFO | Closing 1 open positions...
[23:30:15] INFO | Close order placed: 1000118400450 - BTC-USDT sell 0.0691 (LIMIT @ $90000.50)
[23:30:16] WARNING | Close order placed but not filled yet: 1000118400450 - status: live
[23:30:16] INFO | Position closing complete: 0 filled, 1 pending, 0 failed
[23:30:16] WARNING | ⚠️  1 close order(s) placed but not filled yet - check Blofin exchange for order status
[23:30:16] WARNING | Pending orders may fill in the next few seconds/minutes. Check the 'details' for order IDs.
```

### Scenario 3: Market Price Not Available (Fallback)

```
[23:30:15] INFO | Closing 1 open positions...
[23:30:15] WARNING | Could not get market price, using market order: Connection timeout
[23:30:15] INFO | Close order placed: 1000118400450 - BTC-USDT sell 0.0691 (MARKET)
[23:30:16] WARNING | Close order placed but not filled yet: 1000118400450 - status: live
[23:30:16] INFO | Position closing complete: 0 filled, 1 pending, 0 failed
[23:30:16] WARNING | ⚠️  1 close order(s) placed but not filled yet - check Blofin exchange for order status
```

## Why Order Didn't Fill in Your Case

Based on your logs:
1. ✅ Order was placed successfully (ID: 1000118400444)
2. ✅ Used MARKET order (because market price fetch failed)
3. ⏳ Order not filled after 0.5 seconds
4. ⏳ Position still open, close order pending

**Likely reason**: Blofin doesn't support true market orders for perpetual contracts. The "market" order was probably converted to a limit order at current price, which is now sitting in the order book waiting for a match.

## What to Do About Pending Order 1000118400444

### Check on Blofin Exchange

1. **Go to Blofin Demo Trading**
2. **Click "Orders" tab**
3. **Find order 1000118400444**:
   - Status "Filled"? → Position closed! ✅
   - Status "Live"? → Still waiting to fill ⏳
   - Status "Cancelled"? → Failed ❌
   - Not found? → May have filled and disappeared

4. **Check "Positions" tab**:
   - Position gone? → Order filled ✅
   - Position still there? → Order still pending ⏳

### Options

**Option 1: Wait** (Recommended)
- Order likely to fill in next 1-5 minutes
- Check Blofin in a few minutes
- Position should close automatically

**Option 2: Cancel and Retry**
- Go to Blofin → Orders tab
- Cancel order 1000118400444
- Use "Close All Positions" button again
- New order should get better price

**Option 3: Manual Close**
- Go to Blofin → Positions tab
- Click "Close" on the position
- Instant close at market price

## Bid/Ask vs Close Price Explanation

**For SELL orders** (closing long positions):
- **Best**: Use `bid` price (highest price buyers willing to pay)
- **Fallback**: Use `close` price (last traded price)
- **Why 0.1% below**: Ensures we're below best bid, guarantees fill

**For BUY orders** (closing short positions):
- **Best**: Use `ask` price (lowest price sellers willing to accept)
- **Fallback**: Use `close` price (last traded price)
- **Why 0.1% above**: Ensures we're above best ask, guarantees fill

**Example**:
```
Current market:
  Bid: $90,000 (buyers want to pay this)
  Ask: $90,010 (sellers want to receive this)
  Close: $90,005 (last trade price)

Closing LONG position (SELL):
  Use bid: $90,000
  Our limit: $90,000 * 0.999 = $89,910
  Result: Fills instantly (we're selling below market)

Closing SHORT position (BUY):
  Use ask: $90,010
  Our limit: $90,010 * 1.001 = $90,100
  Result: Fills instantly (we're buying above market)
```

## Testing

**Test 1: Normal close**
1. Open a position
2. Close all positions
3. ✅ Should see "filled" or "pending" status
4. ✅ No AttributeError
5. ✅ Clear log messages

**Test 2: Check pending order**
1. If order shows as pending
2. Wait 1-2 minutes
3. Check Blofin exchange
4. ✅ Order should fill
5. ✅ Position should close

## Success Criteria

After this fix:
- ✅ No more `'MarketData' object has no attribute 'last_price'` error
- ✅ Uses correct attributes: `bid`, `ask`, `close`
- ✅ Clear distinction between "filled" and "pending"
- ✅ Helpful warnings about pending orders
- ✅ Provides order IDs for manual verification
- ✅ Better price selection (bid/ask when available)
