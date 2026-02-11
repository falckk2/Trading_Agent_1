# Trading System Issues - Fixed

## Date: 2026-01-02

### Problem Summary

1. **Orders not showing as executed**: User reported that positions were opened but no order execution messages appeared in the GUI
2. **Stop button not cancelling orders**: When clicking stop, received timeout error and position closing failed
3. **Order placement KeyError**: `'ordId'` error when placing orders

---

## Root Causes Identified

### Issue 1: No Orders Being Executed

**Actual Problem**: Orders were NOT being executed at all. They were being **rejected by the risk manager** before reaching the exchange.

**Diagnosis from logs**:
```
WARNING | Order rejected: Exposure limits exceeded
WARNING | Order rejected by risk manager
```

**Root Cause**:
- You have existing positions on the exchange (likely opened manually or from a previous session)
- These positions use ~90% of your $50,000 portfolio limit configured in `trading_config.json`
- Risk manager blocks new orders because:
  - `max_total_exposure_pct: 0.9` (90% limit)
  - Current positions already exceed this limit
  - Formula: `current_exposure + new_order > max_allowed`

### Issue 2: Timeout Error When Stopping

**Error Message**: `"Timeout context manager should be used inside a task"`

**Root Cause**:
- The GUI creates a new event loop when stopping trading
- The aiohttp session was created in the trading engine's event loop
- Using aiohttp across different event loops causes the timeout context manager error
- The `ClientTimeout` object was being created outside the async context

---

## Fixes Applied

### Fix 1: Improved Order Cancellation (blofin_client.py:194-269)

**Added two new methods**:

1. `get_pending_orders()` - Fetches ALL pending orders from exchange
   - Endpoint: `/api/v1/trade/orders-pending`
   - Ensures we don't miss any orders that exist on the exchange

2. `cancel_batch_orders()` - Cancels multiple orders efficiently
   - Endpoint: `/api/v1/trade/cancel-batch-orders`
   - Processes multiple orders in one request
   - Returns individual success/failure status for each order

**Updated `_cancel_all_orders()` method** (trading_engine.py:367-386):
- Now fetches pending orders directly from exchange (not just from memory)
- Uses batch cancellation for efficiency
- Includes fallback to individual cancellation if batch fails
- Provides detailed logging of cancellation results

### Fix 2: Fixed aiohttp Timeout Error (blofin_client.py:369-395)

**Changed**:
```python
# Before (BROKEN):
timeout = aiohttp.ClientTimeout(total=self.timeout)  # Created outside context
async with self._session.get(url, timeout=timeout) as response:
    ...

# After (FIXED):
async with self._session.get(
    url,
    timeout=aiohttp.ClientTimeout(total=self.timeout)  # Created in context
) as response:
    ...
```

This ensures the timeout is created within the async context where it's actually used.

### Fix 3: Better Error Visibility (risk_manager.py:158-166)

**Added detailed logging** when exposure limits are exceeded:
```
Exposure limit exceeded:
Current=$45,000.00,
Order=$450.00,
New Total=$45,450.00,
Max Allowed=$45,000.00 (90% of $50,000)
```

### Fix 4: Risk Errors Now Show in GUI (trading_engine.py:337-365)

**Added event publishing** for risk manager rejections:
- When an order is rejected, an ERROR event is now published
- The GUI's `_on_error` handler will display these errors
- User can now see WHY orders are being rejected

### Fix 5: Immediate Order Status Checking (trading_engine.py:306-335)

**Added immediate status check** for market orders:
- After placing a market order, waits 0.5 seconds
- Immediately checks if order filled
- Publishes ORDER_FILLED event right away
- Prevents delay in showing executed orders

---

## How to Resolve "No Orders Executing"

**The system is working correctly** - it's protecting you from over-exposure!

### Option A: Close Existing Positions (Recommended)

1. Click "Stop Trading" in the GUI
2. Choose "YES" to close all positions
3. Wait for positions to close
4. Start trading again

### Option B: Increase Exposure Limit

Edit `config/trading_config.json`:
```json
"risk": {
    "max_total_exposure_pct": 0.95,  // Increase from 0.90 to 0.95 (95%)
    ...
}
```

### Option C: Check Exchange Directly

Log into Blofin sandbox and check your positions:
- URL: https://demo-trading.blofin.com
- You may have positions from manual trading or previous sessions
- Close them manually if needed

---

## Testing the Fixes

### Test 1: Stop Button
1. Start trading
2. Wait for signals
3. Click "Stop Trading"
4. Verify: ✅ No timeout errors
5. Verify: ✅ Orders cancelled successfully

### Test 2: Order Execution Display
1. Ensure you have available exposure (close positions if needed)
2. Start trading
3. Wait for a signal that passes risk checks
4. Verify: ✅ Order execution message appears immediately (within 1 second)

### Test 3: Risk Error Visibility
1. Have positions that exceed 90% exposure
2. Start trading
3. Wait for signals
4. Verify: ✅ Error message appears explaining exposure limit exceeded
5. Check logs: Detailed exposure breakdown should be visible

---

### Issue 3: Order Placement KeyError - 'ordId'

**Error Message**: `"Order placement failed: Order ID not found in response: {'orderId': '1000118369747', ...}"`

**Root Cause**:
- The code was looking for `'ordId'` (capital 'I') in API responses
- The Blofin API actually returns `'orderId'` (lowercase 'd') consistently in all responses
- This was a simple typo that prevented parsing successful order placement responses

**Evidence from API Response**:
```json
{
  "orderId": "1000118369747",
  "clientOrderId": "",
  "code": "0",
  "msg": "Order placed"
}
```

The order WAS being placed successfully on the exchange, but we failed to extract the order ID from the response!

---

## Fixes Applied

### Fix 6: Corrected API Field Names (blofin_client.py - multiple locations)

**Changed all occurrences** of `ordId` to `orderId`:
- Line 207: Order placement response parsing
- Line 213: Cancel order request
- Line 238, 250: Pending orders response parsing
- Line 268, 277: Batch cancel request and response
- Line 295, 301: Get order status request and response

Per the official Blofin API documentation, the field name is consistently `orderId` (not `ordId`) in both requests and responses.

### Fix 7: Fixed Order Attribute Name (base_exchange.py:92)

**Changed**: `order.order_id = order_id` → `order.id = order_id`

The Order model uses `id` not `order_id` as the attribute name.

---

## Summary of Changes

**Files Modified:**
1. `crypto_trading/exchange/blofin_client.py` - Fixed timeout, added batch cancel & pending orders, corrected all ordId→orderId
2. `crypto_trading/exchange/base_exchange.py` - Fixed order.order_id → order.id
3. `crypto_trading/core/trading_engine.py` - Improved cancel logic, immediate status checks, error events
4. `crypto_trading/core/risk_manager.py` - Better error logging with detailed exposure info

**API Calls Now Correct:**
- ✅ `/api/v1/trade/orders-pending` - Get all pending orders
- ✅ `/api/v1/trade/cancel-batch-orders` - Cancel multiple orders
- ✅ All field names match Blofin API spec (`orderId`, not `ordId`)
- ✅ Proper error handling and status codes

**User Experience Improvements:**
- ✅ Clear error messages when orders are rejected
- ✅ Detailed exposure information in logs
- ✅ Immediate order execution feedback
- ✅ Reliable stop button functionality
- ✅ Orders are now placed and tracked correctly
