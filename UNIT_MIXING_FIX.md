# Critical Unit Mixing Bug Fix

## Issue: BTC vs Contracts Unit Confusion

The system was mixing **BTC** and **contracts** units when closing positions, causing incorrect rounding that resulted in zero-sized orders.

## Root Cause

### The Data Flow

1. **Blofin API** returns positions in **contracts**
   - Example: `6.0` contracts for BTC-USDT

2. **Blofin Client** (`_get_positions_impl`, line 349) converts to **BTC**
   ```python
   position_size_btc = abs(position_size_contracts) * btc_per_contract  # 6.0 * 0.001 = 0.006
   amount=position_size_btc,  # Store in BTC, not contracts
   ```
   - Position object stores: `amount = 0.006 BTC`

3. **close_all_positions** (OLD CODE - BUGGY) tried to round **BTC** by **contract** lot_size
   ```python
   lot_size = Decimal("0.1")  # This is in CONTRACTS
   rounded_amount = round_size(abs(position.amount), lot_size)
   # Rounds 0.006 BTC by 0.1 contracts → 0.0 BTC ❌
   ```

4. **Blofin Client** (`_place_order_impl`) converts back to **contracts**
   ```python
   contracts = float(order.amount) / btc_per_contract  # 0.0 / 0.001 = 0
   contracts_rounded = max(0.1, round(contracts / 0.1) * 0.1)  # 0.1 (minimum)
   ```

## The Bug

**Problem**: Rounding `0.006 BTC` by `0.1 contracts` is like rounding `$6` by `€10` - **incompatible units!**

**Result**:
- Position: 6.0 contracts (0.006 BTC)
- After rounding: 0.0 BTC
- Order placed: 0.1 contracts (minimum, not the actual position size!)

This would only close 0.1 contracts out of a 6.0 contract position, leaving 5.9 contracts open!

## The Solution

**Remove the rounding from `close_all_positions`** - let the exchange client handle it.

### Before (Buggy):
```python
# Get instrument info for lot_size
inst_info = await self.exchange_client.get_instrument_info(position.symbol)
lot_size = inst_info["lot_size"]  # 0.1 contracts

# Round BTC by contract lot_size (WRONG!)
rounded_amount = self.exchange_client.round_size(abs(position.amount), lot_size)
# 0.006 BTC rounded by 0.1 = 0.0 BTC ❌

close_order = Order(
    amount=rounded_amount,  # 0.0 BTC
    ...
)
```

### After (Fixed):
```python
# Use position amount directly (in BTC)
# The exchange client will handle BTC->contracts conversion and precision rounding
position_amount_btc = abs(position.amount)
logger.debug(f"Closing position: {position.symbol} {position_amount_btc} BTC")

close_order = Order(
    amount=position_amount_btc,  # 0.006 BTC ✅
    ...
)
```

The `blofin_client._place_order_impl` (lines 154-165) already handles:
1. BTC → contracts conversion
2. Rounding to lot_size (0.1 contracts)
3. Enforcing minimum size (0.1 contracts)

## Example Before/After

### Scenario: Close 6.0 contract position (0.006 BTC)

**Before (Buggy):**
1. Position: `amount = 0.006 BTC`
2. Fetch lot_size: `0.1 contracts`
3. Round: `round_size(0.006, 0.1) = 0.0 BTC` ❌
4. Create order: `amount = 0.0 BTC`
5. Blofin client: `0.0 / 0.001 = 0 contracts → bumped to 0.1`
6. **Result**: Only closes 0.1 contracts, leaves 5.9 open! ❌

**After (Fixed):**
1. Position: `amount = 0.006 BTC`
2. Create order: `amount = 0.006 BTC` ✅
3. Blofin client: `0.006 / 0.001 = 6.0 contracts`
4. Round: `round(6.0 / 0.1) * 0.1 = 6.0 contracts`
5. **Result**: Closes full 6.0 contract position! ✅

## Files Modified

**crypto_trading/core/trading_engine.py** (lines 459-482):
- Removed instrument info fetching (not needed for close)
- Removed lot_size rounding (handled by exchange client)
- Pass position amount directly in BTC
- Updated logging to show BTC amounts

## Why This Matters

1. **Correctness**: Positions are now fully closed instead of partially
2. **Simplicity**: Removed unnecessary API call to get instrument info
3. **Separation of Concerns**: Exchange client handles all unit conversions
4. **Single Responsibility**: close_all_positions just creates orders; exchange client handles exchange-specific details

## Related Issues

This fix also resolves:
- ✅ Positions appearing to close but remaining open
- ✅ "0 closed, 1 pending" instead of "1 closed, 0 failed"
- ✅ Unnecessary instrument info API calls

## Testing

After this fix, closing positions should show:
```
INFO | Closing 1 open positions...
DEBUG | Closing position: BTC-USDT 0.006 BTC
INFO | Close order placed: [id] - BTC-USDT sell 0.006 BTC (MARKET)
INFO | ✅ Position closed (filled): BTC-USDT - sell 0.006 BTC
INFO | Position closing complete: 1 closed, 0 failed
```

And verify in Blofin exchange that the full position is closed.
