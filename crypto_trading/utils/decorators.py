"""
Utility decorators for reducing code duplication.
Following DRY principle and SOLID design.
"""

import functools
import logging
from typing import Callable, TypeVar, Any, Optional
from loguru import logger
from ..core.exceptions import (
    TradingSystemError, ExchangeConnectionError, ExchangeAPIError,
    OrderError, RiskManagementError, AgentNotFoundError,
    AgentInitializationError, ConnectionError as TradingConnectionError
)


T = TypeVar('T')


def handle_trading_errors(operation: str):
    """
    Decorator to handle trading errors consistently.

    Args:
        operation: Description of the operation being performed
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except (ExchangeConnectionError, ExchangeAPIError) as e:
                logger.error(f"Exchange error in {operation}: {e}")
                raise
            except TradingSystemError as e:
                logger.error(f"Trading system error in {operation}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in {operation}: {e}", exc_info=True)
                raise TradingSystemError(f"{operation} failed: {e}") from e

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (ExchangeConnectionError, ExchangeAPIError) as e:
                logger.error(f"Exchange error in {operation}: {e}")
                raise
            except TradingSystemError as e:
                logger.error(f"Trading system error in {operation}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in {operation}: {e}", exc_info=True)
                raise TradingSystemError(f"{operation} failed: {e}") from e

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def requires_connection(func: Callable) -> Callable:
    """
    Decorator to ensure database connection before executing method.
    Assumes the class has an _ensure_connection() async method.
    """
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        if hasattr(self, '_ensure_connection'):
            await self._ensure_connection()
        return await func(self, *args, **kwargs)

    return wrapper


def log_execution(log_level: str = "debug"):
    """
    Decorator to log method execution.

    Args:
        log_level: Logging level (debug, info, warning, error)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_name = func.__name__
            logger_func = getattr(logger, log_level)
            logger_func(f"Executing {func_name}")
            try:
                result = await func(*args, **kwargs)
                logger_func(f"Completed {func_name}")
                return result
            except Exception as e:
                logger.error(f"Failed {func_name}: {e}")
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_name = func.__name__
            logger_func = getattr(logger, log_level)
            logger_func(f"Executing {func_name}")
            try:
                result = func(*args, **kwargs)
                logger_func(f"Completed {func_name}")
                return result
            except Exception as e:
                logger.error(f"Failed {func_name}: {e}")
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def validate_order(func: Callable) -> Callable:
    """
    Decorator to validate order before processing.
    Assumes the method receives an order parameter.
    """
    @functools.wraps(func)
    async def async_wrapper(self, order, *args, **kwargs):
        if not order:
            raise ValueError("Order cannot be None")

        if hasattr(order, 'symbol') and not order.symbol:
            raise ValueError("Order must have a symbol")

        if hasattr(order, 'quantity') and order.quantity <= 0:
            raise ValueError("Order quantity must be positive")

        return await func(self, order, *args, **kwargs)

    @functools.wraps(func)
    def sync_wrapper(self, order, *args, **kwargs):
        if not order:
            raise ValueError("Order cannot be None")

        if hasattr(order, 'symbol') and not order.symbol:
            raise ValueError("Order must have a symbol")

        if hasattr(order, 'quantity') and order.quantity <= 0:
            raise ValueError("Order quantity must be positive")

        return func(self, order, *args, **kwargs)

    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry failed operations.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            import asyncio
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed")

            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            import time
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed")

            raise last_exception

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def handle_errors(
    operation: str,
    return_default: Any = None,
    log_traceback: bool = False,
    raise_exception: bool = True
):
    """
    Generic error handler decorator for reducing try-except-log duplication.

    Args:
        operation: Description of the operation (e.g., "get market data", "place order")
        return_default: Default value to return on error (if not raising)
        log_traceback: Whether to log full traceback
        raise_exception: Whether to re-raise exceptions

    Usage:
        @handle_errors("fetching balance", return_default={})
        async def get_balance(self):
            # implementation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error {operation}: {e}", exc_info=log_traceback)
                if raise_exception:
                    raise
                return return_default

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error {operation}: {e}", exc_info=log_traceback)
                if raise_exception:
                    raise
                return return_default

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def handle_operation_failure(operation: str, return_default: Any = None):
    """
    Simplified error handler for "Failed to X" pattern.

    Args:
        operation: Operation name (e.g., "connect", "save data")
        return_default: Default return value on failure

    Usage:
        @handle_operation_failure("save order", return_default=False)
        async def save_order(self, order):
            # implementation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Failed to {operation}: {e}")
                if return_default is None:
                    raise
                return return_default

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Failed to {operation}: {e}")
                if return_default is None:
                    raise
                return return_default

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def suppress_errors(default_return: Any = None, log_error: bool = True):
    """
    Suppress all errors and return default value.
    Useful for non-critical operations.

    Args:
        default_return: Value to return on error
        log_error: Whether to log the error
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.warning(f"Suppressed error in {func.__name__}: {e}")
                return default_return

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.warning(f"Suppressed error in {func.__name__}: {e}")
                return default_return

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def validate_input(validator_func: Optional[Callable] = None, error_message: str = "Invalid input"):
    """
    Validate function inputs before execution.

    Args:
        validator_func: Function to validate inputs, receives (*args, **kwargs)
        error_message: Error message if validation fails
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if validator_func and not validator_func(*args, **kwargs):
                logger.warning(f"Validation failed for {func.__name__}: {error_message}")
                raise ValueError(error_message)
            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if validator_func and not validator_func(*args, **kwargs):
                logger.warning(f"Validation failed for {func.__name__}: {error_message}")
                raise ValueError(error_message)
            return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def ensure_initialized(func: Callable) -> Callable:
    """
    Ensure class is initialized before method execution.
    Assumes class has is_initialized or _is_initialized attribute.
    """
    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        is_init = getattr(self, 'is_initialized', None) or getattr(self, '_is_initialized', None)
        if not is_init:
            raise TradingSystemError(f"{self.__class__.__name__} is not initialized")
        return await func(self, *args, **kwargs)

    @functools.wraps(func)
    def sync_wrapper(self, *args, **kwargs):
        is_init = getattr(self, 'is_initialized', None) or getattr(self, '_is_initialized', None)
        if not is_init:
            raise TradingSystemError(f"{self.__class__.__name__} is not initialized")
        return func(self, *args, **kwargs)

    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
