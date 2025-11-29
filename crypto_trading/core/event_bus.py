"""
Event bus implementation for system-wide event handling.
Supports asynchronous publish-subscribe pattern.
"""

import asyncio
from typing import Dict, List, Callable, Any
from collections import defaultdict
from datetime import datetime
from loguru import logger

from .interfaces import IEventBus, Event, EventType
from ..core.exceptions import TradingSystemError


class EventBus(IEventBus):
    """Asynchronous event bus implementation."""

    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._event_history: List[Event] = []
        self._max_history = 1000
        self._is_running = True

    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """Subscribe to an event type."""
        try:
            if callback not in self._subscribers[event_type]:
                self._subscribers[event_type].append(callback)
                logger.debug(f"Subscribed to {event_type.value}: {callback.__name__}")
            else:
                logger.warning(f"Callback already subscribed to {event_type.value}")

        except Exception as e:
            logger.error(f"Error subscribing to event {event_type.value}: {e}")

    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """Unsubscribe from an event type."""
        try:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                logger.debug(f"Unsubscribed from {event_type.value}: {callback.__name__}")
            else:
                logger.warning(f"Callback not found for {event_type.value}")

        except Exception as e:
            logger.error(f"Error unsubscribing from event {event_type.value}: {e}")

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        if not self._is_running:
            logger.warning("Event bus is not running, event ignored")
            return

        try:
            # Add to history
            self._add_to_history(event)

            # Get subscribers for this event type
            subscribers = self._subscribers.get(event.type, [])

            if not subscribers:
                logger.debug(f"No subscribers for event {event.type.value}")
                return

            # Notify all subscribers asynchronously
            tasks = []
            for callback in subscribers:
                task = self._safe_callback(callback, event)
                tasks.append(task)

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Check for exceptions and log them properly
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        callback_name = subscribers[i].__name__ if hasattr(subscribers[i], '__name__') else str(subscribers[i])
                        logger.error(f"Event callback '{callback_name}' failed for event {event.type.value}: {result}")
                        await self._handle_callback_error(result, callback_name, event)

            logger.debug(f"Published event {event.type.value} to {len(subscribers)} subscribers")

        except Exception as e:
            logger.error(f"Error publishing event {event.type.value}: {e}")

    async def _safe_callback(self, callback: Callable, event: Event) -> None:
        """Safely execute callback with error handling."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                # Run synchronous callback in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, callback, event)

        except Exception as e:
            logger.error(f"Error in event callback {callback.__name__}: {e}")
            raise  # Re-raise to be caught by gather

    async def _handle_callback_error(self, error: Exception, callback_name: str, event: Event) -> None:
        """Handle errors that occur in event callbacks."""
        try:
            # Publish error event if it's not already an error event to avoid loops
            if event.type != EventType.ERROR_OCCURRED:
                error_event = Event(
                    type=EventType.ERROR_OCCURRED,
                    data={
                        "error": str(error),
                        "source": f"event_callback_{callback_name}",
                        "original_event_type": event.type.value,
                        "callback": callback_name
                    },
                    timestamp=datetime.now()
                )
                # Add to history only, don't publish to avoid cascading errors
                self._add_to_history(error_event)

        except Exception as e:
            # If even error handling fails, just log it
            logger.critical(f"Failed to handle callback error: {e}")

    def _add_to_history(self, event: Event) -> None:
        """Add event to history with size management."""
        self._event_history.append(event)

        # Trim history if it gets too large
        if len(self._event_history) > self._max_history:
            # Remove oldest 10% of events
            trim_count = self._max_history // 10
            self._event_history = self._event_history[trim_count:]

    def get_event_history(
        self,
        event_type: EventType = None,
        since: datetime = None,
        limit: int = 100
    ) -> List[Event]:
        """Get event history with optional filtering."""
        try:
            events = self._event_history

            # Filter by event type
            if event_type:
                events = [e for e in events if e.type == event_type]

            # Filter by timestamp
            if since:
                events = [e for e in events if e.timestamp >= since]

            # Apply limit
            if limit and len(events) > limit:
                events = events[-limit:]

            return events

        except Exception as e:
            logger.error(f"Error getting event history: {e}")
            return []

    def get_subscriber_count(self, event_type: EventType = None) -> Dict[str, int]:
        """Get subscriber counts by event type."""
        if event_type:
            return {event_type.value: len(self._subscribers.get(event_type, []))}

        return {
            event_type.value: len(subscribers)
            for event_type, subscribers in self._subscribers.items()
        }

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
        logger.info("Event history cleared")

    def stop(self) -> None:
        """Stop the event bus."""
        self._is_running = False
        logger.info("Event bus stopped")

    def start(self) -> None:
        """Start the event bus."""
        self._is_running = True
        logger.info("Event bus started")

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        total_subscribers = sum(len(subs) for subs in self._subscribers.values())

        # Count events by type in history
        event_counts = defaultdict(int)
        for event in self._event_history:
            event_counts[event.type.value] += 1

        return {
            "is_running": self._is_running,
            "total_subscribers": total_subscribers,
            "subscribers_by_type": self.get_subscriber_count(),
            "event_history_size": len(self._event_history),
            "event_counts": dict(event_counts),
            "max_history_size": self._max_history
        }


class EventPublisher:
    """Helper class for publishing common trading events."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

    async def publish_order_filled(self, order: Any) -> None:
        """Publish order filled event."""
        event = Event(
            type=EventType.ORDER_FILLED,
            data={"order": order},
            timestamp=datetime.now()
        )
        await self.event_bus.publish(event)

    async def publish_order_cancelled(self, order: Any) -> None:
        """Publish order cancelled event."""
        event = Event(
            type=EventType.ORDER_CANCELLED,
            data={"order": order},
            timestamp=datetime.now()
        )
        await self.event_bus.publish(event)

    async def publish_signal_generated(self, signal: Any, agent_name: str) -> None:
        """Publish signal generated event."""
        event = Event(
            type=EventType.SIGNAL_GENERATED,
            data={"signal": signal, "agent": agent_name},
            timestamp=datetime.now()
        )
        await self.event_bus.publish(event)

    async def publish_error(self, error: str, source: str, details: Dict[str, Any] = None) -> None:
        """Publish error event."""
        event_data = {
            "error": error,
            "source": source
        }
        if details:
            event_data.update(details)

        event = Event(
            type=EventType.ERROR_OCCURRED,
            data=event_data,
            timestamp=datetime.now()
        )
        await self.event_bus.publish(event)

    async def publish_custom_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish custom event (extends beyond defined event types)."""
        # For custom events, we can use ERROR_OCCURRED as a generic type
        # and include the actual type in the data
        event = Event(
            type=EventType.ERROR_OCCURRED,  # Using as generic type
            data={"custom_event_type": event_type, **data},
            timestamp=datetime.now()
        )
        await self.event_bus.publish(event)


class EventSubscriber:
    """Helper class for subscribing to trading events."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._subscriptions: List[tuple] = []  # Track subscriptions for cleanup

    def subscribe_to_orders(self, callback: Callable) -> None:
        """Subscribe to all order-related events."""
        self.event_bus.subscribe(EventType.ORDER_FILLED, callback)
        self.event_bus.subscribe(EventType.ORDER_CANCELLED, callback)
        self._subscriptions.extend([
            (EventType.ORDER_FILLED, callback),
            (EventType.ORDER_CANCELLED, callback)
        ])

    def subscribe_to_signals(self, callback: Callable) -> None:
        """Subscribe to signal generation events."""
        self.event_bus.subscribe(EventType.SIGNAL_GENERATED, callback)
        self._subscriptions.append((EventType.SIGNAL_GENERATED, callback))

    def subscribe_to_errors(self, callback: Callable) -> None:
        """Subscribe to error events."""
        self.event_bus.subscribe(EventType.ERROR_OCCURRED, callback)
        self._subscriptions.append((EventType.ERROR_OCCURRED, callback))

    def subscribe_to_all(self, callback: Callable) -> None:
        """Subscribe to all event types."""
        for event_type in EventType:
            self.event_bus.subscribe(event_type, callback)
            self._subscriptions.append((event_type, callback))

    def unsubscribe_all(self) -> None:
        """Unsubscribe from all events."""
        for event_type, callback in self._subscriptions:
            self.event_bus.unsubscribe(event_type, callback)
        self._subscriptions.clear()
        logger.info("Unsubscribed from all events")