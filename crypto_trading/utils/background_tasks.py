"""
Background task management utilities.
Provides base class for managing asyncio background tasks with proper lifecycle management.
"""

import asyncio
from typing import Optional, Callable, Awaitable
from abc import ABC, abstractmethod
from loguru import logger


class BackgroundTaskManager(ABC):
    """
    Base class for components that run background tasks.
    Handles task lifecycle (start, stop, cancel) consistently.
    """

    def __init__(self):
        self._is_running = False
        self._tasks: list[asyncio.Task] = []

    async def start_background_task(
        self,
        coro: Callable[[], Awaitable[None]],
        task_name: str = "background_task"
    ) -> asyncio.Task:
        """
        Start a background task and track it.

        Args:
            coro: Coroutine function to run
            task_name: Name for the task (for logging)

        Returns:
            Created asyncio.Task
        """
        task = asyncio.create_task(coro(), name=task_name)
        self._tasks.append(task)
        logger.debug(f"Started background task: {task_name}")
        return task

    async def stop_all_background_tasks(self) -> None:
        """Stop all tracked background tasks gracefully."""
        self._is_running = False

        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.debug(f"Task {task.get_name()} cancelled successfully")
                except Exception as e:
                    logger.error(f"Error stopping task {task.get_name()}: {e}")

        self._tasks.clear()
        logger.debug("All background tasks stopped")

    def is_running(self) -> bool:
        """Check if background tasks are running."""
        return self._is_running

    @abstractmethod
    async def start(self) -> None:
        """Start the component. Override in subclasses."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the component. Override in subclasses."""
        pass


class PeriodicTaskManager(BackgroundTaskManager):
    """
    Manager for periodic background tasks.
    Handles running tasks at regular intervals.
    """

    def __init__(self, interval: float = 5.0):
        super().__init__()
        self.interval = interval
        self._main_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start periodic task execution."""
        if self._is_running:
            logger.warning(f"{self.__class__.__name__} is already running")
            return

        self._is_running = True

        # Run initialization
        await self._on_start()

        # Start main periodic loop
        self._main_task = await self.start_background_task(
            self._periodic_loop,
            f"{self.__class__.__name__}_loop"
        )

        logger.info(f"{self.__class__.__name__} started (interval: {self.interval}s)")

    async def stop(self) -> None:
        """Stop periodic task execution."""
        if not self._is_running:
            return

        self._is_running = False

        # Stop all tasks
        await self.stop_all_background_tasks()

        # Run cleanup
        await self._on_stop()

        logger.info(f"{self.__class__.__name__} stopped")

    async def _periodic_loop(self) -> None:
        """Main periodic execution loop."""
        while self._is_running:
            try:
                await self._execute_task()
            except Exception as e:
                logger.error(f"Error in periodic task: {e}")
                await self._on_error(e)

            await asyncio.sleep(self.interval)

    @abstractmethod
    async def _execute_task(self) -> None:
        """Execute the periodic task. Override in subclasses."""
        pass

    async def _on_start(self) -> None:
        """Hook called when task manager starts. Override if needed."""
        pass

    async def _on_stop(self) -> None:
        """Hook called when task manager stops. Override if needed."""
        pass

    async def _on_error(self, error: Exception) -> None:
        """Hook called when periodic task encounters error. Override if needed."""
        pass


class ReconnectableTaskManager(BackgroundTaskManager):
    """
    Manager for tasks that need reconnection logic (e.g., websocket connections).
    Handles automatic reconnection on failures.
    """

    def __init__(self, max_reconnect_attempts: int = 5, reconnect_delay: float = 5.0):
        super().__init__()
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self._reconnect_count = 0

    async def start(self) -> None:
        """Start task with reconnection support."""
        if self._is_running:
            logger.warning(f"{self.__class__.__name__} is already running")
            return

        self._is_running = True
        self._reconnect_count = 0

        await self.start_background_task(
            self._reconnectable_loop,
            f"{self.__class__.__name__}_reconnectable"
        )

        logger.info(f"{self.__class__.__name__} started with reconnection support")

    async def stop(self) -> None:
        """Stop reconnectable task."""
        if not self._is_running:
            return

        self._is_running = False
        await self.stop_all_background_tasks()

        logger.info(f"{self.__class__.__name__} stopped")

    async def _reconnectable_loop(self) -> None:
        """Main loop with reconnection logic."""
        while self._is_running:
            try:
                await self._connect_and_run()
                self._reconnect_count = 0  # Reset on successful connection

            except Exception as e:
                self._reconnect_count += 1
                logger.error(f"Connection error (attempt {self._reconnect_count}): {e}")

                if self._reconnect_count >= self.max_reconnect_attempts:
                    logger.error("Max reconnection attempts reached. Stopping.")
                    self._is_running = False
                    break

                logger.info(f"Reconnecting in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)

    @abstractmethod
    async def _connect_and_run(self) -> None:
        """Establish connection and run main logic. Override in subclasses."""
        pass
