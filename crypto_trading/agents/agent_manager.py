"""
Agent manager for handling multiple trading agents.
Implements Factory Pattern and provides centralized agent management.
"""

import asyncio
from typing import Dict, List, Optional, Any, Type
from datetime import datetime
import logging

from ..core.interfaces import ITradingAgent, IStrategy, IExchange, IRiskManager, IPortfolioManager
from ..core.exceptions import AgentException
from .base_agent import BaseTradingAgent


class AgentManager:
    """
    Manager for multiple trading agents.
    Provides centralized control, monitoring, and coordination.
    """

    def __init__(self):
        self.agents: Dict[str, ITradingAgent] = {}
        self.agent_configs: Dict[str, Dict[str, Any]] = {}
        self.agent_factories: Dict[str, Type[BaseTradingAgent]] = {}

        # Manager state
        self.is_running = False
        self.start_time: Optional[datetime] = None

        # Monitoring
        self._monitor_task: Optional[asyncio.Task] = None
        self.monitor_interval = 30  # seconds

        self.logger = logging.getLogger(__name__)

    def register_agent_type(self, agent_type: str, agent_class: Type[BaseTradingAgent]) -> None:
        """Register an agent type for factory creation."""
        self.agent_factories[agent_type] = agent_class
        self.logger.info(f"Registered agent type: {agent_type}")

    async def create_agent(
        self,
        agent_id: str,
        agent_type: str,
        strategy: IStrategy,
        exchange: IExchange,
        risk_manager: IRiskManager,
        portfolio_manager: IPortfolioManager,
        config: Dict[str, Any] = None
    ) -> str:
        """Create and register a new trading agent."""
        try:
            if agent_id in self.agents:
                raise AgentException(f"Agent {agent_id} already exists")

            if agent_type not in self.agent_factories:
                raise AgentException(f"Unknown agent type: {agent_type}")

            # Create agent instance
            agent_class = self.agent_factories[agent_type]
            agent = agent_class(
                strategy=strategy,
                exchange=exchange,
                risk_manager=risk_manager,
                portfolio_manager=portfolio_manager,
                agent_config=config or {}
            )

            # Initialize agent
            await agent.initialize()

            # Register agent
            self.agents[agent_id] = agent
            self.agent_configs[agent_id] = {
                "type": agent_type,
                "config": config or {},
                "created_at": datetime.now()
            }

            self.logger.info(f"Created agent: {agent_id} of type {agent_type}")
            return agent_id

        except Exception as e:
            self.logger.error(f"Failed to create agent {agent_id}: {e}")
            raise AgentException(f"Agent creation failed: {e}")

    async def start_agent(self, agent_id: str) -> None:
        """Start a specific agent."""
        if agent_id not in self.agents:
            raise AgentException(f"Agent {agent_id} not found")

        try:
            agent = self.agents[agent_id]
            await agent.start_trading()
            self.logger.info(f"Started agent: {agent_id}")

        except Exception as e:
            self.logger.error(f"Failed to start agent {agent_id}: {e}")
            raise AgentException(f"Agent start failed: {e}")

    async def stop_agent(self, agent_id: str) -> None:
        """Stop a specific agent."""
        if agent_id not in self.agents:
            raise AgentException(f"Agent {agent_id} not found")

        try:
            agent = self.agents[agent_id]
            await agent.stop_trading()
            self.logger.info(f"Stopped agent: {agent_id}")

        except Exception as e:
            self.logger.error(f"Failed to stop agent {agent_id}: {e}")

    async def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from management."""
        if agent_id not in self.agents:
            raise AgentException(f"Agent {agent_id} not found")

        try:
            # Stop agent if running
            agent = self.agents[agent_id]
            if hasattr(agent, 'is_running') and agent.is_running:
                await agent.stop_trading()

            # Remove from registry
            del self.agents[agent_id]
            del self.agent_configs[agent_id]

            self.logger.info(f"Removed agent: {agent_id}")

        except Exception as e:
            self.logger.error(f"Failed to remove agent {agent_id}: {e}")

    async def start_all_agents(self) -> None:
        """Start all registered agents."""
        if not self.agents:
            self.logger.warning("No agents to start")
            return

        self.logger.info(f"Starting {len(self.agents)} agents...")

        start_tasks = []
        for agent_id, agent in self.agents.items():
            task = asyncio.create_task(self._safe_start_agent(agent_id, agent))
            start_tasks.append(task)

        # Wait for all agents to start
        await asyncio.gather(*start_tasks, return_exceptions=True)

        # Start monitoring
        if not self.is_running:
            await self.start_monitoring()

    async def stop_all_agents(self) -> None:
        """Stop all running agents."""
        if not self.agents:
            self.logger.warning("No agents to stop")
            return

        self.logger.info(f"Stopping {len(self.agents)} agents...")

        stop_tasks = []
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'is_running') and agent.is_running:
                task = asyncio.create_task(self._safe_stop_agent(agent_id, agent))
                stop_tasks.append(task)

        # Wait for all agents to stop
        await asyncio.gather(*stop_tasks, return_exceptions=True)

        # Stop monitoring
        await self.stop_monitoring()

    async def start_monitoring(self) -> None:
        """Start monitoring all agents."""
        if self.is_running:
            self.logger.warning("Monitoring already running")
            return

        self.is_running = True
        self.start_time = datetime.now()
        self._monitor_task = asyncio.create_task(self._monitor_loop())

        self.logger.info("Started agent monitoring")

    async def stop_monitoring(self) -> None:
        """Stop monitoring."""
        if not self.is_running:
            return

        self.is_running = False

        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Stopped agent monitoring")

    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status information for a specific agent."""
        if agent_id not in self.agents:
            return {"error": f"Agent {agent_id} not found"}

        try:
            agent = self.agents[agent_id]
            config = self.agent_configs[agent_id]

            status = {
                "agent_id": agent_id,
                "type": config["type"],
                "created_at": config["created_at"],
                "is_initialized": getattr(agent, 'is_initialized', False),
                "is_running": getattr(agent, 'is_running', False),
                "start_time": getattr(agent, 'start_time', None),
                "performance_metrics": {}
            }

            # Get performance metrics if available
            if hasattr(agent, 'get_performance_metrics'):
                try:
                    status["performance_metrics"] = agent.get_performance_metrics()
                except Exception as e:
                    status["performance_metrics"] = {"error": str(e)}

            return status

        except Exception as e:
            return {"error": f"Failed to get status: {e}"}

    def get_all_agent_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all agents."""
        statuses = {}
        for agent_id in self.agents:
            statuses[agent_id] = self.get_agent_status(agent_id)
        return statuses

    def get_running_agents(self) -> List[str]:
        """Get list of currently running agent IDs."""
        running_agents = []
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'is_running') and agent.is_running:
                running_agents.append(agent_id)
        return running_agents

    def get_manager_statistics(self) -> Dict[str, Any]:
        """Get statistics about the agent manager."""
        total_agents = len(self.agents)
        running_agents = len(self.get_running_agents())

        # Aggregate performance metrics
        total_signals = 0
        total_orders = 0
        total_trades = 0
        total_pnl = 0.0

        for agent in self.agents.values():
            if hasattr(agent, 'get_performance_metrics'):
                try:
                    metrics = agent.get_performance_metrics()
                    total_signals += metrics.get('signals_generated', 0)
                    total_orders += metrics.get('orders_placed', 0)
                    total_trades += metrics.get('trades_executed', 0)
                    total_pnl += metrics.get('total_pnl', 0)
                except Exception:
                    pass

        return {
            "total_agents": total_agents,
            "running_agents": running_agents,
            "manager_uptime_seconds": self._get_uptime_seconds(),
            "aggregate_metrics": {
                "total_signals": total_signals,
                "total_orders": total_orders,
                "total_trades": total_trades,
                "total_pnl": total_pnl
            }
        }

    async def execute_signal_on_agent(self, agent_id: str, signal) -> Optional[Any]:
        """Execute a signal on a specific agent."""
        if agent_id not in self.agents:
            raise AgentException(f"Agent {agent_id} not found")

        agent = self.agents[agent_id]
        if not hasattr(agent, 'execute_signal'):
            raise AgentException(f"Agent {agent_id} doesn't support signal execution")

        try:
            return await agent.execute_signal(signal)
        except Exception as e:
            self.logger.error(f"Failed to execute signal on agent {agent_id}: {e}")
            raise AgentException(f"Signal execution failed: {e}")

    # Private methods

    async def _safe_start_agent(self, agent_id: str, agent: ITradingAgent) -> None:
        """Safely start an agent with error handling."""
        try:
            await agent.start_trading()
            self.logger.info(f"Successfully started agent: {agent_id}")
        except Exception as e:
            self.logger.error(f"Failed to start agent {agent_id}: {e}")

    async def _safe_stop_agent(self, agent_id: str, agent: ITradingAgent) -> None:
        """Safely stop an agent with error handling."""
        try:
            await agent.stop_trading()
            self.logger.info(f"Successfully stopped agent: {agent_id}")
        except Exception as e:
            self.logger.error(f"Failed to stop agent {agent_id}: {e}")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        self.logger.info("Starting monitoring loop")

        try:
            while self.is_running:
                try:
                    await self._check_agent_health()
                    await asyncio.sleep(self.monitor_interval)

                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(5)

        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Monitoring loop failed: {e}")

    async def _check_agent_health(self) -> None:
        """Check health of all agents."""
        for agent_id, agent in self.agents.items():
            try:
                # Check if agent is responsive
                if hasattr(agent, 'is_running') and agent.is_running:
                    # Perform basic health check
                    if hasattr(agent, 'get_performance_metrics'):
                        metrics = agent.get_performance_metrics()
                        # Could add specific health checks based on metrics

            except Exception as e:
                self.logger.warning(f"Health check failed for agent {agent_id}: {e}")

    def _get_uptime_seconds(self) -> float:
        """Get manager uptime in seconds."""
        if not self.start_time:
            return 0
        return (datetime.now() - self.start_time).total_seconds()