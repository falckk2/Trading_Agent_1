"""
Meta-agents module - Agents that use other agents for decision making.
"""

from .consensus_agent import ConsensusAgent
from .validator_agent import ValidatorAgent

__all__ = [
    'ConsensusAgent',
    'ValidatorAgent'
]
