"""Inspect entry point: importing this module registers every AgentCheck task."""

from agentcheck.tasks.inventory import inventory_lookup

__all__ = ["inventory_lookup"]
