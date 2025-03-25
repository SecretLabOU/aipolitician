"""
AI Politician Debate System
==========================

A LangGraph-based system for simulating debates between AI politicians.
"""

from src.models.langgraph.debate.workflow import run_debate, DebateInput, DebateFormat
from src.models.langgraph.debate.cli import main as run_cli

__all__ = ['run_debate', 'DebateInput', 'DebateFormat', 'run_cli'] 