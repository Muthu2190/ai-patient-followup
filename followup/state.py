"""Short-term memory: the LangGraph state shared by the three agents within one run."""
from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, Any, TypedDict


class FollowUpState(TypedDict, total=False):
    # Inputs
    run_id: str
    now: str  # ISO timestamp the run treats as "now" (overridable for tests/demos)
    patient_id: str

    # Loaded by the orchestrator
    patient: dict[str, Any]
    memory: dict[str, Any]  # long-term memory snapshot (past reminders, escalations)
    history: list[dict[str, Any]]  # merged reminder + response history

    # Written by the agents, in order
    summary: dict[str, Any]
    reminder: dict[str, Any]
    escalation: dict[str, Any]

    # Accumulated across nodes
    log_keys: Annotated[list[str], operator.add]


def state_now(state: FollowUpState) -> datetime:
    return datetime.fromisoformat(state["now"])
