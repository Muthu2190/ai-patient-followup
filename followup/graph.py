"""LangGraph orchestrator: load -> Patient Summary -> Reminder -> Escalation -> persist.

The agents run in sequence over one shared state object (short-term memory). The
final node writes the run's reminder and escalation into long-term memory and
re-indexes the patient so the next run's RAG retrieval sees them.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from functools import lru_cache
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from .agents import escalation_agent, patient_summary_agent, reminder_agent
from .guardrails import assert_synthetic, log_agent_output
from .memory import load_memory, reminder_history, save_memory
from .rag import get_index
from .state import FollowUpState
from .storage import get_storage, list_patient_ids, load_patient

log = logging.getLogger(__name__)


def load_context(state: FollowUpState) -> dict[str, Any]:
    patient = load_patient(state["patient_id"])
    if patient is None:
        raise KeyError(f"Patient {state['patient_id']} not found in storage")
    assert_synthetic(patient)
    memory = load_memory(patient["patient_id"])
    return {"patient": patient, "memory": memory, "history": reminder_history(patient, memory), "log_keys": []}


def persist_memory(state: FollowUpState) -> dict[str, Any]:
    patient, memory = state["patient"], state["memory"]
    reminder, escalation = state["reminder"], state["escalation"]
    if reminder.get("action") == "send":
        memory["reminders"].append({k: v for k, v in reminder.items() if k not in ("action", "skipped_duplicates")})
    if escalation.get("alert"):
        alert = escalation["alert"]
        # Keep one open alert per patient: a newer run supersedes the previous open one.
        for old in memory["escalations"]:
            if old.get("status") == "open":
                old["status"] = "superseded"
                old["superseded_by"] = alert["alert_id"]
        memory["escalations"].append(alert)
    memory["last_run"] = {"run_id": state["run_id"], "at": state["now"], "risk_level": escalation["risk_level"]}
    memory["last_summary"] = state["summary"]["text"]
    save_memory(memory)
    get_index().index_patient(patient, memory)
    key = log_agent_output(state["run_id"], patient["patient_id"], "4_memory", {"output": {"memory_key": f"memory/{patient['patient_id']}.json"}})
    return {"memory": memory, "log_keys": [key]}


@lru_cache(maxsize=1)
def build_graph():
    graph = StateGraph(FollowUpState)
    graph.add_node("load_context", load_context)
    graph.add_node("patient_summary_agent", patient_summary_agent)
    graph.add_node("reminder_agent", reminder_agent)
    graph.add_node("escalation_agent", escalation_agent)
    graph.add_node("persist_memory", persist_memory)
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "patient_summary_agent")
    graph.add_edge("patient_summary_agent", "reminder_agent")
    graph.add_edge("reminder_agent", "escalation_agent")
    graph.add_edge("escalation_agent", "persist_memory")
    graph.add_edge("persist_memory", END)
    return graph.compile()


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def run_patient(patient_id: str, run_id: str | None = None, now: datetime | None = None) -> FollowUpState:
    state: FollowUpState = {
        "run_id": run_id or new_run_id(),
        "now": (now or datetime.now()).isoformat(timespec="minutes"),
        "patient_id": patient_id,
    }
    return build_graph().invoke(state)


def run_followups(
    patient_ids: list[str] | None = None,
    now: datetime | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Run the pipeline for many patients and write a run manifest to storage."""
    storage = get_storage()
    patient_ids = patient_ids or list_patient_ids(storage)
    if get_index().count() == 0:  # e.g. fresh container whose data lives in S3
        get_index().rebuild(storage)
    run_id = new_run_id()
    now = now or datetime.now()
    results, errors = [], []
    for i, pid in enumerate(patient_ids, 1):
        if progress:
            progress(i, len(patient_ids), pid)
        try:
            final = run_patient(pid, run_id=run_id, now=now)
        except Exception as exc:  # one bad record should not stop the whole run
            log.exception("Follow-up failed for %s", pid)
            errors.append({"patient_id": pid, "error": str(exc)})
            continue
        results.append({
            "patient_id": pid,
            "name": final["patient"]["name"],
            "diagnosis": final["summary"]["diagnosis"],
            "summary": final["summary"]["text"],
            "reminder_action": final["reminder"]["action"],
            "reminder_text": final["reminder"].get("text") or final["reminder"].get("reason"),
            "channel": final["reminder"].get("channel"),
            "send_at": final["reminder"].get("sent_at"),
            "risk_level": final["escalation"]["risk_level"],
            "reasons": final["escalation"]["reasons"],
        })
    manifest = {
        "run_id": run_id,
        "started_for": now.isoformat(timespec="minutes"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "patients": len(patient_ids),
        "reminders_queued": sum(r["reminder_action"] == "send" for r in results),
        "escalations": sum(r["risk_level"] == "escalate" for r in results),
        "watch": sum(r["risk_level"] == "watch" for r in results),
        "results": results,
        "errors": errors,
    }
    storage.put_json(f"runs/{run_id}.json", manifest)
    return manifest
