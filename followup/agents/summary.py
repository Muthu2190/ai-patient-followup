"""Agent 1 — Patient Summary Agent.

Input: patient record retrieved via RAG. Output: follow-up summary (diagnosis, meds,
due visits, open tasks) written to the shared state.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from ..guardrails import log_agent_output
from ..llm import get_llm
from ..rag import format_context, get_index
from ..state import FollowUpState, state_now

SYSTEM = (
    "You are a clinical documentation assistant preparing post-discharge follow-up summaries for care staff. "
    "Use ONLY the facts in the provided context. Do not invent diagnoses, medicines, doses or dates. "
    "Write 4-6 plain sentences covering: diagnosis, medication changes, upcoming and missed visits, "
    "open follow-up tasks, and the most recent patient response."
)

QUERY = "discharge diagnosis summary medication changes upcoming appointments follow-up tasks patient responses"


def structured_facts(patient: dict[str, Any], history: list[dict[str, Any]], today: date) -> dict[str, Any]:
    d = patient["discharge"]
    due, missed = [], []
    for a in sorted(patient["appointments"], key=lambda a: a["date"]):
        label = f"{a['type']} ({a['department']}) on {a['date']}" + (" [critical]" if a.get("critical") else "")
        appt_date = date.fromisoformat(a["date"])
        if a["status"] == "missed" or (a["status"] == "scheduled" and appt_date < today):
            missed.append({**a, "label": label})
        elif a["status"] == "scheduled":
            due.append({**a, "label": label, "days_until": (appt_date - today).days})
    med_changes = [
        f"{m['change']} {m['name']} {m['dose']} {m['frequency']}" for m in patient["medications"] if m["change"] != "unchanged"
    ]
    open_tasks = [t for t in patient.get("tasks", []) if t["status"] == "open"]
    responses = [r for r in history if r.get("responded")]
    return {
        "name": patient["name"],
        "first_name": patient["name"].split()[0],
        "age": patient["age"],
        "sex": patient["sex"],
        "discharge_date": d["date"],
        "diagnosis": d["diagnosis"],
        "warning_signs": d["warning_signs"],
        "med_changes": med_changes,
        "due": due,
        "missed": missed,
        "due_visits": [a["label"] for a in due],
        "missed_visits": [a["label"] for a in missed],
        "open_task_items": open_tasks,
        "open_tasks": [f"{t['task']} (due {t['due']})" for t in open_tasks],
        "last_response": responses[-1]["response_text"] if responses else None,
    }


def patient_summary_agent(state: FollowUpState) -> dict[str, Any]:
    patient, history = state["patient"], state["history"]
    today = state_now(state).date()
    facts = structured_facts(patient, history, today)

    docs = get_index().retrieve(patient["patient_id"], QUERY, k=6)
    context = format_context(docs)
    prompt = f"Patient context (retrieved records):\n{context}\n\nToday is {today.isoformat()}. Write the follow-up summary."
    llm = get_llm()
    text = llm.generate("summary", SYSTEM, prompt, facts)

    summary = {
        "text": text,
        "diagnosis": facts["diagnosis"],
        "med_changes": facts["med_changes"],
        "due_visits": facts["due_visits"],
        "missed_visits": facts["missed_visits"],
        "open_tasks": facts["open_tasks"],
        "warning_signs": facts["warning_signs"],
        "facts": facts,
        "sources": [f"{d.metadata['section']}#{d.metadata['chunk']}" for d in docs],
        "llm": llm.name,
    }
    key = log_agent_output(state["run_id"], patient["patient_id"], "1_patient_summary", {"output": summary, "context": context})
    return {"summary": summary, "log_keys": [key]}
