"""Long-term memory per patient, persisted to storage (S3 in production).

Short-term memory is the LangGraph state of one run (see graph.py). Long-term memory
keeps what happened across runs: reminders the Reminder Agent queued, the patient's
responses to them, and escalations raised — so the next run can avoid duplicate
reminders and spot repeated non-response.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from .storage import Storage, get_storage

# A reminder with no reply this long after it was sent counts as missed.
RESPONSE_GRACE = timedelta(hours=24)


def _key(patient_id: str) -> str:
    return f"memory/{patient_id}.json"


def empty_memory(patient_id: str) -> dict[str, Any]:
    return {"patient_id": patient_id, "reminders": [], "escalations": []}


def load_memory(patient_id: str, storage: Storage | None = None) -> dict[str, Any]:
    return (storage or get_storage()).get_json(_key(patient_id)) or empty_memory(patient_id)


def save_memory(memory: dict[str, Any], storage: Storage | None = None) -> None:
    (storage or get_storage()).put_json(_key(memory["patient_id"]), memory)


def record_response(
    patient_id: str, reminder_id: str, text: str, when: datetime | None = None, storage: Storage | None = None
) -> bool:
    """Attach a patient reply to a reminder the system sent. Returns False if not found."""
    storage = storage or get_storage()
    memory = load_memory(patient_id, storage)
    for reminder in memory["reminders"]:
        if reminder["reminder_id"] == reminder_id:
            reminder["responded"] = True
            reminder["response_text"] = text
            reminder["response_at"] = (when or datetime.now()).isoformat(timespec="minutes")
            save_memory(memory, storage)
            return True
    return False


def reminder_history(patient: dict[str, Any], memory: dict[str, Any]) -> list[dict[str, Any]]:
    """Merge the historical reminder log from the patient record with reminders from past runs."""
    history = [{**r, "source": "record"} for r in patient.get("reminder_log", [])]
    history += [{**r, "source": "assistant"} for r in memory.get("reminders", [])]
    return sorted(history, key=lambda r: r["sent_at"])


def reminder_outcome(reminder: dict[str, Any], now: datetime) -> str:
    """'responded', 'pending' (still inside the reply window) or 'missed'."""
    if reminder.get("responded"):
        return "responded"
    if datetime.fromisoformat(reminder["sent_at"]) + RESPONSE_GRACE > now:
        return "pending"
    return "missed"


def consecutive_missed(history: list[dict[str, Any]], now: datetime) -> int:
    """Number of most recent reminders in a row the patient did not answer (pending ones are skipped)."""
    count = 0
    for reminder in reversed(history):
        outcome = reminder_outcome(reminder, now)
        if outcome == "pending":
            continue
        if outcome == "responded":
            break
        count += 1
    return count
