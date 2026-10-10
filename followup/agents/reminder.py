"""Agent 2 — Reminder Agent.

Input: summary + reminder history. Output: personalised reminder text, channel and
send time — or no reminder, when the same topic was covered recently (no duplicates).
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, time, timedelta
from typing import Any

from ..config import settings
from ..guardrails import finalize_reminder, log_agent_output
from ..llm import MockLLM, get_llm
from ..state import FollowUpState, state_now

SYSTEM = (
    "You write short, warm follow-up reminders to patients who recently left hospital. "
    "Rules: address the patient by first name; at most 60 words; plain language; mention only the reminder topic given; "
    "NEVER give clinical advice, doses, diagnoses or tell the patient to start/stop/change any medicine; "
    "ask for a simple reply (YES/OK) so the care team knows it was received. Return only the message text."
)

CONTACT_SLOTS = {"morning": time(9, 0), "afternoon": time(14, 0), "evening": time(18, 0)}


def _candidates(facts: dict[str, Any], today: date) -> list[dict[str, Any]]:
    """Reminder topics in priority order."""
    window = settings.upcoming_window_days
    out: list[dict[str, Any]] = []
    upcoming = [a for a in facts["due"] if a["days_until"] <= window]
    for a in sorted(upcoming, key=lambda a: (not a.get("critical"), a["days_until"])):
        when = "tomorrow" if a["days_until"] == 1 else ("today" if a["days_until"] == 0 else a["date"])
        out.append({
            "topic_key": f"appt:{a['appointment_id']}",
            "topic_type": "appointment",
            "topic": f"{a['type']} appointment on {a['date']}",
            "detail": f"{a['type'].lower()} appointment with {a['department']}",
            "when": when,
            "urgent": a["days_until"] <= 1,
        })
    for a in facts["missed"]:
        out.append({
            "topic_key": f"missed:{a['appointment_id']}",
            "topic_type": "missed_appointment",
            "topic": f"rescheduling missed {a['type']} appointment",
            "detail": f"{a['type'].lower()} appointment on {a['date']}",
            "urgent": bool(a.get("critical")),
        })
    if facts["med_changes"]:
        out.append({
            "topic_key": "meds:discharge",
            "topic_type": "medication",
            "topic": "medication changes at discharge",
            "detail": f"{len(facts['med_changes'])} change(s) to your medicines",
            "urgent": False,
        })
    for t in facts["open_task_items"]:
        if date.fromisoformat(t["due"]) <= today + timedelta(days=window):
            out.append({
                "topic_key": f"task:{t['task_id']}",
                "topic_type": "task",
                "topic": t["task"],
                "detail": f"{t['task'].lower()} due {t['due']}",
                "urgent": False,
            })
    return out


def _is_duplicate(candidate: dict[str, Any], history: list[dict[str, Any]], now: datetime) -> str | None:
    cutoff = now - timedelta(days=settings.duplicate_reminder_window_days)
    for r in reversed(history):
        if r.get("topic_key") != candidate["topic_key"]:
            continue
        if r.get("responded"):
            return f"patient already acknowledged '{candidate['topic']}' on {r.get('response_at', r['sent_at'])[:10]}"
        if datetime.fromisoformat(r["sent_at"]) >= cutoff:
            return f"reminder about '{candidate['topic']}' already sent on {r['sent_at'][:16]}"
    return None


def _send_at(preferred: str, now: datetime, urgent: bool) -> datetime:
    if urgent:
        return (now + timedelta(minutes=15)).replace(second=0, microsecond=0)
    slot = CONTACT_SLOTS.get(preferred, CONTACT_SLOTS["morning"])
    candidate = datetime.combine(now.date(), slot)
    return candidate if candidate > now else candidate + timedelta(days=1)


def reminder_agent(state: FollowUpState) -> dict[str, Any]:
    patient, history, summary = state["patient"], state["history"], state["summary"]
    now = state_now(state)
    facts = summary["facts"]

    skipped: list[str] = []
    chosen = None
    for cand in _candidates(facts, now.date()):
        reason = _is_duplicate(cand, history, now)
        if reason:
            skipped.append(reason)
            continue
        chosen = cand
        break

    if chosen is None:
        reminder = {
            "action": "none",
            "reason": "; ".join(skipped) if skipped else "no reminder topics due in the next "
            f"{settings.upcoming_window_days} days",
            "skipped_duplicates": skipped,
        }
    else:
        contact = patient["contact"]
        llm_facts = {"first_name": facts["first_name"], **chosen}
        prompt = (
            f"Patient first name: {facts['first_name']}\n"
            f"Preferred language: {contact.get('language', 'English')}\n"
            f"Reminder topic: {chosen['topic']} ({chosen['detail']}"
            + (f", {chosen['when']}" if chosen.get("when") else "")
            + ")\n"
            f"Previous reminders on file: {len(history)}; last reply: {facts['last_response'] or 'none'}\n"
            "Write the reminder message."
        )
        llm = get_llm()
        draft = llm.generate("reminder", SYSTEM, prompt, llm_facts)
        text, violations = finalize_reminder(draft, MockLLM._reminder(llm_facts))
        reminder = {
            "action": "send",
            "reminder_id": f"R-{uuid.uuid4().hex[:10]}",
            "run_id": state["run_id"],
            "topic_key": chosen["topic_key"],
            "topic_type": chosen["topic_type"],
            "topic": chosen["topic"],
            "text": text,
            "channel": contact["preferred_channel"],
            "to": contact["phone"] if contact["preferred_channel"] in ("sms", "whatsapp") else contact["email"],
            "sent_at": _send_at(contact.get("preferred_time", "morning"), now, chosen["urgent"]).isoformat(timespec="minutes"),
            "created_at": now.isoformat(timespec="minutes"),
            "status": "queued",
            "responded": False,
            "response_text": None,
            "response_at": None,
            "guardrail_violations": violations,
            "skipped_duplicates": skipped,
            "llm": llm.name,
        }

    key = log_agent_output(state["run_id"], patient["patient_id"], "2_reminder", {"output": reminder})
    return {"reminder": reminder, "log_keys": [key]}
