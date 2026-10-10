"""Agent 3 — Escalation Agent.

Input: summary + reminder history + patient responses. Output: risk flag
(none / watch / escalate) and, when raised, an alert for a human clinician.

The risk level comes from explicit rules (auditable); the LLM only words the alert
using retrieved context, and can never lower a rule-based flag.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta
from typing import Any

from ..guardrails import log_agent_output
from ..llm import get_llm
from ..memory import consecutive_missed, reminder_outcome
from ..rag import format_context, get_index
from ..state import FollowUpState, state_now

SYSTEM = (
    "You are a triage assistant writing short alerts for hospital care staff about discharged patients. "
    "State who the patient is, the specific risk reasons given, and recommend that a clinician contacts the patient. "
    "Do not diagnose and do not address the patient. Two to three sentences."
)

RISK_ORDER = {"none": 0, "watch": 1, "escalate": 2}

# Generic red-flag phrases; each patient's discharge warning signs are added on top.
WARNING_PHRASES = [
    "chest pain", "chest tightness", "short of breath", "shortness of breath", "can't breathe", "cannot breathe",
    "breathless", "bleeding", "blood in", "fainted", "passed out", "confused", "confusion", "high fever", "fever",
    "severe pain", "swelling", "swollen", "dizzy", "vomiting", "palpitations", "racing heart", "rash", "wound is red",
]
NEGATIONS = {"no", "not", "without", "never", "denies", "none", "nor"}
RESPONSE_LOOKBACK = timedelta(days=14)


def find_warning_symptoms(text: str, extra_phrases: list[str]) -> list[str]:
    """Return warning phrases present in a patient reply, skipping simple negations ('no chest pain')."""
    lowered = text.lower()
    found = []
    for phrase in sorted(set(WARNING_PHRASES + [p.lower() for p in extra_phrases]), key=len, reverse=True):
        for match in re.finditer(rf"\b{re.escape(phrase)}\b", lowered):
            preceding = re.findall(r"[a-z']+", lowered[: match.start()])[-3:]
            if NEGATIONS.isdisjoint(preceding) and not any(phrase in f for f in found):
                found.append(phrase)
                break
    return found


def assess_risk(patient: dict[str, Any], history: list[dict[str, Any]], facts: dict[str, Any], now: datetime):
    reasons: list[dict[str, str]] = []

    missed_in_row = consecutive_missed(history, now)
    if missed_in_row >= 2:
        reasons.append({"level": "escalate", "rule": "missed_reminders", "text": f"missed {missed_in_row} reminders in a row"})
    elif missed_in_row == 1:
        reasons.append({"level": "watch", "rule": "missed_reminders", "text": "did not reply to the last reminder"})

    for r in history:
        if not r.get("responded") or not r.get("response_text"):
            continue
        when = datetime.fromisoformat(r.get("response_at") or r["sent_at"])
        if now - when > RESPONSE_LOOKBACK:
            continue
        symptoms = find_warning_symptoms(r["response_text"], facts["warning_signs"])
        if symptoms:
            reasons.append({
                "level": "escalate",
                "rule": "warning_symptom",
                "text": f"reported warning symptom ({', '.join(symptoms)}) on {when.date()}: \"{r['response_text']}\"",
            })

    for a in facts["missed"]:
        if a.get("critical"):
            reasons.append({"level": "escalate", "rule": "missed_critical_appointment", "text": f"missed critical {a['label']}"})
        else:
            reasons.append({"level": "watch", "rule": "missed_appointment", "text": f"missed {a['label']}"})

    level = max((r["level"] for r in reasons), key=RISK_ORDER.get, default="none")
    return level, reasons, missed_in_row


def escalation_agent(state: FollowUpState) -> dict[str, Any]:
    patient, history, summary = state["patient"], state["history"], state["summary"]
    now = state_now(state)
    facts = summary["facts"]

    level, reasons, missed_in_row = assess_risk(patient, history, facts, now)
    reason_texts = [r["text"] for r in reasons]

    escalation: dict[str, Any] = {
        "risk_level": level,
        "reasons": reason_texts,
        "rules_triggered": sorted({r["rule"] for r in reasons}),
        "consecutive_missed": missed_in_row,
        "reply_stats": {
            o: sum(1 for r in history if reminder_outcome(r, now) == o) for o in ("responded", "missed", "pending")
        },
    }

    if level != "none":
        docs = get_index().retrieve(patient["patient_id"], "warning symptoms patient replies missed appointments reminders", k=4)
        prompt = (
            f"Patient: {patient['name']} ({patient['patient_id']}), {facts['diagnosis']}, discharged {facts['discharge_date']}.\n"
            f"Risk level: {level}\nRisk reasons: {'; '.join(reason_texts)}\n"
            f"Follow-up summary: {summary['text']}\n\nRetrieved context:\n{format_context(docs)}\n\nWrite the clinician alert."
        )
        llm = get_llm()
        message = llm.generate(
            "escalation", SYSTEM, prompt, {"name": patient["name"], "diagnosis": facts["diagnosis"], "risk_level": level, "reasons": reason_texts}
        )
        escalation["alert"] = {
            "alert_id": f"A-{uuid.uuid4().hex[:10]}",
            "run_id": state["run_id"],
            "patient_id": patient["patient_id"],
            "patient_name": patient["name"],
            "risk_level": level,
            "reasons": reason_texts,
            "message": message,
            "assigned_to": patient.get("care_team", "Discharge follow-up team"),
            "created_at": now.isoformat(timespec="minutes"),
            "status": "open",  # a human clinician acknowledges / resolves it in the dashboard
            "llm": llm.name,
        }

    key = log_agent_output(state["run_id"], patient["patient_id"], "3_escalation", {"output": escalation})
    return {"escalation": escalation, "log_keys": [key]}
