"""Guardrails from the architecture one-pager.

1. Synthetic data only — records must carry `"synthetic": true` and fictional contact details.
2. Reminders avoid clinical advice — LLM output is checked and replaced with a safe template if needed.
3. Escalations always go to a human clinician — the Escalation Agent only raises alerts, never acts.
4. Every agent output is logged to storage (see `log_agent_output`).
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .storage import Storage, get_storage

SAFETY_FOOTER = "If you feel unwell or notice any warning signs, contact your care team or call emergency services."

# Phrases that turn a reminder into clinical advice (dosing, stopping, diagnosis).
_CLINICAL_ADVICE = re.compile(
    r"\b("
    r"increase (your|the) dose|decrease (your|the) dose|double|halve|stop taking|start taking|adjust"
    r"|\d+\s?(mg|mcg|ml|units?)\b"
    r"|you (should|must) (take|stop|avoid)"
    r"|diagnos(e|is|ed)|prognosis|it is (probably|likely) nothing"
    r")",
    re.IGNORECASE,
)


class GuardrailViolation(Exception):
    pass


def assert_synthetic(patient: dict[str, Any]) -> None:
    if patient.get("synthetic") is not True:
        raise GuardrailViolation(
            f"Record {patient.get('patient_id')} is not flagged synthetic; refusing to send it to an LLM."
        )
    email = patient.get("contact", {}).get("email", "")
    if email and not email.endswith(".example"):
        raise GuardrailViolation(f"Record {patient['patient_id']} has a non-synthetic email domain.")


def check_reminder(text: str) -> list[str]:
    """Return the clinical-advice phrases found in a reminder (empty list = safe)."""
    return sorted({m.group(0).lower() for m in _CLINICAL_ADVICE.finditer(text)})


def finalize_reminder(text: str, safe_fallback: str) -> tuple[str, list[str]]:
    """Apply the no-clinical-advice rule; returns (final_text, violations)."""
    text = " ".join(text.split())
    violations = check_reminder(text)
    if violations:
        text = safe_fallback
    if SAFETY_FOOTER not in text:
        text = f"{text} {SAFETY_FOOTER}"
    return text[:640], violations


def log_agent_output(
    run_id: str, patient_id: str, agent: str, output: dict[str, Any], storage: Storage | None = None
) -> str:
    key = f"logs/{run_id}/{patient_id}/{agent}.json"
    (storage or get_storage()).put_json(
        key, {"run_id": run_id, "patient_id": patient_id, "agent": agent, "logged_at": datetime.now().isoformat(), **output}
    )
    return key
