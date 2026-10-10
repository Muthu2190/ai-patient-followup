"""Read models for the Streamlit dashboard (alerts, reminder outbox, runs)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .memory import load_memory, save_memory
from .storage import Storage, get_storage, list_patient_ids


def _memories(storage: Storage) -> list[dict[str, Any]]:
    return [load_memory(pid, storage) for pid in list_patient_ids(storage)]


def all_alerts(storage: Storage | None = None) -> list[dict[str, Any]]:
    storage = storage or get_storage()
    alerts = [a for m in _memories(storage) for a in m.get("escalations", [])]
    return sorted(alerts, key=lambda a: (a["status"] != "open", -{"escalate": 2, "watch": 1}.get(a["risk_level"], 0), a["created_at"]))


def open_alerts(storage: Storage | None = None) -> list[dict[str, Any]]:
    return [a for a in all_alerts(storage) if a["status"] == "open"]


def update_alert(patient_id: str, alert_id: str, status: str, clinician: str, note: str = "", storage: Storage | None = None) -> bool:
    """Human-in-the-loop: a clinician acknowledges or resolves an alert."""
    storage = storage or get_storage()
    memory = load_memory(patient_id, storage)
    for alert in memory["escalations"]:
        if alert["alert_id"] == alert_id:
            alert["status"] = status
            alert.setdefault("actions", []).append(
                {"status": status, "by": clinician, "note": note, "at": datetime.now().isoformat(timespec="minutes")}
            )
            save_memory(memory, storage)
            return True
    return False


def outbox(storage: Storage | None = None) -> list[dict[str, Any]]:
    storage = storage or get_storage()
    rows = []
    for m in _memories(storage):
        for r in m.get("reminders", []):
            rows.append({"patient_id": m["patient_id"], **r})
    return sorted(rows, key=lambda r: r["created_at"], reverse=True)


def list_runs(storage: Storage | None = None) -> list[dict[str, Any]]:
    storage = storage or get_storage()
    keys = sorted(storage.list_keys("runs/"), reverse=True)
    return [m for k in keys if (m := storage.get_json(k))]


def run_logs(run_id: str, patient_id: str, storage: Storage | None = None) -> dict[str, Any]:
    storage = storage or get_storage()
    return {Path(k).stem: storage.get_json(k) for k in storage.list_keys(f"logs/{run_id}/{patient_id}/")}
