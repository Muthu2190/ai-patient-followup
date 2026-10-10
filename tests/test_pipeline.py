from datetime import datetime, timedelta

import pytest

from followup.agents.escalation import find_warning_symptoms
from followup.cli import reset_all
from followup.dashboard_data import open_alerts
from followup.graph import run_followups, run_patient
from followup.guardrails import SAFETY_FOOTER, GuardrailViolation, assert_synthetic, check_reminder
from followup.memory import load_memory, record_response
from followup.rag import get_index
from followup.storage import get_storage, save_patients
from followup.synthetic import generate_patients

NOW = datetime(2026, 10, 9, 10, 0)


@pytest.fixture(autouse=True)
def fresh_data():
    reset_all()
    patients = generate_patients(30, seed=7, now=NOW)
    save_patients(patients)
    get_index().rebuild()
    return {p["patient_id"]: p for p in patients}


def _by_scenario(patients, scenario):
    return next(p for p in patients.values() if p["scenario"] == scenario)


def test_full_run_matches_synthetic_ground_truth(fresh_data):
    manifest = run_followups(now=NOW)
    assert manifest["patients"] == 30 and not manifest["errors"]
    expected = {
        "normal": "none", "non_responder": "escalate", "warning_symptom": "escalate",
        "missed_critical": "escalate", "single_miss": "watch", "missed_routine": "watch",
    }
    for r in manifest["results"]:
        scenario = fresh_data[r["patient_id"]]["scenario"]
        assert r["risk_level"] == expected[scenario], (scenario, r)


def test_every_agent_output_is_logged(fresh_data):
    final = run_patient("P0001", now=NOW)
    keys = get_storage().list_keys(f"logs/{final['run_id']}/P0001/")
    assert [k.rsplit("/", 1)[1] for k in keys] == [
        "1_patient_summary.json", "2_reminder.json", "3_escalation.json", "4_memory.json",
    ]


def test_summary_is_grounded_in_retrieval(fresh_data):
    final = run_patient("P0001", now=NOW)
    summary = final["summary"]
    assert summary["sources"], "summary should cite retrieved chunks"
    assert fresh_data["P0001"]["discharge"]["diagnosis"] in summary["text"]


def test_no_duplicate_reminders_across_runs(fresh_data):
    first = run_patient("P0001", now=NOW)["reminder"]
    second = run_patient("P0001", now=NOW + timedelta(hours=2))["reminder"]
    assert first["action"] == "send"
    assert second.get("topic_key") != first["topic_key"]
    assert any(first["topic"] in s for s in second["skipped_duplicates"])


def test_reminders_pass_guardrails(fresh_data):
    manifest = run_followups(now=NOW)
    sent = [r for r in manifest["results"] if r["reminder_action"] == "send"]
    assert sent
    for r in sent:
        assert SAFETY_FOOTER in r["reminder_text"]
        assert check_reminder(r["reminder_text"].replace(SAFETY_FOOTER, "")) == []


def test_repeated_non_response_escalates_over_runs(fresh_data):
    pid = _by_scenario(fresh_data, "single_miss")["patient_id"]
    assert run_patient(pid, now=NOW)["escalation"]["risk_level"] == "watch"
    # The reminder queued in run 1 goes unanswered -> two misses in a row by run 2.
    later = run_patient(pid, now=NOW + timedelta(days=3))
    assert later["escalation"]["risk_level"] == "escalate"
    assert any(a["patient_id"] == pid and a["risk_level"] == "escalate" for a in open_alerts())


def test_patient_reply_with_symptom_escalates(fresh_data):
    pid = "P0001"
    assert fresh_data[pid]["scenario"] == "normal"
    reminder = run_patient(pid, now=NOW)["reminder"]
    record_response(pid, reminder["reminder_id"], "Since yesterday I get chest pain when walking", when=NOW + timedelta(hours=1))
    final = run_patient(pid, now=NOW + timedelta(hours=3))
    assert final["escalation"]["risk_level"] == "escalate"
    assert "warning_symptom" in final["escalation"]["rules_triggered"]
    assert load_memory(pid)["escalations"][-1]["status"] == "open"


def test_symptom_negation():
    assert find_warning_symptoms("No chest pain and sleeping well", []) == []
    assert find_warning_symptoms("I have chest pain", []) == ["chest pain"]
    assert find_warning_symptoms("Running a high fever", []) == ["high fever"]


def test_clinical_advice_is_blocked():
    assert check_reminder("Please increase your dose of furosemide to 80 mg")
    assert not check_reminder("Hi Avery, reminder of your appointment on Monday. Reply YES to confirm.")


def test_non_synthetic_records_are_refused(fresh_data):
    real_looking = {**fresh_data["P0001"], "synthetic": False}
    with pytest.raises(GuardrailViolation):
        assert_synthetic(real_looking)
