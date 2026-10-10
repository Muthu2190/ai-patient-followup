"""Synthetic patient generator — fictional people only, never real patient data.

Each record has demographics, a discharge summary, medications, appointments,
open tasks and a log of reminder responses. A share of records are deliberate
edge cases (non-responders, warning symptoms, missed critical visits) so every
Escalation Agent rule is exercised.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

FIRST = ["Avery", "Jordan", "Riley", "Morgan", "Casey", "Quinn", "Rowan", "Sasha", "Elliot", "Harper", "Kai", "Noor",
         "Priya", "Mateo", "Lena", "Tomas", "Aisha", "Wei", "Arjun", "Ines", "Kofi", "Mira", "Dev", "Zara", "Leo",
         "Yuki", "Omar", "Nadia", "Felix", "Iris", "Ravi", "Selin", "Hugo", "Amara", "Theo", "Lucia"]
LAST = ["Abbott", "Banerjee", "Castillo", "Dimitrov", "Eriksen", "Fofana", "Garcia", "Hallam", "Iyer", "Jansen",
        "Kowalski", "Lindqvist", "Mbeki", "Nakamura", "Okafor", "Petrov", "Quintero", "Rahman", "Silva", "Tanaka",
        "Umar", "Varga", "Whitlock", "Xu", "Yilmaz", "Zielinski"]

CONDITIONS: list[dict[str, Any]] = [
    {
        "diagnosis": "Acute decompensated heart failure", "ward": "Cardiology",
        "summary": "Admitted with breathlessness and leg swelling; treated with IV diuretics and discharged euvolaemic.",
        "warning_signs": ["shortness of breath", "swelling", "rapid weight gain", "chest pain"],
        "meds": [("Furosemide", "40 mg", "once daily", "changed", "dose increased"), ("Bisoprolol", "2.5 mg", "once daily", "new", None),
                 ("Ramipril", "5 mg", "once daily", "unchanged", None)],
        "appts": [("Heart failure nurse review", "Cardiology", True), ("Blood test (kidney function)", "Phlebotomy", False)],
        "tasks": ["Daily weight diary", "Kidney function blood test"],
    },
    {
        "diagnosis": "Community-acquired pneumonia", "ward": "Respiratory",
        "summary": "Treated with IV then oral antibiotics; oxygen weaned before discharge.",
        "warning_signs": ["high fever", "shortness of breath", "chest pain", "confusion"],
        "meds": [("Amoxicillin", "500 mg", "three times daily", "new", "5 more days"), ("Paracetamol", "1 g", "as needed", "unchanged", None)],
        "appts": [("Chest X-ray follow-up", "Radiology", False), ("Respiratory clinic review", "Respiratory", True)],
        "tasks": ["Finish antibiotic course"],
    },
    {
        "diagnosis": "Type 2 diabetes with hyperglycaemia", "ward": "Endocrinology",
        "summary": "Admitted with high blood glucose; insulin started and education provided.",
        "warning_signs": ["confusion", "vomiting", "fainted", "dizzy"],
        "meds": [("Insulin glargine", "14 units", "at night", "new", None), ("Metformin", "1 g", "twice daily", "unchanged", None)],
        "appts": [("Diabetes nurse review", "Endocrinology", True), ("Retinal screening", "Ophthalmology", False)],
        "tasks": ["Record blood glucose twice daily", "HbA1c blood test"],
    },
    {
        "diagnosis": "Elective total knee replacement", "ward": "Orthopaedics",
        "summary": "Uncomplicated left knee replacement; mobilising with frame at discharge.",
        "warning_signs": ["wound is red", "fever", "swelling", "calf pain"],
        "meds": [("Rivaroxaban", "10 mg", "once daily", "new", "for 14 days"), ("Codeine", "30 mg", "as needed", "new", None)],
        "appts": [("Wound check and suture removal", "Orthopaedics", True), ("Physiotherapy session", "Physiotherapy", False)],
        "tasks": ["Daily physiotherapy exercises"],
    },
    {
        "diagnosis": "COPD exacerbation", "ward": "Respiratory",
        "summary": "Treated with nebulisers, steroids and antibiotics; returned to baseline oxygen saturations.",
        "warning_signs": ["shortness of breath", "chest tightness", "fever", "confusion"],
        "meds": [("Prednisolone", "30 mg", "once daily", "new", "5-day course"), ("Tiotropium", "18 mcg", "once daily", "unchanged", None)],
        "appts": [("Pulmonary rehab assessment", "Respiratory", False), ("COPD nurse review", "Respiratory", True)],
        "tasks": ["Inhaler technique check"],
    },
    {
        "diagnosis": "Non-ST elevation myocardial infarction", "ward": "Cardiology",
        "summary": "Managed with coronary angiography and one stent; started on dual antiplatelet therapy.",
        "warning_signs": ["chest pain", "palpitations", "shortness of breath", "fainted"],
        "meds": [("Aspirin", "75 mg", "once daily", "new", None), ("Ticagrelor", "90 mg", "twice daily", "new", None),
                 ("Atorvastatin", "80 mg", "at night", "new", None)],
        "appts": [("Cardiology outpatient review", "Cardiology", True), ("Cardiac rehab intake", "Cardiac Rehab", False)],
        "tasks": ["Cardiac rehab enrolment"],
    },
    {
        "diagnosis": "Urinary tract infection with sepsis", "ward": "General Medicine",
        "summary": "Treated with IV antibiotics; blood cultures cleared and kidney function recovered.",
        "warning_signs": ["high fever", "confusion", "vomiting", "dizzy"],
        "meds": [("Nitrofurantoin", "100 mg", "twice daily", "new", "3 more days")],
        "appts": [("GP review", "Primary Care", False), ("Repeat urine culture", "Microbiology", True)],
        "tasks": ["Repeat urine sample"],
    },
]

OK_REPLIES = ["Yes, I'll be there.", "OK, thanks for the reminder.", "Got it, feeling a bit better each day.",
              "Yes confirmed.", "OK. No chest pain and sleeping well.", "Thanks, all good here.", "OK will do."]
WARNING_REPLIES = ["I've had chest pain since last night.", "My legs are very swollen and I'm short of breath.",
                   "Felt dizzy and fainted this morning.", "Running a high fever again and feeling confused.",
                   "The wound is red and leaking."]

SCENARIOS = [("normal", 0.50), ("non_responder", 0.15), ("warning_symptom", 0.12), ("missed_critical", 0.10),
             ("single_miss", 0.08), ("missed_routine", 0.05)]


def _pick_scenario(rng: random.Random) -> str:
    return rng.choices([s for s, _ in SCENARIOS], weights=[w for _, w in SCENARIOS])[0]


def generate_patient(index: int, rng: random.Random, now: datetime, scenario: str | None = None) -> dict[str, Any]:
    scenario = scenario or _pick_scenario(rng)
    cond = rng.choice(CONDITIONS)
    first, last = rng.choice(FIRST), rng.choice(LAST)
    pid = f"P{index:04d}"
    discharge = now - timedelta(days=rng.randint(6, 18), hours=rng.randint(0, 8))
    channel = rng.choice(["sms", "sms", "email", "whatsapp"])

    appointments = []
    for j, (atype, dept, critical) in enumerate(cond["appts"]):
        offset = rng.randint(1, 10) if j == 0 else rng.randint(4, 21)
        appt = {"appointment_id": f"{pid}-A{j + 1}", "type": atype, "department": dept,
                "date": (now + timedelta(days=offset)).date().isoformat(), "critical": critical, "status": "scheduled"}
        appointments.append(appt)
    # past appointment outcomes for scenarios
    critical_appt = next(a for a in appointments if a["critical"])
    routine_appt = next((a for a in appointments if not a["critical"]), None)
    if scenario == "missed_critical":
        critical_appt.update(date=(now - timedelta(days=rng.randint(1, 3))).date().isoformat(), status="missed")
    elif scenario == "missed_routine" and routine_appt:
        routine_appt.update(date=(now - timedelta(days=rng.randint(1, 3))).date().isoformat(), status="missed")
    elif rng.random() < 0.3:
        appointments.append({"appointment_id": f"{pid}-A0", "type": "Post-discharge phone call", "department": cond["ward"],
                             "date": (discharge + timedelta(days=2)).date().isoformat(), "critical": False, "status": "attended"})

    # reminder log: sent between discharge and two days ago
    n_reminders = rng.randint(2, 4)
    span = max((now - timedelta(days=2) - discharge).total_seconds(), 3600)
    sent_times = sorted(discharge + timedelta(seconds=rng.uniform(0, span)) for _ in range(n_reminders))
    sent_times[-1] = max(sent_times[-1], now - timedelta(days=3, hours=rng.randint(0, 12)))  # keep the latest one recent
    log = []
    for k, sent in enumerate(sent_times):
        topic, key = ("post-discharge check-in", "checkin:discharge") if k == 0 else ("medication changes at discharge", "meds:discharge")
        last_two = k >= n_reminders - 2
        if scenario == "non_responder" and last_two:
            responded, reply = False, None
        elif scenario == "single_miss" and k == n_reminders - 1:
            responded, reply = False, None
        elif scenario == "warning_symptom" and k == n_reminders - 1:
            responded, reply = True, rng.choice(WARNING_REPLIES)
        else:
            responded, reply = True, rng.choice(OK_REPLIES)
        entry = {"reminder_id": f"{pid}-R{k + 1}", "sent_at": sent.isoformat(timespec="minutes"), "channel": channel,
                 "topic": topic, "topic_key": key, "responded": responded, "response_text": reply,
                 "response_at": (sent + timedelta(hours=rng.randint(1, 10))).isoformat(timespec="minutes") if responded else None}
        log.append(entry)

    tasks = [{"task_id": f"{pid}-T{t + 1}", "task": task, "status": "open",
              "due": (now + timedelta(days=rng.randint(1, 12))).date().isoformat()} for t, task in enumerate(cond["tasks"])]

    return {
        "patient_id": pid,
        "synthetic": True,
        "scenario": scenario,  # ground-truth label, useful for evaluating the Escalation Agent
        "name": f"{first} {last}",
        "age": rng.randint(28, 89),
        "sex": rng.choice(["Female", "Male"]),
        "contact": {
            "phone": f"+1-555-01{rng.randint(0, 99):02d}",  # 555-01xx numbers are reserved for fiction
            "email": f"{first.lower()}.{last.lower()}@patients.example",
            "preferred_channel": channel,
            "preferred_time": rng.choice(["morning", "afternoon", "evening"]),
            "language": "English",
        },
        "care_team": f"{cond['ward']} follow-up team",
        "discharge": {"date": discharge.date().isoformat(), "ward": cond["ward"], "diagnosis": cond["diagnosis"],
                      "summary": cond["summary"], "warning_signs": cond["warning_signs"]},
        "medications": [{"name": n, "dose": d, "frequency": f, "change": c, "note": note} for n, d, f, c, note in cond["meds"]],
        "appointments": appointments,
        "tasks": tasks,
        "reminder_log": log,
    }


def generate_patients(count: int = 60, seed: int = 42, now: datetime | None = None) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    now = now or datetime.now()
    # Guarantee at least one of every edge case, then sample the rest.
    fixed = [s for s, _ in SCENARIOS]
    return [generate_patient(i + 1, rng, now, fixed[i] if i < len(fixed) else None) for i in range(count)]
