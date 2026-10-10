"""Streamlit dashboard for care staff: run follow-ups, review alerts, reminders and agent logs.

    streamlit run streamlit_app.py
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from followup.cli import reset_all
from followup.config import settings
from followup.dashboard_data import all_alerts, list_runs, open_alerts, outbox, run_logs, update_alert
from followup.graph import run_followups
from followup.llm import get_llm
from followup.memory import load_memory, record_response, reminder_history, reminder_outcome
from followup.rag import get_index
from followup.storage import get_storage, list_patient_ids, load_patient, save_patients
from followup.synthetic import generate_patients

st.set_page_config(page_title="Patient Follow-Up Assistant", page_icon="🩺", layout="wide")

RISK_BADGE = {"escalate": "🔴 escalate", "watch": "🟠 watch", "none": "🟢 none"}

storage = get_storage()
index = get_index()

# ---- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.title("🩺 Follow-Up Assistant")
    st.caption("Synthetic data only · escalations go to a human clinician")
    st.markdown(
        f"**LLM:** `{get_llm().name}`  \n**Storage:** `{settings.storage_backend}`  \n"
        f"**Vector DB:** `{index.backend}` ({index.embedding_name}, {index.count()} chunks)"
    )
    st.divider()

    patient_ids = list_patient_ids(storage)
    days_ahead = st.number_input(
        "Simulate run as of N days from now", min_value=0, max_value=14, value=0,
        help="Move the clock forward to see unanswered reminders turn into missed ones.",
    )
    run_now = datetime.now() + timedelta(days=int(days_ahead))
    if st.button("▶ Run follow-ups for all patients", type="primary", disabled=not patient_ids, use_container_width=True):
        bar = st.progress(0.0, text="Starting…")
        manifest = run_followups(
            now=run_now, progress=lambda i, n, pid: bar.progress(i / n, text=f"{pid} ({i}/{n})")
        )
        bar.empty()
        st.session_state["last_run"] = manifest["run_id"]
        st.success(
            f"Run done: {manifest['reminders_queued']} reminders, {manifest['escalations']} escalations, "
            f"{manifest['watch']} watch"
        )

    with st.expander("Synthetic data", expanded=not patient_ids):
        count = st.slider("Patients", 50, 200, 60, step=10)
        seed = st.number_input("Random seed", value=42, step=1)
        reset = st.checkbox("Reset patients, memory, logs and index first", value=True)
        if st.button("Generate synthetic patients", use_container_width=True):
            with st.spinner("Generating and indexing…"):
                if reset:
                    reset_all()
                save_patients(generate_patients(count, int(seed)), storage)
                chunks = index.rebuild(storage)
            st.success(f"{count} patients generated, {chunks} chunks indexed")
            st.rerun()

if not patient_ids:
    st.info("No patients yet — generate synthetic data from the sidebar to get started.")
    st.stop()

tab_overview, tab_alerts, tab_reminders, tab_patient, tab_logs = st.tabs(
    ["Overview", "Clinician alerts", "Reminder outbox", "Patient", "Runs & agent logs"]
)

runs = list_runs(storage)

# ---- Overview ------------------------------------------------------------------
with tab_overview:
    if not runs:
        st.info("No follow-up run yet. Press **Run follow-ups** in the sidebar.")
    else:
        # Latest result per patient across all runs (single-patient runs included).
        latest_by_patient = {}
        for run in reversed(runs):
            for r in run["results"]:
                latest_by_patient[r["patient_id"]] = {**r, "run_id": run["run_id"], "as_of": run["started_for"]}
        results = list(latest_by_patient.values())
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Patients assessed", f"{len(results)} / {len(patient_ids)}")
        c2.metric("Reminders queued", sum(r["reminder_action"] == "send" for r in results))
        c3.metric("Escalate", sum(r["risk_level"] == "escalate" for r in results))
        c4.metric("Watch", sum(r["risk_level"] == "watch" for r in results))
        c5.metric("Open alerts", len(open_alerts(storage)))
        st.caption(f"Latest assessment per patient · most recent run `{runs[0]['run_id']}`")
        df = pd.DataFrame(results)
        if not df.empty:
            df["risk"] = df["risk_level"].map(RISK_BADGE)
            df["reasons"] = df["reasons"].map(lambda r: "; ".join(r))
            risk_filter = st.multiselect("Risk", list(RISK_BADGE.values()), default=list(RISK_BADGE.values()))
            df = df[df["risk"].isin(risk_filter)].sort_values("risk_level", key=lambda s: s.map({"escalate": 0, "watch": 1, "none": 2}))
            st.dataframe(
                df.reindex(columns=["patient_id", "name", "diagnosis", "risk", "reasons", "reminder_action", "channel", "send_at", "as_of", "summary"]),
                use_container_width=True, hide_index=True,
            )

# ---- Alerts --------------------------------------------------------------------
with tab_alerts:
    show_closed = st.toggle("Show acknowledged / resolved alerts", value=False)
    alerts = all_alerts(storage) if show_closed else open_alerts(storage)
    if not alerts:
        st.success("No open alerts.")
    for a in alerts:
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(f"**{RISK_BADGE[a['risk_level']]} · {a['patient_name']}** (`{a['patient_id']}`) — {a['status']}")
            left.write(a["message"])
            left.caption("Reasons: " + "; ".join(a["reasons"]) + f" · assigned to {a['assigned_to']} · {a['created_at']}")
            for act in a.get("actions", []):
                left.caption(f"↳ {act['status']} by {act['by']} at {act['at']}: {act['note']}")
            if a["status"] in ("open", "acknowledged"):
                who = right.text_input("Clinician", key=f"who-{a['alert_id']}", placeholder="Your name")
                note = right.text_input("Note", key=f"note-{a['alert_id']}")
                if a["status"] == "open" and right.button("Acknowledge", key=f"ack-{a['alert_id']}", disabled=not who):
                    update_alert(a["patient_id"], a["alert_id"], "acknowledged", who, note, storage)
                    st.rerun()
                if right.button("Resolve", key=f"res-{a['alert_id']}", disabled=not who):
                    update_alert(a["patient_id"], a["alert_id"], "resolved", who, note, storage)
                    st.rerun()

# ---- Reminders -----------------------------------------------------------------
with tab_reminders:
    rows = outbox(storage)
    if not rows:
        st.info("No reminders generated yet.")
    else:
        df = pd.DataFrame(rows)
        st.dataframe(
            df.reindex(columns=["patient_id", "reminder_id", "channel", "to", "sent_at", "topic", "text", "responded", "response_text"]),
            use_container_width=True, hide_index=True,
        )
        st.subheader("Record a patient reply")
        st.caption("Simulates the patient answering an SMS / email. The next run's Escalation Agent reads it.")
        pending = [r for r in rows if not r.get("responded")]
        if pending:
            choice = st.selectbox(
                "Reminder", pending, format_func=lambda r: f"{r['patient_id']} · {r['reminder_id']} · {r['topic']}"
            )
            reply = st.text_input("Patient reply", placeholder="e.g. Yes I'll be there / I've had chest pain since yesterday")
            if st.button("Save reply", disabled=not reply):
                record_response(choice["patient_id"], choice["reminder_id"], reply, storage=storage)
                st.success("Reply saved to long-term memory. Re-run the follow-up to re-assess risk.")

# ---- Patient -------------------------------------------------------------------
with tab_patient:
    pid = st.selectbox(
        "Patient", patient_ids,
        format_func=lambda p: f"{p} · {(load_patient(p, storage) or {}).get('name', '')}",
    )
    patient = load_patient(pid, storage)
    memory = load_memory(pid, storage)
    if st.button(f"Run follow-up for {pid}"):
        manifest = run_followups([pid], now=run_now)
        r = manifest["results"][0] if manifest["results"] else None
        if r:
            st.success(f"Risk: {RISK_BADGE[r['risk_level']]} · reminder: {r['reminder_action']}")
        memory = load_memory(pid, storage)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"### {patient['name']}")
        st.write(f"{patient['age']} · {patient['sex']} · discharged {patient['discharge']['date']} from {patient['discharge']['ward']}")
        st.write(f"**Diagnosis:** {patient['discharge']['diagnosis']}")
        st.write(patient["discharge"]["summary"])
        st.write("**Warning signs:** " + ", ".join(patient["discharge"]["warning_signs"]))
        st.caption(f"Scenario label (synthetic ground truth): `{patient.get('scenario')}`")
        if memory.get("last_summary"):
            st.markdown("**Latest AI follow-up summary**")
            st.info(memory["last_summary"])
    with c2:
        st.markdown("**Medications**")
        st.dataframe(pd.DataFrame(patient["medications"]), hide_index=True, use_container_width=True)
        st.markdown("**Appointments**")
        st.dataframe(pd.DataFrame(patient["appointments"]), hide_index=True, use_container_width=True)

    st.markdown("**Reminder & response history** (record + assistant memory)")
    now = datetime.now() + timedelta(days=int(days_ahead))
    hist = [{**h, "outcome": reminder_outcome(h, now)} for h in reminder_history(patient, memory)]
    st.dataframe(
        pd.DataFrame(hist).reindex(columns=["sent_at", "source", "channel", "topic", "outcome", "response_text"]),
        hide_index=True, use_container_width=True,
    )

# ---- Runs & logs ---------------------------------------------------------------
with tab_logs:
    if not runs:
        st.info("No runs yet.")
    else:
        run = st.selectbox("Run", runs, format_func=lambda m: f"{m['run_id']} · {m['patients']} patients")
        st.write(
            f"Reminders queued: {run['reminders_queued']} · escalations: {run['escalations']} · watch: {run['watch']}"
            f" · errors: {len(run['errors'])}"
        )
        if run["errors"]:
            st.error(run["errors"])
        pids = [r["patient_id"] for r in run["results"]]
        if pids:
            lpid = st.selectbox("Patient", pids, key="log-pid")
            st.caption(f"Every agent output is logged to `logs/{run['run_id']}/{lpid}/` in {settings.storage_backend} storage")
            for agent, payload in run_logs(run["run_id"], lpid, storage).items():
                with st.expander(agent):
                    st.json(payload)

st.sidebar.caption(f"Model provider setting: `{settings.llm_provider}`")
