# AI Patient Follow-Up Assistant

Working implementation of the *Architecture One-Pager v1.0.0*. Three agents run as a
sequential **LangGraph** pipeline over shared state. They summarise each discharged
patient's case, send personalised reminders and escalate at-risk patients to clinicians.
Every LLM call is grounded in the patient's records through **LangChain RAG** over
**ChromaDB**. Care staff work from a **Streamlit** dashboard.

```
Care staff ─► Streamlit ─► LangGraph orchestrator ──(shared state)──┐
                               │                                     │
     load_context ─► 1 Patient Summary ─► 2 Reminder ─► 3 Escalation ─► persist_memory
                          │      │             │            │              │
                          └──── RAG retriever (ChromaDB) ◄──┘    long-term memory + reindex
                                         ▲                               │
                                  S3 / local storage ◄───────────────────┘
                                  (patients, memory, logs, runs)
```

## Quick start (local, no API key needed)

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # macOS/Linux: cp .env.example .env
python -m followup.cli generate --count 60 --reset
python -m followup.cli run
streamlit run streamlit_app.py    # http://localhost:8501
```

If no `OPENAI_API_KEY` or `GOOGLE_API_KEY` is set, the app uses an **offline mock
LLM** (deterministic templates) and **hashing embeddings**, so the whole pipeline runs
without network access. To use real models, add a key to `.env`. `LLM_PROVIDER=auto`
picks OpenAI first, then Gemini. If an LLM call fails, that single call falls back to the template.

## Project layout

| Path | What it is |
|---|---|
| `followup/graph.py` | LangGraph `StateGraph`: `load_context → patient_summary_agent → reminder_agent → escalation_agent → persist_memory` |
| `followup/state.py` | Short-term memory: the typed graph state shared by the agents |
| `followup/agents/summary.py` | **Agent 1**: retrieves records via RAG and writes the follow-up summary (diagnosis, meds, due visits, open tasks) |
| `followup/agents/reminder.py` | **Agent 2**: picks the next reminder topic, skips duplicates, writes text, channel and send time |
| `followup/agents/escalation.py` | **Agent 3**: rule-based risk flag (none / watch / escalate) plus an LLM-worded clinician alert |
| `followup/rag.py` | Chunking (LangChain splitter), embeddings, ChromaDB / in-memory vector store, per-patient retrieval |
| `followup/memory.py` | Long-term memory per patient (reminders, replies, escalations) persisted to storage |
| `followup/storage.py` | `S3Storage` (boto3, IAM role) and `LocalStorage` with the same key layout |
| `followup/llm.py` | OpenAI / Gemini via LangChain, plus the offline `MockLLM` |
| `followup/guardrails.py` | Synthetic-only check, no-clinical-advice filter, agent output logging |
| `followup/synthetic.py` | Generator for 50–200 fictional patients, including edge cases |
| `followup/cli.py` | `generate`, `index`, `run`, `alerts` commands |
| `streamlit_app.py` | Staff dashboard |
| `tests/` | End-to-end pytest suite (offline) |
| `Dockerfile`, `docker-compose.yml`, `deploy/` | EC2 + S3 deployment |

### Storage layout (S3 bucket prefix or `./storage`)

```
patients/<id>.json             synthetic input records
memory/<id>.json               long-term memory: reminders, replies, escalations, last summary
logs/<run_id>/<id>/<agent>.json  every agent output (guardrail: logged for review)
runs/<run_id>.json             run manifest
```

## How the agents behave

**Patient Summary Agent.** Retrieves the top chunks for the patient from the vector store,
using a `patient_id` metadata filter. It asks the LLM for a 4–6 sentence summary that uses only that context.
Structured facts (due and missed visits, medication changes, open tasks) are computed from the
record and passed down the graph.

**Reminder Agent.** Candidate topics in priority order are upcoming appointments (critical ones first),
missed appointments to reschedule, medication changes and open tasks. A topic is skipped if
the patient already acknowledged it, or if a reminder on it went out in the last
`DUPLICATE_REMINDER_WINDOW_DAYS`. The send time follows the patient's preferred contact slot.
Urgent topics (an appointment within 24 h) go out in 15 minutes.

**Escalation Agent.** These are the rules from the one-pager:

| Rule | Level |
|---|---|
| 2+ reminders in a row with no reply (reply window is 24 h) | escalate |
| Reply mentions a warning symptom (generic list + the patient's discharge warning signs; simple negation such as "no chest pain" is ignored) | escalate |
| Missed critical appointment | escalate |
| Last reminder unanswered / missed routine appointment | watch |

The risk level is rule-based so it can be audited. The LLM only words the alert and cannot lower the level.
Alerts go to a human clinician, who acknowledges or resolves them in the dashboard.

**Memory.** Short-term memory is the graph state. The `persist_memory` node appends the new reminder and alert
to long-term memory and re-indexes the patient. Replies recorded later (dashboard → *Reminder outbox*)
are read by the next run. The sidebar's *Simulate run as of N days from now* lets you watch an
unanswered reminder turn into a missed one and escalate.

## Guardrails

- **Synthetic data only.** Records must have `"synthetic": true` and a `*.example` email, or the pipeline refuses them. Phone numbers use the fictional 555-01xx range.
- **Reminders give no clinical advice.** LLM output is checked for dosing, start/stop and diagnosis language. If any is found, it is replaced with a safe template. A safety footer is always appended.
- **Escalations go to humans.** The system only raises alerts. Clinicians act on them.
- **Everything is logged.** Each agent's output, and the retrieved context, is written to `logs/`.

## Tests

```bash
pytest -q
```

The suite runs fully offline and checks that:

- every synthetic scenario gets its expected risk level;
- all four logs are written;
- summaries cite retrieved chunks;
- reminders are not duplicated across runs;
- the reminder guardrails hold;
- repeated non-response escalates across runs;
- a recorded symptom reply escalates;
- negation handling works;
- the advice filter works;
- non-synthetic records are refused.

## Deploying to AWS (EC2 + S3)

1. Create an S3 bucket (block public access, SSE on). Create a Secrets Manager secret `followup/llm-keys` with `{"OPENAI_API_KEY": "..."}` (and/or `GOOGLE_API_KEY`).
2. Create an IAM role for EC2. A role takes two different policy documents:
   - **Trust policy** (who can assume the role): [`deploy/ec2-trust-policy.json`](deploy/ec2-trust-policy.json). In the console, choosing *AWS service → EC2* as the trusted entity creates it for you.
   - **Permissions policy** (what the role may do): [`deploy/iam-policy.json`](deploy/iam-policy.json). Replace `YOUR-FOLLOWUP-BUCKET` first, then attach it with *Add permissions → Create inline policy → JSON*. Never paste it into the trust policy editor.

   With the AWS CLI:
   ```bash
   aws iam create-role --role-name followup-ec2-role --assume-role-policy-document file://deploy/ec2-trust-policy.json
   aws iam put-role-policy --role-name followup-ec2-role --policy-name followup-s3-secrets --policy-document file://deploy/iam-policy.json
   aws iam create-instance-profile --instance-profile-name followup-ec2-profile
   aws iam add-role-to-instance-profile --instance-profile-name followup-ec2-profile --role-name followup-ec2-role
   ```
3. Create a security group that allows inbound **TCP 8501 only from your known IPs**, plus SSH if you need it.
4. Launch a **t3.medium** (Amazon Linux 2023) with that role and security group. Use [`deploy/ec2-user-data.sh`](deploy/ec2-user-data.sh) as user data, after filling in the repo URL and bucket.
5. Seed data into S3 once, from the instance or any machine with bucket access:
   ```bash
   docker compose exec followup python -m followup.cli generate --count 100 --reset
   ```
6. Open `http://<ec2-public-ip>:8501`.

With `STORAGE_BACKEND=s3`, patients, memory, logs and runs live in S3. The ChromaDB index lives
on the container volume. If the index is empty (for example on a fresh instance), it is rebuilt from S3 automatically on the first run.
API keys are read from the environment, which is populated from Secrets Manager at boot. They are never stored in code.

To run locally in Docker:

```bash
docker compose up --build
```
