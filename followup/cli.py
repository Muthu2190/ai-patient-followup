"""Command line entry point.

    python -m followup.cli generate --count 60 --reset   # synthetic data -> storage (S3 or local)
    python -m followup.cli index                         # (re)build the vector index
    python -m followup.cli run [--patient P0001]         # run the 3-agent pipeline
    python -m followup.cli alerts                        # list open clinician alerts
"""
from __future__ import annotations

import argparse
import json
import logging

from .llm import get_llm
from .storage import clear_prefix, get_storage, save_patients


def cmd_generate(args: argparse.Namespace) -> None:
    from .synthetic import generate_patients

    if args.reset:
        reset_all()
    patients = generate_patients(args.count, args.seed)
    n = save_patients(patients)
    print(f"Wrote {n} synthetic patients to {get_storage().location}/patients/")
    cmd_index(args)


def reset_all() -> None:
    for prefix in ("patients/", "memory/", "logs/", "runs/"):
        clear_prefix(prefix)
    from .rag import get_index

    get_index().clear()


def cmd_index(_: argparse.Namespace) -> None:
    from .rag import get_index

    index = get_index()
    chunks = index.rebuild()
    print(f"Indexed {chunks} chunks into {index.backend} ({index.embedding_name} embeddings)")


def cmd_run(args: argparse.Namespace) -> None:
    from .graph import run_followups

    print(f"LLM: {get_llm().name} | storage: {get_storage().location}")
    manifest = run_followups(
        [args.patient] if args.patient else None,
        progress=lambda i, n, pid: print(f"  [{i}/{n}] {pid}", end="\r"),
    )
    print()
    for r in manifest["results"]:
        flag = {"escalate": "!!", "watch": "! ", "none": "  "}[r["risk_level"]]
        print(f"{flag} {r['patient_id']} {r['name']:<22} {r['risk_level']:<8} reminder={r['reminder_action']:<4} "
              + ("; ".join(r["reasons"]) if r["reasons"] else ""))
    print(
        f"\nRun {manifest['run_id']}: {manifest['patients']} patients, {manifest['reminders_queued']} reminders queued, "
        f"{manifest['escalations']} escalations, {manifest['watch']} on watch, {len(manifest['errors'])} errors"
    )
    if args.patient and manifest["results"]:
        print(json.dumps(manifest["results"][0], indent=2))


def cmd_alerts(_: argparse.Namespace) -> None:
    from .dashboard_data import open_alerts

    alerts = open_alerts()
    for a in alerts:
        print(f"[{a['risk_level'].upper()}] {a['patient_id']} {a['patient_name']}: {a['message']}")
    print(f"{len(alerts)} open alert(s)")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="followup", description="AI Patient Follow-Up Assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate synthetic patients and index them")
    g.add_argument("--count", type=int, default=60)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--reset", action="store_true", help="delete existing patients, memory, logs and index first")
    g.set_defaults(func=cmd_generate)

    sub.add_parser("index", help="rebuild the vector index").set_defaults(func=cmd_index)

    r = sub.add_parser("run", help="run the follow-up pipeline")
    r.add_argument("--patient", help="run a single patient id")
    r.set_defaults(func=cmd_run)

    sub.add_parser("alerts", help="list open clinician alerts").set_defaults(func=cmd_alerts)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
