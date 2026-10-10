"""LLM access: OpenAI or Gemini through LangChain, or an offline mock.

Every agent calls `generate(task, system, prompt, facts)`. Real providers get the
system + user prompt (which already contains the RAG context). The mock provider
ignores the prompt and writes deterministic text from `facts`, so the whole
pipeline runs end-to-end without network access or API keys.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Protocol

from .config import settings

log = logging.getLogger(__name__)


class LLM(Protocol):
    name: str

    def generate(self, task: str, system: str, prompt: str, facts: dict[str, Any]) -> str: ...


class LangChainLLM:
    def __init__(self, provider: str):
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            self.model = ChatOpenAI(model=settings.openai_model, temperature=0.2, api_key=settings.openai_api_key)
            self.name = f"openai:{settings.openai_model}"
        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            self.model = ChatGoogleGenerativeAI(
                model=settings.gemini_model, temperature=0.2, google_api_key=settings.google_api_key
            )
            self.name = f"gemini:{settings.gemini_model}"
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
        self._fallback = MockLLM()

    def generate(self, task: str, system: str, prompt: str, facts: dict[str, Any]) -> str:
        try:
            result = self.model.invoke([("system", system), ("human", prompt)])
            text = result.content if isinstance(result.content, str) else str(result.content)
            return text.strip()
        except Exception as exc:  # network / quota errors should not stop the follow-up run
            log.warning("LLM call failed for %s (%s); using template fallback", task, exc)
            return self._fallback.generate(task, system, prompt, facts)


class MockLLM:
    """Deterministic template 'LLM' used offline and as a fallback."""

    name = "mock:templates"

    def generate(self, task: str, system: str, prompt: str, facts: dict[str, Any]) -> str:
        handler = getattr(self, f"_{task}", None)
        if handler is None:
            raise ValueError(f"MockLLM has no template for task '{task}'")
        return handler(facts)

    @staticmethod
    def _summary(f: dict[str, Any]) -> str:
        lines = [
            f"{f['name']} ({f['age']}{f['sex'][:1]}) was discharged on {f['discharge_date']} "
            f"after treatment for {f['diagnosis']}."
        ]
        if f.get("med_changes"):
            lines.append("Medication changes at discharge: " + "; ".join(f["med_changes"]) + ".")
        if f.get("due_visits"):
            lines.append("Upcoming visits: " + "; ".join(f["due_visits"]) + ".")
        if f.get("missed_visits"):
            lines.append("Missed visits: " + "; ".join(f["missed_visits"]) + ".")
        if f.get("open_tasks"):
            lines.append("Open follow-up tasks: " + "; ".join(f["open_tasks"]) + ".")
        if f.get("last_response"):
            lines.append(f"Most recent patient response: \"{f['last_response']}\".")
        return " ".join(lines)

    @staticmethod
    def _reminder(f: dict[str, Any]) -> str:
        first = f["first_name"]
        topic = f["topic_type"]
        if topic == "appointment":
            body = (
                f"this is a friendly reminder of your {f['detail']} on {f['when']}. "
                "Please reply YES to confirm or call us to reschedule."
            )
        elif topic == "medication":
            body = (
                f"your care team updated your medicines when you left hospital ({f['detail']}). "
                "Please keep taking them as listed on your discharge letter and reply OK once you have checked."
            )
        elif topic == "missed_appointment":
            body = (
                f"we missed you at your {f['detail']}. "
                "Please call the clinic so we can find a new time that suits you."
            )
        else:
            body = f"we're checking in after your hospital stay. Reminder: {f['detail']}. Reply OK to let us know you saw this."
        return f"Hi {first}, {body}"

    @staticmethod
    def _escalation(f: dict[str, Any]) -> str:
        if f["risk_level"] == "none":
            return "No escalation criteria met; routine follow-up continues."
        reasons = "; ".join(f["reasons"])
        action = (
            "Please contact the patient today and review their follow-up plan."
            if f["risk_level"] == "escalate"
            else "Keep under observation; re-check after the next reminder cycle."
        )
        return f"{f['name']} ({f['diagnosis']}): {reasons}. {action}"


@lru_cache(maxsize=1)
def get_llm() -> LLM:
    provider = settings.resolved_llm_provider
    if provider == "mock":
        return MockLLM()
    try:
        return LangChainLLM(provider)
    except Exception as exc:
        log.warning("Could not initialise %s LLM (%s); falling back to mock", provider, exc)
        return MockLLM()
