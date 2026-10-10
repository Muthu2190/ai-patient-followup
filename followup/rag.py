"""RAG layer: turn patient records + long-term memory into chunks, embed, retrieve.

Vector store: ChromaDB (persistent, via langchain-chroma) or LangChain's in-memory
store. Embeddings: OpenAI / Gemini when a key is configured, otherwise a local
feature-hashing embedder so retrieval works fully offline.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .memory import load_memory, reminder_history
from .storage import Storage, get_storage, load_patients

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbeddings(Embeddings):
    """Offline embedder: hashed unigrams + bigrams, L2-normalised. No model download needed."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        tokens = _TOKEN.findall(text.lower())
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        vec = [0.0] * self.dim
        for feat in features:
            digest = hashlib.md5(feat.encode()).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            vec[idx] += 1.0 if digest[4] % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings() -> tuple[Embeddings, str]:
    provider = settings.resolved_llm_provider
    try:
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=settings.openai_api_key), "openai"
        if provider == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return (
                GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=settings.google_api_key),
                "gemini",
            )
    except Exception as exc:
        log.warning("Embedding provider %s unavailable (%s); using hashing embeddings", provider, exc)
    return HashingEmbeddings(), "hashing"


# ---- Record -> documents --------------------------------------------------------

def patient_documents(patient: dict[str, Any], memory: dict[str, Any]) -> list[Document]:
    pid = patient["patient_id"]
    d = patient["discharge"]
    sections: dict[str, str] = {
        "discharge": (
            f"Patient {pid}: {patient['name']}, age {patient['age']}, {patient['sex']}. "
            f"Discharged {d['date']} from {d['ward']} ward. Primary diagnosis: {d['diagnosis']}. "
            f"Discharge summary: {d['summary']} "
            f"Warning signs to watch for: {', '.join(d['warning_signs'])}. "
            f"Follow-up tasks: {'; '.join(t['task'] + ' (' + t['status'] + ', due ' + t['due'] + ')' for t in patient.get('tasks', []))}."
        ),
        "medications": "Medications at discharge for patient {}: ".format(pid)
        + " ".join(
            f"{m['name']} {m['dose']} {m['frequency']} — status: {m['change']}"
            + (f" ({m['note']})" if m.get("note") else "")
            + "."
            for m in patient["medications"]
        ),
        "appointments": "Appointments for patient {}: ".format(pid)
        + " ".join(
            f"{a['type']} with {a['department']} on {a['date']} — {a['status']}"
            + (" (critical)" if a.get("critical") else "")
            + "."
            for a in patient["appointments"]
        ),
    }
    history = reminder_history(patient, memory)
    if history:
        sections["history"] = "Reminder and response history for patient {}: ".format(pid) + " ".join(
            f"[{r['sent_at']}] {r['channel']} reminder about {r['topic']}: "
            + (f"patient replied \"{r['response_text']}\"." if r.get("responded") else "no reply.")
            for r in history
        )
    if memory.get("escalations"):
        sections["escalations"] = "Past escalations for patient {}: ".format(pid) + " ".join(
            f"[{e['created_at']}] {e['risk_level']}: {'; '.join(e['reasons'])}." for e in memory["escalations"]
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
    docs: list[Document] = []
    for section, text in sections.items():
        for i, chunk in enumerate(splitter.split_text(text)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"patient_id": pid, "section": section, "chunk": i},
                    id=f"{pid}:{section}:{i}",
                )
            )
    return docs


# ---- Vector index ---------------------------------------------------------------

class VectorIndex:
    def __init__(self) -> None:
        self.embeddings, self.embedding_name = get_embeddings()
        self.backend = settings.vector_backend
        self._ids_by_patient: dict[str, list[str]] = {}
        if self.backend == "chroma":
            try:
                from langchain_chroma import Chroma

                self.store = Chroma(
                    collection_name=f"patients_{self.embedding_name}",
                    embedding_function=self.embeddings,
                    persist_directory=str(settings.chroma_dir),
                    collection_metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:
                log.warning("ChromaDB unavailable (%s); using in-memory vector store", exc)
                self.backend = "memory"
        if self.backend != "chroma":
            from langchain_core.vectorstores import InMemoryVectorStore

            self.backend = "memory"
            self.store = InMemoryVectorStore(self.embeddings)

    def count(self) -> int:
        if self.backend == "chroma":
            return self.store._collection.count()
        return len(self.store.store)

    def clear(self) -> None:
        if self.backend == "chroma":
            self.store.reset_collection()
        else:
            self.store.store.clear()

    def _delete_patient(self, patient_id: str) -> None:
        if self.backend == "chroma":
            self.store._collection.delete(where={"patient_id": patient_id})
        else:
            ids = [k for k, v in self.store.store.items() if v["metadata"].get("patient_id") == patient_id]
            if ids:
                self.store.delete(ids)

    def index_patient(self, patient: dict[str, Any], memory: dict[str, Any]) -> int:
        """(Re)index one patient's record and history. Idempotent."""
        self._delete_patient(patient["patient_id"])
        docs = patient_documents(patient, memory)
        self.store.add_documents(docs, ids=[d.id for d in docs])
        return len(docs)

    def rebuild(self, storage: Storage | None = None) -> int:
        storage = storage or get_storage()
        total = 0
        for patient in load_patients(storage):
            total += self.index_patient(patient, load_memory(patient["patient_id"], storage))
        return total

    def retrieve(self, patient_id: str, query: str, k: int = 6) -> list[Document]:
        if self.backend == "chroma":
            return self.store.similarity_search(query, k=k, filter={"patient_id": patient_id})
        return self.store.similarity_search(
            query, k=k, filter=lambda doc: doc.metadata.get("patient_id") == patient_id
        )


@lru_cache(maxsize=1)
def get_index() -> VectorIndex:
    return VectorIndex()


def format_context(docs: list[Document]) -> str:
    return "\n".join(f"[{d.metadata['section']}] {d.page_content}" for d in docs)
