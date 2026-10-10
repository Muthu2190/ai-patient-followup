import os
import tempfile

# Configure an isolated, offline environment before `followup` is imported.
_tmp = tempfile.mkdtemp(prefix="followup-tests-")
os.environ.update(
    LLM_PROVIDER="mock",
    STORAGE_BACKEND="local",
    LOCAL_STORAGE_DIR=os.path.join(_tmp, "storage"),
    VECTOR_BACKEND="memory",
    OPENAI_API_KEY="",
    GOOGLE_API_KEY="",
)
