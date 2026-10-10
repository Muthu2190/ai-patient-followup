"""Object storage: AWS S3 in production, a local folder with the same key layout for dev.

Key layout (relative to the bucket prefix):
    patients/<patient_id>.json          synthetic patient records (input)
    memory/<patient_id>.json            long-term memory: reminders, responses, escalations
    logs/<run_id>/<patient_id>/<agent>.json   every agent output, for review and evaluation
    runs/<run_id>.json                  run manifest with per-patient results
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Protocol

from .config import settings


class Storage(Protocol):
    def put_json(self, key: str, data: Any) -> None: ...
    def get_json(self, key: str, default: Any = None) -> Any: ...
    def list_keys(self, prefix: str) -> list[str]: ...
    def delete(self, key: str) -> None: ...
    @property
    def location(self) -> str: ...


class LocalStorage:
    def __init__(self, root: Path):
        Path(root).mkdir(parents=True, exist_ok=True)
        self.root = Path(root).resolve()

    @property
    def location(self) -> str:
        return str(self.root)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root not in path.parents and path != self.root:
            raise ValueError(f"Key escapes storage root: {key}")
        return path

    def put_json(self, key: str, data: Any) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)

    def get_json(self, key: str, default: Any = None) -> Any:
        path = self._path(key)
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def list_keys(self, prefix: str) -> list[str]:
        base = self._path(prefix) if prefix else self.root
        if not base.exists():
            return []
        return sorted(
            p.relative_to(self.root).as_posix() for p in base.rglob("*") if p.is_file() and not p.name.endswith(".tmp")
        )

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()


class S3Storage:
    def __init__(self, bucket: str, prefix: str, region: str):
        import boto3  # imported lazily so local dev does not need AWS credentials

        if not bucket:
            raise ValueError("S3_BUCKET must be set when STORAGE_BACKEND=s3")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        # On EC2 the client picks up credentials from the instance IAM role.
        self.client = boto3.client("s3", region_name=region)

    @property
    def location(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}"

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_json(self, key: str, data: Any) -> None:
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._key(key),
            Body=body,
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )

    def get_json(self, key: str, default: Any = None) -> Any:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except self.client.exceptions.NoSuchKey:
            return default
        return json.loads(obj["Body"].read())

    def list_keys(self, prefix: str) -> list[str]:
        full_prefix = self._key(prefix)
        strip = len(self.prefix) + 1 if self.prefix else 0
        keys: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            keys.extend(item["Key"][strip:] for item in page.get("Contents", []))
        return sorted(keys)

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._key(key))


@lru_cache(maxsize=1)
def get_storage() -> Storage:
    if settings.storage_backend == "s3":
        return S3Storage(settings.s3_bucket, settings.s3_prefix, settings.aws_region)
    return LocalStorage(settings.local_storage_dir)


# ---- Typed helpers -------------------------------------------------------------

def list_patient_ids(storage: Storage | None = None) -> list[str]:
    storage = storage or get_storage()
    return [Path(k).stem for k in storage.list_keys("patients/") if k.endswith(".json")]


def load_patient(patient_id: str, storage: Storage | None = None) -> dict | None:
    return (storage or get_storage()).get_json(f"patients/{patient_id}.json")


def load_patients(storage: Storage | None = None) -> list[dict]:
    storage = storage or get_storage()
    return [p for pid in list_patient_ids(storage) if (p := load_patient(pid, storage))]


def save_patients(patients: Iterable[dict], storage: Storage | None = None) -> int:
    storage = storage or get_storage()
    count = 0
    for patient in patients:
        storage.put_json(f"patients/{patient['patient_id']}.json", patient)
        count += 1
    return count


def clear_prefix(prefix: str, storage: Storage | None = None) -> None:
    storage = storage or get_storage()
    for key in storage.list_keys(prefix):
        storage.delete(key)
