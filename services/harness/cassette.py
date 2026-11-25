from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from services.harness.llm import ChatClient, ChatClientError, ChatResponse


class CassetteError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class CassetteEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_key: str = Field(min_length=64, max_length=64)
    model: str
    provider: str
    system_prompt_sha256: str = Field(min_length=64, max_length=64)
    user_prompt_sha256: str = Field(min_length=64, max_length=64)
    max_tokens: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0.0)
    response: ChatResponse | None = None
    error_code: str | None = None
    error_message: str | None = None
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _exactly_one_result(self) -> "CassetteEntry":
        has_response = self.response is not None
        has_error = self.error_code is not None
        if has_response == has_error:
            raise ValueError("entry must contain exactly one of response or error_code")
        return self


class CassetteFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    model_id: str
    provider: str
    input_price_per_million: float = Field(default=0.0, ge=0.0)
    output_price_per_million: float = Field(default=0.0, ge=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    entries: tuple[CassetteEntry, ...] = ()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _request_key(
    *,
    model: str,
    provider: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    timeout_seconds: float,
) -> str:
    payload = {
        "model": model,
        "provider": provider,
        "system_prompt_sha256": _sha256(system_prompt),
        "user_prompt_sha256": _sha256(user_prompt),
        "max_tokens": max_tokens,
        "timeout_seconds": timeout_seconds,
    }
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")))


class CassetteChatClient:
    """Record or replay chat responses without storing raw prompts."""

    def __init__(
        self,
        path: str | Path,
        *,
        mode: Literal["record", "replay"],
        inner: ChatClient | None = None,
        input_price_per_million: float = 0.0,
        output_price_per_million: float = 0.0,
        overwrite: bool = False,
    ) -> None:
        self._path = Path(path)
        self._mode = mode
        self._inner = inner
        self._indices: dict[str, int] = {}
        self._entries: list[CassetteEntry] = []

        if mode == "record":
            if inner is None:
                raise CassetteError(
                    "CASSETTE_INNER_REQUIRED", "Record mode needs a client"
                )
            if self._path.exists() and not overwrite:
                raise CassetteError(
                    "CASSETTE_EXISTS",
                    f"Cassette already exists: {self._path}",
                )
            self.model_id = inner.model_id
            self.provider = inner.provider
            self._input_price = input_price_per_million
            self._output_price = output_price_per_million
            return

        if inner is not None:
            raise CassetteError(
                "CASSETTE_INNER_FORBIDDEN", "Replay mode does not call a live client"
            )
        cassette = self._load()
        self.model_id = cassette.model_id
        self.provider = cassette.provider
        self._input_price = input_price_per_million or (
            cassette.input_price_per_million
        )
        self._output_price = output_price_per_million or (
            cassette.output_price_per_million
        )
        self._entries = list(cassette.entries)

    @property
    def entries(self) -> tuple[CassetteEntry, ...]:
        return tuple(self._entries)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self._input_price + output_tokens * self._output_price
        ) / 1_000_000.0

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        timeout_seconds: float,
    ) -> ChatResponse:
        key = _request_key(
            model=self.model_id,
            provider=self.provider,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        if self._mode == "record":
            return self._record(
                key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
        return self._replay(key)

    def _record(
        self,
        key: str,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        timeout_seconds: float,
    ) -> ChatResponse:
        assert self._inner is not None
        try:
            response = self._inner.complete(
                system_prompt, user_prompt, max_tokens, timeout_seconds
            )
        except ChatClientError as exc:
            self._entries.append(
                CassetteEntry(
                    request_key=key,
                    model=self.model_id,
                    provider=self.provider,
                    system_prompt_sha256=_sha256(system_prompt),
                    user_prompt_sha256=_sha256(user_prompt),
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    error_code=exc.code,
                    error_message=str(exc),
                )
            )
            self._persist()
            raise

        self._entries.append(
            CassetteEntry(
                request_key=key,
                model=self.model_id,
                provider=self.provider,
                system_prompt_sha256=_sha256(system_prompt),
                user_prompt_sha256=_sha256(user_prompt),
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                response=response,
            )
        )
        self._persist()
        return response

    def _replay(self, key: str) -> ChatResponse:
        matches = [entry for entry in self._entries if entry.request_key == key]
        index = self._indices.get(key, 0)
        if index >= len(matches):
            raise ChatClientError(
                "CASSETTE_MISS", "No recorded response for this request"
            )
        self._indices[key] = index + 1
        entry = matches[index]
        if entry.error_code is not None:
            raise ChatClientError(entry.error_code, "Replayed recorded error")
        assert entry.response is not None
        return entry.response

    def _load(self) -> CassetteFile:
        if not self._path.is_file():
            raise CassetteError(
                "CASSETTE_NOT_FOUND", f"Cassette not found: {self._path}"
            )
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CassetteError(
                "CASSETTE_INVALID", f"Invalid cassette: {self._path}"
            ) from exc
        try:
            return CassetteFile.model_validate(payload)
        except Exception as exc:
            raise CassetteError(
                "CASSETTE_INVALID", f"Invalid cassette: {self._path}"
            ) from exc

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        cassette = CassetteFile(
            model_id=self.model_id,
            provider=self.provider,
            input_price_per_million=self._input_price,
            output_price_per_million=self._output_price,
            entries=tuple(self._entries),
        )
        temporary = self._path.with_name(f"{self._path.name}.tmp")
        temporary.write_text(
            json.dumps(cassette.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self._path)
