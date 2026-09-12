from __future__ import annotations

import re
from typing import Any, Sequence
from pydantic import BaseModel, ConfigDict, Field


DEFAULT_MAX_TOKENS = 448
DEFAULT_OVERLAP_TOKENS = 64
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class TokenSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    token: str
    token_index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    chunk_index: int = Field(ge=0)
    total_chunks: int = Field(ge=1)
    text: str
    token_count: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    start_token: int = Field(ge=0)
    end_token: int = Field(ge=0)
    source_message_id: str | None = None
    source_message_ids: tuple[str, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_global_char_offset(self, local_offset: int) -> int:
        if local_offset < 0 or local_offset > len(self.text):
            raise ValueError(
                f"local_offset {local_offset} out of bounds for chunk text of length {len(self.text)}"
            )
        return self.start_char + local_offset

    def to_local_char_offset(self, global_offset: int) -> int | None:
        if self.start_char <= global_offset <= self.end_char:
            return global_offset - self.start_char
        return None

    def contains_global_offset(self, global_offset: int) -> bool:
        return self.start_char <= global_offset <= self.end_char


def tokenize_with_offsets(text: str) -> list[TokenSpan]:
    tokens: list[TokenSpan] = []
    for idx, match in enumerate(TOKEN_PATTERN.finditer(text)):
        tokens.append(
            TokenSpan(
                token=match.group(0),
                token_index=idx,
                start_char=match.start(),
                end_char=match.end(),
            )
        )
    return tokens


def chunk_text(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap: int = DEFAULT_OVERLAP_TOKENS,
    source_message_id: str | None = None,
    chunk_id_prefix: str = "chk",
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")
    if overlap < 0 or overlap >= max_tokens:
        raise ValueError("overlap must be non-negative and less than max_tokens")

    base_metadata = metadata or {}
    tokens = tokenize_with_offsets(text)

    if not tokens:
        return [
            Chunk(
                chunk_id=f"{chunk_id_prefix}-0",
                chunk_index=0,
                total_chunks=1,
                text="",
                token_count=0,
                start_char=0,
                end_char=0,
                start_token=0,
                end_token=0,
                source_message_id=source_message_id,
                source_message_ids=(source_message_id,) if source_message_id else (),
                metadata=base_metadata,
            )
        ]

    total_tokens = len(tokens)
    step = max_tokens - overlap
    slices: list[tuple[int, int]] = []

    start = 0
    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        slices.append((start, end))
        if end >= total_tokens:
            break
        start += step

    total_chunks = len(slices)
    chunks: list[Chunk] = []

    for idx, (token_start, token_end) in enumerate(slices):
        start_char = tokens[token_start].start_char
        end_char = tokens[token_end - 1].end_char
        chunk_text_slice = text[start_char:end_char]
        chunk_id = f"{chunk_id_prefix}-{idx}"

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                chunk_index=idx,
                total_chunks=total_chunks,
                text=chunk_text_slice,
                token_count=token_end - token_start,
                start_char=start_char,
                end_char=end_char,
                start_token=token_start,
                end_token=token_end,
                source_message_id=source_message_id,
                source_message_ids=(source_message_id,) if source_message_id else (),
                metadata=dict(base_metadata),
            )
        )

    return chunks


def chunk_messages(
    messages: Sequence[Any],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap: int = DEFAULT_OVERLAP_TOKENS,
    case_id: str | None = None,
) -> list[Chunk]:
    if not messages:
        return []

    prefix = f"case-{case_id}" if case_id else "msg"
    combined_parts: list[str] = []
    message_spans: list[tuple[str, int, int]] = []
    current_char = 0

    for msg in messages:
        if isinstance(msg, dict):
            source_id = str(msg.get("source_message_id") or msg.get("message_id") or "unknown")
            text = str(msg.get("text", ""))
        else:
            source_id = getattr(msg, "source_message_id", getattr(msg, "message_id", "unknown"))
            text = getattr(msg, "text", "")

        start_c = current_char
        combined_parts.append(text)
        current_char += len(text)
        end_c = current_char
        message_spans.append((source_id, start_c, end_c))

        combined_parts.append("\n\n")
        current_char += 2

    full_text = "".join(combined_parts)
    raw_chunks = chunk_text(
        full_text,
        max_tokens=max_tokens,
        overlap=overlap,
        chunk_id_prefix=prefix,
    )

    enriched_chunks: list[Chunk] = []
    for chk in raw_chunks:
        associated_ids: list[str] = []
        for src_id, m_start, m_end in message_spans:
            if max(chk.start_char, m_start) < min(chk.end_char, m_end):
                associated_ids.append(src_id)

        primary_id = associated_ids[0] if associated_ids else None
        enriched_chunks.append(
            Chunk(
                chunk_id=chk.chunk_id,
                chunk_index=chk.chunk_index,
                total_chunks=chk.total_chunks,
                text=chk.text,
                token_count=chk.token_count,
                start_char=chk.start_char,
                end_char=chk.end_char,
                start_token=chk.start_token,
                end_token=chk.end_token,
                source_message_id=primary_id,
                source_message_ids=tuple(associated_ids),
                metadata=dict(chk.metadata),
            )
        )

    return enriched_chunks
