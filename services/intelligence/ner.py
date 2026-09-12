from __future__ import annotations

import re
from typing import Any, Sequence
from pydantic import BaseModel, ConfigDict, Field

from services.intelligence.chunking import TokenSpan, tokenize_with_offsets


SUPPORTED_ENTITY_TYPES = {
    "LOCATION",
    "OBJECT",
    "TIME",
    "LANDMARK",
    "DURATION",
    "AGENCY",
    "SERVICE_ID",
}


class EntitySpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    text: str
    start_token: int = Field(ge=0)
    end_token: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    chunk_id: str | None = None
    source_message_id: str | None = None


def parse_bio_tags(
    tokens: Sequence[str],
    tags: Sequence[str],
    text: str | None = None,
    token_offsets: Sequence[tuple[int, int]] | None = None,
    confidences: Sequence[float] | None = None,
    global_char_offset: int = 0,
    global_token_offset: int = 0,
    chunk_id: str | None = None,
    source_message_id: str | None = None,
) -> list[EntitySpan]:
    if len(tokens) != len(tags):
        raise ValueError(
            f"tokens length ({len(tokens)}) does not match tags length ({len(tags)})"
        )
    if confidences is not None and len(confidences) != len(tokens):
        raise ValueError(
            f"confidences length ({len(confidences)}) does not match tokens length ({len(tokens)})"
        )

    offsets: list[tuple[int, int]]
    if token_offsets is not None:
        if len(token_offsets) != len(tokens):
            raise ValueError("token_offsets length must match tokens length")
        offsets = list(token_offsets)
    elif text is not None:
        offsets = []
        cursor = 0
        for tok in tokens:
            pos = text.find(tok, cursor)
            if pos == -1:
                pos = cursor
            start_pos = pos
            end_pos = pos + len(tok)
            cursor = end_pos
            offsets.append((start_pos, end_pos))
    else:
        offsets = []
        cur = 0
        for tok in tokens:
            offsets.append((cur, cur + len(tok)))
            cur += len(tok) + 1

    spans: list[EntitySpan] = []
    current_label: str | None = None
    current_start_token: int = -1
    current_confidences: list[float] = []

    def close_current_span(end_token_idx: int) -> None:
        nonlocal current_label, current_start_token, current_confidences
        if current_label is None or current_start_token == -1:
            return

        c_start_char = offsets[current_start_token][0]
        c_end_char = offsets[end_token_idx - 1][1]
        
        if text is not None and c_end_char <= len(text):
            span_text = text[c_start_char:c_end_char]
        else:
            span_text = " ".join(tokens[current_start_token:end_token_idx])

        mean_conf = (
            sum(current_confidences) / len(current_confidences)
            if current_confidences
            else 1.0
        )
        mean_conf = max(0.0, min(1.0, mean_conf))

        spans.append(
            EntitySpan(
                label=current_label,
                text=span_text,
                start_token=current_start_token + global_token_offset,
                end_token=end_token_idx + global_token_offset,
                start_char=c_start_char + global_char_offset,
                end_char=c_end_char + global_char_offset,
                confidence=round(mean_conf, 4),
                chunk_id=chunk_id,
                source_message_id=source_message_id,
            )
        )
        current_label = None
        current_start_token = -1
        current_confidences = []

    for idx, (token, tag) in enumerate(zip(tokens, tags)):
        token_conf = confidences[idx] if confidences is not None else 1.0

        if not tag or tag == "O":
            close_current_span(idx)
            continue

        parts = tag.split("-", 1)
        prefix = parts[0].upper()
        label = parts[1].upper() if len(parts) > 1 else "UNKNOWN"

        if prefix == "B":
            close_current_span(idx)
            current_label = label
            current_start_token = idx
            current_confidences = [token_conf]
        elif prefix == "I":
            if current_label == label and current_start_token != -1:
                current_confidences.append(token_conf)
            else:
                close_current_span(idx)
                current_label = label
                current_start_token = idx
                current_confidences = [token_conf]
        else:
            close_current_span(idx)

    close_current_span(len(tokens))
    return spans


def _merge_span_texts(
    prev_text: str,
    prev_start: int,
    prev_end: int,
    span_text: str,
    span_start: int,
    span_end: int,
) -> str:
    if not prev_text:
        return span_text
    if not span_text:
        return prev_text

    if span_start >= prev_start and span_end <= prev_end:
        return prev_text

    if prev_start >= span_start and prev_end <= span_end:
        return span_text

    if prev_start <= span_start and prev_end < span_end:
        if span_start <= prev_end:
            overlap = prev_end - span_start
            suffix = span_text[overlap:] if 0 <= overlap <= len(span_text) else ""
        else:
            gap = span_start - prev_end
            suffix = (" " * gap) + span_text
        return prev_text + suffix

    if span_start < prev_start and span_end < prev_end:
        if prev_start <= span_end:
            overlap = span_end - prev_start
            suffix = prev_text[overlap:] if 0 <= overlap <= len(prev_text) else ""
        else:
            gap = prev_start - span_end
            suffix = (" " * gap) + prev_text
        return span_text + suffix

    return prev_text if len(prev_text) >= len(span_text) else span_text


def merge_overlapping_spans(
    spans: Sequence[EntitySpan],
    text: str | None = None,
) -> list[EntitySpan]:
    if not spans:
        return []

    sorted_spans = sorted(
        spans,
        key=lambda s: (s.start_char, -(s.end_char - s.start_char), -s.confidence),
    )

    merged: list[EntitySpan] = []
    for span in sorted_spans:
        if not merged:
            merged.append(span)
            continue

        prev = merged[-1]
        is_same_label = prev.label == span.label
        has_overlap = max(prev.start_char, span.start_char) < min(prev.end_char, span.end_char)
        is_adjacent = prev.end_char == span.start_char or (prev.end_char + 1 == span.start_char and prev.label == span.label)

        if is_same_label and (has_overlap or is_adjacent):
            new_start_char = min(prev.start_char, span.start_char)
            new_end_char = max(prev.end_char, span.end_char)
            new_start_token = min(prev.start_token, span.start_token)
            new_end_token = max(prev.end_token, span.end_token)
            new_conf = max(prev.confidence, span.confidence)

            if text is not None and 0 <= new_start_char <= new_end_char <= len(text):
                new_text = text[new_start_char:new_end_char]
            else:
                new_text = _merge_span_texts(
                    prev.text,
                    prev.start_char,
                    prev.end_char,
                    span.text,
                    span.start_char,
                    span.end_char,
                )

            merged[-1] = EntitySpan(
                label=prev.label,
                text=new_text,
                start_token=new_start_token,
                end_token=new_end_token,
                start_char=new_start_char,
                end_char=new_end_char,
                confidence=new_conf,
                chunk_id=prev.chunk_id or span.chunk_id,
                source_message_id=prev.source_message_id or span.source_message_id,
            )
        else:
            merged.append(span)

    return merged


_LOCATION_REGEX = re.compile(
    r"\b(?:Jl\.|Jalan|Kelurahan|Kecamatan|Kabupaten|Kota|Desa|RT\s*\d+\s*/?\s*RW\s*\d+|KM\s*\d+)"
    r"(?:\s+[A-Z0-9a-z\.\-]+){1,6}",
    re.UNICODE,
)

_OBJECT_REGEX = re.compile(
    r"\b(?:jalan\s+berlubang|jalan\s+rusak|aspal\s+ambles|lubang\s+jalan|lampu\s+penerangan|lampu\s+jalan|"
    r"tiang\s+listrik|kabel\s+menjuntai|saluran\s+drainase|gorong-gorong|trotoar\s+rusak|"
    r"jembatan\s+retak|pohon\s+tumbang|tumpukan\s+sampah|pipa\s+pdam\s+bocor)\b",
    re.IGNORECASE | re.UNICODE,
)

_TIME_REGEX = re.compile(
    r"\b(?:kemarin(?:\s+(?:pagi|siang|sore|malam))?|tadi\s+(?:pagi|siang|sore|malam)|"
    r"hari\s+ini|seminggu\s+terakhir|sejak\s+\d+\s+(?:hari|minggu|bulan)\s+lalu|"
    r"pukul\s+\d{1,2}[:\.]\d{2}|jam\s+\d{1,2}(?:[:\.]\d{2})?)\b",
    re.IGNORECASE | re.UNICODE,
)


def extract_entities_from_text(
    text: str,
    global_char_offset: int = 0,
    chunk_id: str | None = None,
    source_message_id: str | None = None,
) -> list[EntitySpan]:
    raw_spans: list[EntitySpan] = []

    for match in _LOCATION_REGEX.finditer(text):
        raw_spans.append(
            EntitySpan(
                label="LOCATION",
                text=match.group(0).strip(),
                start_token=0,
                end_token=0,
                start_char=match.start() + global_char_offset,
                end_char=match.end() + global_char_offset,
                confidence=0.95,
                chunk_id=chunk_id,
                source_message_id=source_message_id,
            )
        )

    for match in _OBJECT_REGEX.finditer(text):
        raw_spans.append(
            EntitySpan(
                label="OBJECT",
                text=match.group(0).strip(),
                start_token=0,
                end_token=0,
                start_char=match.start() + global_char_offset,
                end_char=match.end() + global_char_offset,
                confidence=0.92,
                chunk_id=chunk_id,
                source_message_id=source_message_id,
            )
        )

    for match in _TIME_REGEX.finditer(text):
        raw_spans.append(
            EntitySpan(
                label="TIME",
                text=match.group(0).strip(),
                start_token=0,
                end_token=0,
                start_char=match.start() + global_char_offset,
                end_char=match.end() + global_char_offset,
                confidence=0.90,
                chunk_id=chunk_id,
                source_message_id=source_message_id,
            )
        )

    return merge_overlapping_spans(raw_spans, text=text if global_char_offset == 0 else None)
