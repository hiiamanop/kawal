from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from contracts.models import Category


@dataclass(frozen=True)
class CausalAnalysisResult:
    has_causal_relation: bool
    cause_clause: str | None = None
    effect_clause: str | None = None
    root_cause_category: Category | None = None
    secondary_category: Category | None = None
    hazard_escalation_category: Category | None = None
    explanation: str | None = None


_CAUSE_AFTER_PATTERN = re.compile(
    r"\b(?:gara-gara|gara2|karena|krn|akibat|sebab)\b", re.IGNORECASE
)
_CAUSE_BEFORE_PATTERN = re.compile(
    r"\b(?:bikin|menyebabkan|mengakibatkan|menimbulkan|memicu)\b", re.IGNORECASE
)
_ESCALATION_PATTERN = re.compile(
    r"\b(?:sampe|sampai|hingga)\b", re.IGNORECASE
)

_FIRE_HAZARD_KEYWORDS = (
    "percikan api",
    "kabel meletup",
    "kabel listrik meletup",
    "tiang listrik meletup",
    "kebakaran",
    "meledak",
    "kobaran api",
    "terpercik api",
)

_PUBLIC_ORDER_KEYWORDS = (
    "pkl",
    "pedagang kaki lima",
    "lapak liar",
    "tenda liar",
    "jualan di trotoar",
    "bahu jalan dipake jualan",
    "trotoar dipake jualan",
    "balap liar",
    "pungli",
    "premanisme",
)


def _detect_priority_domain(text: str) -> Category | None:
    text_lower = text.lower()
    if any(kw in text_lower for kw in _FIRE_HAZARD_KEYWORDS):
        return Category.FIRE_RESCUE
    if any(kw in text_lower for kw in _PUBLIC_ORDER_KEYWORDS):
        # Only route to PUBLIC_ORDER if not explicitly about broken physical pavement
        if not any(dmg in text_lower for dmg in ("trotoar hancur", "paving hancur", "trotoar jebol", "aspal bolong")):
            return Category.PUBLIC_ORDER
    return None


def analyze_causality(
    text: str,
    ml_runtime: Any = None,
    llm_adapter: Any = None,
) -> CausalAnalysisResult:
    """Analyze Indonesian citizen complaint text to disentangle root cause from secondary impacts."""
    priority_cat = _detect_priority_domain(text)

    cause_clause: str | None = None
    effect_clause: str | None = None

    # Check GARA-GARA / KARENA pattern: [Effect] connector [Cause]
    parts = _CAUSE_AFTER_PATTERN.split(text, maxsplit=1)
    if len(parts) == 2 and len(parts[0].strip()) >= 8 and len(parts[1].strip()) >= 8:
        effect_clause = parts[0].strip()
        cause_clause = parts[1].strip()

    # Check BIKIN / MENYEBABKAN pattern: [Cause] connector [Effect]
    if cause_clause is None:
        parts = _CAUSE_BEFORE_PATTERN.split(text, maxsplit=1)
        if len(parts) == 2 and len(parts[0].strip()) >= 8 and len(parts[1].strip()) >= 8:
            cause_clause = parts[0].strip()
            effect_clause = parts[1].strip()

    # Check SAMPE / SAMPAI escalation pattern: [Trigger] connector [Escalation]
    if cause_clause is None:
        parts = _ESCALATION_PATTERN.split(text, maxsplit=1)
        if len(parts) == 2 and len(parts[0].strip()) >= 8 and len(parts[1].strip()) >= 8:
            cause_clause = parts[0].strip()
            effect_clause = parts[1].strip()

    if cause_clause is not None and effect_clause is not None and ml_runtime is not None:
        try:
            cause_pred = ml_runtime.predict(cause_clause).prediction
            effect_pred = ml_runtime.predict(effect_clause).prediction

            if cause_pred is not None and effect_pred is not None:
                cat_cause = Category(cause_pred.category)
                cat_effect = Category(effect_pred.category)

                # If priority category (e.g. FIRE_RESCUE or PUBLIC_ORDER) was detected, preserve it
                effective_root = priority_cat or cat_cause

                if effective_root != cat_effect:
                    return CausalAnalysisResult(
                        has_causal_relation=True,
                        cause_clause=cause_clause,
                        effect_clause=effect_clause,
                        root_cause_category=effective_root,
                        secondary_category=cat_effect,
                        hazard_escalation_category=priority_cat,
                        explanation=f"Akar penyebab: {effective_root.value} ({cause_clause}); Dampak permukaan: {cat_effect.value} ({effect_clause})",
                    )
        except Exception:
            pass

    # Semantic Fallback: If no explicit lexical connector matched, use LLM disambiguation
    if llm_adapter is not None and hasattr(llm_adapter, "disambiguate_complaint"):
        try:
            dis = llm_adapter.disambiguate_complaint(text)
            if dis and isinstance(dis, dict):
                root_str = dis.get("root_cause_category")
                sec_str = dis.get("secondary_category")
                all_cats = {c.value for c in Category}
                if root_str and root_str in all_cats:
                    root_cat = Category(root_str)
                    sec_cat = Category(sec_str) if sec_str in all_cats else None
                    if sec_cat is not None and root_cat != sec_cat:
                        effective_root = priority_cat or root_cat
                        return CausalAnalysisResult(
                            has_causal_relation=True,
                            root_cause_category=effective_root,
                            secondary_category=sec_cat,
                            hazard_escalation_category=priority_cat,
                            explanation=dis.get("reasoning", "Disambiguasi semantik kausalitas via OmniRoute"),
                        )
        except Exception:
            pass

    return CausalAnalysisResult(
        has_causal_relation=False,
        hazard_escalation_category=priority_cat,
    )
