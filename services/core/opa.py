from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from contracts.models import PolicyInput, PolicyResult


class OpaPolicyClient:
    """Synchronous OPA Data API client that denies ticket creation on any failure."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8181",
        decision_path: str = "kawal/policy/result",
        timeout_seconds: float = 2.0,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/v1/data/{decision_path.strip('/')}"
        self._timeout_seconds = timeout_seconds

    def evaluate_ticket_creation(self, input_: PolicyInput) -> PolicyResult:
        payload = {"input": input_.model_dump(mode="json")}
        request = Request(
            self._url,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                if response.status != 200:
                    return self._deny("OPA_HTTP_ERROR")
                body = json.loads(response.read().decode("utf-8"))
            result = body.get("result")
            if not isinstance(result, dict):
                return self._deny("OPA_INVALID_RESPONSE")
            return PolicyResult.model_validate(result)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            return self._deny("OPA_UNAVAILABLE")
        except Exception:
            return self._deny("OPA_EVALUATION_FAILED")

    @staticmethod
    def _deny(reason_code: str) -> PolicyResult:
        return PolicyResult(
            decision="DENY",
            rule_ids=("POL-OPA-FAIL-CLOSED",),
            reason_codes=(reason_code,),
        )
