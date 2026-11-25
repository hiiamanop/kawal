from __future__ import annotations

from contracts.harness import AgentUsage, CostReport


class BudgetExceededError(RuntimeError):
    pass


class BudgetTracker:
    def __init__(self, max_cost_usd: float) -> None:
        if max_cost_usd < 0:
            raise ValueError("max_cost_usd must be non-negative")
        self._max_cost_usd = max_cost_usd
        self._spent_usd = 0.0
        self._usages: list[tuple[str, AgentUsage]] = []

    @property
    def max_cost_usd(self) -> float:
        return self._max_cost_usd

    @property
    def spent_usd(self) -> float:
        return self._spent_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self._max_cost_usd - self._spent_usd)

    @property
    def calls(self) -> int:
        return len(self._usages)

    def can_afford(self, estimate_usd: float) -> bool:
        if estimate_usd <= 0:
            return True
        return estimate_usd <= self.remaining_usd + 1e-12

    def record(self, role: str, usage: AgentUsage) -> None:
        self._spent_usd += usage.cost_usd
        self._usages.append((role, usage))
        if self._spent_usd > self._max_cost_usd + 1e-12:
            raise BudgetExceededError(
                f"Cost {usage.cost_usd:.6f} pushed spend to {self._spent_usd:.6f}, "
                f"above budget {self._max_cost_usd:.6f}"
            )

    def report(
        self,
        *,
        arbiter_calls: int = 0,
        budget_status: str = "OK",
    ) -> CostReport:
        by_role: dict[str, float] = {}
        total_latency_ms = 0.0
        for role, usage in self._usages:
            by_role[role] = by_role.get(role, 0.0) + usage.cost_usd
            total_latency_ms += usage.latency_ms
        return CostReport(
            total_cost_usd=self._spent_usd,
            total_latency_ms=total_latency_ms,
            calls=len(self._usages),
            arbiter_calls=arbiter_calls,
            by_role=by_role,
            budget_status=budget_status,  # type: ignore[arg-type]
        )
