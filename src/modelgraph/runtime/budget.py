"""Hard run budgets."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from modelgraph.compiler.ir import CompiledNode
from modelgraph.contracts.envelope import BudgetState
from modelgraph.contracts.graph import Defaults
from modelgraph.errors import BudgetExceeded
from modelgraph.providers.base import Usage
from modelgraph.util import estimate_tokens


@dataclass
class Budget:
    max_steps: int
    max_depth: int
    max_total_tokens: int
    max_cost_usd: float
    max_wall_seconds: float
    on_budget_exhausted: str = "fail"
    used_steps: int = 0
    used_tokens: int = 0
    used_input_tokens: int = 0
    used_output_tokens: int = 0
    used_cost_usd: float = 0.0

    _started: float = 0.0

    def __post_init__(self) -> None:
        self._started = time.monotonic()

    @classmethod
    def from_defaults(cls, defaults: Defaults) -> Budget:
        return cls(
            max_steps=defaults.max_steps,
            max_depth=defaults.max_depth,
            max_total_tokens=defaults.max_total_tokens,
            max_cost_usd=defaults.max_cost_usd,
            max_wall_seconds=defaults.max_wall_seconds,
            on_budget_exhausted=defaults.on_budget_exhausted,
        )

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._started

    def begin_step(self) -> None:
        self.check_wall()
        self.used_steps += 1
        if self.used_steps > self.max_steps:
            raise BudgetExceeded(
                f"step budget exhausted ({self.max_steps})", details={"code": "MG-4001"}
            )

    def check_depth(self, depth: int) -> None:
        if depth > self.max_depth:
            raise BudgetExceeded(
                f"depth budget exhausted ({self.max_depth})", details={"code": "MG-4006"}
            )

    def check_wall(self) -> None:
        if self.elapsed_seconds > self.max_wall_seconds:
            raise BudgetExceeded(
                f"wall-clock budget exhausted ({self.max_wall_seconds}s)",
                details={"code": "MG-4004"},
            )

    def estimate_tokens(self, messages: Sequence[object], max_output_tokens: int) -> int:
        text = "\n".join(str(getattr(message, "content", message)) for message in messages)
        return estimate_tokens(text) + min(max_output_tokens, 256)

    def check_tokens(self, estimated: int) -> None:
        if self.used_tokens + estimated > self.max_total_tokens:
            raise BudgetExceeded(
                f"token budget exhausted ({self.max_total_tokens})",
                details={"code": "MG-4002"},
            )

    def charge(self, usage: Usage, node: CompiledNode) -> float:
        self.used_tokens += usage.total_tokens
        self.used_input_tokens += usage.input_tokens
        self.used_output_tokens += usage.output_tokens
        cost = self._cost(usage, node)
        self.used_cost_usd += cost
        if self.used_tokens > self.max_total_tokens:
            raise BudgetExceeded(
                f"token budget exceeded ({self.used_tokens} > {self.max_total_tokens})",
                details={"code": "MG-4002"},
            )
        if self.used_cost_usd > self.max_cost_usd:
            raise BudgetExceeded(
                f"cost budget exceeded (${self.used_cost_usd:.4f} > ${self.max_cost_usd:.4f})",
                details={"code": "MG-4003"},
            )
        return cost

    def _cost(self, usage: Usage, node: CompiledNode) -> float:
        if node.model is None:
            return 0.0
        cost = node.model.spec.cost
        if cost.pricing == "free":
            return 0.0
        if cost.pricing == "per_request":
            return cost.per_request_usd
        if cost.pricing == "unknown":
            return 0.0
        return (
            usage.input_tokens / 1_000_000 * cost.input_per_million_usd
            + usage.output_tokens / 1_000_000 * cost.output_per_million_usd
        )

    def check_cost(self, estimated_usd: float) -> None:
        if self.used_cost_usd + estimated_usd > self.max_cost_usd:
            raise BudgetExceeded(
                f"cost budget exhausted (${self.max_cost_usd:.4f})",
                details={"code": "MG-4003"},
            )

    def snapshot(self) -> BudgetState:
        return BudgetState(
            max_steps=self.max_steps,
            used_steps=self.used_steps,
            max_depth=self.max_depth,
            max_total_tokens=self.max_total_tokens,
            used_tokens=self.used_tokens,
            max_cost_usd=self.max_cost_usd,
            used_cost_usd=round(self.used_cost_usd, 8),
            max_wall_seconds=self.max_wall_seconds,
            elapsed_seconds=round(self.elapsed_seconds, 3),
            on_budget_exhausted=self.on_budget_exhausted,
        )
