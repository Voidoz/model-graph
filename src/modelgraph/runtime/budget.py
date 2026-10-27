"""Hard run budgets.

A budget is shared by every branch, subgraph, and (under fan-out) worker thread of one
run. All mutations and pre-flight checks take the same re-entrant lock, so two concurrent
branches cannot both pass a check and then overspend it.

Pre-flight uses a reservation rather than a bare check: a call reserves the tokens and
cost it may consume before the provider is invoked, and the reservation is settled with
the provider's reported usage afterwards. A scheduler can reserve a whole fan-out batch
atomically before starting any branch, which is what prevents a partial-start overspend.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from modelgraph.compiler.ir import CompiledNode
from modelgraph.contracts.envelope import BudgetState
from modelgraph.contracts.graph import Defaults
from modelgraph.errors import BudgetExceeded
from modelgraph.providers.base import Usage
from modelgraph.util import estimate_tokens


@dataclass
class Reservation:
    """A pre-flight hold on some tokens and cost.

    The object is consumed exactly once by :meth:`Budget.settle` or released by
    :meth:`Budget.release`. Both calls are idempotent so an executor can put a release in
    a ``finally`` block without masking a successful settle.
    """

    tokens: int
    cost_usd: float
    consumed: bool = False


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
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    reserved_tokens: int = 0
    reserved_cost_usd: float = 0.0

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
        with self._lock:
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

    def estimate_input_tokens(self, messages: Sequence[object]) -> int:
        text = "\n".join(str(getattr(message, "content", message)) for message in messages)
        return estimate_tokens(text)

    def estimate_cost(self, node: CompiledNode, *, input_tokens: int, output_tokens: int) -> float:
        """Estimate a call's cost from the manifest prices (never above the actual model).

        ``free`` and ``unknown`` pricing estimate zero, matching :meth:`cost_for`. A
        ``per_request`` model is charged once per attempt, so its estimate is its price.
        """
        model = node.model
        if model is None:
            return 0.0
        cost = model.spec.cost
        if cost.pricing == "free":
            return 0.0
        if cost.pricing == "per_request":
            return cost.per_request_usd
        if cost.pricing == "unknown":
            return 0.0
        return (
            input_tokens / 1_000_000 * cost.input_per_million_usd
            + output_tokens / 1_000_000 * cost.output_per_million_usd
        )

    def check_tokens(self, estimated: int) -> None:
        with self._lock:
            if self.used_tokens + self.reserved_tokens + estimated > self.max_total_tokens:
                raise BudgetExceeded(
                    f"token budget exhausted ({self.max_total_tokens})",
                    details={"code": "MG-4002"},
                )

    def check_cost(self, estimated_usd: float) -> None:
        with self._lock:
            if self.used_cost_usd + self.reserved_cost_usd + estimated_usd > self.max_cost_usd:
                raise BudgetExceeded(
                    f"cost budget exhausted (${self.max_cost_usd:.4f})",
                    details={"code": "MG-4003"},
                )

    def reserve(self, tokens: int, cost_usd: float) -> Reservation:
        """Atomically hold ``tokens``/``cost_usd`` for a call that is about to start."""

        with self._lock:
            if self.used_tokens + self.reserved_tokens + tokens > self.max_total_tokens:
                raise BudgetExceeded(
                    f"token budget exhausted ({self.max_total_tokens})",
                    details={
                        "code": "MG-4002",
                        "estimated_tokens": tokens,
                        "used_tokens": self.used_tokens,
                        "reserved_tokens": self.reserved_tokens,
                    },
                )
            if self.used_cost_usd + self.reserved_cost_usd + cost_usd > self.max_cost_usd:
                raise BudgetExceeded(
                    f"cost budget exhausted (${self.max_cost_usd:.4f})",
                    details={
                        "code": "MG-4003",
                        "estimated_cost_usd": cost_usd,
                        "used_cost_usd": self.used_cost_usd,
                        "reserved_cost_usd": self.reserved_cost_usd,
                    },
                )
            self.reserved_tokens += tokens
            self.reserved_cost_usd += cost_usd
            return Reservation(tokens=tokens, cost_usd=cost_usd)

    def reserve_many(self, estimates: Sequence[tuple[int, float]]) -> list[Reservation]:
        """Reserve a batch atomically; either every item fits or none is reserved."""

        with self._lock:
            total_tokens = sum(item[0] for item in estimates)
            total_cost = sum(item[1] for item in estimates)
            if self.used_tokens + self.reserved_tokens + total_tokens > self.max_total_tokens:
                raise BudgetExceeded(
                    f"fan-out pre-flight exceeded the token budget "
                    f"({self.max_total_tokens}); no branch was started",
                    details={
                        "code": "MG-4002",
                        "estimated_tokens": total_tokens,
                        "used_tokens": self.used_tokens,
                        "reserved_tokens": self.reserved_tokens,
                    },
                )
            if self.used_cost_usd + self.reserved_cost_usd + total_cost > self.max_cost_usd:
                raise BudgetExceeded(
                    f"fan-out pre-flight exceeded the cost budget "
                    f"(${self.max_cost_usd:.4f}); no branch was started",
                    details={
                        "code": "MG-4003",
                        "estimated_cost_usd": total_cost,
                        "used_cost_usd": self.used_cost_usd,
                        "reserved_cost_usd": self.reserved_cost_usd,
                    },
                )
            self.reserved_tokens += total_tokens
            self.reserved_cost_usd += total_cost
            return [Reservation(tokens=tokens, cost_usd=cost) for tokens, cost in estimates]

    def release(self, reservation: Reservation | None) -> None:
        if reservation is None:
            return
        with self._lock:
            if reservation.consumed:
                return
            reservation.consumed = True
            self.reserved_tokens = max(0, self.reserved_tokens - reservation.tokens)
            self.reserved_cost_usd = max(0.0, self.reserved_cost_usd - reservation.cost_usd)

    def settle(self, reservation: Reservation | None, usage: Usage, node: CompiledNode) -> float:
        """Replace a reservation with the provider's reported usage and return the cost."""

        with self._lock:
            if reservation is not None and not reservation.consumed:
                reservation.consumed = True
                self.reserved_tokens = max(0, self.reserved_tokens - reservation.tokens)
                self.reserved_cost_usd = max(0.0, self.reserved_cost_usd - reservation.cost_usd)
            self.used_tokens += usage.total_tokens
            self.used_input_tokens += usage.input_tokens
            self.used_output_tokens += usage.output_tokens
            cost = self.cost_for(usage, node)
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

    def charge(self, usage: Usage, node: CompiledNode) -> float:
        """Charge usage without a prior reservation (kept for direct callers)."""

        return self.settle(None, usage, node)

    def cost_for(self, usage: Usage, node: CompiledNode) -> float:
        """Estimated cost of ``usage`` without charging it.

        Used to report avoided cost for a cache hit; a hit must not mutate budget usage.
        """
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

    def remaining(self) -> tuple[int, float]:
        with self._lock:
            return (
                self.max_total_tokens - self.used_tokens - self.reserved_tokens,
                self.max_cost_usd - self.used_cost_usd - self.reserved_cost_usd,
            )

    def usage(self) -> dict[str, int | float]:
        """A consistent usage snapshot for manifests and trace events."""

        with self._lock:
            return {
                "steps": self.used_steps,
                "input_tokens": self.used_input_tokens,
                "output_tokens": self.used_output_tokens,
                "total_tokens": self.used_tokens,
                "cost_usd": round(self.used_cost_usd, 8),
                "elapsed_seconds": round(self.elapsed_seconds, 3),
            }

    def snapshot(self) -> BudgetState:
        with self._lock:
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
