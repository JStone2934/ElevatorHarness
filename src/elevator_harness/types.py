"""Hardware-independent contracts. Distances are metres; angles are radians."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from threading import Event
from typing import Any, Literal, Protocol
import uuid


def vector(value, size):
    if not isinstance(value, (tuple, list)) or len(value) != size or any(
        type(x) not in (int, float) or not math.isfinite(x) for x in value
    ):
        raise ValueError(f'expected {size} finite numbers')
    return tuple(float(x) for x in value)


@dataclass(frozen=True)
class OperationFrame:
    reference: str
    u: tuple[float, float, float]
    v: tuple[float, float, float]
    n: tuple[float, float, float]

    def __post_init__(self):
        axes = [vector(getattr(self, k), 3) for k in ('u', 'v', 'n')]
        if not self.reference or any(abs(sum(x*x for x in a)-1) > 1e-6 for a in axes):
            raise ValueError('unit axes and reference frame required')
        if any(abs(sum(x*y for x, y in zip(axes[i], axes[j]))) > 1e-6
               for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError('operation axes must be orthogonal')

    def to_robot(self, delta):
        du, dv, dn = vector(delta, 3)
        return tuple(du*self.u[i] + dv*self.v[i] + dn*self.n[i] for i in range(3))

    def normal_position(self, position):
        return sum(x*y for x, y in zip(vector(position, 3), self.n))


@dataclass(frozen=True)
class ToolConfiguration:
    name: str
    contact_reference: str
    calibration_version: str
    local_contact_uv: tuple[float, float] | None = None
    visibility_note: str = ''

    def __post_init__(self):
        if not all((self.name, self.contact_reference, self.calibration_version)):
            raise ValueError('tool identity, contact reference and calibration version required')
        if self.local_contact_uv is not None:
            uv = vector(self.local_contact_uv, 2)
            if any(not 0 <= x <= 1 for x in uv):
                raise ValueError('contact pixel must use normalized [0,1] coordinates')


@dataclass(frozen=True)
class Capabilities:
    frame: OperationFrame
    cameras: tuple[str, ...]
    tools: tuple[ToolConfiguration, ...]
    movable_views: tuple[str, ...] = ()
    has_depth: bool = False
    can_move: bool = True
    manipulator: str = 'primary'
    alignment_guidance: str = ''

    def tool(self, name):
        return next(t for t in self.tools if t.name == name)


@dataclass(frozen=True)
class Image:
    data: bytes
    mime_type: str = 'image/jpeg'


@dataclass(frozen=True)
class Observation:
    id: str
    timestamp_ns: int
    images: dict[str, Image]
    tool: str
    calibration_version: str
    state: dict[str, Any] = field(default_factory=dict)
    depth_m: float | None = None
    depth_valid: bool = False

    def __post_init__(self):
        if not self.id or self.timestamp_ns <= 0:
            raise ValueError('observation identity and timestamp required')
        if self.depth_m is not None and (not math.isfinite(self.depth_m) or self.depth_m < 0):
            raise ValueError('invalid depth')


@dataclass(frozen=True)
class ActionProposal:
    observation_id: str
    kind: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ''
    command_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    assessment: dict[str, Any] = field(default_factory=dict)

    def validate(self, capabilities):
        if not isinstance(self.command_id, str) or not self.command_id or not isinstance(self.observation_id, str) or not self.observation_id:
            raise ValueError('action identity required')
        fields = {'move': {'delta'}, 'configure_tool': {'configuration'},
                  'adjust_view': {'camera', 'angles'}, 'observe': set(),
                  'done': set(), 'unable': set()}
        if not isinstance(self.kind, str) or not isinstance(self.arguments, dict) or self.kind not in fields or set(self.arguments) != fields[self.kind]:
            raise ValueError('unsupported action or arguments')
        if self.kind == 'move':
            if not capabilities.can_move or not any(vector(self.arguments['delta'], 3)):
                raise ValueError('nonzero supported movement required')
        if self.kind == 'configure_tool' and self.arguments['configuration'] not in [t.name for t in capabilities.tools]:
            raise ValueError('unsupported tool configuration')
        if self.kind == 'adjust_view':
            if self.arguments['camera'] not in capabilities.movable_views:
                raise ValueError('camera cannot be adjusted')
            vector(self.arguments['angles'], 2)


@dataclass(frozen=True)
class ExecutionResult:
    command_id: str
    accepted: bool
    sent: bool
    measured_state: dict[str, Any] = field(default_factory=dict)
    capture_status: str = 'pending'
    observation: Observation | None = None
    error: str | None = None


@dataclass(frozen=True)
class Evidence:
    observation_id: str
    tool: str
    calibration_version: str
    source: str
    values: dict[str, Any]

    def matches(self, obs):
        return (self.observation_id, self.tool, self.calibration_version) == (
            obs.id, obs.tool, obs.calibration_version)


@dataclass(frozen=True)
class Verification:
    verdict: Literal['confirmed', 'denied', 'unknown']
    source: str
    evidence_ids: tuple[str, ...] = ()
    reason: str = ''


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ''


@dataclass
class Context:
    observation: Observation
    capabilities: Capabilities
    goal: str
    history: list[dict]
    task_state: dict
    evidence: Evidence | None = None
    feedback: str = ''
    policy_state: dict = field(default_factory=dict)
    decision_data: dict = field(default_factory=dict)
    supplementary_images: list[dict] = field(default_factory=list)


class RobotAdapter(Protocol):
    capabilities: Capabilities
    def observe(self, cancel: Event, timeout: float) -> Observation: ...
    def execute(self, action: ActionProposal, cancel: Event, timeout: float) -> ExecutionResult: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


class Agent(Protocol):
    def decide(self, context: Context, cancel: Event) -> ActionProposal: ...


class Task(Protocol):
    goal: str
    def observe(self, observation: Observation) -> None: ...
    def update(self, action: ActionProposal, result: ExecutionResult) -> None: ...
    def describe(self) -> dict: ...


class ExecutionPolicy(Protocol):
    def check(self, action: ActionProposal, context: Context) -> Decision: ...
    def record(self, action: ActionProposal, before: Observation, result: ExecutionResult) -> None: ...
    def describe(self) -> dict: ...


class Verifier(Protocol):
    def verify(self, observations: list[Observation], goal: str, cancel: Event) -> Verification: ...


class ModelClient(Protocol):
    def complete(self, messages: list[dict], tool: dict, cancel: Event) -> dict: ...
