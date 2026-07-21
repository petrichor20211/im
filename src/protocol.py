"""Protocol types, serialization, and validation for text Omni-Flow sessions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable


class Control(str, Enum):
    LISTEN = "listen"
    SPEAK = "speak"
    CONTINUE = "continue"
    STOP = "stop"


CONTROL_TOKEN = {
    Control.LISTEN: "<listen>",
    Control.SPEAK: "<speak>",
    Control.CONTINUE: "<continue>",
    Control.STOP: "<stop>",
}


@dataclass
class Tick:
    input_text: str
    input_token_ids: list[int]
    control: Control
    output_text: str = ""
    output_token_ids: list[int] = field(default_factory=list)
    out_end: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["control"] = self.control.value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tick":
        return cls(
            input_text=data["input_text"],
            input_token_ids=list(data["input_token_ids"]),
            control=Control(data["control"]),
            output_text=data.get("output_text", ""),
            output_token_ids=list(data.get("output_token_ids", [])),
            out_end=bool(data.get("out_end", False)),
        )


@dataclass
class Session:
    session_id: str
    rule_family: str
    instruction: str
    ticks: list[Tick]
    template_id: str
    split: str
    random_seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ticks"] = [tick.to_dict() for tick in self.ticks]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        return cls(
            session_id=data["session_id"],
            rule_family=data["rule_family"],
            instruction=data["instruction"],
            ticks=[Tick.from_dict(tick) for tick in data["ticks"]],
            template_id=data["template_id"],
            split=data["split"],
            random_seed=int(data["random_seed"]),
            metadata=dict(data.get("metadata", {})),
        )


class ProtocolValidationError(ValueError):
    pass


def serialize_tick(tick: Tick) -> str:
    parts = [
        "<tick>",
        "<input>",
        tick.input_text,
        "</input>",
        CONTROL_TOKEN[tick.control],
    ]
    if tick.control in {Control.SPEAK, Control.CONTINUE}:
        parts.extend(["<output>", tick.output_text, "</output>"])
        if tick.out_end:
            parts.append("<out_end>")
    parts.append("<tick_end>")
    return "".join(parts)


def serialize_session(session: Session) -> str:
    return session.instruction + "\n" + "\n".join(serialize_tick(tick) for tick in session.ticks)


def emitted_output_ids(ticks: Iterable[Tick]) -> list[int]:
    return [token_id for tick in ticks for token_id in tick.output_token_ids]


def validate_session(
    session: Session,
    *,
    max_input_tokens: int = 16,
    max_output_tokens: int = 4,
    min_ticks: int = 8,
    max_ticks: int = 64,
) -> None:
    errors: list[str] = []
    if not min_ticks <= len(session.ticks) <= max_ticks:
        errors.append(f"tick count {len(session.ticks)} is outside [{min_ticks}, {max_ticks}]")

    state = "idle"
    for index, tick in enumerate(session.ticks):
        prefix = f"tick {index}"
        if len(tick.input_token_ids) > max_input_tokens:
            errors.append(f"{prefix}: input has {len(tick.input_token_ids)} tokens")
        if len(tick.output_token_ids) > max_output_tokens:
            errors.append(f"{prefix}: output has {len(tick.output_token_ids)} tokens")

        has_output = bool(tick.output_token_ids) or bool(tick.output_text)
        if tick.control in {Control.LISTEN, Control.STOP} and has_output:
            errors.append(f"{prefix}: {tick.control.value} must not carry output")
        if tick.control in {Control.LISTEN, Control.STOP} and tick.out_end:
            errors.append(f"{prefix}: {tick.control.value} must not carry out_end")

        if state == "idle":
            if tick.control == Control.LISTEN:
                pass
            elif tick.control == Control.SPEAK:
                state = "speaking"
            else:
                errors.append(f"{prefix}: illegal {tick.control.value} while idle")
        else:
            if tick.control == Control.CONTINUE:
                pass
            elif tick.control == Control.STOP:
                state = "idle"
            else:
                errors.append(f"{prefix}: illegal {tick.control.value} while speaking")

        if tick.out_end and tick.control in {Control.SPEAK, Control.CONTINUE}:
            state = "idle"

    expected = session.metadata.get("expected_emitted_output_token_ids")
    if expected is not None and emitted_output_ids(session.ticks) != expected:
        errors.append("emitted output tokens do not match metadata expectation")

    if errors:
        raise ProtocolValidationError(f"{session.session_id}: " + "; ".join(errors))
