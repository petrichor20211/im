from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from protocol import (  # noqa: E402
    Control,
    ProtocolValidationError,
    Session,
    Tick,
    serialize_session,
    validate_session,
)


class ProtocolTest(unittest.TestCase):
    def make_session(self, ticks: list[Tick]) -> Session:
        return Session(
            session_id="test-session",
            rule_family="unit",
            instruction="test",
            ticks=ticks,
            template_id="unit-v0",
            split="train",
            random_seed=1,
            metadata={},
        )

    def test_valid_state_transitions_and_round_trip(self) -> None:
        ticks = [Tick("", [], Control.LISTEN) for _ in range(5)]
        ticks.extend(
            [
                Tick("question", [1], Control.SPEAK, "part", [2, 3]),
                Tick("", [], Control.CONTINUE, "end", [4], out_end=True),
                Tick("", [], Control.LISTEN),
            ]
        )
        session = self.make_session(ticks)
        validate_session(session)
        restored = Session.from_dict(json.loads(json.dumps(session.to_dict())))
        self.assertEqual(restored, session)
        rendered = serialize_session(session)
        self.assertIn("<speak><output>part</output><tick_end>", rendered)
        self.assertIn("<out_end><tick_end>", rendered)

    def test_continue_is_illegal_while_idle(self) -> None:
        ticks = [Tick("", [], Control.LISTEN) for _ in range(7)]
        ticks.append(Tick("", [], Control.CONTINUE, "bad", [1]))
        with self.assertRaisesRegex(ProtocolValidationError, "illegal continue"):
            validate_session(self.make_session(ticks))

    def test_listen_cannot_have_output(self) -> None:
        ticks = [Tick("", [], Control.LISTEN) for _ in range(7)]
        ticks.append(Tick("", [], Control.LISTEN, "bad", [1]))
        with self.assertRaisesRegex(ProtocolValidationError, "must not carry output"):
            validate_session(self.make_session(ticks))

    def test_output_budget_is_enforced(self) -> None:
        ticks = [Tick("", [], Control.LISTEN) for _ in range(7)]
        ticks.append(Tick("", [], Control.SPEAK, "too long", [1, 2, 3, 4, 5], out_end=True))
        with self.assertRaisesRegex(ProtocolValidationError, "output has 5 tokens"):
            validate_session(self.make_session(ticks))


if __name__ == "__main__":
    unittest.main()
