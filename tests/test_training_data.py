from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from protocol import Control, Session, Tick  # noqa: E402
from training_data import IGNORE_INDEX, WeightedCollator, encode_session  # noqa: E402


class FakeTokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [10 + ord(character) % 50 for character in text]


TOKENS = {
    "<tick>": 100,
    "<input>": 101,
    "</input>": 102,
    "<output>": 103,
    "</output>": 104,
    "<listen>": 105,
    "<speak>": 106,
    "<continue>": 107,
    "<stop>": 108,
    "<out_end>": 109,
    "<tick_end>": 110,
}
WEIGHTS = {
    "output_text": 1.0,
    "first_or_non_repeated_listen": 1.0,
    "repeated_listen": 0.3,
    "speak": 2.0,
    "stop": 2.0,
    "continue": 1.5,
    "out_end": 1.5,
    "tick_end": 1.0,
}


class TrainingDataTest(unittest.TestCase):
    def session(self) -> Session:
        ticks = [Tick("x", [7], Control.LISTEN), Tick("", [], Control.LISTEN)]
        ticks += [Tick("", [], Control.LISTEN) for _ in range(4)]
        ticks += [
            Tick("q", [8], Control.SPEAK, "a", [9]),
            Tick("", [], Control.CONTINUE, "b", [11], out_end=True),
        ]
        return Session("s", "unit", "instruction", ticks, "t", "train", 1)

    def test_independent_control_targets_and_weights(self) -> None:
        encoded = encode_session(self.session(), FakeTokenizer(), TOKENS, WEIGHTS, independent_control=True)
        first_listen = encoded.input_ids.index(TOKENS["<listen>"])
        second_listen = encoded.input_ids.index(TOKENS["<listen>"], first_listen + 1)
        self.assertEqual(encoded.labels[first_listen], TOKENS["<listen>"])
        self.assertEqual(encoded.loss_weights[first_listen], 1.0)
        self.assertEqual(encoded.loss_weights[second_listen], 0.3)
        input_position = encoded.input_ids.index(7)
        self.assertEqual(encoded.labels[input_position], IGNORE_INDEX)
        speak_position = encoded.input_ids.index(TOKENS["<speak>"])
        self.assertEqual(encoded.loss_weights[speak_position], 2.0)

    def test_control_less_group_has_no_control_tokens(self) -> None:
        encoded = encode_session(self.session(), FakeTokenizer(), TOKENS, WEIGHTS, independent_control=False)
        self.assertTrue(set(encoded.input_ids).isdisjoint({105, 106, 107, 108}))
        self.assertIn(TOKENS["<output>"], encoded.input_ids)
        self.assertIn(TOKENS["<out_end>"], encoded.input_ids)

    def test_collator_masks_padding(self) -> None:
        first = encode_session(self.session(), FakeTokenizer(), TOKENS, WEIGHTS, independent_control=True)
        second = encode_session(self.session(), FakeTokenizer(), TOKENS, WEIGHTS, independent_control=False)
        batch = WeightedCollator(0)([first, second])
        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertTrue((batch["labels"][batch["attention_mask"] == 0] == IGNORE_INDEX).all())
        self.assertTrue((batch["loss_weights"][batch["attention_mask"] == 0] == 0).all())


if __name__ == "__main__":
    unittest.main()
