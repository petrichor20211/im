#!/usr/bin/env python3
"""Generate deterministic pilot sessions for the text Omni-Flow protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from protocol import Control, Session, Tick, validate_session


TASKS = ("wait", "trigger", "periodic", "streaming", "interrupt", "distractor")
TASK_WEIGHTS = (0.20, 0.15, 0.15, 0.30, 0.10, 0.10)
SPLITS = (
    ("train", 0.70),
    ("validation", 0.10),
    ("test_in_domain", 0.04),
    ("test_template_ood", 0.04),
    ("test_timing_ood", 0.04),
    ("test_length_ood", 0.04),
    ("test_distractor_ood", 0.04),
)


class SessionGenerator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, rng: random.Random, seed: int):
        self.tokenizer = tokenizer
        self.rng = rng
        self.seed = seed

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)

    def chunks(self, token_ids: list[int], maximum: int) -> list[list[int]]:
        result: list[list[int]] = []
        cursor = 0
        while cursor < len(token_ids):
            width = self.rng.randint(1, min(maximum, len(token_ids) - cursor))
            result.append(token_ids[cursor : cursor + width])
            cursor += width
        return result or [[]]

    def input_tick(self, ids: list[int], control: Control = Control.LISTEN) -> Tick:
        return Tick(self.decode(ids), ids, control)

    def output_ticks(self, ids: list[int], first_input: list[int] | None = None) -> list[Tick]:
        pieces = self.chunks(ids, 4)
        ticks = []
        for index, piece in enumerate(pieces):
            ticks.append(
                Tick(
                    input_text=self.decode(first_input or []) if index == 0 else "",
                    input_token_ids=list(first_input or []) if index == 0 else [],
                    control=Control.SPEAK if index == 0 else Control.CONTINUE,
                    output_text=self.decode(piece),
                    output_token_ids=piece,
                    out_end=index == len(pieces) - 1,
                )
            )
        return ticks

    def pad(self, ticks: list[Tick], *, long: bool = False) -> list[Tick]:
        low = max(65 if long else 8, len(ticks))
        high = 128 if long else 64
        target = self.rng.randint(low, high)
        extra = target - len(ticks)
        before = self.rng.randint(0, extra)
        empty = lambda: Tick("", [], Control.LISTEN)
        return [empty() for _ in range(before)] + ticks + [empty() for _ in range(extra - before)]

    def make_session(self, index: int, split: str, task: str) -> Session:
        builder: Callable[[str], tuple[str, str, list[Tick], dict]] = getattr(self, f"build_{task}")
        instruction, template_id, ticks, metadata = builder(split)
        if task != "periodic":
            ticks = self.pad(ticks, long=split == "test_length_ood")
        metadata["expected_emitted_output_token_ids"] = [
            token_id for tick in ticks for token_id in tick.output_token_ids
        ]
        session = Session(
            session_id=f"synthetic-{index:06d}",
            rule_family=task,
            instruction=instruction,
            ticks=ticks,
            template_id=template_id,
            split=split,
            random_seed=self.seed,
            metadata=metadata,
        )
        validate_session(session, max_ticks=128 if split == "test_length_ood" else 64)
        return session

    def build_wait(self, split: str):
        a, b = self.rng.randint(0, 99), self.rng.randint(0, 99)
        templates = ["请等待算式完整后再回答。", "只有收到完整计算题后才能给答案。"]
        if split == "test_template_ood":
            templates = ["信息没说完时保持安静，完整后仅返回计算结果。"]
        instruction = self.rng.choice(templates)
        expression = f"计算 {a} 加上 {b}"
        pieces = self.chunks(self.encode(expression), 8)
        ticks = [self.input_tick(piece) for piece in pieces[:-1]]
        answer = self.encode(str(a + b))
        ticks.extend(self.output_ticks(answer, pieces[-1]))
        template_group = "ood" if split == "test_template_ood" else "train"
        return instruction, f"wait-{template_group}-{templates.index(instruction)}", ticks, {"answer_token_ids": answer}

    def build_trigger(self, split: str):
        templates = ["只有看到 RED 才输出 alert。", "在 RED 出现前保持沉默，出现后输出 alert。"]
        if split == "test_template_ood":
            templates = ["监控输入流；目标标记 RED 到达时报告 alert。"]
        instruction = self.rng.choice(templates)
        distractors = ["blue green", "yellow READ", "redder", "GREEN blue"]
        ticks = [self.input_tick(self.encode(self.rng.choice(distractors))) for _ in range(self.rng.randint(1, 5))]
        trigger = self.encode(self.rng.choice(["yellow RED", "RED", "value=RED"]))
        answer = self.encode("alert")
        ticks.extend(self.output_ticks(answer, trigger))
        template_group = "ood" if split == "test_template_ood" else "train"
        return instruction, f"trigger-{template_group}-{templates.index(instruction)}", ticks, {"answer_token_ids": answer}

    def build_periodic(self, split: str):
        if split == "test_timing_ood":
            period = self.rng.choice([7, 9])
        else:
            period = self.rng.choice([2, 3, 4, 5])
        length = self.rng.randint(65, 128) if split == "test_length_ood" else self.rng.randint(8, 64)
        first_delay = self.rng.randrange(period)
        start_tick = self.rng.randint(0, min(5, length - 1))
        instruction = (
            f"看到 START 后先等待 {first_delay} 个完整 tick，然后输出 ping；"
            f"此后每隔 {period} 个 tick 再输出一次 ping。"
        )
        ping = self.encode("ping")
        ticks: list[Tick] = []
        trigger_ticks = []
        first_trigger = start_tick + first_delay
        for index in range(length):
            input_text = "START" if index == start_tick else ""
            input_ids = self.encode(input_text) if input_text else []
            should_trigger = index >= first_trigger and (index - first_trigger) % period == 0
            if should_trigger:
                trigger_ticks.append(index)
                ticks.append(
                    Tick(input_text, input_ids, Control.SPEAK, self.decode(ping), ping, out_end=True)
                )
            else:
                ticks.append(Tick(input_text, input_ids, Control.LISTEN))
        return instruction, "periodic-v1", ticks, {
            "period": period,
            "first_delay": first_delay,
            "start_tick": start_tick,
            "trigger_ticks": trigger_ticks,
        }

    def build_streaming(self, split: str):
        subtype = self.rng.choice(("repeat", "uppercase", "arithmetic"))
        if subtype == "repeat":
            subject = self.rng.choice(("机器人", "研究员", "小猫", "快递员", "学生", "工程师"))
            verb = self.rng.choice(("搬运了", "记录了", "发现了", "整理了", "检查了"))
            number = self.rng.randint(10, 999)
            obj = self.rng.choice(("个蓝色箱子", "条实验数据", "本旧书", "颗红色按钮", "张车票"))
            answer_text = f"{subject}{verb}{number}{obj}。"
            question = f"请原样重复冒号后的句子：{answer_text}"
        elif subtype == "uppercase":
            alphabet = "abcdefghjkmnpqrstuvwxyz"
            source = "".join(self.rng.choice(alphabet) for _ in range(self.rng.randint(5, 12)))
            question = f"把字符串 {source} 转成大写。"
            answer_text = source.upper()
        else:
            a, b = self.rng.randint(10, 999), self.rng.randint(10, 999)
            question = f"计算 {a} 加 {b}，只输出结果。"
            answer_text = str(a + b)
        instruction = "收到完整问题后回答，并将答案按每 tick 最多四个 token 输出。"
        question_chunks = self.chunks(self.encode(question), 8)
        ticks = [self.input_tick(piece) for piece in question_chunks[:-1]]
        answer = self.encode(answer_text)
        ticks.extend(self.output_ticks(answer, question_chunks[-1]))
        return instruction, f"streaming-{subtype}-v1", ticks, {"answer_token_ids": answer, "subtype": subtype}

    def build_interrupt(self, split: str):
        instruction = "回答需要分块输出；用户明确要求停止时立即停止。"
        request = self.encode("请详细介绍太阳系中的八颗行星。")
        request_chunks = self.chunks(request, 8)
        ticks = [self.input_tick(piece) for piece in request_chunks[:-1]]
        answer = self.encode("太阳系的八颗行星依次包括水星金星地球火星木星土星天王星和海王星。")
        answer_chunks = self.chunks(answer, 4)
        emitted_count = self.rng.randint(1, min(3, len(answer_chunks) - 1))
        first = answer_chunks[0]
        ticks.append(Tick(self.decode(request_chunks[-1]), request_chunks[-1], Control.SPEAK, self.decode(first), first))
        for chunk_index, piece in enumerate(answer_chunks[1:emitted_count], start=1):
            backchannel = self.rng.choice(["嗯，继续", "你说得对", "没错", ""]) if chunk_index == 1 else ""
            ticks.append(
                Tick(backchannel, self.encode(backchannel) if backchannel else [], Control.CONTINUE, self.decode(piece), piece)
            )
        interrupt_text = self.rng.choice(["停一下，我换个问题", "别说了", "等等，不是这个意思"])
        ticks.append(Tick(interrupt_text, self.encode(interrupt_text), Control.STOP))
        return instruction, "interrupt-v0", ticks, {"interrupted": True, "full_answer_token_ids": answer}

    def build_distractor(self, split: str):
        instruction = "只有同一个 tick 的输入同时包含 ALPHA 和 OMEGA 时才输出 done，否则保持沉默。"
        if split == "test_distractor_ood":
            pool = ["ALPHA ... OMEG", "ALPHA_OMEG", "ALPHABET OMEGA3", "alpha OMEGA", "PREDALPHA", "Ω"]
        else:
            pool = ["ALPHA", "OMEG", "omega", "random 123", "ALPHABET", "nothing"]
        count = self.rng.randint(2, 12)
        ticks = [self.input_tick(self.encode(self.rng.choice(pool))) for _ in range(count)]
        return instruction, "distractor-v0", ticks, {"no_trigger": True}


def choose_splits(count: int) -> list[str]:
    boundaries = []
    running = 0.0
    for name, ratio in SPLITS:
        running += ratio
        boundaries.append((name, running))
    result = []
    for index in range(count):
        position = (index + 0.5) / count
        result.append(next(name for name, boundary in boundaries if position <= boundary + 1e-12))
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/120090727/yutong/model/Qwen3.5-0.8B")
    parser.add_argument("--output", type=Path, default=Path("artifacts/data/synthetic_preview.jsonl"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    generator = SessionGenerator(tokenizer, rng, args.seed)
    splits = choose_splits(args.count)
    tasks = rng.choices(TASKS, weights=TASK_WEIGHTS, k=args.count)
    for index, split in enumerate(splits):
        if split == "test_template_ood":
            tasks[index] = rng.choice(("wait", "trigger"))
        elif split == "test_timing_ood":
            tasks[index] = "periodic"
        elif split == "test_distractor_ood":
            tasks[index] = rng.choice(("trigger", "interrupt", "distractor"))

    sessions = [generator.make_session(i, split, task) for i, (split, task) in enumerate(zip(splits, tasks))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for session in sessions:
            handle.write(json.dumps(session.to_dict(), ensure_ascii=False) + "\n")

    manifest = {
        "count": len(sessions),
        "seed": args.seed,
        "model": args.model,
        "output": str(args.output),
        "sha256": file_sha256(args.output),
        "task_counts": Counter(session.rule_family for session in sessions),
        "split_counts": Counter(session.split for session in sessions),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=dict) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
