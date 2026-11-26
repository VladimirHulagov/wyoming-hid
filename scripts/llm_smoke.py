#!/usr/bin/env python3
"""Send raw utterances through the formatter against the real API and print timings."""

import asyncio
import logging
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm_formatter import build_formatter

UTTERANCES = [
    "дохнешьнет оро все иет там ну туда да да все телефон ушко есть",
    "радио слушаешь радио нет а ну точно да",
    "может попоавит нет",
    "идешь в ухо оно у тебя тут сейчас вот видишь серенькое значит не подключено вот на ухо надо нажать а на пару на пару а все наверное разрядился у тебя",
    "заряд",
    "ну ты возьми вот эту книжку и посмотри с какой ты и некдоя найду где",
    "фломастеры почему я закрываю ну значит забыл а он подсохаа закрывать у меня вот рука очень туго держится хорошо хорошо ничего страшного просто надо марину тогда просить или меня мы будем проверять периодически",
    "вот я все три и сделал ошибки ну ничего страшного я тебе еще принесу хо вот ы эти вот три сделай еще раз хорошо сейчас сделаю только те одну последнюю оставь последнюю вот да я тебе скажу какую",
    "угу",
]
GARBAGE_INDEX = 0


class OutcomeTracker(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.outcome = {}
        self.unchanged = 0

    def emit(self, record):
        message = record.getMessage()
        if not message.startswith("llm: seq="):
            return
        seq_text = message.split(" ", 2)[1]
        if message.endswith(" formatted"):
            self.outcome[int(seq_text[4:])] = "formatted"
        elif "typing raw" in message:
            self.outcome[int(seq_text[4:])] = "raw"


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    tracker = OutcomeTracker()
    logging.getLogger("wyoming-hid").addHandler(tracker)

    typed = []
    started = time.monotonic()
    done = asyncio.Event()
    current = {"raw": ""}

    def on_text(text):
        if current["raw"] and text.strip() == current["raw"].strip():
            tracker.unchanged += 1
        typed.append(text)
        print(f"{time.monotonic() - started:6.2f}s  typed: {text}", flush=True)
        if len(typed) >= len(UTTERANCES):
            done.set()

    config = yaml.safe_load((ROOT / "config.yaml").read_text()) or {}
    formatter = build_formatter(config.get("llm"), on_text)
    if formatter is None:
        print("LLM formatting is disabled (check the llm section of config.yaml)")
        return 1

    seqs = {}
    for index, raw in enumerate(UTTERANCES):
        seqs[formatter.submit(raw)] = index
        await asyncio.sleep(0.4)
    await asyncio.wait_for(done.wait(), timeout=120)

    started = time.monotonic()
    typed.clear()
    done.clear()
    for index, raw in enumerate(UTTERANCES):
        current["raw"] = raw
        before = len(typed)
        seqs[formatter.submit(raw)] = index
        while len(typed) == before:
            await asyncio.sleep(0.05)
    current["raw"] = ""
    await formatter.close()

    clean = [i for i in range(len(UTTERANCES)) if i != GARBAGE_INDEX]
    clean_total = len(clean) * 2
    clean_formatted = sum(
        1 for seq, index in seqs.items()
        if index in clean and tracker.outcome.get(seq) == "formatted"
    )
    garbage = [tracker.outcome.get(seq) for seq, index in seqs.items() if index == GARBAGE_INDEX]
    print(
        f"clean formatted={clean_formatted}/{clean_total} "
        f"unchanged={tracker.unchanged} garbage outcome={garbage}"
    )
    return 0 if clean_formatted * 10 >= clean_total * 9 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
