# LLM Formatting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use the subagent-driven-development skill (from brainstorming). Steps use checkbox (`- [ ]`) syntax for tracking.

**Execution Mode:** subagent-driven (set during brainstorming)

**Goal:** Format each finalized dictation utterance with GLM-5.3-Flash before typing it: requests in parallel, results typed in order, raw text on any failure.

**Architecture:** A new `llm_formatter.py` holds text validation helpers, an `LLMClient` for the z.ai API and an `OrderedFormatter` pipeline (one task per utterance, strictly ordered emission, cooldown after repeated failures). `wyoming-hid.py` submits finalized utterances to the formatter instead of typing them directly; `StreamingSTT` gets a flag to keep the overlay visible until the formatted text is typed.

**Tech Stack:** Python 3.12.3, asyncio, aiohttp 3.13.3, unittest (run through `python3 -m pytest`, pytest 9.0.3), xdotool.

**Repo facts:** `/home/lacitis/CONFIGURATION/TOOLS/wyoming-hid`, branch `feature/streaming-stt` at `84f03af`, clean working tree. No worktree: the repo is small and clean, the work happens on a new branch `feature/llm-formatting`.

**Rules for every task:**
- Work in the repo root; run all commands from there.
- TDD is mandatory: write the failing test first, run it and see it fail, implement, run it and see it pass, commit.
- No code comments, no features beyond the task.
- `config.yaml` holds credentials: never read it, print it or modify it, except the explicit step in Task 8. Never print API keys.
- Commit only the files listed in the task.
- If a command fails or a test does not behave as the plan says, stop and report instead of improvising.

---

### Task 1: Text helpers

**Files:**
- Create: `llm_formatter.py`
- Create: `tests/test_llm_formatter.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create the branch**

Run:
```bash
git checkout -b feature/llm-formatting
```
Expected: `Switched to a new branch 'feature/llm-formatting'`

Append `.pytest_cache/` as a new last line of `.gitignore`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_llm_formatter.py`:
```python
import unittest

from llm_formatter import ends_sentence, looks_faithful, normalize, separator


class TestTextHelpers(unittest.TestCase):
    def test_faithful_identical(self):
        self.assertTrue(looks_faithful("привет как дела", "Привет, как дела?"))

    def test_faithful_glued_words(self):
        self.assertTrue(looks_faithful("дохнешьнет все", "Дохнешь? Нет, все."))

    def test_faithful_filler_removal(self):
        self.assertTrue(
            looks_faithful("ну ты возьми вот эту книжку", "Ты возьми эту книжку.")
        )

    def test_faithful_asr_correction(self):
        self.assertTrue(looks_faithful("может попоавит нет", "Может поправит? Нет."))

    def test_unfaithful_dropped_words(self):
        self.assertFalse(
            looks_faithful("возьми вот эту книжку и посмотри", "Возьми книжку.")
        )

    def test_unfaithful_too_long(self):
        out = "Привет, привет, привет. " * 3
        self.assertFalse(looks_faithful("привет привет привет", out))

    def test_unfaithful_invented_words(self):
        self.assertFalse(
            looks_faithful(
                "скажи ему чтобы он зашёл ко мне в кабинет завтра утром",
                "Конечно! Вот отформатированный текст: Скажи ему, чтобы он зашёл ко мне в кабинет завтра утром.",
            )
        )

    def test_unfaithful_preamble_on_long_output(self):
        self.assertFalse(
            looks_faithful(
                "скажи ему чтобы он зашёл ко мне в кабинет завтра утром и взял с собой все документы которые лежат на столе",
                "Конечно! Отлично! Скажи ему, чтобы он зашёл ко мне в кабинет завтра утром и взял с собой все документы, которые лежат на столе.",
            )
        )

    def test_unfaithful_dropped_negation(self):
        self.assertFalse(
            looks_faithful("не забудь отправить письмо", "Забудь отправить письмо.")
        )

    def test_unfaithful_dropped_negation_long(self):
        self.assertFalse(
            looks_faithful(
                "пожалуйста не забудь отправить ему письмо завтра утром до совещания",
                "Пожалуйста, забудь отправить ему письмо завтра утром до совещания.",
            )
        )

    def test_unfaithful_dropped_ni(self):
        self.assertFalse(
            looks_faithful(
                "пожалуйста я ни в коем случае не пойду туда завтра утром",
                "Пожалуйста, я в коем случае не пойду туда завтра утром.",
            )
        )

    def test_unfaithful_added_negation(self):
        self.assertFalse(
            looks_faithful("забудь отправить письмо", "Не забудь отправить письмо.")
        )

    def test_unfaithful_dropped_negation_prefix(self):
        self.assertFalse(
            looks_faithful("неважно отправить письмо сейчас", "Важно отправить письмо сейчас.")
        )

    def test_unfaithful_dropped_digits(self):
        self.assertFalse(
            looks_faithful("скажи ему чтобы перезвонил мне на 8 950", "Скажи ему, чтобы перезвонил мне.")
        )

    def test_unfaithful_changed_digits(self):
        self.assertFalse(looks_faithful("код 2023", "Код 2024."))

    def test_unfaithful_coverage_below_threshold(self):
        self.assertFalse(
            looks_faithful(
                "отправь письмо василию завтра утром в девять часов",
                "Отправь письмо Василию завтра утром в девять.",
            )
        )

    def test_unfaithful_control_characters(self):
        self.assertFalse(looks_faithful("привет мир", "Привет\nмир."))
        self.assertFalse(looks_faithful("привет мир", "Привет\tмир."))

    def test_unfaithful_one_word_preamble(self):
        self.assertFalse(looks_faithful("привет как дела", "Отлично, привет, как дела?"))

    def test_faithful_digits_kept(self):
        self.assertTrue(
            looks_faithful(
                "скажи ему чтобы перезвонил мне на 8 950",
                "Скажи ему, чтобы перезвонил мне на 8 950.",
            )
        )

    def test_unfaithful_half_of_the_words_dropped(self):
        self.assertFalse(looks_faithful("иван петр", "Иван."))

    def test_faithful_stutter_removed(self):
        self.assertTrue(looks_faithful("мы пошли пошли в магазин", "Мы пошли в магазин."))

    def test_unfaithful_dropped_non_adjacent_repeat(self):
        self.assertFalse(
            looks_faithful(
                "включи камеру и выключи микрофон и включи запись",
                "Включи камеру, выключи микрофон и запись.",
            )
        )

    def test_faithful_glued_words_with_short_pieces(self):
        self.assertTrue(looks_faithful("даяснял или не снял", "Да, я снял или не снял."))

    def test_faithful_only_fillers(self):
        self.assertTrue(looks_faithful("ну вот а да", ""))
        self.assertTrue(looks_faithful("угу", "Ну вот."))
        self.assertFalse(looks_faithful("угу", "Привет"))

    def test_ends_sentence(self):
        self.assertTrue(ends_sentence("Привет."))
        self.assertTrue(ends_sentence("Как дела?"))
        self.assertTrue(ends_sentence("Так..."))
        self.assertTrue(ends_sentence("Он сказал: «Привет»."))
        self.assertTrue(ends_sentence("(Да.)"))
        self.assertFalse(ends_sentence("Привет"))
        self.assertFalse(ends_sentence("Он сказал: «Привет»"))

    def test_separator(self):
        self.assertEqual(separator(None, "Привет"), "")
        self.assertEqual(separator("Привет", "как дела"), " ")
        self.assertEqual(separator("мы пошли", ", а потом"), "")

    def test_normalize_first_fragment(self):
        self.assertEqual(normalize(None, "привет"), "привет")

    def test_normalize_adds_space(self):
        self.assertEqual(normalize("мы пошли", "и увидели"), " и увидели")

    def test_normalize_capitalizes_after_sentence(self):
        self.assertEqual(normalize("Привет.", "как дела"), " Как дела")

    def test_normalize_keeps_leading_punctuation(self):
        self.assertEqual(normalize("мы пошли", ", а потом"), ", а потом")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run:
```bash
python3 -m pytest tests/test_llm_formatter.py -q
```
Expected: collection error, `ModuleNotFoundError: No module named 'llm_formatter'`

- [ ] **Step 4: Implement the helpers**

Create `llm_formatter.py`:
```python
#!/usr/bin/env python3
"""Format finalized dictation utterances with an LLM before typing."""

from __future__ import annotations

import re
import string
import unicodedata
from difflib import SequenceMatcher
from typing import Optional

DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
DEFAULT_MODEL = "glm-5.3-flash"
DEFAULT_TIMEOUT = 8.0
DEFAULT_CONCURRENCY = 4
DEFAULT_CONTEXT_UTTERANCES = 2
COOLDOWN_AFTER = 3
COOLDOWN_SECONDS = 60.0

FILLERS = frozenset("ну вот а да угу ага это то и же там тут".split())
MUST_KEEP = frozenset(("не", "ни"))
MIN_TOKEN_LEN = 3
FUZZY_MIN_LEN = 4
FUZZY_RATIO = 0.6
FAITHFUL_RATIO = 0.9
EXTRA_DIVISOR = 7
MAX_OUTPUT_FACTOR = 2.5
MAX_OUTPUT_EXTRA = 20
SENTENCE_FINAL = ".!?…"
CLOSERS = "\"'»)]}❞“”„"
LEAD_PUNCT = frozenset(string.punctuation)

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    return [token.replace("ё", "е") for token in _TOKEN_RE.findall(text.lower())]


def _content_words(text: str) -> list[str]:
    return [token for token in _tokens(text) if token not in FILLERS]


def _has_digits(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def _is_negation(token: str) -> bool:
    return token in MUST_KEEP or (len(token) >= 5 and token.startswith(("не", "ни")))


def _match(word: str, token: str) -> bool:
    if word == token:
        return True
    if _has_digits(word) or _has_digits(token):
        return False
    if _is_negation(word) != _is_negation(token):
        return False
    if len(word) >= MIN_TOKEN_LEN and len(token) >= MIN_TOKEN_LEN:
        if word in token or token in word:
            return True
    if len(word) >= FUZZY_MIN_LEN and len(token) >= FUZZY_MIN_LEN:
        return SequenceMatcher(None, word, token).ratio() >= FUZZY_RATIO
    return False


def looks_faithful(raw: str, out: str) -> bool:
    if len(out) > MAX_OUTPUT_FACTOR * len(raw) + MAX_OUTPUT_EXTRA:
        return False
    if any(unicodedata.category(ch) == "Cc" for ch in out):
        return False
    raw_words = _content_words(raw)
    if not raw_words:
        return not _content_words(out)
    out_tokens = _tokens(out)
    if any(word in raw_words and word not in out_tokens for word in MUST_KEEP):
        return False
    if sum(1 for token in out_tokens if _is_negation(token)) > sum(
        1 for word in raw_words if _is_negation(word)
    ):
        return False
    counted = [w for i, w in enumerate(raw_words) if i == 0 or w != raw_words[i - 1]]
    matched, used = 0, set()
    for word in counted:
        for index, token in enumerate(out_tokens):
            if index not in used and _match(word, token):
                matched += 1
                used.add(index)
                break
    if matched / len(counted) < FAITHFUL_RATIO:
        return False
    long_out = [t for t in out_tokens if len(t) >= MIN_TOKEN_LEN and t not in FILLERS]
    extra = [t for t in long_out if not any(_match(t, word) for word in raw_words)]
    return len(extra) <= min(len(long_out) // EXTRA_DIVISOR, 1)


def ends_sentence(text: str) -> bool:
    text = text.rstrip()
    while text and text[-1] in CLOSERS:
        text = text[:-1]
    return bool(text) and text[-1] in SENTENCE_FINAL


def separator(prev: Optional[str], new: str) -> str:
    if prev is None:
        return ""
    if new[:1] in LEAD_PUNCT:
        return ""
    return " "


def normalize(prev: Optional[str], text: str) -> str:
    text = text.strip()
    if prev is not None:
        if ends_sentence(prev) and text[:1].islower():
            text = text[0].upper() + text[1:]
        text = separator(prev, text) + text
    return text
```

- [ ] **Step 5: Run the tests to verify they pass**

Run:
```bash
python3 -m pytest tests/test_llm_formatter.py -q
```
Expected: `30 passed`

- [ ] **Step 6: Commit**

```bash
git add llm_formatter.py tests/test_llm_formatter.py .gitignore
git commit -m "feat: add text helpers for LLM formatting"
```

---

### Task 2: Prompt and messages

**Files:**
- Modify: `llm_formatter.py`
- Modify: `tests/test_llm_formatter.py`

- [ ] **Step 1: Write the failing tests**

Replace the import block at the top of `tests/test_llm_formatter.py` with:
```python
import unittest

from llm_formatter import (
    SYSTEM_PROMPT,
    build_messages,
    ends_sentence,
    looks_faithful,
    normalize,
    separator,
)
```

Append to `tests/test_llm_formatter.py`:
```python
class TestBuildMessages(unittest.TestCase):
    def test_without_context(self):
        messages = build_messages("  привет мир  ", [])
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Новая реплика без пунктуации:\nпривет мир"},
            ],
        )

    def test_with_context(self):
        messages = build_messages("как дела", ["  Привет.  "])
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(
            messages[1],
            {
                "role": "user",
                "content": "Уже напечатанный текст:\nПривет.\n\nНовая реплика без пунктуации:\nкак дела",
            },
        )

    def test_with_two_context_entries(self):
        messages = build_messages("река", ["Мы пошли и увидели", " море "])
        self.assertEqual(
            messages[1]["content"],
            "Уже напечатанный текст:\nМы пошли и увидели\nморе\n\nНовая реплика без пунктуации:\nрека",
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
python3 -m pytest tests/test_llm_formatter.py -q
```
Expected: `ImportError: cannot import name 'SYSTEM_PROMPT'`

- [ ] **Step 3: Implement**

In `llm_formatter.py` change `from typing import Optional` to `from typing import Optional, Sequence`.

Append to `llm_formatter.py`:
```python
SYSTEM_PROMPT = (
    "Ты — редактор голосового ввода. На входе — фраза, распознанная из речи: без пунктуации, "
    "со словами-паразитами и, возможно, со слитными словами. Оформи её в связный читаемый русский "
    "текст: поставь знаки препинания и заглавные буквы, раздели слитные слова, убери слова-паразиты "
    "(ну, вот, да-да, угу, ага; «а» — только если это не союз) и бессмысленные повторы соседних слов. "
    "Союзы, предлоги и ответы «да»/«нет» сохраняй. Очевидные ошибки распознавания исправляй, "
    "остальное не переписывай. "
    "Ничего не добавляй и не придумывай: все содержательные слова фразы должны остаться. Если фраза "
    "неразборчива — верни её без изменений. Если уже напечатанного текста нет или он заканчивается "
    "знаком конца предложения, начни с заглавной буквы; иначе — со строчной. Блок «Уже напечатанный "
    "текст» приведён только для ориентира: не повторяй его и не изменяй. Не рассуждай. Выводи только "
    "готовый текст новой реплики в одну строку, без пояснений, кавычек и разметки."
)


def build_messages(raw: str, context: Sequence[str]) -> list[dict]:
    lines = []
    if context:
        lines.append("Уже напечатанный текст:")
        lines.extend(entry.strip() for entry in context)
        lines.append("")
    lines.append("Новая реплика без пунктуации:")
    lines.append(raw.strip())
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
python3 -m pytest tests/test_llm_formatter.py -q
```
Expected: `33 passed`

- [ ] **Step 5: Commit**

```bash
git add llm_formatter.py tests/test_llm_formatter.py
git commit -m "feat: add system prompt and message builder"
```

---

### Task 3: API key resolution

**Files:**
- Modify: `llm_formatter.py`
- Modify: `tests/test_llm_formatter.py`

- [ ] **Step 1: Write the failing tests**

Replace the import block at the top of `tests/test_llm_formatter.py` with:
```python
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_formatter import (
    LLMConfigError,
    SYSTEM_PROMPT,
    build_messages,
    ends_sentence,
    looks_faithful,
    normalize,
    resolve_api_key,
    separator,
)
```

Append to `tests/test_llm_formatter.py`:
```python
class TestResolveApiKey(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Path(self.tmp.name) / "auth.json"
        store_patch = patch("llm_formatter.AUTH_STORE_PATH", self.store)
        store_patch.start()
        self.addCleanup(store_patch.stop)

    def test_explicit_wins(self):
        with patch.dict(os.environ, {"ZAI_API_KEY": "env"}):
            self.assertEqual(resolve_api_key("explicit"), "explicit")

    def test_env(self):
        with patch.dict(os.environ, {"ZAI_API_KEY": "env"}):
            self.assertEqual(resolve_api_key(), "env")

    def test_env_beats_store(self):
        self.store.write_text(json.dumps({"zai-coding-plan": {"key": "file"}}))
        with patch.dict(os.environ, {"ZAI_API_KEY": "env"}):
            self.assertEqual(resolve_api_key(), "env")

    def test_auth_store(self):
        self.store.write_text(json.dumps({"zai-coding-plan": {"key": " file "}}))
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_api_key(), "file")

    def test_auth_store_disabled(self):
        self.store.write_text(json.dumps({"zai-coding-plan": {"key": "file"}}))
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMConfigError):
                resolve_api_key(allow_auth_store=False)

    def test_blank_values_fall_through(self):
        self.store.write_text(json.dumps({"zai-coding-plan": {"key": "file"}}))
        with patch.dict(os.environ, {"ZAI_API_KEY": "   "}):
            self.assertEqual(resolve_api_key("   "), "file")

    def test_malformed_store_raises(self):
        for text in ("[]", "null", '{"zai-coding-plan": "key"}', '{"zai-coding-plan": {"key": 123}}', "{"):
            self.store.write_text(text)
            with self.subTest(text=text), patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(LLMConfigError):
                    resolve_api_key()

    def test_unreadable_store_raises(self):
        if os.geteuid() == 0:
            self.skipTest("root ignores file permissions")
        self.store.write_text(json.dumps({"zai-coding-plan": {"key": "file"}}))
        self.addCleanup(Path(self.tmp.name).chmod, 0o700)
        Path(self.tmp.name).chmod(0)
        with patch.dict(os.environ, {}, clear=True), self.assertLogs("wyoming-hid", level="WARNING") as logs:
            with self.assertRaises(LLMConfigError):
                resolve_api_key()
        self.assertNotIn("file", "\n".join(logs.output))

    def test_no_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMConfigError):
                resolve_api_key()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
python3 -m pytest tests/test_llm_formatter.py -q
```
Expected: `ImportError: cannot import name 'LLMConfigError'`

- [ ] **Step 3: Implement**

In `llm_formatter.py` make the import block look like this (the constants below it stay as they are, with `AUTH_STORE_PATH` and `AUTH_STORE_PROVIDER` added after `CLOSERS`, and `log = logging.getLogger("wyoming-hid")` added before `_TOKEN_RE`):
```python
from __future__ import annotations

import json
import logging
import os
import re
import string
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Sequence
```

Append after `CLOSERS = ...`:
```python
AUTH_STORE_PATH = Path("~/.local/share/opencode/auth.json").expanduser()
AUTH_STORE_PROVIDER = "zai-coding-plan"
```

Append before `_TOKEN_RE = ...`:
```python
log = logging.getLogger("wyoming-hid")
```

Append to the end of `llm_formatter.py`:
```python
class LLMConfigError(Exception):
    pass


def resolve_api_key(explicit: Optional[str] = None, allow_auth_store: bool = True) -> str:
    for candidate, source in ((explicit, "config"), (os.environ.get("ZAI_API_KEY"), "env")):
        if isinstance(candidate, str) and candidate.strip():
            log.info("llm: API key source: %s", source)
            return candidate.strip()
    if allow_auth_store:
        try:
            if AUTH_STORE_PATH.is_file():
                data = json.loads(AUTH_STORE_PATH.read_text(encoding="utf-8"))
                entry = data.get(AUTH_STORE_PROVIDER) if isinstance(data, dict) else None
                key = entry.get("key") if isinstance(entry, dict) else None
                if isinstance(key, str) and key.strip():
                    log.info("llm: API key source: opencode auth store")
                    return key.strip()
        except (OSError, ValueError) as e:
            log.warning("Could not read %s (%s)", AUTH_STORE_PATH, type(e).__name__)
    raise LLMConfigError(
        "No API key: set llm.api_key in config.yaml, export ZAI_API_KEY or log in to opencode"
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
python3 -m pytest tests/test_llm_formatter.py -q
```
Expected: `42 passed` (pytest 9 appends `, 5 subtests passed` because of the subtests in `test_malformed_store_raises`; both forms are acceptable)

- [ ] **Step 5: Commit**

```bash
git add llm_formatter.py tests/test_llm_formatter.py
git commit -m "feat: resolve z.ai API key from config, env or opencode auth store"
```

---

### Task 4: LLMClient

**Files:**
- Modify: `llm_formatter.py`
- Modify: `tests/test_llm_formatter.py`

- [ ] **Step 1: Write the failing tests**

Replace the import block at the top of `tests/test_llm_formatter.py` with:
```python
import asyncio
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer

from llm_formatter import (
    LLMClient,
    LLMConfigError,
    LLMError,
    SYSTEM_PROMPT,
    build_messages,
    ends_sentence,
    looks_faithful,
    normalize,
    resolve_api_key,
    separator,
)
```

Append to `tests/test_llm_formatter.py`:
```python
async def make_server(state):
    calls = []

    async def handler(request):
        calls.append((await request.json(), request.headers.get("Authorization")))
        if state.get("delay"):
            await asyncio.sleep(state["delay"])
        return web.Response(status=state["status"], text=state["body"])

    app = web.Application()
    app.router.add_post("/chat/completions", handler)
    server = TestServer(app)
    await server.start_server()
    return server, calls


def ok_body(content, finish_reason="stop"):
    return json.dumps(
        {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}
    )


class TestLLMClient(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.state = {"status": 200, "body": ok_body("Привет."), "delay": 0}
        self.server, self.calls = await make_server(self.state)
        self.client = LLMClient(str(self.server.make_url("")), "secret", "m")
        self.client.start()

    async def asyncTearDown(self):
        await self.client.close()
        await self.server.close()

    async def test_success(self):
        self.assertEqual(await self.client.format("привет", []), "Привет.")
        body, auth = self.calls[0]
        self.assertEqual(body["model"], "m")
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["stream"], False)
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["max_tokens"], 512)
        self.assertEqual(body["messages"][0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(auth, "Bearer secret")

    async def test_http_error(self):
        self.state.update(status=500, body="boom")
        with self.assertRaises(LLMError) as ctx:
            await self.client.format("привет", [])
        self.assertIn("HTTP 500", str(ctx.exception))

    async def test_empty_content(self):
        self.state["body"] = ok_body("   ")
        with self.assertRaises(LLMError) as ctx:
            await self.client.format("привет", [])
        self.assertEqual(str(ctx.exception), "empty content")

    async def test_truncated(self):
        self.state["body"] = ok_body("Обрез", finish_reason="length")
        with self.assertRaises(LLMError) as ctx:
            await self.client.format("привет", [])
        self.assertIn("finish_reason", str(ctx.exception))

    async def test_malformed(self):
        self.state["body"] = "not json"
        with self.assertRaises(LLMError) as ctx:
            await self.client.format("привет", [])
        self.assertIn("malformed JSON", str(ctx.exception))

    async def test_content_is_not_a_string(self):
        self.state["body"] = json.dumps(
            {"choices": [{"message": {"content": [{"type": "text", "text": "Привет."}]},
                          "finish_reason": "stop"}]}
        )
        with self.assertRaises(LLMError) as ctx:
            await self.client.format("привет", [])
        self.assertIn("unexpected response", str(ctx.exception))

    async def test_null_choice(self):
        self.state["body"] = json.dumps({"choices": [None]})
        with self.assertRaises(LLMError) as ctx:
            await self.client.format("привет", [])
        self.assertIn("unexpected response", str(ctx.exception))

    async def test_timeout(self):
        self.state["delay"] = 1.0
        client = LLMClient(str(self.server.make_url("")), "secret", "m", timeout=0.1)
        client.start()
        try:
            with self.assertRaises(LLMError) as ctx:
                await client.format("привет", [])
        finally:
            await client.close()
        self.assertEqual(str(ctx.exception), "timeout after 0.1s")

    async def test_connection_refused(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        client = LLMClient(f"http://127.0.0.1:{port}", "secret", "m")
        client.start()
        try:
            with self.assertRaises(LLMError):
                await client.format("привет", [])
        finally:
            await client.close()

    async def test_not_started(self):
        client = LLMClient(str(self.server.make_url("")), "secret", "m")
        with self.assertRaises(LLMError) as ctx:
            await client.format("привет", [])
        self.assertEqual(str(ctx.exception), "client is not started")

    async def test_closed(self):
        await self.client.close()
        with self.assertRaises(LLMError) as ctx:
            await self.client.format("привет", [])
        self.assertEqual(str(ctx.exception), "client is not started")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
python3 -m pytest tests/test_llm_formatter.py -q
```
Expected: `ImportError: cannot import name 'LLMClient'`

- [ ] **Step 3: Implement**

In `llm_formatter.py` make the import block look like this:
```python
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import string
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Optional, Sequence

import aiohttp
```

Append to `llm_formatter.py`:
```python
OnText = Callable[[str], None]


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    def start(self) -> None:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def format(self, raw: str, context: Sequence[str]) -> str:
        if self._session is None:
            raise LLMError("client is not started")
        body = {
            "model": self.model,
            "messages": build_messages(raw, context),
            "temperature": 0.2,
            "max_tokens": 512,
            "stream": False,
            "thinking": {"type": "disabled"},
        }
        try:
            async with self._session.post(self.base_url + "/chat/completions", json=body) as resp:
                if resp.status != 200:
                    raise LLMError(
                        f"HTTP {resp.status}: {(await resp.text(errors='replace'))[:200]}"
                    )
                payload = await resp.read()
        except asyncio.TimeoutError as e:
            raise LLMError(f"timeout after {self.timeout:.1f}s") from e
        except aiohttp.ClientError as e:
            raise LLMError(str(e)) from e
        try:
            data = json.loads(payload)
        except ValueError as e:
            raise LLMError(f"malformed JSON in response: {e}") from e
        return _extract_content(data)


def _extract_content(data: object) -> str:
    summary = str(data)[:200]
    if not isinstance(data, dict):
        raise LLMError(f"unexpected response: {summary}")
    choices = data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    if not isinstance(choice, dict):
        raise LLMError(f"unexpected response: {summary}")
    finish_reason = choice.get("finish_reason")
    if finish_reason not in (None, "stop"):
        raise LLMError(f"unexpected finish_reason: {finish_reason}")
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise LLMError(f"unexpected response: {summary}")
    if not content.strip():
        raise LLMError("empty content")
    return content.strip()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
python3 -m pytest tests/test_llm_formatter.py -q
```
Expected: `53 passed` (pytest 9 appends `, 5 subtests passed`; both forms are acceptable)

- [ ] **Step 5: Commit**

```bash
git add llm_formatter.py tests/test_llm_formatter.py
git commit -m "feat: add z.ai chat completions client"
```

---

### Task 5: OrderedFormatter

**Files:**
- Modify: `llm_formatter.py`
- Modify: `tests/test_llm_formatter.py`

- [ ] **Step 1: Write the failing tests**

Replace the import block at the top of `tests/test_llm_formatter.py` with:
```python
import asyncio
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer

from llm_formatter import (
    LLMClient,
    LLMConfigError,
    LLMError,
    OrderedFormatter,
    SYSTEM_PROMPT,
    build_messages,
    ends_sentence,
    looks_faithful,
    normalize,
    resolve_api_key,
    separator,
)
```

Append to `tests/test_llm_formatter.py`:
```python
class FakeClient:
    def __init__(self, results, gates=None):
        self.calls = []
        self._results = list(results)
        self._gates = list(gates) if gates is not None else [None] * len(results)
        self.closed = False

    def start(self):
        pass

    async def format(self, raw, context):
        self.calls.append((raw, list(context)))
        index = len(self.calls) - 1
        gate = self._gates[index]
        if gate is not None:
            await gate.wait()
        result = self._results[index]
        if isinstance(result, Exception):
            raise result
        return result

    async def close(self):
        self.closed = True


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class TestOrderedFormatter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.typed = []

    def on_text(self, text):
        self.typed.append(text)

    async def wait_typed(self, count):
        for _ in range(300):
            if len(self.typed) >= count:
                return
            await asyncio.sleep(0.01)
        self.fail(f"only {len(self.typed)} of {count} texts typed")

    async def test_emits_in_order(self):
        gate_a = asyncio.Event()
        gate_b = asyncio.Event()
        client = FakeClient(["Первый.", "Второй."], gates=[gate_a, gate_b])
        formatter = OrderedFormatter(client, self.on_text)
        formatter.submit("первый")
        formatter.submit("второй")
        await asyncio.sleep(0.01)
        self.assertEqual(self.typed, [])
        gate_b.set()
        await asyncio.sleep(0.01)
        self.assertEqual(self.typed, [])
        gate_a.set()
        await self.wait_typed(2)
        self.assertEqual(self.typed, ["Первый.", " Второй."])
        await formatter.close()
        self.assertTrue(client.closed)

    async def test_failure_types_raw(self):
        client = FakeClient([LLMError("boom"), "Мир."])
        formatter = OrderedFormatter(client, self.on_text)
        formatter.submit("привет")
        formatter.submit("мир")
        await self.wait_typed(2)
        self.assertEqual(self.typed, ["привет", " Мир."])
        self.assertEqual(client.calls[0], ("привет", []))

    async def test_timeout_types_raw(self):
        gate = asyncio.Event()
        client = FakeClient(["x"], gates=[gate])
        formatter = OrderedFormatter(client, self.on_text, timeout=0.05)
        formatter.submit("первый")
        await self.wait_typed(1)
        self.assertEqual(self.typed, ["первый"])
        await formatter.close()

    async def test_rejected_output_types_raw(self):
        client = FakeClient(["Как дела сегодня"])
        formatter = OrderedFormatter(client, self.on_text)
        formatter.submit("привет")
        await self.wait_typed(1)
        self.assertEqual(self.typed, ["привет"])

    async def test_close_types_pending_raw(self):
        gate = asyncio.Event()
        client = FakeClient(["Первый.", "x"], gates=[None, gate])
        formatter = OrderedFormatter(client, self.on_text)
        formatter.submit("первый")
        formatter.submit("второй")
        await self.wait_typed(1)
        self.assertEqual(self.typed, ["Первый."])
        await formatter.close()
        self.assertEqual(self.typed, ["Первый.", " Второй"])
        self.assertTrue(client.closed)

    async def test_reset_clears_context(self):
        client = FakeClient(["A.", "B.", "C."])
        formatter = OrderedFormatter(client, self.on_text, context_utterances=1)
        formatter.submit("a")
        await self.wait_typed(1)
        formatter.reset()
        formatter.submit("b")
        await self.wait_typed(2)
        formatter.submit("c")
        await self.wait_typed(3)
        self.assertEqual(
            client.calls,
            [("a", []), ("b", []), ("c", ["B."])],
        )

    async def test_context_passed(self):
        client = FakeClient(["A.", "B.", "C."])
        formatter = OrderedFormatter(client, self.on_text, context_utterances=2)
        formatter.submit("a")
        await self.wait_typed(1)
        formatter.submit("b")
        await self.wait_typed(2)
        formatter.submit("c")
        await self.wait_typed(3)
        self.assertEqual(client.calls[1], ("b", ["A."]))
        self.assertEqual(client.calls[2], ("c", ["A.", " B."]))

    async def test_concurrency_cap(self):
        release = asyncio.Event()
        client = FakeClient(["1", "2", "3"], gates=[release, release, release])
        formatter = OrderedFormatter(client, self.on_text, concurrency=2)
        for raw in ("a", "b", "c"):
            formatter.submit(raw)
        await asyncio.sleep(0.01)
        self.assertEqual(len(client.calls), 2)
        release.set()
        await self.wait_typed(3)
        self.assertEqual([call[0] for call in client.calls], ["a", "b", "c"])

    async def test_cooldown(self):
        clock = FakeClock()
        client = FakeClient(
            [LLMError("a"), LLMError("b"), LLMError("c"), "Пять."]
        )
        formatter = OrderedFormatter(
            client, self.on_text, cooldown_after=3, cooldown_seconds=60.0, clock=clock
        )
        for raw in ("один", "два", "три"):
            formatter.submit(raw)
        await self.wait_typed(3)
        self.assertEqual(len(client.calls), 3)
        formatter.submit("четыре")
        await self.wait_typed(4)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(self.typed[3], " четыре")
        clock.now += 61
        formatter.submit("пять")
        await self.wait_typed(5)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(self.typed[4], " Пять.")

    async def test_close_types_current_before_pending(self):
        gate = asyncio.Event()
        client = FakeClient(["x", "x", "x"], gates=[gate, gate, gate])
        formatter = OrderedFormatter(client, self.on_text)
        for raw in ("один", "два", "три"):
            formatter.submit(raw)
        while not client.calls:
            await asyncio.sleep(0)
        await formatter.close()
        self.assertEqual(self.typed, ["один", " два", " три"])

    async def test_process_exception_types_raw(self):
        client = FakeClient([RuntimeError("bug"), "Мир."])
        formatter = OrderedFormatter(client, self.on_text)
        formatter.submit("привет")
        formatter.submit("мир")
        await self.wait_typed(2)
        self.assertEqual(self.typed, ["привет", " Мир."])

    async def test_cooldown_ignores_rejections(self):
        client = FakeClient(["Несвязанный текст совсем"] * 4)
        formatter = OrderedFormatter(client, self.on_text)
        for raw in ("первый", "второй", "третий", "четвертый"):
            formatter.submit(raw)
        await self.wait_typed(4)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(self.typed, ["первый", " второй", " третий", " четвертый"])

    async def test_reset_clears_tail(self):
        client = FakeClient(["Первый", "Второй"])
        formatter = OrderedFormatter(client, self.on_text)
        formatter.submit("первый")
        await self.wait_typed(1)
        formatter.reset()
        formatter.submit("второй")
        await self.wait_typed(2)
        self.assertEqual(self.typed, ["Первый", "Второй"])

    async def test_on_text_error_is_logged_and_continues(self):
        typed = []
        state = {"fail": True}

        def on_text(text):
            typed.append(text)
            if state["fail"]:
                state["fail"] = False
                raise RuntimeError("typing failed")

        client = FakeClient(["Первый.", "Второй."])
        formatter = OrderedFormatter(client, on_text)
        formatter.submit("первый")
        formatter.submit("второй")
        for _ in range(300):
            if len(typed) >= 2:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(typed, ["Первый.", " Второй."])

    async def test_submit_after_close_types_raw(self):
        client = FakeClient(["Первый."])
        formatter = OrderedFormatter(client, self.on_text)
        formatter.submit("первый")
        await self.wait_typed(1)
        await formatter.close()
        self.assertTrue(client.closed)
        formatter.submit("второй")
        await self.wait_typed(2)
        self.assertEqual(self.typed, ["Первый.", "второй"])
        self.assertEqual(len(client.calls), 1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
python3 -m pytest tests/test_llm_formatter.py -q
```
Expected: `ImportError: cannot import name 'OrderedFormatter'`

- [ ] **Step 3: Implement**

In `llm_formatter.py` add `import time` after `import string` and `from collections import deque` directly before `from difflib import SequenceMatcher`.

Append to `llm_formatter.py`:
```python
class OrderedFormatter:
    def __init__(
        self,
        client: LLMClient,
        on_text: OnText,
        timeout: float = DEFAULT_TIMEOUT,
        concurrency: int = DEFAULT_CONCURRENCY,
        context_utterances: int = DEFAULT_CONTEXT_UTTERANCES,
        cooldown_after: int = COOLDOWN_AFTER,
        cooldown_seconds: float = COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.client = client
        self.timeout = timeout
        self.concurrency = concurrency
        self.context_size = context_utterances
        self._on_text = on_text
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._context: deque[str] = deque(maxlen=max(0, context_utterances))
        self._cooldown_after = cooldown_after
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._pending: dict[int, tuple[str, asyncio.Task]] = {}
        self._next_seq = 0
        self._consumer: Optional[asyncio.Task] = None
        self._current: Optional[tuple[int, str, asyncio.Task]] = None
        self._typing: Optional[asyncio.Task] = None
        self._tail: Optional[str] = None
        self._failures = 0
        self._disabled_until: Optional[float] = None
        self._closed = False

    @property
    def pending_count(self) -> int:
        return len(self._pending) + (1 if self._current is not None else 0)

    def submit(self, raw: str) -> int:
        if self._closed:
            log.warning("llm: formatter is closed, typing raw: %r", raw)
            asyncio.get_running_loop().create_task(asyncio.to_thread(self._on_text, raw))
            return -1
        self.client.start()
        seq = self._next_seq
        self._next_seq += 1
        self._pending[seq] = (raw, asyncio.create_task(self._process(seq, raw)))
        if self._consumer is None:
            self._consumer = asyncio.create_task(self._run())
        return seq

    def reset(self) -> None:
        self._context.clear()
        self._tail = None

    def _in_cooldown(self) -> bool:
        until = self._disabled_until
        if until is None:
            return False
        if self._clock() >= until:
            self._disabled_until = None
            self._failures = 0
            return False
        return True

    async def _process(self, seq: int, raw: str) -> Optional[str]:
        t_submit = self._clock()
        if self._in_cooldown():
            log.info("llm: seq=%d skipped (cooldown), typing raw", seq)
            return None
        async with self._sem:
            if self._in_cooldown():
                log.info("llm: seq=%d skipped (cooldown), typing raw", seq)
                return None
            waited = self._clock() - t_submit
            t_start = self._clock()
            try:
                text = await asyncio.wait_for(
                    self.client.format(raw, list(self._context)), self.timeout
                )
            except asyncio.TimeoutError:
                self._register_failure(seq, f"timeout after {self.timeout:.1f}s")
                return None
            except LLMError as e:
                self._register_failure(seq, str(e))
                return None
            except Exception as e:
                self._register_failure(seq, repr(e))
                return None
        api = self._clock() - t_start
        total = self._clock() - t_submit
        faithful = False
        try:
            text = text.strip() if isinstance(text, str) else ""
            if text:
                faithful = await asyncio.to_thread(looks_faithful, raw, text)
        except Exception:
            log.exception("llm: seq=%d validation failed", seq)
            faithful = False
        if not faithful:
            log.warning(
                "llm: seq=%d wait=%.2fs api=%.2fs total=%.2fs rejected, typing raw",
                seq, waited, api, total,
            )
            return None
        self._failures = 0
        log.info(
            "llm: seq=%d wait=%.2fs api=%.2fs total=%.2fs formatted",
            seq, waited, api, total,
        )
        return text

    def _register_failure(self, seq: int, error: str) -> None:
        self._failures += 1
        log.warning("llm: seq=%d failed (%s), typing raw", seq, error)
        if self._failures >= self._cooldown_after:
            self._disabled_until = self._clock() + self._cooldown_seconds
            self._failures = 0
            log.warning(
                "llm: %d failures in a row, formatting paused for %.0fs",
                self._cooldown_after, self._cooldown_seconds,
            )

    async def _run(self) -> None:
        try:
            while self._pending:
                seq = min(self._pending)
                raw, task = self._pending.pop(seq)
                self._current = (seq, raw, task)
                try:
                    formatted = await task
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("llm: seq=%d failed unexpectedly", seq)
                    formatted = None
                self._current = None
                await self._emit(formatted if formatted is not None else raw)
        finally:
            self._consumer = None

    async def _emit(self, text: str) -> None:
        out = normalize(self._tail, text)
        self._tail = out
        if self._context.maxlen:
            self._context.append(out)
        typing_task = asyncio.ensure_future(asyncio.to_thread(self._on_text, out))
        self._typing = typing_task
        try:
            await asyncio.shield(typing_task)
        except Exception:
            log.exception("llm: typing failed, text dropped: %r", out)
        finally:
            if typing_task.done():
                self._typing = None

    async def close(self) -> None:
        self._closed = True
        consumer, self._consumer = self._consumer, None
        if consumer is not None:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        typing_task, self._typing = self._typing, None
        if typing_task is not None:
            await asyncio.gather(typing_task, return_exceptions=True)
        if self._current is not None:
            seq, raw, task = self._current
            self._current = None
            task.cancel()
            await self._emit(raw)
        for seq in sorted(self._pending):
            raw, task = self._pending.pop(seq)
            task.cancel()
            await self._emit(raw)
        await self.client.close()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
python3 -m pytest tests/test_llm_formatter.py -q
```
Expected: `68 passed`

- [ ] **Step 5: Commit**

```bash
git add llm_formatter.py tests/test_llm_formatter.py
git commit -m "feat: add ordered formatter pipeline"
```

---

### Task 6: StreamingSTT keeps the overlay

**Files:**
- Modify: `streaming.py`
- Create: `tests/test_streaming.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_streaming.py`:
```python
import unittest

from streaming import State, StreamingSTT


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, data):
        self.sent.append(data)


class TestDeferOverlayHide(unittest.IsolatedAsyncioTestCase):
    async def test_final_hides_overlay_by_default(self):
        partials = []
        stt = StreamingSTT(FakeWS(), wake_phrase="голосовой ввод", stop_phrase="", on_partial=partials.append)
        stt.state = State.DICTATING
        stt._last_partial_text = "привет"
        await stt._finalize_utterance()
        self.assertEqual(partials, [""])

    async def test_final_keeps_overlay_with_defer_flag(self):
        partials = []
        finals = []
        stt = StreamingSTT(
            FakeWS(),
            wake_phrase="голосовой ввод",
            stop_phrase="",
            on_partial=partials.append,
            on_final=finals.append,
            defer_overlay_hide=True,
        )
        stt.state = State.DICTATING
        stt._last_partial_text = "привет"
        await stt._finalize_utterance()
        stt._last_partial_text = "как дела"
        await stt._handle_final("как дела")
        self.assertEqual(finals, ["привет", " как дела"])
        self.assertEqual(partials, [])
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python3 -m pytest tests/test_streaming.py -q
```
Expected: `1 failed, 1 passed` (`TypeError: __init__() got an unexpected keyword argument 'defer_overlay_hide'`)

- [ ] **Step 3: Implement**

In `streaming.py` replace the complete `__init__` method with:
```python
    def __init__(
        self,
        ws,
        wake_phrase: str,
        stop_phrase: str,
        on_partial=None,
        on_final=None,
        on_state_change=None,
        defer_overlay_hide: bool = False,
    ):
        self.ws = ws
        self.wake_norm = self._normalize(wake_phrase)
        self.wake_words = set(self.wake_norm.split())
        self.stop_norm = self._normalize(stop_phrase) if stop_phrase else None
        self.state = State.IDLE
        self._on_partial = on_partial
        self._on_final = on_final
        self._on_state_change = on_state_change
        self._defer_overlay_hide = defer_overlay_hide
        self._idle_counter = 0
        self._has_typed = False
        self._running = True
        self._recent_partials = []
        self._max_partials = 5
        self._last_partial_text = ""
        self._last_partial_norm = ""
        self._dictated_texts = set()
        self._wake_time = 0
```

Replace the complete `_finalize_utterance` and `_handle_final` methods with:
```python
    async def _finalize_utterance(self):
        if not self._last_partial_text:
            return
        if self._is_wake_remnant(self._last_partial_text):
            self._last_partial_text = ""
            return

        text = self._last_partial_text
        self._dictated_texts.add(self._last_partial_norm)
        self._last_partial_text = ""
        self._last_partial_norm = ""

        typed_text = text
        if self._has_typed:
            typed_text = " " + typed_text
        if self._on_final:
            self._on_final(typed_text)
        self._has_typed = True

        if self._on_partial and not self._defer_overlay_hide:
            self._on_partial("")

        log.info("Finalized: '%s'", text)

    async def _handle_final(self, text: str):
        norm = self._normalize(text)

        if self.state == State.IDLE:
            if self.wake_norm in norm:
                self._set_state(State.DICTATING)
                self._idle_counter = 0
                self._has_typed = False
                self._dictated_texts.clear()
            return

        if self.state == State.DICTATING:
            self._idle_counter = 0

            if self.stop_norm and self.stop_norm in norm:
                self._set_state(State.IDLE)
                self._dictated_texts.clear()
                self._recent_partials.clear()
                return

            if self._is_wake_remnant(text):
                return

            if norm in self._dictated_texts:
                return

            self._dictated_texts.add(norm)

            typed_text = text
            if self._has_typed:
                typed_text = " " + typed_text
            if self._on_final:
                self._on_final(typed_text)
            self._has_typed = True

            if self._on_partial and not self._defer_overlay_hide:
                self._on_partial("")

            log.info("Final (from server): '%s'", text)
            await self.ws.send(json.dumps({"type": "start"}))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
python3 -m pytest tests/test_streaming.py -q
```
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add streaming.py tests/test_streaming.py
git commit -m "feat: keep overlay until formatted text is typed"
```

---

### Task 7: type_text fix and wiring

**Files:**
- Modify: `wyoming-hid.py`
- Modify: `llm_formatter.py`
- Modify: `tests/test_llm_formatter.py`
- Create: `tests/test_entry.py`

- [ ] **Step 1: Write the failing tests**

Replace the import block at the top of `tests/test_llm_formatter.py` with:
```python
import asyncio
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer

from llm_formatter import (
    LLMClient,
    LLMConfigError,
    LLMError,
    OrderedFormatter,
    SYSTEM_PROMPT,
    build_formatter,
    build_messages,
    ends_sentence,
    looks_faithful,
    normalize,
    resolve_api_key,
    separator,
)
```

Append to `tests/test_llm_formatter.py`:
```python
class TestBuildFormatter(unittest.TestCase):
    def setUp(self):
        self.typed = []

    def on_text(self, text):
        self.typed.append(text)

    def test_disabled_returns_none(self):
        self.assertIsNone(build_formatter(None, self.on_text))
        self.assertIsNone(build_formatter({"enabled": False}, self.on_text))

    def test_defaults(self):
        with patch("llm_formatter.resolve_api_key", return_value="k"):
            formatter = build_formatter({}, self.on_text)
        self.assertEqual(
            (formatter.timeout, formatter.concurrency, formatter.context_size,
             formatter.client.model, formatter.client.base_url),
            (8.0, 4, 2, "glm-5.3-flash", "https://api.z.ai/api/coding/paas/v4"),
        )

    def test_overrides(self):
        cfg = {
            "base_url": "http://x/",
            "model": "m",
            "timeout": 3,
            "concurrency": 2,
            "context_utterances": 1,
        }
        with patch("llm_formatter.resolve_api_key", return_value="k"):
            formatter = build_formatter(cfg, self.on_text)
        self.assertEqual(
            (formatter.timeout, formatter.concurrency, formatter.context_size,
             formatter.client.model, formatter.client.base_url),
            (3.0, 2, 1, "m", "http://x"),
        )

    def test_missing_key_returns_none(self):
        with patch("llm_formatter.resolve_api_key", side_effect=LLMConfigError("no key")):
            with self.assertLogs("wyoming-hid", level="ERROR"):
                self.assertIsNone(build_formatter({}, self.on_text))
```

Create `tests/test_entry.py`:
```python
import asyncio
import importlib.util
import signal
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRY = REPO_ROOT / "wyoming-hid.py"


def load_entry_module():
    spec = importlib.util.spec_from_file_location("wyoming_hid_entry", ENTRY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTypeText(unittest.TestCase):
    def test_preserves_leading_space(self):
        module = load_entry_module()
        with patch.object(module.subprocess, "run") as run:
            module.type_text(" мир")
        run.assert_called_once_with(
            ["xdotool", "type", "--clearmodifiers", "--delay", "0", " мир"],
            check=True,
            timeout=10,
        )

    def test_skips_whitespace_only(self):
        module = load_entry_module()
        with patch.object(module.subprocess, "run") as run:
            module.type_text("   ")
        run.assert_not_called()


class TestRunStreamingWiring(unittest.IsolatedAsyncioTestCase):
    async def test_final_goes_to_formatter(self):
        module = load_entry_module()
        events = []
        original = signal.getsignal(signal.SIGINT)

        class FakeFormatter:
            def submit(self, text):
                events.append(("submit", text))

            def reset(self):
                events.append(("reset",))

            async def close(self):
                events.append(("close",))

        callbacks = {}

        class FakeStreamingSTT:
            def __init__(self, ws, wake_phrase, stop_phrase, on_partial, on_final, on_state_change, defer_overlay_hide=False):
                callbacks["on_final"] = on_final
                callbacks["on_state_change"] = on_state_change
                callbacks["defer"] = defer_overlay_hide

            async def start_session(self):
                pass

            async def result_reader(self, idle_timeout_chunks=6):
                await asyncio.sleep(3600)

            async def send_audio(self, data):
                raise module.websockets.ConnectionClosed(None, None)

        class FakeStream:
            def read(self, n, exception_on_overflow=False):
                return b"\x00" * n

            def stop_stream(self):
                pass

            def close(self):
                pass

        class FakePyAudio:
            def open(self, **kwargs):
                return FakeStream()

            def terminate(self):
                pass

        class FakeWSContext:
            def __init__(self, ws):
                self.ws = ws

            async def __aenter__(self):
                return self.ws

            async def __aexit__(self, *exc):
                return False

        class FakeConnect:
            def __init__(self):
                self.calls = 0

            def __call__(self, url, **kwargs):
                self.calls += 1
                if self.calls > 1:
                    raise KeyboardInterrupt
                return FakeWSContext(object())

        fake_pyaudio = SimpleNamespace(PyAudio=FakePyAudio, paInt16=8)

        try:
            with patch.object(module, "StreamingSTT", FakeStreamingSTT), patch.object(module, "pyaudio", fake_pyaudio), patch.object(module.websockets, "connect", FakeConnect()), patch.object(module, "build_formatter", lambda cfg, on_text: FakeFormatter()):
                task = asyncio.ensure_future(module.run_streaming(
                    "ws://x/stt", "", "xdotool", wake_phrase="w", stop_phrase="s",
                    idle_timeout=15, use_overlay=False, overlay_opacity=0.85,
                    overlay_font_size=18, llm_cfg={},
                ))
                for _ in range(300):
                    if "on_final" in callbacks:
                        break
                    await asyncio.sleep(0.01)
                self.assertIn("on_final", callbacks)
                self.assertTrue(callbacks["defer"])
                callbacks["on_final"]("привет")
                callbacks["on_state_change"](module.State.IDLE, module.State.DICTATING)
                await asyncio.wait_for(task, timeout=10)
        finally:
            signal.signal(signal.SIGINT, original)
        self.assertEqual(events, [("submit", "привет"), ("reset",), ("close",)])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
python3 -m pytest tests/test_entry.py tests/test_llm_formatter.py -q
```
Expected: `python3 -m pytest tests/test_entry.py tests/test_llm_formatter.py -q` stops at the collection error of `tests/test_llm_formatter.py` (`ImportError: cannot import name 'build_formatter'`); run `python3 -m pytest tests/test_entry.py -q` separately: `2 failed, 1 passed` (`test_preserves_leading_space`: the command was called with `'мир'` instead of `' мир'`; `test_final_goes_to_formatter`: `AttributeError: ... does not have the attribute 'build_formatter'`, because the entry module only gets that attribute in Step 3)

- [ ] **Step 3: Implement**

Append to `llm_formatter.py`:
```python
def build_formatter(
    llm_cfg: Optional[dict], on_text: OnText
) -> Optional[OrderedFormatter]:
    if llm_cfg is None:
        return None
    if not llm_cfg.get("enabled", True):
        return None
    base_url = llm_cfg.get("base_url", DEFAULT_BASE_URL)
    try:
        api_key = resolve_api_key(
            llm_cfg.get("api_key"), allow_auth_store=base_url == DEFAULT_BASE_URL
        )
    except LLMConfigError as e:
        log.error("LLM formatting is off: %s", e)
        return None
    client = LLMClient(
        base_url=base_url,
        api_key=api_key,
        model=llm_cfg.get("model", DEFAULT_MODEL),
        timeout=float(llm_cfg.get("timeout", DEFAULT_TIMEOUT)),
    )
    return OrderedFormatter(
        client,
        on_text,
        timeout=client.timeout,
        concurrency=int(llm_cfg.get("concurrency", DEFAULT_CONCURRENCY)),
        context_utterances=int(llm_cfg.get("context_utterances", DEFAULT_CONTEXT_UTTERANCES)),
    )
```

In `wyoming-hid.py` add `from llm_formatter import build_formatter` directly below `from streaming import State, StreamingSTT`, and replace `type_text`, `run_streaming` and `main` with the versions below.
```python
def type_text(text: str, method: str = "xdotool"):
    if not text.strip():
        return
    log.info("Typing: %s", text)
    try:
        if method == "xdotool":
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "0", text],
                check=True,
                timeout=10,
            )
        elif method == "ydotool":
            subprocess.run(
                ["ydotool", "type", "--key-delay", "0", text],
                check=True,
                timeout=10,
            )
    except FileNotFoundError:
        log.error("Typing tool '%s' not found", method)
    except subprocess.TimeoutExpired:
        log.error("Typing timed out")
    except subprocess.CalledProcessError as e:
        log.error("Typing failed: %s", e)


async def run_streaming(
    ws_url: str,
    auth_header: str,
    typing_method: str,
    wake_phrase: str,
    stop_phrase: str,
    idle_timeout: float,
    use_overlay: bool,
    overlay_opacity: float,
    overlay_font_size: int,
    llm_cfg=None,
):
    overlay = None
    if use_overlay:
        try:
            overlay = GTKOverlay(opacity=overlay_opacity, font_size=overlay_font_size)
            overlay.start()
            log.info("Overlay started")
        except Exception as e:
            log.warning("Overlay failed, falling back to console: %s", e)
            overlay = None

    def on_text(text: str):
        type_text(text, typing_method)
        if overlay and formatter.pending_count == 0:
            overlay.hide()

    formatter = build_formatter(llm_cfg, on_text)

    def on_partial(text: str):
        if overlay:
            if text:
                overlay.update(text)
                overlay.show(text)
            else:
                overlay.hide()

    def on_final(text: str):
        if formatter:
            formatter.submit(text)
        else:
            type_text(text, typing_method)

    def on_state_change(old: State, new_state: State):
        if overlay:
            if new_state == State.IDLE:
                overlay.hide()
            elif new_state == State.DICTATING:
                overlay.show("🎤 ...")
        if formatter and new_state == State.DICTATING:
            formatter.reset()

    extra_headers = {"Proxy-Authorization": auth_header} if auth_header else {}

    backoff = 1
    max_backoff = 30
    shutting_down = False

    def _signal_handler(sig, frame):
        nonlocal shutting_down
        shutting_down = True
        log.info("Shutting down...")

    import signal
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        while not shutting_down:
            try:
                async with websockets.connect(
                    ws_url,
                    extra_headers=extra_headers,
                    ping_interval=None,
                    max_size=None,
                    compression=None,
                ) as ws:
                    log.info("Connected to %s", ws_url)
                    backoff = 1

                    _dn = open(os.devnull, "w")
                    _oe = os.dup(2)
                    os.dup2(_dn.fileno(), 2)
                    pa = pyaudio.PyAudio()
                    stream = pa.open(
                        format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=NATIVE_RATE,
                        input=True,
                        frames_per_buffer=NATIVE_CHUNK,
                    )
                    os.dup2(_oe, 2)
                    os.close(_oe)
                    _dn.close()
                    log.info("Mic: %dHz -> %dHz", NATIVE_RATE, TARGET_RATE)

                    stt = StreamingSTT(
                        ws,
                        wake_phrase=wake_phrase,
                        stop_phrase=stop_phrase,
                        on_partial=on_partial,
                        on_final=on_final,
                        on_state_change=on_state_change,
                        defer_overlay_hide=formatter is not None,
                    )

                    await stt.start_session()

                    loop = asyncio.get_event_loop()
                    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=120)
                    mic_stop = threading.Event()

                    resample_state = None
                    audio_buffer = bytearray()

                    def mic_reader():
                        nonlocal resample_state
                        try:
                            while not mic_stop.is_set():
                                raw = stream.read(NATIVE_CHUNK, exception_on_overflow=False)
                                resampled, resample_state = audioop.ratecv(
                                    raw, SAMPLE_WIDTH, CHANNELS,
                                    NATIVE_RATE, TARGET_RATE, resample_state,
                                )
                                if not audio_queue.full():
                                    loop.call_soon_threadsafe(
                                        audio_queue.put_nowait, resampled
                                    )
                        except Exception as e:
                            log.error("mic_reader: %s", e)

                    mic_thread = threading.Thread(target=mic_reader, daemon=True)
                    mic_thread.start()

                    result_task = asyncio.create_task(stt.result_reader(
                        idle_timeout_chunks=max(1, int(idle_timeout / 2))
                    ))

                    try:
                        while not shutting_down:
                            try:
                                chunk = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
                            except asyncio.TimeoutError:
                                continue
                            audio_buffer.extend(chunk)
                            while len(audio_buffer) >= SEND_BYTES:
                                send_data = bytes(audio_buffer[:SEND_BYTES])
                                audio_buffer = audio_buffer[SEND_BYTES:]
                                await stt.send_audio(send_data)
                    except websockets.ConnectionClosed:
                        log.warning("Disconnected")
                    except asyncio.CancelledError:
                        pass
                    finally:
                        result_task.cancel()
                        try:
                            await result_task
                        except asyncio.CancelledError:
                            pass
                        stream.stop_stream()
                        stream.close()
                        mic_stop.set()
                        pa.terminate()
                        if overlay:
                            overlay.hide()

            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                if shutting_down:
                    break
                log.warning("Connection error: %s, retrying in %ds", e, backoff)
                if overlay:
                    overlay.show(f"⚠ Reconnecting ({backoff}s)...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
            except KeyboardInterrupt:
                shutting_down = True
    finally:
        if formatter:
            await formatter.close()


def main():
    parser = argparse.ArgumentParser(description="Wyoming HID - Streaming STT to keyboard")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--url")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--method", choices=["xdotool", "ydotool"])
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = load_config(Path(args.config))

    ws_url = args.url or cfg.get("url", "ws://localhost:8080/stt")
    user = args.user or cfg.get("user", "")
    password = args.password or cfg.get("password", "")
    typing_method = args.method or cfg.get("method", "xdotool")
    wake_phrase = cfg.get("wake_phrase", "голосовой ввод")
    stop_phrase = cfg.get("stop_phrase", "стоп ввод")
    idle_timeout = float(cfg.get("idle_timeout", 15))
    use_overlay = not args.no_overlay and cfg.get("overlay", True)
    overlay_opacity = float(cfg.get("overlay_opacity", 0.85))
    overlay_font_size = int(cfg.get("overlay_font_size", 18))

    if user and password:
        auth = f"Basic {base64.b64encode(f'{user}:{password}'.encode()).decode()}"
    else:
        auth = ""

    llm_cfg = cfg.get("llm") or {}

    log.info(
        "URL: %s | Method: %s | Overlay: %s | LLM: %s",
        ws_url, typing_method, use_overlay,
        "on" if llm_cfg.get("enabled", True) else "off",
    )

    try:
        asyncio.run(run_streaming(
            ws_url, auth, typing_method,
            wake_phrase=wake_phrase,
            stop_phrase=stop_phrase,
            idle_timeout=idle_timeout,
            use_overlay=use_overlay,
            overlay_opacity=overlay_opacity,
            overlay_font_size=overlay_font_size,
            llm_cfg=llm_cfg,
        ))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
python3 -m pytest tests -q
```
Expected: `77 passed` (pytest 9 may append `, 5 subtests passed` because of the subtests in `test_malformed_store_raises`)

- [ ] **Step 5: Check that the module still starts**

Run:
```bash
python3 -m py_compile wyoming-hid.py && python3 wyoming-hid.py --help
```
Expected: usage text, exit code 0

- [ ] **Step 6: Commit**

```bash
git add wyoming-hid.py llm_formatter.py tests/test_entry.py tests/test_llm_formatter.py
git commit -m "feat: format finalized utterances with GLM-5.3-Flash before typing"
```

---

### Task 8: Config and docs

**Files:**
- Modify: `config.yaml.example`
- Modify: `README.md`

- [ ] **Step 1: Add the section to the example config**

Append to `config.yaml.example`:
```yaml

# LLM formatting of finalized utterances before typing
llm:
  enabled: true                     # false to type the raw text
  base_url: https://api.z.ai/api/coding/paas/v4
  model: glm-5.3-flash
  timeout: 8                        # seconds per request
  concurrency: 4                    # requests in flight
  context_utterances: 2             # formatted utterances passed as context
  # api_key: ...                    # default: $ZAI_API_KEY, then the opencode auth store (zai-coding-plan)
```

- [ ] **Step 2: Update the README**

In `README.md` make three changes.

Change line 23 to:
```markdown
- Python packages: `pyaudio`, `websockets`, `pyyaml`, `aiohttp`
```

Change line 29 to:
```bash
pip3 install websockets pyyaml aiohttp
```

Insert the following section between the closing fence of the configuration block (line 57) and `## Usage` (line 59), with one blank line before and after it:
``````markdown
## LLM Formatting

With an `llm` section in `config.yaml`, every finalized utterance is sent to GLM-5.3-Flash on z.ai before typing: punctuation, capitalization, glued words split and fillers removed. Requests run in parallel (up to `concurrency` in flight) and results are typed in the order the utterances were finalized. On any failure (timeout, HTTP error, unfaithful output) the raw text is typed instead.

```yaml
llm:
  enabled: true
  base_url: https://api.z.ai/api/coding/paas/v4
  model: glm-5.3-flash
  timeout: 8            # seconds per request
  concurrency: 4        # requests in flight
  context_utterances: 2 # formatted utterances passed as context
  # api_key: ...        # default: $ZAI_API_KEY, then the opencode auth store (zai-coding-plan)
```
``````

- [ ] **Step 3: Enable formatting in the local config**

The local `config.yaml` is ignored by git and holds credentials: do not print it. Run:
```bash
python3 - <<'EOF'
from pathlib import Path

path = Path("config.yaml")
text = path.read_text()
block = "\n# LLM formatting of finalized utterances before typing\nllm:\n  enabled: true\n"
if any(line.startswith("llm:") for line in text.splitlines()):
    print("llm section already present, config.yaml left untouched")
else:
    path.write_text(text.rstrip("\n") + "\n" + block)
    print("llm section added")
EOF
```
Expected: `llm section added`

- [ ] **Step 4: Commit**

```bash
git add config.yaml.example README.md
git commit -m "docs: describe the llm config section"
```

---

### Task 9: Smoke test and final verification

**Files:**
- Create: `scripts/llm_smoke.py`

- [ ] **Step 1: Create the smoke script**

Create `scripts/llm_smoke.py`:
```python
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


class ModeCounter(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.formatted = 0
        self.raw = 0
        self.unchanged = 0

    def emit(self, record):
        message = record.getMessage()
        if not message.startswith("llm: seq="):
            return
        if message.endswith(" formatted"):
            self.formatted += 1
        elif "typing raw" in message:
            self.raw += 1


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    counter = ModeCounter()
    logging.getLogger("wyoming-hid").addHandler(counter)

    typed = []
    started = time.monotonic()
    done = asyncio.Event()
    current = {"raw": ""}

    def on_text(text):
        if current["raw"] and text.strip() == current["raw"].strip():
            counter.unchanged += 1
        typed.append(text)
        elapsed = time.monotonic() - started
        print(f"{elapsed:6.2f}s  typed: {text}", flush=True)
        if len(typed) >= len(UTTERANCES):
            done.set()

    config = yaml.safe_load((ROOT / "config.yaml").read_text()) or {}
    formatter = build_formatter(config.get("llm") or {}, on_text)
    if formatter is None:
        print("LLM formatting is disabled (check the llm section of config.yaml)")
        return 1
    for raw in UTTERANCES:
        formatter.submit(raw)
        await asyncio.sleep(0.4)
    await asyncio.wait_for(done.wait(), timeout=120)

    started = time.monotonic()
    typed.clear()
    for raw in UTTERANCES:
        current["raw"] = raw
        before = len(typed)
        formatter.submit(raw)
        while len(typed) == before:
            await asyncio.sleep(0.05)
    current["raw"] = ""
    await formatter.close()
    total = len(UTTERANCES) * 2
    print(f"formatted={counter.formatted} unchanged={counter.unchanged} raw={counter.raw}")
    return 0 if (counter.formatted + counter.unchanged) * 10 >= total * 9 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 2: Run it**

Run:
```bash
python3 scripts/llm_smoke.py
```
Expected: eighteen `typed:` lines, nine from the parallel pass (submitted 0.4 s apart, each line appears roughly 2-3 s after its submission) and nine from the sequential pass (each line appears roughly 2-3 s after the previous one); the last line is `formatted=9 unchanged=0 raw=0` (a single `raw=1` or `unchanged=1` is acceptable); exit code 0. The typed text itself depends on the model.

If the log shows `llm: API key source: env` followed by `HTTP 401` failures, the `ZAI_API_KEY` of the current environment is stale (this is the case on the development laptop: the key exported in `~/.bashrc` is rejected by z.ai, while the key in the opencode auth store works). In that case run `env -u ZAI_API_KEY python3 scripts/llm_smoke.py`, and report both outputs.

- [ ] **Step 3: Final verification**

Run:
```bash
python3 -m pytest tests -q
```
Expected: `77 passed` (pytest 9 may append `, 5 subtests passed` because of the subtests in `test_malformed_store_raises`)

Run:
```bash
python3 -m py_compile wyoming-hid.py streaming.py overlay.py llm_formatter.py scripts/llm_smoke.py
```
Expected: no output, exit code 0

Run:
```bash
git status --short
```
Expected: empty output

- [ ] **Step 4: Commit**

```bash
git add scripts/llm_smoke.py
git commit -m "test: add smoke script for the real API"
```

---

## Self-Review

**1. Spec coverage**

- System prompt, messages (constant, context lines): Task 2
- `looks_faithful`, length guard: Task 1
- `separator`, `ends_sentence`, capitalization after a sentence end: Task 1
- `LLMClient` (non-streaming, thinking disabled, temperature 0.2, max_tokens 512, `LLMError` on non-200 / malformed / empty / truncated): Task 4
- `resolve_api_key` (explicit, env, opencode auth store, `LLMConfigError`): Task 3
- `OrderedFormatter` (ordered parallel emission, timeout, raw on failure, cooldown, `reset()`, `close()`, logging): Task 5
- `StreamingSTT` overlay flag: Task 6
- `type_text` strip fix, wiring, `build_formatter` returning `None` when the section is missing, disabled or without a key: Task 7
- Config example, README, local config: Task 8
- Acceptance run against the real API: Task 9

**2. Placeholder scan:** no placeholders; every step has complete code and exact commands.

**3. Type consistency:** `OrderedFormatter(client, on_text, timeout, concurrency, context_utterances, cooldown_after, cooldown_seconds, clock)` with attributes `client`, `timeout`, `concurrency`, `context_size`, `pending_count` and methods `submit(raw)`, `reset()`, `close()`; `LLMClient(base_url, api_key, model, timeout)` with attributes `base_url`, `api_key`, `model`, `timeout` and methods `start()`, `close()`, `format(raw, context)`; `build_formatter(llm_cfg, on_text)`; `StreamingSTT(..., defer_overlay_hide=False)`; `resolve_api_key(explicit=None)`, `AUTH_STORE_PATH`, `AUTH_STORE_PROVIDER`, `LLMConfigError`, `LLMError`, `OnText`.
