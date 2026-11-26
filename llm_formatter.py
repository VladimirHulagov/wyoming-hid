#!/usr/bin/env python3
"""Format finalized dictation utterances with an LLM before typing."""

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
AUTH_STORE_PATH = Path("~/.local/share/opencode/auth.json").expanduser()
AUTH_STORE_PROVIDER = "zai-coding-plan"
LEAD_PUNCT = frozenset(string.punctuation)

log = logging.getLogger("wyoming-hid")

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
