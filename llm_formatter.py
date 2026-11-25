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
    matched, used = 0, set()
    for word in raw_words:
        for index, token in enumerate(out_tokens):
            if index not in used and _match(word, token):
                matched += 1
                used.add(index)
                break
    if matched / len(raw_words) < FAITHFUL_RATIO:
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
