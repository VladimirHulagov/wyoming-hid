# LLM Formatting Design

Turn each finalized utterance of the dictation into readable Russian text (punctuation,
capitalization, glued words split, fillers removed) with GLM-5.3-Flash **before** it is typed.
Budget: 3-5 seconds per utterance. Nothing is ever lost: on any failure the raw text is typed.

## Architecture

```
StreamingSTT._finalize_utterance() / _handle_final()
        |
        | on_final(raw text)
        v
OrderedFormatter.submit(raw)          one task per utterance, seq number
        |
        |  (<= concurrency in flight, asyncio.Semaphore)
        v
LLMClient.format(raw, context)        POST {base_url}/chat/completions
        |                             thinking disabled, a short timeout
        |  returns a formatted fragment or raises LLMError
        v
consumer task                         awaits tasks strictly in seq order
        |
        |  separator + first-char case normalization
        v
on_text(text) -> type_text() in a worker thread (xdotool) + overlay.hide()
```

Measured on this laptop (3 utterances, ~100 output tokens): TTFT 1.1-1.4 s, total 2.2-2.4 s,
82-100 tokens/s. Non-streaming requests are used here; verify the total time stays in the budget.

## Components

### `llm_formatter.py`

- `SYSTEM_PROMPT`, `build_messages(raw, context)` - system prompt is a constant Russian text;
  the user message contains up to N formatted context lines and the raw utterance.
- `looks_faithful(raw, out) -> bool` - a guard against invented or dropped content:
  - tokens are runs of Cyrillic/Latin letters in the lowercased text (ё -> е); fillers are not
    counted as content words: ну, вот, а, да, угу, ага, это, то, и, же, там, тут;
  - a content word of `raw` is matched in `out` by exact match, by substring (both >= 3 chars) or by
    `SequenceMatcher` ratio >= 0.6 (both >= 4 chars);
  - at least 70 % must match (ASR error corrections such as "попоавит" -> "поправит" pass);
  - `len(out) <= 2.5 * len(raw) + 20`;
  - when `raw` has no content words, `out` must have no content words either.
- `separator(prev, new)` - `""` when `prev` is None or `new` starts with punctuation, else `" "`.
- `ends_sentence(text)` - last non-closer character is one of `. ! ? …`.
- `LLMClient(base_url, api_key, model, timeout)`:
  - `start()`, `close()`, `async format(raw, context) -> str`;
  - non-streaming POST to `{base_url}/chat/completions`, `temperature 0.2`, `max_tokens 512`,
    `thinking: {"type": "disabled"}`;
  - raises `LLMError` on non-200 status, malformed JSON, empty content or `finish_reason == "length"`.
- `resolve_api_key(explicit) -> str` - `explicit` from config, then `ZAI_API_KEY`,
  then `~/.local/share/opencode/auth.json` key `zai-coding-plan`; raises `LLMConfigError` if none.
- `OrderedFormatter(client, on_text, timeout=8.0, concurrency=4, context_utterances=2,
  cooldown_after=3, cooldown_seconds=60.0, clock=time.monotonic)`:
  - `submit(raw) -> seq` - synchronous, never blocks: stores the raw text, assigns `seq`,
    creates the task and makes sure the consumer task is running;
  - the consumer awaits tasks strictly in `seq` order and calls `await on_text(text)` sequentially
    (`on_text` is an async callable); the context for the next requests is the last
    `context_utterances` emitted texts;
  - `_process` - cooldown check, request under the semaphore with `asyncio.wait_for(timeout)`,
    validation (`looks_faithful`); returns the formatted text or the raw text; only request
    errors and timeouts count towards the cooldown;
  - after emitting, if `ends_sentence(prev)` and the new text starts with a lowercase letter,
    the first character is uppercased; text starting with punctuation is left as is;
  - one INFO log line per utterance: seq, queue wait, API time, total time, mode (formatted/raw).
  - every failure path returns the raw text and logs a warning: HTTP error, timeout, empty
    output, unfaithful output, truncated output;
  - after `cooldown_after` consecutive failures formatting is skipped for `cooldown_seconds`
    (raw text is typed immediately); a success resets the counter;
  - `reset()` - clears the context only; queue, separator state and in-flight tasks are untouched;
  - `close()` - cancels the consumer, types the raw text for every pending utterance in order,
    closes the client.

### Wiring

- `load_config`: new optional section `llm` (defaults below).
- `run_streaming(..., llm_cfg)`:
  - the formatter is created once, after the overlay, outside the reconnect loop;
  - `on_final` -> `formatter.submit(raw)`;
  - `on_state_change` to DICTATING -> `formatter.reset()` (a new dictation session starts);
  - `on_text` -> `type_text(text)` in a worker thread, then `overlay.hide()`;
  - after the outer loop: `await formatter.close()`.
- `StreamingSTT(..., defer_overlay_hide=False)`: with the formatter, `on_partial("")` after a
  final is skipped; the overlay keeps the raw text until the formatted text is typed.
- `type_text` no longer strips the text. Bug fix: the leading separator that the state machine
  adds was stripped, so consecutive utterances were glued together.

### Config

```yaml
llm:
  enabled: true
  base_url: https://api.z.ai/api/coding/paas/v4
  model: glm-5.3-flash
  timeout: 8            # seconds per request
  concurrency: 4        # requests in flight
  context_utterances: 2
  # api_key: ...        # optional; default: $ZAI_API_KEY, then opencode auth store (zai-coding-plan)
```

`config.yaml` stays gitignored; the section is documented in `config.yaml.example` and the README.

## Error handling

| Situation | Behaviour |
|---|---|
| HTTP error, 429, network error, timeout | raw text typed, warning logged |
| Empty, unfaithful or truncated output | raw text typed |
| 3 failures in a row | formatting skipped for 60 s, raw typed immediately |
| Connection to STT lost | in-flight results are still typed in order |
| Ctrl+C | pending utterances typed raw immediately, nothing lost |

## Testing

`tests/test_llm_formatter.py` - stdlib `unittest` (`IsolatedAsyncioTestCase`), fake client:

- in-order emission with out-of-order completion;
- timeout, HTTP error, empty, unfaithful and truncated output -> raw;
- separator rule; capitalization after a sentence end; no separator for the first fragment;
- context is passed to the client; `reset()` clears the context and keeps the queue;
- concurrency cap; cooldown after 3 failures and its expiry; `close()` types pending raw.

`tests/test_streaming.py` - `defer_overlay_hide` flag.

Manual acceptance: run against the real API with 20 utterances; at least 90 % formatted, each
typed within ~3 s of finalization, order preserved; a log line per utterance.

## Risks

- The coding-plan endpoint is, by z.ai policy, limited to supported tools; usage in unsupported
  tools may lead to restricted benefits. The endpoint and key are configurable.
- The model may correct misrecognized words; the faithfulness guard allows corrections but rejects
  invented content.
- A stalled request (observed once: 134 s to first token) is cut by the per-request timeout.

## Out of scope

- `wyoming-hid.service` still points to `%h/wyoming-hid/`; not changed.
- Batch formatting of logs, timestamps in the typed text, wake-word detection changes.
