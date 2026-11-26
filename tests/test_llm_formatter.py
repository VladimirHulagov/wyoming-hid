import asyncio
import json
import os
import socket
import sys
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
        for raw, out in (
            ("привет мир", "Привет\u2028мир."),
            ("привет мир", "Привет\u200bмир."),
            ("привет мир", "Привет\u202eмир."),
        ):
            with self.subTest(out=out):
                self.assertFalse(looks_faithful(raw, out))

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

    async def test_finish_reason_must_be_stop(self):
        for finish_reason in (None, "length", "sensitive"):
            with self.subTest(finish_reason=finish_reason):
                self.state["body"] = ok_body("Обрез", finish_reason=finish_reason)
                with self.assertRaises(LLMError):
                    await self.client.format("привет", [])

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

    async def test_reset_with_pending_utterance(self):
        gate = asyncio.Event()
        client = FakeClient(["Первый.", "Второй."], gates=[gate, None])
        formatter = OrderedFormatter(client, self.on_text)
        formatter.submit("первый")
        await asyncio.sleep(0.01)
        formatter.reset()
        formatter.submit("второй")
        gate.set()
        await self.wait_typed(2)
        self.assertEqual(self.typed, ["Первый.", "Второй."])
        self.assertEqual(client.calls[1], ("второй", []))

    async def test_separator_kept_for_pending_utterance_of_old_session(self):
        gate = asyncio.Event()
        client = FakeClient(["Первый.", "Второй."], gates=[None, gate])
        formatter = OrderedFormatter(client, self.on_text)
        formatter.submit("первый")
        await self.wait_typed(1)
        formatter.submit(" второй")
        await asyncio.sleep(0.01)
        formatter.reset()
        gate.set()
        await self.wait_typed(2)
        self.assertEqual(self.typed, ["Первый.", " Второй."])


class TestBuildFormatter(unittest.TestCase):
    def setUp(self):
        self.typed = []

    def on_text(self, text):
        self.typed.append(text)

    def test_disabled_returns_none(self):
        self.assertIsNone(build_formatter(None, self.on_text))
        self.assertIsNone(build_formatter({"enabled": False}, self.on_text))
        self.assertIsNone(build_formatter({"enabled": "false"}, self.on_text))
        self.assertIsNone(build_formatter({"enabled": 0}, self.on_text))

    def test_non_dict_section_returns_none(self):
        with patch("llm_formatter.resolve_api_key", return_value="k"):
            self.assertIsNone(build_formatter("yes", self.on_text))
            self.assertIsNone(build_formatter(False, self.on_text))
            self.assertIsNone(build_formatter([], self.on_text))

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

    def test_malformed_values_return_none(self):
        for cfg in ({"timeout": "abc"}, {"timeout": None}, {"concurrency": "two"}, {"context_utterances": "many"}, {"timeout": 0}):
            with self.subTest(cfg=cfg), patch("llm_formatter.resolve_api_key", return_value="k"), self.assertLogs("wyoming-hid", level="ERROR"):
                self.assertIsNone(build_formatter(cfg, self.on_text))

    def test_aiohttp_missing_disables_formatting(self):
        with patch.dict(sys.modules, {"aiohttp": None}), self.assertLogs("wyoming-hid", level="ERROR"):
            self.assertIsNone(build_formatter({}, self.on_text))
