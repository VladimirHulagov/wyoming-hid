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
