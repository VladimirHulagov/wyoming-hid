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

    def test_unfaithful_dropped_negation(self):
        self.assertFalse(
            looks_faithful("не забудь отправить письмо", "Забудь отправить письмо.")
        )

    def test_unfaithful_dropped_digits(self):
        self.assertFalse(
            looks_faithful("скажи ему чтобы перезвонил мне на 8 950", "Скажи ему, чтобы перезвонил мне.")
        )

    def test_unfaithful_one_word_preamble(self):
        self.assertFalse(looks_faithful("привет как дела", "Отлично, привет, как дела?"))

    def test_faithful_digits_kept(self):
        self.assertTrue(
            looks_faithful(
                "скажи ему чтобы перезвонил мне на 8 950",
                "Скажи ему, чтобы перезвонил мне на 8 950.",
            )
        )

    def test_unfaithful_repeated_word(self):
        self.assertFalse(looks_faithful("про про про про про", "Проверка."))

    def test_faithful_glued_words_with_short_pieces(self):
        self.assertTrue(looks_faithful("даянеснял или не снял", "Да, я не снял или не снял."))

    def test_faithful_only_fillers(self):
        self.assertTrue(looks_faithful("ну вот а да", ""))
        self.assertFalse(looks_faithful("угу", "Привет"))

    def test_ends_sentence(self):
        self.assertTrue(ends_sentence("Привет."))
        self.assertTrue(ends_sentence("Как дела?"))
        self.assertTrue(ends_sentence("Так..."))
        self.assertTrue(ends_sentence("Он сказал: «Привет»."))
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
