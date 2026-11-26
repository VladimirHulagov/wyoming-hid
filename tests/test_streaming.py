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
