import asyncio
import importlib.util
import signal
import sys
import time
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


class FakeFormatter:
    def __init__(self, events):
        self.events = events
        self.pending = []

    @property
    def pending_count(self):
        return len(self.pending)

    def submit(self, text):
        self.pending.append(text)
        self.events.append(("submit", text))

    def reset(self):
        self.events.append(("reset",))

    async def close(self):
        self.events.append(("close",))


class FakeOverlay:
    def __init__(self):
        self.calls = []

    def update(self, text):
        self.calls.append(("update", text))

    def show(self, text):
        self.calls.append(("show", text))

    def hide(self):
        self.calls.append(("hide",))


class TestTypist(unittest.IsolatedAsyncioTestCase):
    async def test_overlay_kept_until_formatted_text_is_typed(self):
        module = load_entry_module()
        events = []
        overlay = FakeOverlay()
        typing = module.Typist("xdotool", overlay)
        typing.formatter = FakeFormatter(events)
        with patch.object(module.subprocess, "run") as run:
            typing.on_partial("привет")
            typing.on_final("привет")
            self.assertEqual(overlay.calls, [("update", "привет"), ("show", "привет")])
            self.assertEqual(events, [("submit", "привет")])
            typing.formatter.pending.clear()
            typing("Привет.")
        run.assert_called_once_with(
            ["xdotool", "type", "--clearmodifiers", "--delay", "0", "Привет."],
            check=True,
            timeout=10,
        )
        self.assertEqual(overlay.calls, [("update", "привет"), ("show", "привет"), ("hide",)])

    async def test_overlay_hidden_on_idle_when_nothing_pending(self):
        module = load_entry_module()
        overlay = FakeOverlay()
        typing = module.Typist("xdotool", overlay)
        typing.formatter = FakeFormatter([])
        typing.on_state_change(module.State.DICTATING, module.State.IDLE)
        self.assertEqual(overlay.calls, [("hide",)])

    async def test_overlay_kept_on_idle_while_pending(self):
        module = load_entry_module()
        overlay = FakeOverlay()
        typing = module.Typist("xdotool", overlay)
        typing.formatter = FakeFormatter([])
        typing.formatter.pending.append("привет")
        typing.on_state_change(module.State.DICTATING, module.State.IDLE)
        self.assertEqual(overlay.calls, [])
        typing.formatter.pending.clear()
        with patch.object(module.subprocess, "run") as run:
            typing("Привет.")
        run.assert_called_once_with(
            ["xdotool", "type", "--clearmodifiers", "--delay", "0", "Привет."],
            check=True,
            timeout=10,
        )
        self.assertEqual(overlay.calls, [("hide",)])

    async def test_raw_path_types_and_hides(self):
        module = load_entry_module()
        overlay = FakeOverlay()
        typing = module.Typist("xdotool", overlay)
        with patch.object(module.subprocess, "run") as run:
            typing.on_final("привет")
        run.assert_called_once_with(
            ["xdotool", "type", "--clearmodifiers", "--delay", "0", "привет"],
            check=True,
            timeout=10,
        )
        self.assertEqual(overlay.calls, [("hide",)])

    async def test_live_partial_keeps_overlay(self):
        module = load_entry_module()
        overlay = FakeOverlay()
        typing = module.Typist("xdotool", overlay)
        typing.formatter = FakeFormatter([])
        typing.on_partial("привет")
        with patch.object(module.subprocess, "run"):
            typing("Привет.")
        self.assertEqual(overlay.calls, [("update", "привет"), ("show", "привет")])

    async def test_shutdown_types_pending_and_closes_formatter(self):
        module = load_entry_module()
        original = signal.getsignal(signal.SIGINT)
        events = []

        class FakeStreamingSTT:
            started = False
            defer = None

            def __init__(self, ws, wake_phrase, stop_phrase, on_partial, on_final, on_state_change, defer_overlay_hide=False):
                FakeStreamingSTT.started = True
                FakeStreamingSTT.defer = defer_overlay_hide

            async def start_session(self):
                pass

            async def result_reader(self, idle_timeout_chunks=6):
                await asyncio.sleep(3600)

            async def send_audio(self, data):
                pass

        class FakeStream:
            def read(self, n, exception_on_overflow=False):
                time.sleep(0.005)
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
            async def __aenter__(self):
                return object()

            async def __aexit__(self, *exc):
                return False

        class FakeConnect:
            def __call__(self, url, **kwargs):
                return FakeWSContext()

        fake_pyaudio = SimpleNamespace(PyAudio=FakePyAudio, paInt16=8)
        typing = module.Typist("xdotool", None)
        typing.formatter = FakeFormatter(events)
        try:
            with patch.object(module, "StreamingSTT", FakeStreamingSTT), patch.object(module, "pyaudio", fake_pyaudio), patch.object(module.websockets, "connect", FakeConnect()):
                task = asyncio.ensure_future(module.run_streaming(
                    "ws://x/stt", "", typing, wake_phrase="w", stop_phrase="s",
                    idle_timeout=15,
                ))
                for _ in range(300):
                    if FakeStreamingSTT.started:
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(FakeStreamingSTT.started)
                self.assertTrue(FakeStreamingSTT.defer)
                signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
                await asyncio.wait_for(task, timeout=10)
        finally:
            signal.signal(signal.SIGINT, original)
        self.assertEqual(events, [("close",)])
