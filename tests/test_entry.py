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
