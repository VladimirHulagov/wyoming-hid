# Streaming STT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use the specified execution mode skill (from brainstorming). Steps use checkbox (`- [ ]`) syntax for tracking.

**Execution Mode:** subagent-driven

**Goal:** Implement real-time streaming speech recognition with GTK3 overlay for partial results, replacing the current batch polling approach.

**Architecture:** Single persistent WebSocket connection continuously streams audio to the STT server. A coroutine reads partial/final results in parallel. A state machine (IDLE → LISTENING → DICTATING) manages wake-word detection and typing. GTK3 overlay shows partial text in a floating window. Final text is typed via xdotool/ydotool.

**Tech Stack:** Python 3.8+, PyGObject (GTK3), websockets, pyaudio, audioop

---

### Task 1: Create `overlay.py` — GTK3 Overlay Module

**Files:**
- Create: `overlay.py`

- [ ] **Step 1: Create `overlay.py` with GTKOverlay class**

```python
#!/usr/bin/env python3
"""GTK3 overlay window for streaming STT partial results."""

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk, Gdk, Pango

import threading


class GTKOverlay:
    def __init__(self, opacity: float = 0.85, font_size: int = 18):
        self._opacity = opacity
        self._font_size = font_size
        self._window = None
        self._label = None
        self._thread = None
        self._visible = False

    def start(self):
        self._thread = threading.Thread(target=self._run_gtk, daemon=True)
        self._thread.start()

    def _run_gtk(self):
        self._window = Gtk.Window(type=Gtk.WindowType.POPUP)
        self._window.set_keep_above(True)
        self._window.set_decorated(False)
        self._window.set_skip_taskbar_hint(True)
        self._window.set_skip_pager_hint(True)

        screen = self._window.get_screen()
        visual = screen.get_rgba_visual()
        if visual and self._window.is_composited():
            self._window.set_visual(visual)

        self._window.set_app_paintable(True)
        self._window.connect("draw", self._on_draw)
        self._window.connect("screen-changed", self._on_screen_changed)

        self._label = Gtk.Label()
        self._label.set_text("")
        self._label.set_ellipsize(Pango.EllipsizeMode.END)
        self._label.set_max_width_chars(80)
        self._label.set_margin_top(12)
        self._label.set_margin_bottom(12)
        self._label.set_margin_start(20)
        self._label.set_margin_end(20)

        css_provider = Gtk.CssProvider()
        css = f"""
        label {{
            color: white;
            font-size: {self._font_size}px;
            font-weight: bold;
        }}
        window {{
            background-color: rgba(30, 30, 30, {self._opacity});
            border-radius: 12px;
        }}
        """
        css_provider.load_from_data(css.encode())
        self._label.get_style_context().add_provider(
            css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        self._window.get_style_context().add_provider(
            css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        self._window.add(self._label)
        self._window.set_default_size(500, 50)
        self._window.set_size_request(300, 50)
        self._window.show_all()
        self._window.hide()

        display = Gdk.Display.get_default()
        if display:
            monitor = display.get_primary_monitor()
            if monitor is None:
                monitor = display.get_monitor(0)
            if monitor:
                geom = monitor.get_geometry()
                win_w, win_h = self._window.get_size()
                x = geom.x + (geom.width - win_w) // 2
                y = geom.y + geom.height - win_h - 80
                self._window.move(x, y)

        Gtk.main()

    def _on_draw(self, widget, cr):
        cr.set_source_rgba(0, 0, 0, 0)
        cr.set_operator(getattr(cr, "OPERATOR_SOURCE", 0))
        cr.paint()
        cr.set_operator(getattr(cr, "OPERATOR_OVER", 0))
        return False

    def _on_screen_changed(self, widget, old_screen):
        screen = widget.get_screen()
        visual = screen.get_rgba_visual()
        if visual and widget.is_composited():
            widget.set_visual(visual)
        else:
            widget.set_visual(screen.get_system_visual())

    def show(self, text: str = "🎤 ..."):
        self._visible = True
        if self._window:
            GLib.idle_add(self._do_show, text)

    def hide(self):
        self._visible = False
        if self._window:
            GLib.idle_add(self._do_hide)

    def update(self, text: str):
        if self._visible and self._window:
            GLib.idle_add(self._do_update, text)

    def _do_show(self, text):
        if self._label:
            self._label.set_text(text)
        if self._window:
            self._window.show_all()
        return False

    def _do_hide(self):
        if self._window:
            self._window.hide()
        return False

    def _do_update(self, text):
        if self._label:
            self._label.set_text(text)
        display = Gdk.Display.get_default()
        if display and self._window:
            monitor = display.get_primary_monitor()
            if monitor is None:
                monitor = display.get_monitor(0)
            if monitor:
                geom = monitor.get_geometry()
                win_w, win_h = self._window.get_size()
                x = geom.x + (geom.width - win_w) // 2
                y = geom.y + geom.height - win_h - 80
                self._window.move(x, y)
                self._window.resize(win_w, win_h)
        return False
```

- [ ] **Step 2: Verify overlay.py syntax**

Run: `python3 -c "import ast; ast.parse(open('overlay.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add overlay.py
git commit -m "feat: add GTK3 overlay module for partial results"
```

---

### Task 2: Create `streaming.py` — Wyoming Streaming Protocol + State Machine

**Files:**
- Create: `streaming.py`

This module handles the streaming state machine, audio chunk sending, and result reading. It does NOT handle microphone or typing — those are injected via callbacks.

- [ ] **Step 1: Create `streaming.py`**

```python
#!/usr/bin/env python3
"""Wyoming streaming STT protocol handler with state machine."""

import asyncio
import json
import logging
from enum import Enum, auto

log = logging.getLogger("wyoming-hid")


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    DICTATING = auto()


class StreamingSTT:
    def __init__(
        self,
        ws,
        wake_phrase: str,
        stop_phrase: str,
        on_partial=None,
        on_final=None,
        on_state_change=None,
    ):
        self.ws = ws
        self.wake_norm = self._normalize(wake_phrase)
        self.stop_norm = self._normalize(stop_phrase) if stop_phrase else None
        self.state = State.IDLE
        self._on_partial = on_partial
        self._on_final = on_final
        self._on_state_change = on_state_change
        self._idle_counter = 0
        self._has_typed = False
        self._running = True

    @staticmethod
    def _normalize(text: str) -> str:
        return text.lower().replace("ё", "е").strip()

    def _set_state(self, new_state: State):
        if self.state != new_state:
            old = self.state
            self.state = new_state
            log.info("State: %s -> %s", old.name, new_state.name)
            if self._on_state_change:
                self._on_state_change(old, new_state)

    async def send_audio(self, chunk: bytes):
        await self.ws.send(chunk)

    async def start_session(self):
        await self.ws.send(json.dumps({"type": "start"}))

    async def result_reader(self, idle_timeout_chunks: int = 6):
        max_idle = max(1, idle_timeout_chunks)
        try:
            while self._running:
                try:
                    raw = await asyncio.wait_for(self.ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    if self.state == State.DICTATING:
                        self._idle_counter += 1
                        if self._idle_counter >= max_idle:
                            self._set_state(State.IDLE)
                            self._idle_counter = 0
                    continue

                try:
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if result.get("type") != "transcript":
                    continue

                text = result.get("text", "").strip()
                if not text:
                    continue

                is_final = result.get("is_final", False)

                if not is_final:
                    await self._handle_partial(text)
                else:
                    await self._handle_final(text)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error("result_reader error: %s", e)

    async def _handle_partial(self, text: str):
        norm = self._normalize(text)

        if self.state == State.IDLE:
            if self.wake_norm in norm:
                self._set_state(State.LISTENING)
                self._idle_counter = 0
                self._has_typed = False
                log.info("Wake detected: '%s'", text)
                if self._on_partial:
                    self._on_partial("🎤 " + text)
            else:
                log.debug("Ignored partial: '%s'", text)

        elif self.state == State.LISTENING:
            if self.stop_norm and self.stop_norm in norm:
                self._set_state(State.IDLE)
                self._idle_counter = 0
                return
            self._set_state(State.DICTATING)
            self._idle_counter = 0
            if self._on_partial:
                self._on_partial(text)

        elif self.state == State.DICTATING:
            self._idle_counter = 0
            if self.stop_norm and self.stop_norm in norm:
                self._set_state(State.IDLE)
                self._idle_counter = 0
                if self._on_partial:
                    self._on_partial("")
                return
            if self._on_partial:
                self._on_partial(text)

    async def _handle_final(self, text: str):
        norm = self._normalize(text)

        if self.state == State.IDLE:
            if self.wake_norm in norm:
                self._set_state(State.LISTENING)
                self._idle_counter = 0
                self._has_typed = False
            return

        if self.state == State.LISTENING:
            if self.stop_norm and self.stop_norm in norm:
                self._set_state(State.IDLE)
                self._idle_counter = 0
                return
            self._set_state(State.DICTATING)

        if self.state == State.DICTATING:
            self._idle_counter = 0
            if self.stop_norm and self.stop_norm in norm:
                self._set_state(State.IDLE)
                self._idle_counter = 0
                if self._on_partial:
                    self._on_partial("")
                return

            typed_text = text
            if self._has_typed:
                typed_text = " " + typed_text
            if self._on_final:
                self._on_final(typed_text)
            self._has_typed = True

            if self._on_partial:
                self._on_partial("")

            await self.ws.send(json.dumps({"type": "start"}))

    def stop(self):
        self._running = False
```

- [ ] **Step 2: Verify streaming.py syntax**

Run: `python3 -c "import ast; ast.parse(open('streaming.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add streaming.py
git commit -m "feat: add streaming STT protocol handler with state machine"
```

---

### Task 3: Refactor `wyoming-hid.py` — Main Entry Point

**Files:**
- Modify: `wyoming-hid.py` (full rewrite)

Replace the entire file with a new version that uses `streaming.py` and `overlay.py`.

- [ ] **Step 1: Rewrite `wyoming-hid.py`**

```python
#!/usr/bin/env python3
"""Stream microphone audio to STT WebSocket server and type recognized text."""

import argparse
import asyncio
import base64
import ctypes
import json
import logging
import subprocess
import sys
import threading
from pathlib import Path

import audioop
import pyaudio
import websockets
import yaml

from overlay import GTKOverlay
from streaming import State, StreamingSTT

try:
    ctypes.cdll.LoadLibrary("libasound.so.2").snd_lib_error_set_handler(None)
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("wyoming-hid")

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"

NATIVE_RATE = 44100
TARGET_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
NATIVE_CHUNK = 4410
SEND_CHUNK_MS = 100
SEND_SAMPLES = TARGET_RATE * SEND_CHUNK_MS // 1000
SEND_BYTES = SEND_SAMPLES * SAMPLE_WIDTH


def load_config(config_path: Path) -> dict:
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def type_text(text: str, method: str = "xdotool"):
    if not text.strip():
        return
    text = text.strip()
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

    def on_partial(text: str):
        if overlay:
            if text:
                overlay.update(text)
                overlay.show(text)
            else:
                overlay.hide()

    def on_final(text: str):
        type_text(text, typing_method)

    def on_state_change(old: State, new: State):
        if overlay:
            if new == State.IDLE:
                overlay.hide()
            elif new == State.LISTENING:
                overlay.show("🎤 ...")

    extra_headers = {"Proxy-Authorization": auth_header} if auth_header else {}

    backoff = 1
    max_backoff = 30

    while True:
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

                pa = pyaudio.PyAudio()
                stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=NATIVE_RATE,
                    input=True,
                    frames_per_buffer=NATIVE_CHUNK,
                )
                log.info("Mic: %dHz -> %dHz", NATIVE_RATE, TARGET_RATE)

                stt = StreamingSTT(
                    ws,
                    wake_phrase=wake_phrase,
                    stop_phrase=stop_phrase,
                    on_partial=on_partial,
                    on_final=on_final,
                    on_state_change=on_state_change,
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
                    while True:
                        chunk = await audio_queue.get()
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
            log.warning("Connection error: %s, retrying in %ds", e, backoff)
            if overlay:
                overlay.show(f"⚠ Reconnecting ({backoff}s)...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
        except KeyboardInterrupt:
            if overlay:
                overlay.hide()
            break


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

    log.info("URL: %s | Method: %s | Overlay: %s", ws_url, typing_method, use_overlay)

    try:
        asyncio.run(run_streaming(
            ws_url, auth, typing_method,
            wake_phrase=wake_phrase,
            stop_phrase=stop_phrase,
            idle_timeout=idle_timeout,
            use_overlay=use_overlay,
            overlay_opacity=overlay_opacity,
            overlay_font_size=overlay_font_size,
        ))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify wyoming-hid.py syntax**

Run: `python3 -c "import ast; ast.parse(open('wyoming-hid.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add wyoming-hid.py
git commit -m "feat: rewrite main for streaming mode with overlay support"
```

---

### Task 4: Update `config.yaml.example`

**Files:**
- Modify: `config.yaml.example`

- [ ] **Step 1: Update config with new options**

```yaml
# Wyoming HID configuration
url: wss://your-stt-server.example.com/stt
user: your_username
password: your_password
method: xdotool    # xdotool | ydotool

# Voice activation
wake_phrase: "голосовой ввод"   # say this to start dictation
stop_phrase: "стоп ввод"        # say this to stop dictation (or set "" to disable)
idle_timeout: 15                 # auto-deactivate after N seconds of silence

# Overlay settings
overlay: true                    # show GTK3 overlay for partial results
overlay_opacity: 0.85            # background opacity (0.0 - 1.0)
overlay_font_size: 18            # text size in pixels
```

- [ ] **Step 2: Commit**

```bash
git add config.yaml.example
git commit -m "docs: update config example with overlay settings"
```

---

### Task 5: Update `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README to reflect streaming mode**

Update the Requirements section to add PyGObject, update How It Works to describe streaming, and add overlay configuration docs. Key changes:

1. Requirements: add `gir1.2-gtk-3.0`, `python3-gi`
2. How It Works: describe continuous streaming instead of batch polling
3. Configuration: document new overlay options
4. Command Line Options: add `--no-overlay`

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for streaming mode"
```

---

### Task 6: Integration Test

**Files:** None (manual testing)

- [ ] **Step 1: Verify syntax of all Python files**

Run: `python3 -c "import ast; [ast.parse(open(f).read()) for f in ['overlay.py', 'streaming.py', 'wyoming-hid.py']]; print('All OK')"`
Expected: `All OK`

- [ ] **Step 2: Test with `--verbose --no-overlay` against STT server**

Run: `python3 wyoming-hid.py --verbose --no-overlay`
Expected: Connects to server, streams audio, shows partial/final results in console.

- [ ] **Step 3: Test with overlay**

Run: `python3 wyoming-hid.py --verbose`
Expected: GTK overlay appears at bottom center when dictation is active.

- [ ] **Step 4: Test wake-word and stop-word**

Say wake phrase -> overlay should appear with "🎤 ...". Dictate text -> partial shows in overlay, final typed. Say stop phrase -> overlay hides.

- [ ] **Step 5: Test reconnection**

Restart STT server while client is running -> should reconnect with backoff, overlay shows "⚠ Reconnecting...".

- [ ] **Step 6: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: integration test fixes"
```
