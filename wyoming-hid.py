#!/usr/bin/env python3
"""Stream microphone audio to STT WebSocket server and type recognized text."""

import argparse
import asyncio
import base64
import ctypes
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

os.environ.setdefault("ALSA_PCM_PLUGINS", "")
os.environ.setdefault("JACK_NO_START_SERVER", "1")

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

_devnull = open(os.devnull, "w")
_old_stderr = os.dup(2)
os.dup2(_devnull.fileno(), 2)

import pyaudio

os.dup2(_old_stderr, 2)
os.close(_old_stderr)
_devnull.close()

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
    shutting_down = False

    def _signal_handler(sig, frame):
        nonlocal shutting_down
        shutting_down = True
        log.info("Shutting down...")

    import signal
    signal.signal(signal.SIGINT, _signal_handler)

    while not shutting_down:
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

                _dn = open(os.devnull, "w")
                _oe = os.dup(2)
                os.dup2(_dn.fileno(), 2)
                pa = pyaudio.PyAudio()
                stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=NATIVE_RATE,
                    input=True,
                    frames_per_buffer=NATIVE_CHUNK,
                )
                os.dup2(_oe, 2)
                os.close(_oe)
                _dn.close()
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
                    while not shutting_down:
                        try:
                            chunk = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue
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
            if shutting_down:
                break
            log.warning("Connection error: %s, retrying in %ds", e, backoff)
            if overlay:
                overlay.show(f"⚠ Reconnecting ({backoff}s)...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
        except KeyboardInterrupt:
            shutting_down = True


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
