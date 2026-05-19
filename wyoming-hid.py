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
SEND_CHUNK = 2560
BUFFER_CHUNKS = 32    # ~2.56s rolling buffer for wake detection overlap
MIN_CHUNKS = 10       # minimum chunks to send (~0.8s)


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


def normalize(text: str) -> str:
    return text.lower().replace("ё", "е").strip()


async def stt_session(
    ws_url: str,
    auth_header: str,
    typing_method: str,
    wake_phrase: str,
    stop_phrase: str,
    idle_timeout: float,
):
    extra_headers = {"Proxy-Authorization": auth_header} if auth_header else {}
    wake_norm = normalize(wake_phrase)
    stop_norm = normalize(stop_phrase) if stop_phrase else None

    async with websockets.connect(
        ws_url,
        extra_headers=extra_headers,
        ping_interval=None,
        max_size=None,
        compression=None,
    ) as ws:
        log.info("Connected")

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=NATIVE_RATE,
            input=True,
            frames_per_buffer=NATIVE_CHUNK,
        )
        log.info("Mic: %dHz → %dHz", NATIVE_RATE, TARGET_RATE)

        loop = asyncio.get_event_loop()
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=60)
        mic_stop = threading.Event()

        def mic_reader():
            resample_state = None
            try:
                while not mic_stop.is_set():
                    raw = stream.read(NATIVE_CHUNK, exception_on_overflow=False)
                    resampled, resample_state = audioop.ratecv(
                        raw, SAMPLE_WIDTH, CHANNELS,
                        NATIVE_RATE, TARGET_RATE, resample_state,
                    )
                    if not audio_queue.full():
                        loop.call_soon_threadsafe(audio_queue.put_nowait, resampled)
            except Exception as e:
                log.error("mic_reader: %s", e)

        mic_thread = threading.Thread(target=mic_reader, daemon=True)
        mic_thread.start()

        dictation_active = False
        idle_counter = 0
        max_idle = max(1, int(idle_timeout / 2.5))
        has_typed = False

        rolling_buffer = []
        sent_up_to = 0

        log.info("Wake: '%s' | Stop: '%s'", wake_phrase, stop_phrase or "-")

        try:
            while True:
                await asyncio.sleep(1.0)

                while not audio_queue.empty():
                    try:
                        rolling_buffer.append(audio_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                if len(rolling_buffer) > BUFFER_CHUNKS * 2:
                    excess = len(rolling_buffer) - BUFFER_CHUNKS
                    rolling_buffer = rolling_buffer[excess:]
                    sent_up_to = max(0, sent_up_to - excess)

                if not dictation_active:
                    send_chunks = rolling_buffer[-BUFFER_CHUNKS:]
                else:
                    send_chunks = rolling_buffer[sent_up_to:]
                    sent_up_to = len(rolling_buffer)

                if len(send_chunks) < MIN_CHUNKS:
                    if dictation_active:
                        idle_counter += 1
                        if idle_counter >= max_idle:
                            dictation_active = False
                            idle_counter = 0
                            rolling_buffer = []
                            sent_up_to = 0
                            log.info("Deactivated: idle")
                    continue

                await ws.send(json.dumps({"type": "start"}))
                for chunk in send_chunks:
                    await ws.send(chunk)
                await ws.send(json.dumps({"type": "stop"}))

                transcripts = []
                try:
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.5)
                        try:
                            result = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if result.get("type") == "transcript" and result.get("text", "").strip():
                            transcripts.append(result["text"].strip())
                except asyncio.TimeoutError:
                    pass

                if not transcripts:
                    if dictation_active:
                        idle_counter += 1
                        if idle_counter >= max_idle:
                            dictation_active = False
                            idle_counter = 0
                            rolling_buffer = []
                            sent_up_to = 0
                            log.info("Deactivated: idle")
                    continue

                full_text = " ".join(transcripts)
                norm = normalize(full_text)

                if not dictation_active:
                    if wake_norm in norm:
                        dictation_active = True
                        idle_counter = 0
                        has_typed = False
                        sent_up_to = len(rolling_buffer)
                        log.info("Activated: '%s'", full_text)
                    else:
                        log.debug("Ignored: '%s'", full_text)
                else:
                    idle_counter = 0
                    if stop_norm and stop_norm in norm:
                        dictation_active = False
                        idle_counter = 0
                        rolling_buffer = []
                        sent_up_to = 0
                        log.info("Deactivated: '%s'", full_text)
                    else:
                        text = full_text
                        if has_typed:
                            text = " " + text
                        type_text(text, typing_method)
                        has_typed = True
                        sent_up_to = len(rolling_buffer)

        except websockets.ConnectionClosed:
            log.warning("Disconnected")
        except KeyboardInterrupt:
            pass
        finally:
            log.info("Stopping...")
            stream.stop_stream()
            stream.close()
            mic_stop.set()
            pa.terminate()
            log.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Wyoming HID - STT to keyboard")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--url")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--method", choices=["xdotool", "ydotool"])
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

    if user and password:
        auth = f"Basic {base64.b64encode(f'{user}:{password}'.encode()).decode()}"
    else:
        auth = ""

    log.info("URL: %s | Method: %s", ws_url, typing_method)

    try:
        asyncio.run(stt_session(
            ws_url, auth, typing_method,
            wake_phrase=wake_phrase,
            stop_phrase=stop_phrase,
            idle_timeout=idle_timeout,
        ))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
