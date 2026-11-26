# Wyoming HID

Voice dictation tool that captures microphone audio, streams it in real-time to a Wyoming-compatible STT server via WebSocket, and types the recognized text using `xdotool` or `ydotool`. Partial recognition results are shown in a floating GTK3 overlay during dictation.

Supports wake-word activation and stop-word deactivation for hands-free operation.

## How It Works

1. Captures microphone audio at 44100 Hz, downsamples to 16000 Hz
2. Continuously streams audio to a Wyoming STT server via WebSocket
3. State machine: **IDLE** (listening for wake-word) → **LISTENING** → **DICTATING**
4. Partial recognition results displayed in a GTK3 overlay in real-time
5. Final text typed via `xdotool` (X11) or `ydotool` (Wayland)
6. Deactivates on stop phrase or idle timeout
7. Auto-reconnects to the server with exponential backoff

## Requirements

- Python 3.8+
- PortAudio (`libportaudio2`, `python3-pyaudio`)
- GTK3 (`gir1.2-gtk-3.0`, `python3-gi`)
- `xdotool` (X11) or `ydotool` (Wayland)
- Python packages: `pyaudio`, `websockets`, `pyyaml`, `aiohttp`

Install dependencies:

```bash
sudo apt install xdotool libportaudio2 python3-pyaudio gir1.2-gtk-3.0 python3-gi
pip3 install websockets pyyaml aiohttp
```

## Configuration

Copy the example config and edit with your STT server credentials:

```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml`:

```yaml
url: wss://your-stt-server.example.com/stt
user: your_username
password: your_password
method: xdotool    # xdotool | ydotool

# Voice activation
wake_phrase: "голосовой ввод"   # say this to start dictation
stop_phrase: "стоп ввод"        # say this to stop dictation ("" to disable)
idle_timeout: 15                 # auto-deactivate after N seconds of silence

# Overlay
overlay: true          # show floating overlay with partial results
overlay_opacity: 0.8   # overlay transparency (0.0–1.0)
overlay_font_size: 24  # font size for overlay text
```

## LLM Formatting

Formatting is opt-in: without an `llm` section in `config.yaml` the recognized text is typed as is. With the section, every finalized utterance is sent to GLM-5.3-Flash on z.ai before typing: punctuation, capitalization, glued words split and fillers removed. Requests run in parallel (up to `concurrency` in flight) and results are typed in the order the utterances were finalized. The overlay keeps the raw text until the formatted text is typed. On any failure (timeout, HTTP error, output that does not match the utterance) the raw text is typed instead. The startup log line shows `LLM: on` or `LLM: off`.

```yaml
llm:
  enabled: true
  base_url: https://api.z.ai/api/coding/paas/v4
  model: glm-5.3-flash
  timeout: 8            # seconds per request
  concurrency: 4        # requests in flight
  context_utterances: 2 # formatted utterances passed as context
  # api_key: ...        # default: $ZAI_API_KEY, then the opencode auth store (zai-coding-plan)
```

The API key is read from `llm.api_key`, then from `ZAI_API_KEY`, then from the opencode auth store (the last one only for the default `base_url`, so that the stored key is never sent to another host). The default `base_url` is the z.ai coding-plan endpoint, which z.ai limits to supported tools; for a regular z.ai API key use `https://api.z.ai/api/paas/v4`.

### Limitations

- A request takes about 2 s. Results are typed strictly in order, so one slow request (up to the timeout) delays the utterances queued behind it.
- The model sees the previous one or two formatted utterances as context, as of the start of each request.
- The guard that protects against invented text is strict but not complete: a stutter may be merged, one unexplained word is tolerated in longer outputs, and up to 10 % of the content words may be dropped by the model. Unintelligible speech may be typed with punctuation instead of raw.

## Usage

### Direct Run

```bash
python3 wyoming-hid.py                          # uses config.yaml
python3 wyoming-hid.py -c /path/to/config.yaml  # custom config
python3 wyoming-hid.py --url ws://host/stt      # override URL
python3 wyoming-hid.py --no-overlay             # disable overlay
python3 wyoming-hid.py --verbose                # debug output
```

### As systemd User Service

```bash
# Install, enable and start
./install.sh install

# Other commands
./install.sh start
./install.sh stop
./install.sh restart
./install.sh status
./install.sh uninstall
./install.sh check    # check dependencies only
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `-c, --config` | Path to config file (default: `config.yaml` in script directory) |
| `--url` | WebSocket STT server URL (overrides config) |
| `--user` | Username for STT server (overrides config) |
| `--password` | Password for STT server (overrides config) |
| `--method` | Typing method: `xdotool` or `ydotool` (overrides config) |
| `--no-overlay` | Disable the GTK3 overlay window |
| `-v, --verbose` | Enable debug logging |

## License

MIT
