# Wyoming HID

Voice dictation tool that captures microphone audio, sends it to a Wyoming-compatible STT (Speech-to-Text) WebSocket server, and types the recognized text as keyboard input using `xdotool` or `ydotool`.

Supports wake-word activation and stop-word deactivation for hands-free operation.

## How It Works

1. Captures microphone audio at 44100 Hz, downsamples to 16000 Hz
2. Maintains a rolling audio buffer for overlap-based wake-word detection
3. Sends audio chunks to a WebSocket STT server (Wyoming protocol)
4. Listens for wake phrase to activate dictation mode
5. Types recognized text using `xdotool` (X11) or `ydotool` (Wayland)
6. Deactivates on stop phrase or idle timeout

## Requirements

- Python 3.8+
- PortAudio (`libportaudio2`, `python3-pyaudio`)
- `xdotool` (X11) or `ydotool` (Wayland)
- Python packages: `pyaudio`, `websockets`, `pyyaml`, `audioop`

Install dependencies:

```bash
sudo apt install xdotool libportaudio2 python3-pyaudio
pip3 install websockets pyyaml
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
```

## Usage

### Direct Run

```bash
python3 wyoming-hid.py                          # uses config.yaml
python3 wyoming-hid.py -c /path/to/config.yaml  # custom config
python3 wyoming-hid.py --url ws://host/stt      # override URL
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
| `-v, --verbose` | Enable debug logging |

## License

MIT
