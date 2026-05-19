# Streaming STT Design

Real-time streaming speech recognition with GTK3 overlay for partial results.

## Architecture

```
Microphone -> [mic_reader thread] -> audio_queue
                                        |
                            [async main loop]
                                   |
                         WebSocket (persistent connection)
                          | continuous stream
                         STT Server (Wyoming)
                          |
                    [result_reader coroutine]
                          |
              state machine (idle -> listening -> dictating)
                          |
                    +------+------+
                    |             |
              GTK Overlay    xdotool/ydotool
              (partials)     (final text)
```

## State Machine

Three states:

- **IDLE**: Audio streams to server, partial results analyzed for wake-word. Overlay hidden.
- **LISTENING**: Wake-word detected, overlay shows microphone icon with "Listening...". Waiting for first phrase.
- **DICTATING**: Partial text shown in overlay. Final text typed to active field. Stop-word or idle timeout returns to IDLE.

Transitions:

- IDLE -> LISTENING: wake-word found in partial transcript
- LISTENING -> DICTATING: first partial with actual speech content
- DICTATING -> IDLE: stop-word detected or idle timeout

## Streaming Protocol

Current flow (batch): `start -> chunks -> stop -> wait for results -> repeat`

New flow (streaming):

1. Send `{"type": "start"}` once at connection
2. Audio chunks sent continuously via `ws.send()` as they arrive from mic_reader
3. Results read in parallel via separate coroutine (`result_reader`)
4. Partial: `{"type": "transcript", "is_final": false}` -> show in overlay
5. Final: `{"type": "transcript", "is_final": true}` -> type via xdotool/ydotool
6. After final, send new `{"type": "start"}` to begin next utterance

## GTK3 Overlay

- Window type: `GTK_WINDOW_POPUP`, always on top (`keep_above`)
- Position: bottom center of screen
- Content: current partial text, updated in real-time
- Style: semi-transparent dark background, white text, rounded corners
- Visibility: shown only in LISTENING/DICTATING states
- Updates via `GLib.idle_add()` from async loop (thread-safe)
- GTK runs in its own thread with `Gtk.main()`

## Reconnection

On WebSocket disconnect: exponential backoff (1s, 2s, 4s, max 30s). Overlay shows "Reconnecting...". Microphone continues recording to buffer.

## File Structure

```
wyoming-hid.py          -> main entry point (refactored)
overlay.py              -> GTK3 overlay class
streaming.py            -> WS streaming + state machine
config.yaml.example     -> updated config with new options
```

## Configuration Additions

```yaml
# Streaming settings
streaming: true              # enable streaming mode (default: true)
overlay: true                # show GTK overlay (default: true)
overlay_opacity: 0.85        # overlay background opacity
overlay_font_size: 18        # text font size in pixels
```

## Dependencies

New: `PyGObject` (GTK3 Python bindings) - `python3-gi`, `gir1.2-gtk-3.0`

Existing: `pyaudio`, `websockets`, `pyyaml`, `audioop`

## Error Handling

- WebSocket disconnect: auto-reconnect with backoff, overlay notification
- GTK initialization failure: fall back to console-only mode
- STT server returns errors: log and reconnect
- Audio device issues: log error, exit gracefully

## Testing

- Manual testing: run with `--verbose`, check overlay updates and typing
- Test wake-word detection from streaming partials
- Test stop-word and idle timeout
- Test reconnection after server restart
- Test on both X11 and Wayland
