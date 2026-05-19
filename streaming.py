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

                msg_type = result.get("type", "")
                text = result.get("text", "").strip()
                if not text:
                    continue

                if msg_type == "partial":
                    await self._handle_partial(text)
                elif msg_type == "transcript":
                    is_final = result.get("is_final", True)
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
