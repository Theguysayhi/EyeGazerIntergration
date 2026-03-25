"""
core/buttplug/script_player.py
Funscript playback engine for Buttplug devices.

A .funscript file is a JSON file with the schema:
    {
        "version": "1.0",
        "actions": [
            {"at": <milliseconds>, "pos": <0-100>},
            ...
        ]
    }

For vibrators  – "pos" maps directly to vibration intensity (0 = off, 100 = full).
For strokers   – "pos" maps to linear position (0 = bottom, 100 = top).
                 The move duration sent to the device is the gap (ms) to the next
                 action, which gives the smoothest stroker motion.

buttplug-py 0.2.0 API used here
---------------------------------
Vibrator  : await actuator.command(speed: float)          0.0 – 1.0
Stroker   : await actuator.command(duration: int,         milliseconds
                                   position: float)       0.0 – 1.0
Stop      : await device.stop()

Usage
-----
    player = FunscriptPlayer(device, DeviceType.VIBRATOR)
    await player.play("/path/to/script.funscript")   # runs until end or cancelled
    player.cancel()                                  # stop mid-play
"""

from __future__ import annotations

import asyncio
import json
import os
from enum import Enum, auto
from typing import Any

_DEFAULT_DURATION_MS = 400   # fallback move-duration for last stroker action


class DeviceType(Enum):
    VIBRATOR = auto()
    STROKER  = auto()


class FunscriptPlayer:
    """
    Plays a single .funscript file on a Buttplug device.

    Parameters
    ----------
    device      : buttplug Device object
    device_type : DeviceType – determines which actuator type to drive
    """

    def __init__(self, device: Any, device_type: DeviceType) -> None:
        self._device      = device
        self._device_type = device_type
        self._cancel_evt  = asyncio.Event()

    # ------------------------------------------------------------------
    def cancel(self) -> None:
        """Signal the in-progress play() to stop after the current action."""
        self._cancel_evt.set()

    # ------------------------------------------------------------------
    @staticmethod
    def load(path: str) -> list[dict]:
        """
        Load a .funscript file and return the sorted action list.

        Each action is ``{"at": int, "pos": int}``.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        actions = data.get("actions", [])
        # Ensure sorted by timestamp
        return sorted(actions, key=lambda a: a["at"])

    # ------------------------------------------------------------------
    async def play(self, path: str, loop: bool = False) -> None:
        """
        Play a funscript file, optionally looping until cancelled.

        Parameters
        ----------
        path : str   – path to the .funscript file
        loop : bool  – if True, restart from the beginning after the last action
        """
        self._cancel_evt.clear()

        actions = self.load(path)
        if not actions:
            print(f"[player] {os.path.basename(path)}: no actions found, skipping")
            return

        print(f"[player] Playing {os.path.basename(path)} "
              f"({'loop' if loop else 'once'})  device={self._device.name}")

        while True:
            start_time = asyncio.get_event_loop().time()

            for i, action in enumerate(actions):
                if self._cancel_evt.is_set():
                    await self._send_stop()
                    return

                target_t = start_time + action["at"] / 1000.0
                now      = asyncio.get_event_loop().time()
                wait     = target_t - now
                if wait > 0:
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(asyncio.ensure_future(
                                self._cancel_evt.wait()
                            )),
                            timeout=wait,
                        )
                        # Cancel event fired during sleep
                        await self._send_stop()
                        return
                    except asyncio.TimeoutError:
                        pass  # normal – timeout means the sleep completed

                if self._cancel_evt.is_set():
                    await self._send_stop()
                    return

                intensity = action["pos"] / 100.0

                # For linear (stroker) devices the API needs a duration (ms)
                # telling the device how long to take to reach the position.
                # Use the gap to the next action; fall back to a default for
                # the final action (or when looping back to the start).
                if self._device_type == DeviceType.STROKER:
                    if i + 1 < len(actions):
                        duration_ms = max(1, actions[i + 1]["at"] - action["at"])
                    elif loop:
                        # Looping: gap from last action to start of next loop.
                        # Use the gap between the last two actions as an estimate.
                        if len(actions) >= 2:
                            duration_ms = max(1, actions[-1]["at"] - actions[-2]["at"])
                        else:
                            duration_ms = _DEFAULT_DURATION_MS
                    else:
                        duration_ms = _DEFAULT_DURATION_MS
                else:
                    duration_ms = _DEFAULT_DURATION_MS  # unused for vibrators

                await self._send(intensity, duration_ms)

            if not loop:
                break

        await self._send_stop()

    # ------------------------------------------------------------------
    async def _send(self, intensity: float, duration_ms: int = _DEFAULT_DURATION_MS) -> None:
        """
        Send a single intensity/position command to the device.

        Parameters
        ----------
        intensity   : 0.0 – 1.0
        duration_ms : movement duration for linear actuators (ignored for vibrators)
        """
        try:
            if self._device_type == DeviceType.VIBRATOR:
                for actuator in self._device.actuators:
                    await actuator.command(intensity)

            elif self._device_type == DeviceType.STROKER:
                for actuator in self._device.linear_actuators:
                    await actuator.command(duration_ms, intensity)

        except Exception as exc:
            print(f"[player] Send error on {self._device.name}: {exc}")

    async def _send_stop(self) -> None:
        """Stop the device cleanly."""
        try:
            await self._device.stop()
        except Exception as exc:
            print(f"[player] Stop error on {self._device.name}: {exc}")
