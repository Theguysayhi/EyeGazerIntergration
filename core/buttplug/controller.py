"""
core/buttplug/controller.py
Buttplug.io integration controller.

Manages the full lifecycle of Buttplug device connections and funscript
playback, reacting to NSFW/SFW state transitions signalled by NSFWConsumer.

Playback sequence
-----------------
On NSFW detected
    1. Cancel any in-progress script task.
    2. Randomly pick one starter script from scripts/starter/vib/ and one
       from scripts/starter/stroker/.
    3. Play both simultaneously (one per device category) — once, no loop.
    4. Randomly pick one main script from scripts/main/vib/ and one from
       scripts/main/stroker/.
    5. Play both in a loop until on_sfw() cancels the task.

    If the play task ends for any reason other than an explicit cancellation
    (e.g. device error, unhandled exception) while NSFW content is still
    active, the sequence is automatically restarted so devices never go
    silent while NSFW is showing.

On SFW detected (after dwell_ms of sustained SFW content)
    Cancel the current task (if any) and stop all devices (pause).
    The main loop applies the dwell delay — rapid SFW flickers do not
    interrupt playback.  When NSFW is detected again the full starter →
    main sequence restarts immediately.

Threading
---------
ButtplugController runs its own asyncio event loop on a dedicated daemon
thread.  The sync callbacks on_nsfw() / on_sfw() are safe to call from any
thread — they submit coroutines to that loop via run_coroutine_threadsafe().

buttplug-py 0.2.0 API notes
----------------------------
- Client("name")                             — create client
- WebsocketConnector(url)                    — create connector
- await client.connect(connector)            — connect
- await client.start_scanning()              — begin scan (returns Future)
- await client.stop_scanning()               — stop scan
- await client.stop_all()                    — stop all devices
- await client.disconnect()                  — disconnect
- client.devices -> dict[int, Device]        — discovered devices
- device.actuators      -> tuple[Actuator]   — vibration actuators
- device.linear_actuators -> tuple[Actuator] — linear/stroker actuators
- await device.stop()                        — stop this device
"""

from __future__ import annotations

import asyncio
import glob
import os
import random
import threading
from typing import Any

from .script_player import DeviceType, FunscriptPlayer


def _log_future_exception(fut: Any) -> None:
    """Done-callback that prints any exception raised inside a threadsafe future."""
    try:
        exc = fut.exception()
        if exc is not None:
            print(f"[buttplug] Unhandled exception in coroutine: {exc!r}")
    except Exception:
        pass


class ButtplugController:
    """
    Manages Buttplug device connections and NSFW-driven script playback.

    Parameters
    ----------
    server_url   : WebSocket URL of the Intiface Central server.
    scripts_dir  : Root directory that contains starter/ and main/ sub-folders.
    """

    def __init__(self, server_url: str = "ws://127.0.0.1:12345",
                 scripts_dir: str = "scripts",
                 starter_loops: int = 1) -> None:
        self._server_url    = server_url
        self._scripts_dir   = os.path.abspath(scripts_dir)
        self._starter_loops = max(1, starter_loops)

        self._loop:   asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None          = None
        self._client: Any                              = None   # buttplug.Client

        # Vibrators and strokers discovered during scanning.
        self._vibrators: list[Any] = []
        self._strokers:  list[Any] = []

        # The currently running playback coroutine wrapped as a Task.
        self._play_task: asyncio.Task | None = None
        self._task_lock = asyncio.Lock()   # created inside the loop thread

        # True while NSFW content is active (set by _handle_nsfw / _handle_sfw).
        # Used by _on_play_task_done to decide whether to auto-restart playback.
        self._nsfw_mode: bool = False

        self._ready = threading.Event()    # set when the loop is running

    # ------------------------------------------------------------------
    # Public sync API (callable from any thread)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the background asyncio thread and connect to Intiface Central.
        Returns immediately; connection happens asynchronously.
        """
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="ButtplugController",
        )
        self._thread.start()
        self._ready.wait(timeout=5)  # wait until the event loop is running

    def stop(self) -> None:
        """Gracefully cancel playback, stop devices, and disconnect."""
        if self._loop and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
            try:
                future.result(timeout=10)
            except Exception as exc:
                print(f"[buttplug] Shutdown error: {exc}")
            self._loop.call_soon_threadsafe(self._loop.stop)

    def on_nsfw(self) -> None:
        """
        Thread-safe callback: NSFW content detected.
        Schedules the starter → main script sequence.
        """
        print("[buttplug] on_nsfw() received")
        if self._loop and self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self._handle_nsfw(), self._loop)
            fut.add_done_callback(_log_future_exception)
        else:
            print("[buttplug] on_nsfw() called but loop is not running — "
                  f"loop={self._loop!r}")

    def on_sfw(self) -> None:
        """
        Thread-safe callback: SFW content detected.
        Cancels playback and stops all devices.
        """
        print("[buttplug] on_sfw() received")
        if self._loop and self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self._handle_sfw(), self._loop)
            fut.add_done_callback(_log_future_exception)
        else:
            print("[buttplug] on_sfw() called but loop is not running — "
                  f"loop={self._loop!r}")

    # ------------------------------------------------------------------
    # Background event-loop thread
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        try:
            self._loop.run_until_complete(self._main())
        finally:
            self._loop.close()

    async def _main(self) -> None:
        """
        Connect to Intiface Central, scan for devices, then idle until
        stop() shuts down the loop.
        """
        self._task_lock = asyncio.Lock()
        await self._connect_and_scan()
        # Keep the loop alive — tasks submitted via run_coroutine_threadsafe
        # will execute here.
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    async def _connect_and_scan(self) -> None:
        try:
            from buttplug import Client, WebsocketConnector
        except ImportError:
            print("[buttplug] buttplug-py is not installed — "
                  "run: pip install buttplug-py")
            return

        self._client = Client("EyeGazer")

        print(f"[buttplug] Connecting to {self._server_url} …")
        try:
            connector = WebsocketConnector(self._server_url)
            await self._client.connect(connector)
        except Exception as exc:
            print(f"[buttplug] Could not connect to Intiface Central: {exc}")
            print("[buttplug] Make sure Intiface Central is running and the "
                  "server is started.")
            self._client = None
            return

        print("[buttplug] Connected — scanning for devices …")
        try:
            await self._client.start_scanning()
            await asyncio.sleep(5)          # give devices time to appear
            await self._client.stop_scanning()
        except Exception as exc:
            print(f"[buttplug] Scan error: {exc}")

        self._categorise_devices()

        if not self._vibrators and not self._strokers:
            print("[buttplug] No supported devices found.  Make sure your "
                  "devices are turned on and paired before starting.")
        else:
            print(f"[buttplug] Ready — "
                  f"{len(self._vibrators)} vibrator(s), "
                  f"{len(self._strokers)} stroker(s)")

    def _categorise_devices(self) -> None:
        """Sort discovered devices into vibrators and strokers."""
        if self._client is None:
            return

        self._vibrators.clear()
        self._strokers.clear()

        for device in self._client.devices.values():
            if device.linear_actuators:
                # Device has linear actuators → treat as stroker
                self._strokers.append(device)
                print(f"[buttplug]   stroker  → {device.name}")
            elif device.actuators:
                # Device has scalar/vibrate actuators → treat as vibrator
                self._vibrators.append(device)
                print(f"[buttplug]   vibrator → {device.name}")
            else:
                print(f"[buttplug]   ignored  → {device.name} "
                      "(no vibrate/linear output)")

    # ------------------------------------------------------------------
    # NSFW / SFW handlers (run inside the event loop)
    # ------------------------------------------------------------------

    async def _handle_nsfw(self) -> None:
        self._nsfw_mode = True
        async with self._task_lock:
            # Cancel whatever is currently playing.
            if self._play_task and not self._play_task.done():
                self._play_task.cancel()
                try:
                    await self._play_task
                except (asyncio.CancelledError, Exception):
                    pass

            self._play_task = asyncio.create_task(self._play_sequence())
            # Auto-restart if the task ends unexpectedly while NSFW is active.
            self._play_task.add_done_callback(self._on_play_task_done)

    async def _handle_sfw(self) -> None:
        self._nsfw_mode = False
        async with self._task_lock:
            if self._play_task and not self._play_task.done():
                self._play_task.cancel()
                try:
                    await self._play_task
                except (asyncio.CancelledError, Exception):
                    pass
            await self._stop_all_devices()
            print("[buttplug] SFW dwell elapsed — playback paused, devices stopped")

    def _on_play_task_done(self, task: asyncio.Task) -> None:
        """
        Called automatically when a play task finishes.

        If the task ended without being cancelled (e.g. due to an unhandled
        exception, no scripts found, or other premature exit) and NSFW content
        is still active, restart the full sequence so the device is never
        silent while the debug screen shows NSFW.
        """
        if task.cancelled():
            return   # intentional stop — do nothing

        if task.exception() is not None:
            print(f"[buttplug] Play task ended with error: {task.exception()}")

        if self._nsfw_mode and self._loop and self._loop.is_running():
            print("[buttplug] Play task ended unexpectedly while NSFW active — restarting sequence")
            asyncio.run_coroutine_threadsafe(self._handle_nsfw(), self._loop)

    # ------------------------------------------------------------------
    # Playback sequence
    # ------------------------------------------------------------------

    async def _play_sequence(self) -> None:
        """
        Full NSFW playback sequence:
            starter scripts (once) → main scripts (loop).
        """
        try:
            # --- Starter phase ---
            starter_vib     = self._random_script("starter", "vib")
            starter_stroker = self._random_script("starter", "stroker")

            if not starter_vib and not starter_stroker:
                print("[buttplug] No starter scripts found — skipping to main")
            else:
                print(f"[buttplug] Playing starter scripts ({self._starter_loops}×) …")
                for _ in range(self._starter_loops):
                    tasks = []
                    if starter_vib:
                        for device in self._vibrators:
                            p = FunscriptPlayer(device, DeviceType.VIBRATOR)
                            tasks.append(asyncio.create_task(p.play(starter_vib, loop=False)))
                    if starter_stroker:
                        for device in self._strokers:
                            p = FunscriptPlayer(device, DeviceType.STROKER)
                            tasks.append(asyncio.create_task(p.play(starter_stroker, loop=False)))
                    if tasks:
                        await asyncio.gather(*tasks)

            # --- Main phase (looping) ---
            main_vib     = self._random_script("main", "vib")
            main_stroker = self._random_script("main", "stroker")

            if not main_vib and not main_stroker:
                print("[buttplug] No main scripts found")
                return

            print("[buttplug] Playing main scripts (looping) …")
            tasks = []
            if main_vib:
                for device in self._vibrators:
                    p = FunscriptPlayer(device, DeviceType.VIBRATOR)
                    tasks.append(asyncio.create_task(p.play(main_vib, loop=True)))
            if main_stroker:
                for device in self._strokers:
                    p = FunscriptPlayer(device, DeviceType.STROKER)
                    tasks.append(asyncio.create_task(p.play(main_stroker, loop=True)))

            if tasks:
                await asyncio.gather(*tasks)

        except asyncio.CancelledError:
            await self._stop_all_devices()
            raise  # re-raise so the task is properly marked cancelled

    # ------------------------------------------------------------------
    # Script selection helpers
    # ------------------------------------------------------------------

    def _random_script(self, phase: str, device_kind: str) -> str | None:
        """
        Return the path to a randomly chosen .funscript from the given folder.

        Parameters
        ----------
        phase       : "starter" or "main"
        device_kind : "vib" or "stroker"

        Returns None if the folder is empty or does not exist.
        """
        folder  = os.path.join(self._scripts_dir, phase, device_kind)
        pattern = os.path.join(folder, "*.funscript")
        scripts = glob.glob(pattern)
        if not scripts:
            return None
        choice = random.choice(scripts)
        print(f"[buttplug] Selected {phase}/{device_kind}: "
              f"{os.path.basename(choice)}")
        return choice

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    async def _stop_all_devices(self) -> None:
        """Stop all devices, returning strokers to position 0 first."""
        if self._client is None:
            return
        _RESET_MS = 500
        try:
            for device in self._strokers:
                for actuator in device.linear_actuators:
                    await actuator.command(_RESET_MS, 0.0)
            if self._strokers:
                await asyncio.sleep(_RESET_MS / 1000)
            await self._client.stop_all()
        except Exception as exc:
            print(f"[buttplug] stop_all error: {exc}")

    async def _shutdown(self) -> None:
        """Cancel playback and disconnect cleanly."""
        async with self._task_lock:
            if self._play_task and not self._play_task.done():
                self._play_task.cancel()
                try:
                    await self._play_task
                except (asyncio.CancelledError, Exception):
                    pass

        await self._stop_all_devices()

        if self._client is not None:
            try:
                await self._client.disconnect()
                print("[buttplug] Disconnected.")
            except Exception as exc:
                print(f"[buttplug] Disconnect error: {exc}")
            finally:
                self._client = None
