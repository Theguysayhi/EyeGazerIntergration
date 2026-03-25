"""
core/buttplug — integration package for buttplug-py + local controller/player.

Name-shadowing bootstrap
------------------------
When core/ is on sys.path this package is importable as plain "buttplug",
which shadows the installed buttplug-py library.  We work around this by
temporarily stripping core/ from sys.path during module initialisation,
importing the real library, and re-exporting its public symbols as attributes
of this module.

That way, deferred `from buttplug import Client` calls inside
script_player.py and controller.py (which run AFTER this __init__ has
already been registered in sys.modules) transparently resolve to the real
third-party symbols.

If this package is loaded as `core.buttplug` there is no shadowing; the
bootstrap is skipped in that branch.
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Any


def _load_real_buttplug_symbols() -> dict:
    """
    Temporarily remove core/ from sys.path, import the real installed
    buttplug-py package, extract public symbols, and return them in a dict.

    Real sub-modules such as ``buttplug.errors`` are kept in sys.modules
    so that later ``from buttplug.errors import ButtplugError`` calls still
    work even though ``sys.modules['buttplug']`` is now *this* module.
    """
    real_pkg   = "buttplug"
    this_pkg   = __name__       # 'buttplug' when core/ is on sys.path

    this_module = sys.modules.get(this_pkg)

    # Compute the core/ directory (parent of this package's directory).
    core_dir = os.path.normcase(os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ))

    # Build a clean sys.path that excludes core/.
    saved_path = sys.path[:]
    clean_path = [
        p for p in sys.path
        if os.path.normcase(os.path.abspath(p or os.getcwd())) != core_dir
    ]
    sys.path = clean_path

    # Remove *all* our own entries from sys.modules so importlib starts fresh.
    stashed: dict[str, object] = {}
    for k in list(sys.modules):
        if k == this_pkg or k.startswith(this_pkg + "."):
            stashed[k] = sys.modules.pop(k)

    out: dict = {}
    try:
        real = importlib.import_module(real_pkg)
        # buttplug-py 0.2.0 exports: Client, WebsocketConnector
        for sym in ("Client", "WebsocketConnector"):
            out[sym] = getattr(real, sym, None)

        # Also load buttplug.errors so the sub-module stays in sys.modules.
        try:
            real_err = importlib.import_module(f"{real_pkg}.errors")
            for err_sym in ("ButtplugError", "ClientError", "ServerError",
                            "ConnectorError", "DeviceServerError"):
                out[err_sym] = getattr(real_err, err_sym, None)
        except ImportError:
            pass

        # Also keep buttplug.client and buttplug.connectors accessible
        try:
            importlib.import_module(f"{real_pkg}.client")
        except ImportError:
            pass
        try:
            importlib.import_module(f"{real_pkg}.connectors")
        except ImportError:
            pass

    except ImportError:
        print(
            "[buttplug] WARNING: buttplug-py is not installed.\n"
            "           Run:  pip install buttplug-py"
        )

    finally:
        # Restore sys.path.
        sys.path = saved_path

        # Remove the real top-level entry; we will take that slot back below.
        sys.modules.pop(real_pkg, None)

        # Keep any freshly-loaded real sub-modules (e.g. buttplug.errors)
        # that did NOT exist in our stash.  They are already in sys.modules
        # from the import above, so no action needed for new entries.

        # Restore stashed sub-module entries (using setdefault so we never
        # overwrite a freshly-loaded real sub-module with a stale stashed one).
        for k, v in stashed.items():
            if k != real_pkg and k.startswith(real_pkg + "."):
                if k not in sys.modules:
                    sys.modules[k] = v  # type: ignore[assignment]

        # Put ourselves back as the top-level entry.
        if this_module is not None:
            sys.modules[this_pkg] = this_module

    return out


# ---------------------------------------------------------------------------
# Run the bootstrap only when we are shadowing the real package.
# ---------------------------------------------------------------------------
if __name__ == "buttplug":
    _s = _load_real_buttplug_symbols()
    Client:           Any = _s.get("Client")
    WebsocketConnector: Any = _s.get("WebsocketConnector")
    ButtplugError:    Any = _s.get("ButtplugError")
    ClientError:      Any = _s.get("ClientError")
    ServerError:      Any = _s.get("ServerError")
    ConnectorError:   Any = _s.get("ConnectorError")
    DeviceServerError: Any = _s.get("DeviceServerError")
    del _s
else:
    # Loaded as core.buttplug — no shadowing; real package symbols are
    # accessible directly via `import buttplug`.
    Client: Any = None
    WebsocketConnector: Any = None
    ButtplugError: Any = None
    ClientError: Any = None
    ServerError: Any = None
    ConnectorError: Any = None
    DeviceServerError: Any = None

del _load_real_buttplug_symbols  # keep module namespace tidy

# ---------------------------------------------------------------------------
# Local exports
# ---------------------------------------------------------------------------
from .controller import ButtplugController   # noqa: E402
from .script_player import DeviceType, FunscriptPlayer  # noqa: E402

__all__ = [
    # Third-party re-exports (buttplug-py 0.2.0)
    "Client",
    "WebsocketConnector",
    "ButtplugError",
    "ClientError",
    "ServerError",
    "ConnectorError",
    "DeviceServerError",
    # Local
    "ButtplugController",
    "DeviceType",
    "FunscriptPlayer",
]
