"""3D visualisation + integration API for UmuhinziEnv.

A tiny stdlib HTTP server (no extra dependencies) that:
  * serves the three.js front-end in ``environment/web`` (three.js is
    vendored, so it works offline);
  * streams every environment step as JSON over Server-Sent Events
    (``GET /events``) - the same payload an SMS/mobile back-end would use;
  * exposes a small REST API:
        GET  /api/spec       action + observation definitions
        GET  /api/state      latest environment snapshot
        POST /api/recommend  {"observation": [18 floats]} -> advisory
    ``/api/recommend`` is the serialisation path from a trained policy to a
    real product: a farmer-facing service posts sensor/weather/farmer-report
    features and receives the advisory to send by SMS.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import numpy as np

from .custom_env import ACTION_LABELS, ACTION_NAMES, OBS_NAMES, Action

WEB_DIR = Path(__file__).resolve().parent / "web"
mimetypes.add_type("text/javascript", ".js")


class SimulationServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host, self.port = host, port
        self.frames: list[dict] = []
        self.meta: dict = {}
        self.cond = threading.Condition()
        self.policy_fn: Callable[[np.ndarray], int] | None = None
        self.httpd: ThreadingHTTPServer | None = None

    # ------------------------------------------------------------- producer
    def set_meta(self, **meta):
        with self.cond:
            self.meta.update(meta)

    def push(self, state: dict):
        with self.cond:
            state = {**state, "seq": len(self.frames), "meta": dict(self.meta)}
            self.frames.append(state)
            self.cond.notify_all()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    # --------------------------------------------------------------- server
    def start(self, open_browser: bool = True):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the terminal for agent output
                pass

            def _json(self, obj, code=200):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/events":
                    return self._sse()
                if path == "/api/spec":
                    return self._json(
                        {
                            "actions": [{"id": int(a), "name": a.name, "label": ACTION_LABELS[a]} for a in Action],
                            "observations": OBS_NAMES,
                        }
                    )
                if path == "/api/state":
                    with server.cond:
                        return self._json(server.frames[-1] if server.frames else {})
                if path == "/":
                    path = "/index.html"
                f = (WEB_DIR / path.lstrip("/")).resolve()
                if WEB_DIR not in f.parents or not f.is_file():
                    self.send_error(404)
                    return
                data = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(f.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                if self.path != "/api/recommend":
                    self.send_error(404)
                    return
                n = int(self.headers.get("Content-Length", 0))
                try:
                    payload = json.loads(self.rfile.read(n) or b"{}")
                    obs = np.asarray(payload["observation"], dtype=np.float32)
                    assert obs.shape == (len(OBS_NAMES),)
                except Exception as e:  # noqa: BLE001
                    return self._json({"error": f"bad request: {e}"}, 400)
                if server.policy_fn is None:
                    return self._json({"error": "no policy loaded"}, 503)
                a = Action(int(server.policy_fn(obs)))
                return self._json({"action_id": int(a), "action": a.name, "advisory": ACTION_LABELS[a]})

            def _sse(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                with server.cond:
                    last = len(server.frames) - 2  # replay the latest frame, then stream
                try:
                    while True:
                        with server.cond:
                            server.cond.wait_for(lambda: len(server.frames) - 1 > last, timeout=15)
                            new = server.frames[max(last + 1, 0) :]
                            last = len(server.frames) - 1
                        if not new:
                            self.wfile.write(b": keep-alive\n\n")
                        for fr in new:
                            self.wfile.write(f"data: {json.dumps(fr)}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return

        self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        print(f"[viewer] 3D simulation running at {self.url}")
        if open_browser:
            webbrowser.open(self.url)
        return self

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()


_VIEWER: SimulationServer | None = None


def get_viewer(port: int = 8765, open_browser: bool = True) -> SimulationServer:
    global _VIEWER
    if _VIEWER is None:
        _VIEWER = SimulationServer(port=port).start(open_browser=open_browser)
    return _VIEWER


__all__ = ["SimulationServer", "get_viewer", "ACTION_NAMES"]
