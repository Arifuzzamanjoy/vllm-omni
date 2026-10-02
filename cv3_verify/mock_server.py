#!/usr/bin/env python3
"""Minimal stand-in for the vLLM-Omni speech API, for testing the kit without a GPU.

/v1/audio/speech answers with the *held-out* utterance of the requested voice
(same speaker, different recording, so scoring is not trivially cos=1). Any
request index listed in --swap gets another voice's held-out clip instead,
which the scorers must flag. The request index is recovered from the seed
(seed = run_seed*1000 + idx).
"""

import argparse
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_voice_leak import load_voices  # noqa: E402


def make_handler(clips: dict[str, bytes], swaps: dict[int, str]):
    registered: set[str] = set()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {})
            if self.path == "/v1/audio/voices":
                return self._json(200, {"voices": [], "uploaded_voices": [{"name": n} for n in sorted(registered)]})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("content-length", 0)))
            if self.path == "/v1/audio/voices":
                m = re.search(rb'name="name"\r\n\r\n([^\r]+)\r\n', body)
                if not m:
                    return self._json(400, {"error": "name missing"})
                registered.add(m.group(1).decode().lower())
                return self._json(200, {"success": True, "voice": {"name": m.group(1).decode()}})
            if self.path == "/v1/audio/speech":
                req = json.loads(body)
                voice = req["voice"].lower()
                if voice not in registered:
                    return self._json(400, {"error": f"unknown voice {voice}"})
                idx = int(req.get("seed", 0)) % 1000
                served = swaps.get(idx, voice)
                data = clips[served]
                self.send_response(200)
                self.send_header("content-type", "audio/wav")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self._json(404, {"error": "not found"})

    return H


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--voices", required=True)
    p.add_argument("--port", type=int, default=18091)
    p.add_argument("--swap", action="append", default=[], help="IDX:VOICE_NAME, serve VOICE_NAME for request IDX")
    args = p.parse_args()
    voices = load_voices(args.voices)
    clips = {v["name"].lower(): Path(v.get("heldout_path") or v["wav_path"]).read_bytes() for v in voices}
    swaps = {int(s.split(":")[0]): s.split(":", 1)[1].lower() for s in args.swap}
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(clips, swaps))
    print(f"mock server on :{args.port}, swaps={swaps}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
