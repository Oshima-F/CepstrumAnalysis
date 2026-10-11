"""Synthesize speech with a local AivisSpeech engine (VOICEVOX-compatible HTTP API, port 10101).

Start the engine first, e.g. ...\\AivisSpeech\\AivisSpeech-Engine\\run.exe (or the AivisSpeech app).
The settings below are the defaults of the command-line options and of synthesize().

    python synth_aivis.py --list
    python synth_aivis.py "今日はいい天気ですね。" --speaker 888753760 --out data/mao.wav
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:10101"   # engine URL
SPEAKER = 888753760               # style id (まお / ノーマル); --list shows all
SPEED = 1.0                       # speedScale
INTONATION = 1.0                  # intonationScale
OUT = "data/speech.wav"           # output path


def _post(path, params, body=None):
    """POST to the engine and return the raw response bytes."""
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def list_speakers():
    """Print every speaker with its (style name, style id) pairs. Takes no arguments."""
    with urllib.request.urlopen(f"{BASE}/speakers", timeout=10) as r:
        for sp in json.load(r):
            print(sp["name"], [(st["name"], st["id"]) for st in sp["styles"]])


def synthesize(text, speaker=SPEAKER, speed=SPEED, intonation=INTONATION):
    """Synthesize speech with the engine.

    Args:
        text: Japanese text to speak. Required.
        speaker: Style id. Default ``SPEAKER``.
        speed: Speaking rate (speedScale). Default ``SPEED``.
        intonation: Intonation strength (intonationScale). Default ``INTONATION``.

    Returns:
        WAV file contents as bytes (44.1 kHz, 16-bit, mono as returned by the engine).
    """
    query = json.loads(_post("/audio_query", {"text": text, "speaker": speaker}))
    query["speedScale"] = speed
    query["intonationScale"] = intonation
    return _post("/synthesis", {"speaker": speaker}, json.dumps(query).encode())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--speaker", type=int, default=SPEAKER)
    ap.add_argument("--speed", type=float, default=SPEED)
    ap.add_argument("--intonation", type=float, default=INTONATION)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.text:
        list_speakers()
    else:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(synthesize(a.text, a.speaker, a.speed, a.intonation))
        print("saved", out)
