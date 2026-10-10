"""Synthesize speech with a local AivisSpeech engine (VOICEVOX-compatible HTTP API, port 10101).

Start the engine first, e.g. ...\\AivisSpeech\\AivisSpeech-Engine\\run.exe (or the AivisSpeech app).

Input
    text            Japanese text to speak (positional); --list prints the available speakers instead
Output
    WAV file at --out (44.1 kHz, 16-bit, mono as returned by the engine); synthesize() returns the bytes
Parameters (command line / synthesize() arguments)
    --speaker       style id, e.g. 888753760 (まお / ノーマル)         default 888753760
    --speed         speedScale                                          default 1.0
    --intonation    intonationScale                                     default 1.0
    --out           output path                                         default data/speech.wav
    BASE            engine URL (module constant)                        default http://127.0.0.1:10101

    python synth_aivis.py --list
    python synth_aivis.py "今日はいい天気ですね。" --speaker 888753760 --out data/mao.wav
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:10101"


def _post(path, params, body=None):
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def list_speakers():
    with urllib.request.urlopen(f"{BASE}/speakers", timeout=10) as r:
        for sp in json.load(r):
            print(sp["name"], [(st["name"], st["id"]) for st in sp["styles"]])


def synthesize(text, speaker, speed=1.0, intonation=1.0):
    """Return WAV bytes."""
    query = json.loads(_post("/audio_query", {"text": text, "speaker": speaker}))
    query["speedScale"] = speed
    query["intonationScale"] = intonation
    return _post("/synthesis", {"speaker": speaker}, json.dumps(query).encode())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--speaker", type=int, default=888753760)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--intonation", type=float, default=1.0)
    ap.add_argument("--out", default="data/speech.wav")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.text:
        list_speakers()
    else:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(synthesize(a.text, a.speaker, a.speed, a.intonation))
        print("saved", out)
