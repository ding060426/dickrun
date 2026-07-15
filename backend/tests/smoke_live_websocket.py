"""Send an audio file through the same WebSocket protocol as the microphone."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import websockets


async def run(url: str, audio_path: Path, pace: bool) -> dict:
    samples, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    messages: list[dict] = []

    async with websockets.connect(url, max_size=4 * 1024 * 1024) as socket:
        messages.append(json.loads(await socket.recv()))
        await socket.send(json.dumps({
            "action": "configure",
            "sample_rate": 16000,
            "channels": 1,
            "sample_format": "pcm_s16le",
            "browser_sample_rate": 48000,
        }))
        messages.append(json.loads(await socket.recv()))

        async def receive_until_stopped():
            while True:
                message = json.loads(await socket.recv())
                messages.append(message)
                if message.get("type") == "stopped":
                    return

        receiver = asyncio.create_task(receive_until_stopped())
        for offset in range(0, len(pcm), 640):
            await socket.send(pcm[offset:offset + 640].tobytes())
            if pace:
                await asyncio.sleep(0.04)
        await socket.send(json.dumps({"action": "stop"}))
        await receiver

    counts = Counter(message.get("type") for message in messages)
    results = [message["data"] for message in messages if message.get("type") == "live_result"]
    stopped = next(message["data"] for message in messages if message.get("type") == "stopped")
    return {
        "audio_seconds": round(len(pcm) / sample_rate, 2),
        "message_counts": dict(counts),
        "partial_results": sum(bool(item.get("is_partial")) for item in results),
        "final_results": sum(bool(item.get("is_final")) for item in results),
        "final_texts": [item.get("text", "") for item in results if item.get("is_final")],
        "server": stopped,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--url", default="ws://127.0.0.1:8766/ws/live")
    parser.add_argument("--pace", action="store_true", help="send in real time instead of as fast as possible")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.url, args.audio_path, args.pace)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
