#!/usr/bin/env python3
"""Transparent string baseline for the ViStoryBench-derived visibility set."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    predictions = []
    for row in rows:
        shot = row["text"].split("画面描述：", 1)[1]
        predictions.append({
            "scene_id": row["scene_id"],
            "candidates": [
                {"candidate_id": c["candidate_id"],
                 "state": "visible" if c["mention"] in shot else "referenced_only",
                 "evidence": [c["mention"]]}
                for c in row["candidates"]
            ],
        })
    Path(args.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in predictions), encoding="utf-8")
    print(json.dumps({"written": len(predictions), "out": args.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
