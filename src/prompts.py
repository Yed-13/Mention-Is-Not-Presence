#!/usr/bin/env python3
"""Create frozen direct/guideline prompts without calling a model."""
import argparse, json
from pathlib import Path
from scriptbreak_eval import read_jsonl, write_jsonl

SCHEMA = '{"scene_id":"...","candidates":[{"candidate_id":"...","state":"visible|audible_only|depicted|referenced_only|uncertain","evidence":["exact substring"]}]}'
DIRECT = "Classify every supplied candidate in the Chinese screenplay scene. Return JSON only and preserve every candidate_id. Required schema: " + SCHEMA
GUIDELINES = """Classify every supplied candidate in the Chinese screenplay scene. Return JSON only and preserve every candidate_id.
Use exactly one state per candidate:
- visible: the entity itself is directly present in the current scene space; a visible speaker remains visible.
- audible_only: the entity is currently heard but not directly visible.
- depicted: the entity occurs only inside a photograph, poster, television image, or comparable representation.
- referenced_only: negated, past, hypothetical, reported, or otherwise mentioned without current visual or auditory realization.
- uncertain: the text itself leaves realization unresolved.
Do not infer entities, sounds, or equipment absent from the text. Evidence must contain one or more shortest exact substrings from the supplied scene. Required schema: """ + SCHEMA

GUIDELINES_CONCISE = """Assign one realization state to every supplied candidate in the Chinese screenplay scene and return JSON only. Use visible when the entity itself is in the current scene; audible_only when it is heard but unseen; depicted when it exists only in an image or screen; referenced_only for absent, negated, past, hypothetical, or reported mentions; and uncertain only when the scene explicitly leaves realization unresolved. A visible speaker is visible. Preserve every candidate_id and cite the shortest exact scene substring as evidence. Required schema: """ + SCHEMA

GUIDELINES_REORDERED = """Return JSON only for every supplied candidate in the Chinese screenplay scene and preserve every candidate_id.
Apply the following decision rules:
- uncertain: choose this only when the screenplay explicitly withholds whether the entity is realized.
- referenced_only: the entity is absent from the current scene and occurs only in negation, memory, report, plan, or supposition.
- depicted: the entity appears only through a photograph, poster, television, or another representation.
- audible_only: the entity is heard now but is not directly seen.
- visible: the entity itself occupies the current scene; speech does not override visible presence.
Use only information stated in the scene. Each evidence list must contain the shortest exact substring or substrings that license the decision. Required schema: """ + SCHEMA

CONDITIONS = {
    "direct": DIRECT,
    "guideline": GUIDELINES,
    "guideline_concise": GUIDELINES_CONCISE,
    "guideline_reordered": GUIDELINES_REORDERED,
}

def user_payload(row):
    return json.dumps({"scene_id":row["scene_id"], "text":row["text"],
        "candidates":[{"candidate_id":c["candidate_id"],"mention":c["mention"],"entity_type":c["entity_type"]} for c in row["candidates"]]}, ensure_ascii=False)

def build(rows, condition):
    system = CONDITIONS[condition]
    return [{"scene_id":r["scene_id"], "family_id":r["family_id"], "condition":condition,
             "messages":[{"role":"system","content":system},{"role":"user","content":user_payload(r)}]} for r in rows]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--condition",choices=sorted(CONDITIONS),required=True); p.add_argument("--out",required=True)
    a=p.parse_args(); write_jsonl(a.out, build(read_jsonl(a.data),a.condition))
    print(json.dumps({"written":a.out,"condition":a.condition},ensure_ascii=False))
if __name__ == "__main__": main()
