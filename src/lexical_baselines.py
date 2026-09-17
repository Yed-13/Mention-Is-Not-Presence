#!/usr/bin/env python3
"""Reproducible majority and character n-gram baselines.

The character model is a candidate-conditioned multinomial naive Bayes
classifier implemented with the Python standard library. It measures how much
of the controlled task can be recovered from surface form alone.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

from scriptbreak_eval import confusion_and_scores, read_jsonl, write_jsonl


def instances(rows):
    for row in rows:
        for candidate in row["candidates"]:
            marked = row["text"].replace(candidate["mention"], f"〈实体〉{candidate['mention']}〈/实体〉")
            yield row, candidate, f"类型={candidate['entity_type']}\n{marked}"


def ngrams(text, low=1, high=4):
    normalized = "".join(text.split())
    counts = collections.Counter()
    for n in range(low, high + 1):
        counts.update(normalized[i:i+n] for i in range(max(0, len(normalized) - n + 1)))
    return counts


class CharNgramNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.label_docs = collections.Counter()
        self.label_tokens = collections.defaultdict(collections.Counter)
        self.totals = collections.Counter()
        self.vocab = set()

    def fit(self, rows):
        for _, candidate, text in instances(rows):
            label = candidate["state"]
            feats = ngrams(text)
            self.label_docs[label] += 1
            self.label_tokens[label].update(feats)
            self.totals[label] += sum(feats.values())
            self.vocab.update(feats)
        return self

    def predict(self, text):
        feats = ngrams(text)
        n_docs = sum(self.label_docs.values())
        n_labels = len(self.label_docs)
        vocab_size = len(self.vocab)
        scores = {}
        for label, docs in self.label_docs.items():
            score = math.log((docs + self.alpha) / (n_docs + self.alpha * n_labels))
            denom = self.totals[label] + self.alpha * vocab_size
            token_counts = self.label_tokens[label]
            score += sum(count * math.log((token_counts[token] + self.alpha) / denom)
                         for token, count in feats.items() if token in self.vocab)
            scores[label] = score
        return max(scores, key=scores.get)


def predict_rows(train, test, method):
    majority = collections.Counter(
        candidate["state"] for row in train for candidate in row["candidates"]
    ).most_common(1)[0][0]
    model = CharNgramNB().fit(train) if method == "char_ngram_nb" else None
    output = []
    for row in test:
        candidates = []
        for candidate in row["candidates"]:
            if model is None:
                state = majority
            else:
                marked = row["text"].replace(candidate["mention"], f"〈实体〉{candidate['mention']}〈/实体〉")
                state = model.predict(f"类型={candidate['entity_type']}\n{marked}")
            candidates.append({
                "candidate_id": candidate["candidate_id"],
                "state": state,
                "evidence": [candidate["mention"]],
            })
        output.append({"scene_id": row["scene_id"], "candidates": candidates})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--method", choices=["majority", "char_ngram_nb"], required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()
    train = read_jsonl(args.train)
    test = read_jsonl(args.test)
    predictions = predict_rows(train, test, args.method)
    write_jsonl(args.predictions, predictions)
    metrics = confusion_and_scores(test, predictions)
    Path(args.metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"method": args.method, "accuracy": metrics["accuracy"],
                      "macro_f1": metrics["macro_f1"],
                      "family_exact_match": metrics["family_exact_match"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
