#!/usr/bin/env python3

import argparse
import json
import matplotlib.pyplot as plt
import fig_utils
from utils import text_to_vocab

args = argparse.ArgumentParser()
args.add_argument(
    "-f", "--files", nargs="+",
    default=["corpus/backtranslation.json"]
)
args.add_argument(
    "-v", "--vocab", default="corpus/vocab_f3c9084cd6.json"
)
args = args.parse_args()

with open(args.vocab, "r") as f:
    vocab_set = set(json.load(f))

for f in args.files:
    print(f"Processing main {f}")
    with open(f, "r") as f:
        data = json.load(f)
    abstract_txt = data["abstract_en"]
    paper_txt = data["paper_en"]

    abstract_set = text_to_vocab(abstract_txt)

    print(abstract_set)
    print(f"Abstract set size: {len(abstract_set)}")
    print(f"Vocab set size: {len(vocab_set)}")
    print(
        f"Paper & abstract overlap size:    \
        {len(abstract_set & vocab_set)} ({len(abstract_set & vocab_set)/len(abstract_set):.0%})"
    )
    print(f"Missing words from vocab: {abstract_set - vocab_set}")