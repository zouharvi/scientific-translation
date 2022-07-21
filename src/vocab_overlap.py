#!/usr/bin/env python3

import argparse
import json
import matplotlib.pyplot as plt
import fig_utils
from utils import text_to_vocab

args = argparse.ArgumentParser()
args.add_argument(
    "-f", "--files", nargs="+",
    default=["corpus/nmt_pe.json"]
)
args.add_argument(
    "-pf", "--pos-files", nargs="+",
    default=[
        "corpus/cubitt.json", "corpus/pe_corrections.json",
        "corpus/markables.json", "corpus/mt_subtitling_pe.json",
        "corpus/fiskmo.json", "corpus/casamat.json",
    ]
)
args.add_argument(
    "-nf", "--neg-files", nargs="+",
    default=[
        "corpus/cell_growth_1.json", "corpus/cell_growth_2.json",
        "corpus/cell_growth_3.json", "corpus/cell_growth_4.json",
        "corpus/cell_growth_5.json", "corpus/cell_growth_6.json",
        "corpus/cell_growth_7.json",
    ]
)
args = args.parse_args()

if len(args.pos_files) != len(args.neg_files) - 1:
    print("WARNING: potential statistical imbalance due to the positive and negative set having different sizes")

if len(args.files) != 1:
    raise Exception("Currently able to handle only one main file")

positive_overlap = []
negative_overlap = []
positive_set = set()
negative_set = set()
positive_set_size = []
negative_set_size = []

for f in args.files:
    print(f"Processing main {f}")
    with open(f, "r") as f:
        data = json.load(f)
    abstract_txt = data["abstract_en"]
    paper_txt = data["paper_en"]

    abstract_set = text_to_vocab(abstract_txt)
    paper_set = text_to_vocab(paper_txt)

    print(abstract_set)
    print(f"Abstract set size: {len(abstract_set)}")
    print(f"Paper set size: {len(paper_set)}")
    print(
        f"Paper & abstract overlap size:    \
        {len(abstract_set & paper_set)} ({len(abstract_set & paper_set)/len(abstract_set):.0%})"
    )
    positive_overlap.append(
        len(abstract_set & paper_set) / len(abstract_set) * 100
    )
    print(f"Missing words from paper: {abstract_set - paper_set}")

    positive_set |= paper_set
    positive_set_size.append(len(positive_set))

for f in args.pos_files:
    print(f"Processing positive {f}")
    with open(f, "r") as f:
        data = json.load(f)

    # | text_to_vocab(data["abstract_en"])
    positive_set |= text_to_vocab(data["paper_en"])

    print(
        f"Positive & abstract overlap size: \
        {len(abstract_set & positive_set)} ({len(abstract_set & positive_set)/len(abstract_set):.0%})"
    )

    positive_overlap.append(
        len(abstract_set & positive_set) / len(abstract_set) * 100
    )
    positive_set_size.append(len(positive_set))


print(f"Missing words from positives: {abstract_set - positive_set}")

for f in args.neg_files:
    print(f"Processing negative {f}")
    with open(f, "r") as f:
        data = json.load(f)

    # | text_to_vocab(data["abstract_en"])
    negative_set |= text_to_vocab(data["paper_en"])

    print(
        f"Negative & abstract overlap size: \
        {len(abstract_set & negative_set)} ({len(abstract_set & negative_set)/len(abstract_set):.0%})"
    )
    negative_overlap.append(
        len(abstract_set & negative_set) / len(abstract_set) * 100
    )
    negative_set_size.append(len(negative_set))

# plot results

XTICKS = list(range(len(positive_overlap)))

plt.figure(figsize=(6, 4))
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.plot(
    XTICKS, positive_overlap, marker="x", label="Related overlap",
    color=fig_utils.COLORS[2],
)
ax1.plot(
    XTICKS, negative_overlap, marker="x", label="Unrelated overlap",
    color=fig_utils.COLORS[1],
)
ax2.scatter(
    XTICKS, positive_set_size, marker="s", s=10, label="Related vocab size",
    color=fig_utils.COLORS[2],
)
ax2.scatter(
    XTICKS, negative_set_size, marker="s", s=10, label="Unrelated vocab size",
    color=fig_utils.COLORS[1],
)

# the first item in positive_overlap is the paper itself
ax1.hlines(
    positive_overlap[0], xmin=0, xmax=len(positive_overlap) - 1,
    color="dimgray",
    linestyle=":",
    label="Original paper"
)
hndl1, lbls1 = ax1.get_legend_handles_labels()
hndl2, lbls2 = ax2.get_legend_handles_labels()
ax1.set_xlabel("Docs (cummulative)")
ax1.set_ylabel("Overlap between an abstract and other papers (%)")
ax2.set_ylabel("Vocabulary size")
plt.legend(
    hndl1 + hndl2, lbls1 + lbls2,
    loc="upper left",
    bbox_to_anchor=(0.05, 1.32),
    bbox_transform=ax1.transAxes,
    ncol=2,
)
plt.tight_layout(rect=(0, 0, 1, 1.05))
plt.savefig("figures/vocab_overlap.pdf")
plt.show()
