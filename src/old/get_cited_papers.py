#!/usr/bin/env python3

import argparse
import semanticscholar
import arxiv
from tika import parser
import json
from utils import text_to_vocab

args = argparse.ArgumentParser()
args.add_argument(
    "-p", "--paper", help="Paper id (from semantic scholar)",
    default="f3c9084cd61dfcdc51c9d846a587863c17475540"
)
args = args.parse_args()

# get names of relevant papers
sch = semanticscholar.SemanticScholar(timeout=2)
paper = sch.paper(args.paper)
relevant_titles = [
    x["title"]
    for x in paper["references"]+paper["citations"]
]
print("\n".join(relevant_titles))
print(paper.keys())

vocab = set()

# download pdfs
print()
for title in relevant_titles:
    print("Searching arxiv for", title)
    found_paper = next(arxiv.Search(query=title, max_results=1).results())
    # even if it's a mismatch, we want to use it
    print("Downloading", found_paper.title)
    filename = found_paper.download_pdf(dirpath="corpus_raw/", filename=found_paper.get_short_id()+".pdf")

    text = parser.from_file(filename)["content"]
    vocab |= text_to_vocab(text)

    print("Vocab size", len(vocab))
    print()

    with open(f"corpus/vocab_{args.paper[:10]}.json", "w") as f:
        json.dump(list(vocab), f)