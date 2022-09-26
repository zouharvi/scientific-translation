#!/usr/bin/env python3

import torch
import numpy as np
import json
import pandas as pd
import tqdm
import semanticscholar
from utils import pdf_to_vocab, text_to_vocab_simple
import argparse
import time

ss = semanticscholar.SemanticScholar()

args = argparse.ArgumentParser()
args.add_argument(
    "-di", "--data-in",
    default="data_raw/acl_corpus_full-text.parquet"
)
args.add_argument(
    "-do", "--data-out",
    default="data_raw/acl_corpus.jsonl"
)
args = args.parse_args()
data = pd.read_parquet(args.data_in)

model = torch.hub.load(
    "pytorch/fairseq", 'transformer.wmt19.en-de',
    checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
    tokenizer='moses', bpe='fastbpe'
)
model.eval()
model.cuda()

# clean up & open again
with open(args.data_out, "w"):
    pass
fout = open(args.data_out, "a")

line_count = 0

for row_i, row in tqdm.tqdm(data.iterrows()):
    if row_i == 50:
        break
    if len(row["abstract"]) == 0:
        continue

    line = {}
    vocab_abstract = text_to_vocab_simple(row["abstract"])
    vocab_self = text_to_vocab_simple(row["full_text"][len(row["abstract"]):])
    line["vocab_abstract"] = list(vocab_abstract)
    line["vocab_fulltext"] = list(vocab_self)

    ss_paper = ss.paper("ACL:" + row["acl_id"])
    vocab_ref = set()
    for ref in ss_paper["references"]:
        # overwrite ref object with fuller one
        ref = ss.paper(ref["paperId"])
        if "externalIds" not in ref:
            continue
        if "ACL" not in ref["externalIds"]:
            continue
        acl_id = ref["externalIds"]["ACL"]
        ref_url = f"https://aclanthology.org/{acl_id}.pdf"
        vocab_ref |= pdf_to_vocab(ref_url)
        time.sleep(5)

    line["vocab_ref"] = list(vocab_ref)
    line["abstract"] = row["abstract"]
    
    # translate into German
    abstract_de = model.translate(
        row["abstract"], beam=10
    )
    line["abstract_de"] = abstract_de
    fout.write(json.dumps(line, ensure_ascii=False) + "\n")

    line_count += 1

print("Total", line_count, "saved")