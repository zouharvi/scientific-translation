#!/usr/bin/env python3

import fairseq
import fairseq.models.transformer
import glob
import json
from nltk import sent_tokenize
import tqdm

SUPPORTED_LANGS = {"de"}

with open("corpus/vocab_f3c9084cd6.json", "r") as f:
    vocab_allowmask = json.load(f)

data = []
for fname in glob.glob("corpus_manual/*.json"):
    with open(fname, "r") as f:
        data_local = json.load(f)

        for lang, abstract in data_local["abstract_ref"].items():
            if lang in SUPPORTED_LANGS:
                data.append(sent_tokenize(abstract))

model = fairseq.models.transformer.TransformerModel.from_pretrained(
    'models/wmt19.de-en.joined-dict.ensemble/',
    # checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
    checkpoint_file='model2.pt',
    tokenizer='moses', bpe='fastbpe',
)
model.eval()
model.cuda()

# print(model.encode("hello how are you?"))
# print(model.encode("I'm here hello"))
# print(model.encode(""))
# print(model.encode(" "))
# output = model.translate(
#     "Hello, how are you?",
#     lexical_mask={
#         "Hello", "hello", ",", "how", "are", "we", "you", "I", "Good", "day", "?", ".", "!"
#     },
#     # TODO: change beam size
#     beam=50,
# )
# print(output)

for abstract in data:
    print(abstract)

    for x in abstract:
        translation = model.translate(
            x, lexical_mask=vocab_allowmask,
            # x, lexical_mask={},
            beam=10
        )
        print(translation)