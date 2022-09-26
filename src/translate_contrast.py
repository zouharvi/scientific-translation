#!/usr/bin/env python3

import copy
import fairseq
import fairseq.models.transformer
import tqdm
from utils import json_readl, json_dumpl
import sacrebleu
import numpy as np

bleu_metric = sacrebleu.metrics.BLEU(effective_order=True)

model = fairseq.models.transformer.TransformerModel.from_pretrained(
    'models/wmt19.de-en.joined-dict.ensemble/',
    checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
    # checkpoint_file='model2.pt',
    tokenizer='moses', bpe='fastbpe',
)
model.eval()
model.cuda()

BEAM_SIZE = 5
scores_orig = []
scores_self = []
scores_ref = []
scores_self_ref = []
scores_oracle = []

data = json_readl("data_raw/acl_corpus.jsonl")

all_result = []
kwparams = {}

for step_subtract_frequency in [5, 4, 3, 2, 1, 6, 7, 8, 9, 10]:
    kwparams["step_subtract_frequency"] = step_subtract_frequency
    for strategy in ["linear", "ratio"]:
        kwparams["strategy"] = strategy
        for coefficient in [0.01, 0.05, 0.1, 0.5, 1, 5]:
            kwparams["coefficient"] = coefficient

            line_result = {}
            for line in tqdm.tqdm(data):
                # don't use vocabulary restriction
                translation = model.translate(
                    line["abstract_de"], beam=BEAM_SIZE,
                )
                score = bleu_metric.sentence_score(translation, [line["abstract"]]).score
                scores_orig.append(score)

                # use only self vocabulary
                translation = model.translate(
                    line["abstract_de"], beam=BEAM_SIZE,
                    lexical_mask=line["vocab_fulltext"],
                    **kwparams
                )
                score = bleu_metric.sentence_score(translation, [line["abstract"]]).score
                scores_self.append(score)

                # use only ref vocabulary
                translation = model.translate(
                    line["abstract_de"], beam=BEAM_SIZE,
                    lexical_mask=line["vocab_ref"],
                    **kwparams
                )
                score = bleu_metric.sentence_score(translation, [line["abstract"]]).score
                scores_ref.append(score)

                # use only self+ref vocabulary
                translation = model.translate(
                    line["abstract_de"], beam=BEAM_SIZE,
                    lexical_mask=line["vocab_ref"] + line["vocab_fulltext"],
                    **kwparams
                )
                score = bleu_metric.sentence_score(translation, [line["abstract"]]).score
                scores_self_ref.append(score)

                # use only abstract vocabulary
                translation = model.translate(
                    line["abstract_de"], beam=BEAM_SIZE,
                    lexical_mask=line["vocab_abstract"],
                    **kwparams
                )
                score = bleu_metric.sentence_score(translation, [line["abstract"]]).score
                scores_oracle.append(score)

            line_result["orig"] = np.average(scores_orig)
            line_result["self"] = np.average(scores_self)
            line_result["ref"] = np.average(scores_ref)
            line_result["self+ref"] = np.average(scores_self_ref)
            line_result["oracle"] = np.average(scores_oracle)
            line_result["kwparams"] = copy.deepcopy(kwparams)
            all_result.append(line_result)
            json_dumpl("computed/gridsearch.jsonl", all_result)

            print(f"BLEU:             {np.average(scores_orig):.1f}")
            print(f"BLEU (self):      {np.average(scores_self):.1f}")
            print(f"BLEU (ref):       {np.average(scores_ref):.1f}")
            print(f"BLEU (self+ref):  {np.average(scores_self_ref):.1f}")
            print(f"BLEU (oracle):    {np.average(scores_oracle):.1f}")