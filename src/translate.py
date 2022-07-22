#!/usr/bin/env python3

import torch
import fairseq

model = torch.hub.load(
    'pytorch/fairseq', 'transformer.wmt19.de-en',
    checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
    tokenizer='moses', bpe='fastbpe'
)
model.eval()  # disable dropout

# the underlying model is available under the *models* attribute
assert isinstance(model.models[0], fairseq.models.transformer.TransformerModel)

# Move model to GPU for faster translation
# en2de.cuda()

# Translate a sentence
output = model.translate('Halloda')
print(output)
# 'Hallo Welt!'
