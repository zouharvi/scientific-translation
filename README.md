# scientific-translation
Translation of abstracts, scientific papers and textbooks using general-purpose MT models.

Progress:
- Manually translate (with PE) to Czech & German and see how well the algorithm works to translation to English



## Random notes

Decoder stacktrace:
- `model.translator` of `GeneratorHubInterface` in `fairseq/hub_utils.py` calls `sample`
- `sample` of `GeneratorHubInterface` in `fairseq/hub_utils.py` calls `generate`
- `generate` of `GeneratorHubInterface` in `fairseq/hub_utils.py` calls `self.task.build_generator`
- `build_generator` of `FairseqTask` in `fairseq/tasks/fairseq_task.py` builds `search.BeamSearch`
- `step` of `BeamSearch` in `fairseq/search.py` does the search

Pipeline:
- `translate_synth.py`
- `translate_contrast.py`