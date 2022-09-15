#!/usr/bin/env python3

import json
from bs4 import BeautifulSoup
import semanticscholar
import collections
from tqdm import tqdm
import sch_utils
import time

with open("data_raw/publications.xml", "r") as f:
    data_bs = f.read()
    data_bs = BeautifulSoup(data_bs, "xml")

data = []
data_lang = collections.Counter()

for record in tqdm(list(data_bs.find_all("Record"))):
    lang = record.find("Field", attrs={"Name": "Language"}).text
    abstract_cs = record.find("Field", attrs={"Name": "CzechAbstract"}).text
    abstract_en = record.find("Field", attrs={"Name": "EnglishAbstract"}).text
    abstract_orig = record.find(
        "Field", attrs={"Name": "OriginalLanguageAbstract"}
    ).text
    title_cs = record.find("Field", attrs={"Name": "CzechTitle"}).text
    title_en = record.find("Field", attrs={"Name": "EnglishTitle"}).text
    title_orig = record.find(
        "Field", attrs={"Name": "Title"}
    ).text

    # log distribution of languages
    data_lang.update({lang})
    if lang not in {"eng", "cze"}:
        continue

    # if the title is unavailable, set it to the original
    if title_en is None and lang == "eng":
        title_en = title_orig
    elif title_cs is None and lang == "cze":
        title_cs = title_orig

    if len(title_en) == 0:
        continue

    # sometimes the fields are not filled properly
    if len(abstract_cs) == 0 and lang == "cze":
        abstract_cs = abstract_orig
    elif len(abstract_en) == 0 and lang == "eng":
        abstract_en = abstract_orig
    elif len(abstract_en) == 0 or len(abstract_cs) == 0:
        # this happens only 7 times
        continue

    sch = semanticscholar.SemanticScholar(timeout=2)
    print("Searching for:", title_en)
    paper_data = sch_utils.paper_search(title_en)
    time.sleep(5)
    paper_sames = [x for x in paper_data if sch_utils.is_same_paper(
        x["title"], title_en)]

    # TODO: we are not using paper_sames
    print("Using search results of", len(paper_data))
    if len(paper_data) == 0:
        continue

    papers_to_download = sch_utils.crawl_until_exhausted(paper_data)
    print(".. extended to", len(papers_to_download))
    print("=" * 5)

    data.append({
        "lang": lang,
        "title_cs": title_cs,
        "title_en": title_en,
        "related_papers": papers_to_download,
        "abstract_cs": abstract_cs,
        "abstract_en": abstract_en,
    })

print("Language distribution:", data_lang)
print("Total examples:", len(data))


with open("data_raw/publications.jsonl", "w") as f:
    f.write("\n".join([
        json.dumps(x, ensure_ascii=False)
        for x in data
    ]))
