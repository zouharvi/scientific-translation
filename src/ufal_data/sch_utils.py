#!/usr/bin/env python3
import random
import time
from urllib.parse import quote
import urllib.request
import json


def paper_search(query, limit=10):
    query = quote(query)
    url_base = f"https://api.semanticscholar.org/graph/v1/paper/search?limit={limit}&fields=title,authors,url&query="
    url = url_base + query
    data = urllib.request.urlopen(url)
    data = data.read().decode("utf-8")
    return json.loads(data)["data"]


def paper_references(paper_id, limit=10):
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references?fields=title,url&limit={limit}"
    data = urllib.request.urlopen(url)
    data = data.read().decode("utf-8")
    return [x["citedPaper"] for x in json.loads(data)["data"]]


def is_same_paper(a, b):
    a = "".join(a.lower().split())
    b = "".join(b.lower().split())
    return a == b


def crawl_until_exhausted(papers, paper_ids=set(), total=30, depth=0):
    # we shouldn't go too far down
    if len(papers) == 0 or depth >= 3:
        return []

    papers = [x for x in papers if x["paperId"] is not None]
    paper_ids |= {x["paperId"] for x in papers}
    papers_new = []
    # print("Crawling", len(papers), "papers")
    for paper in papers:
        refs = paper_references(paper["paperId"], limit=total)
        time.sleep(3)

        for new_x in refs:
            if new_x["paperId"] in paper_ids:
                continue
            papers_new.append(new_x)
            paper_ids.add(new_x["paperId"])

        # we may be done
        if len(papers_new) + len(papers) >= total:
            break

    total_papers = len(papers + papers_new)
    if total_papers < total:
        # not += because the result contains itself
        papers_new = crawl_until_exhausted(
            papers_new, paper_ids=paper_ids,
            total=total - total_papers + len(papers_new),
            depth=depth + 1,
        )

    # shuffle to not be skewed towards the first paper references
    random.shuffle(papers_new)
    return (papers + papers_new)[:total]
