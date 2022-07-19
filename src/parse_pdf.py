#!/usr/bin/env python3

from tika import parser
import argparse
import json
import pathlib

args = argparse.ArgumentParser()
args.add_argument("-f", "--files", nargs="+")
args = args.parse_args()

for f in args.files:
    path_json = f"corpus/{pathlib.Path(f).stem}.json"
    if pathlib.Path(path_json).is_file():
        print(f"Skipping {f}")
        continue

    print(f"Processing {f}")
    rawText = parser.from_file(f)

    with open(path_json, "w") as f_json:
        json.dump({"paper_en": rawText["content"]}, f_json)