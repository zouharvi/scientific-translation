#!/usr/bin/env python3

import sys
sys.path.append("src")
import utils
import collections
import numpy as np

data = utils.json_readl("computed/gridsearch.jsonl")

for metric in ["orig", "self", "ref", "self+ref", "oracle"]:
    values = [line[metric] for line in data]
    print(f"{metric:>10}: {min(values):.2f} {max(values):.2f}")

print()
for kwparam in ["step_subtract_frequency", "strategy", "coefficient"]:
    data_local = collections.defaultdict(list)

    for line in data:
        key = line["kwparams"][kwparam]
        data_local[key].append(line["ref"])

    for key, values in data_local.items():
        print(f"{kwparam}/{key:>7}: {np.average(values):.2f} ({max(values):.2f})")
    
    print()


# Notes:
# linear > ratio
# 0.05 optimal coefficient
# step_subtract_frequency 5 vastly better

#       orig: 50.81 50.81
#       self: 49.78 51.21
#        ref: 48.81 50.66
#   self+ref: 49.14 50.80
#     oracle: 51.01 54.15

# step_subtract_frequency/      5: 49.76 (50.66)
# step_subtract_frequency/      4: 49.71 (49.96)
# step_subtract_frequency/      3: 49.74 (49.93)
# step_subtract_frequency/      2: 49.60 (49.81)
# step_subtract_frequency/      1: 49.32 (49.63)
# step_subtract_frequency/      6: 49.26 (49.31)
# step_subtract_frequency/      7: 49.34 (49.39)
# step_subtract_frequency/      8: 49.46 (49.52)
# step_subtract_frequency/      9: 49.56 (49.60)
# step_subtract_frequency/     10: 49.63 (49.66)

# strategy/ linear: 49.60 (50.66)
# strategy/  ratio: 49.47 (49.80)

# coefficient/   0.01: 49.53 (50.63)
# coefficient/   0.05: 49.56 (50.66)
# coefficient/    0.1: 49.58 (50.50)
# coefficient/    0.5: 49.57 (50.11)
# coefficient/      1: 49.54 (49.81)
# coefficient/      5: 49.44 (49.80)