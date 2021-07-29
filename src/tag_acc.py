import csv
import json
import argparse
import json
import logging
import tqdm
import os
import pickle
import sys
import traceback
import csv

import torch
from pathlib import Path



def main(args):
    

    a = []
    # with open('./models/seq_tag/predict-valid-'+args.epoch+'.csv', newline='') as f:
    with open('output_seq_tag_', newline='') as f:
        rows = csv.reader(f)
        for idx, row in enumerate(rows):
            if idx != 0:
                a.append(row[1])

    with open('./data/slot/eval.json') as f:
        valid = json.load(f)

    acc = 0
    n = len(a)
    for index, val in enumerate(valid):
        a[index] = a[index].split()
        if a[index] == val['tags']:
            acc+=1

    print(acc/n)


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--epoch', type=str, default="10")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    main(args)