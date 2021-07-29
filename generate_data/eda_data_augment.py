import json
import csv
import random

# file path
first_turn_file = "./data/recipe_first_turn.csv"

train_slot_file = "./data/slot/train.json"
train_aug_file = "../eda_nlp/data/aug.txt"


# save to  data/slot/*.json file 
with open(train_slot_file, 'r') as f:
    train_slot = json.load(f)
    
sents = []
for sample in train_slot:
    tokens = ""
    for idx, token in enumerate(sample['tokens']):
        if idx != len(sample['tokens'])-1:
            tokens+=token+" "
        else:
            tokens+=token
    sents.append(tokens)
    # print(tokens)

with open(train_aug_file, 'w') as f:
    for idx, sent in enumerate(sents):
        if idx%2==0:
            f.write('0\t'+sent+'\n')
        else:
            f.write('1\t'+sent+'\n')
