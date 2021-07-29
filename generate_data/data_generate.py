import json
import csv
import random

# file path
first_turn_file = "./data/recipe_first_turn.csv"

train_schema_file = "./data/schema/train.json"
dev_schema_file = "./data/schema/dev.json"
test_schema_file = "./data/schema/test.json"

train_save_file = "./data/input/train.json"
dev_save_file = "./data/input/dev.json"
test_save_file = "./data/input/test.json"




# read first turn csv file -> list
first_turn_sen = []
with open(first_turn_file) as f:
    rows = csv.reader(f)
    for idx, row in enumerate(rows):
        if idx==0:
            continue
        first_turn_sen.append(row[0])
print("first turn sen len: {}".format(len(first_turn_sen))) # 33

# shuffle the first turn list
random.shuffle(first_turn_sen)


# split first_trun_sen -> 22/6/5
train  = first_turn_sen[0:22] # 22
dev    = first_turn_sen[22:28] # 6
test   = first_turn_sen[28:33] # 5

# open data/input/*.json file 
with open(train_schema_file) as f:
    train_schema = json.load(f)

with open(dev_schema_file) as f:
    dev_schema = json.load(f)

with open(test_schema_file) as f:
    test_schema = json.load(f)


for idx, item in enumerate(train):
    train_schema[idx]['tokens'] = train[idx].split()
    train_schema[idx]['tags'] = ["O"]*len(train_schema[idx]['tokens'])

for idx, item in enumerate(dev):
    dev_schema[idx]['tokens'] = dev[idx].split()
    dev_schema[idx]['tags'] = ["O"]*len(dev_schema[idx]['tokens'])

for idx, item in enumerate(test):
    test_schema[idx]['tokens'] = test[idx].split()
    test_schema[idx]['tags'] = ["O"]*len(test_schema[idx]['tokens'])


# save to  data/slot/*.json file 
with open(train_save_file, 'w') as f:
    json.dump(train_schema, f, indent=2)
with open(dev_save_file, 'w') as f:
    json.dump(dev_schema, f, indent=2)
with open(test_save_file, 'w') as f:
    json.dump(test_schema, f, indent=2)

