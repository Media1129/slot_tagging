import json
import csv
import random
import copy


# file path
test_slot_file = "./data/slot/test.json"
test_result_file = "output_seq_tag"
test_analyze_file = "./data/analyze.txt"

with open(test_slot_file, 'r') as f:
    test_slot = json.load(f)

test_result = []
with open(test_result_file, 'r') as f:
    for line in f:
        test_result.append(line)




        

# save augment sentence
with open(test_analyze_file, 'w') as f:
    for idx, sent in enumerate(test_slot):
        f.write(' '.join(sent['tokens'])+'\n')
        f.write('ground truth: '+' '.join(sent['tags'])+'\n')
        f.write(test_result[idx+1]+'\n')
        

