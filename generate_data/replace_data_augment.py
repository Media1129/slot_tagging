import json
import csv
import random
import copy


# input file path
train_slot_file = "./data/slot/train.json"

# output file path
aug_sen_file = "./data/aug_sen_v2.txt"
aug_train_slot_file = "./data/slot/train_aug_v2.json"


# load data/slot/*.json file 
with open(train_slot_file, 'r') as f:
    train_slot = json.load(f)


ingredients = []
dishnames = []

for sen in train_slot:
    for index in range(len(sen['tags'])):
        if sen['tags'][index] == "ingredient":
            ingredients.append(sen['tokens'][index].strip('.'))
        elif sen['tags'][index] == "dishname":
            dishnames.append(sen['tokens'][index].strip('.'))

print("Ingredient list: {}".format(ingredients))
print("Dishname list: {}".format(dishnames))

aug_sens = []
ori_ingredient = ""
aug_train_slot = []

for sen in train_slot:
    # print(sen) # {'tokens': ['Help', 'me', 'cook', 'something', 'with', 'strawberries'], 'tags': ['O', 'O', 'O', 'O', 'O', 'ingredient'], 'id': 'train-0'}
    # aug_sens.append(' '.join(sen['tokens']))
    # print(aug_sens) # ['Help me cook something with strawberries']
    copy_sen = copy.deepcopy(sen)
    aug_sens.append(' '.join(copy_sen['tokens']))
    aug_train_slot.append(copy_sen)

    for index in range(len(sen['tags'])):
        if sen['tags'][index] == "dishname":
            ori_dishname = sen['tokens'][index]
            for dishname in dishnames:
                if dishname != ori_dishname:
                    # sen['tokens'][index] = dishname
                    # aug_sens.append(' '.join(sen['tokens']))
                    copy_sen = copy.deepcopy(sen)
                    copy_sen['tokens'][index] = dishname
                    aug_sens.append(' '.join(copy_sen['tokens']))
                    aug_train_slot.append(copy_sen)

        elif sen['tags'][index] == "ingredient":
            ori_ingredient = sen['tokens'][index]
            for ingredient in ingredients:
                if ingredient != ori_ingredient:
                    # sen['tokens'][index] = ingredient
                    # aug_sens.append(' '.join(sen['tokens']))
                    copy_sen = copy.deepcopy(sen)
                    copy_sen['tokens'][index] = ingredient
                    aug_sens.append(' '.join(copy_sen['tokens']))
                    aug_train_slot.append(copy_sen)

        

# save augment sentence
with open(aug_sen_file, 'w') as f:
    for idx, sent in enumerate(aug_sens):
        f.write(sent+'\n')


for idx, aug_train in enumerate(aug_train_slot):
    id_value = "train-{}".format(idx)
    aug_train['id'] = id_value
# save augment train slot json
with open(aug_train_slot_file, 'w') as f:
    json.dump(aug_train_slot, f, indent=2)