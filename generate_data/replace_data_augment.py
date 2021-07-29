import json
import csv
import random
import copy


# file path
train_slot_file = "./data/slot/train.json"

aug_sen_file = "./data/aug_sen.txt"
aug_train_slot_file = "./data/slot/train_aug.json"


# load data/slot/*.json file 
with open(train_slot_file, 'r') as f:
    train_slot = json.load(f)


ingredients = []
dishnames = []

for sen in train_slot:
    for index in range(len(sen['tags'])):
        if sen['tags'][index] == "ingredient":
            ingredients.append(sen['tokens'][index])
        elif sen['tags'][index] == "dishname":
            dishnames.append(sen['tokens'][index])

print("Ingredient list: {}".format(ingredients))
print("Dishname list: {}".format(dishnames))

aug_sens = []
ori_ingredient = ""
aug_train_slot = []

for sen in train_slot:
    aug_sens.append(' '.join(sen['tokens']))
    copy_sen = copy.deepcopy(sen)
    aug_train_slot.append(copy_sen)
    for index in range(len(sen['tags'])):
        if sen['tags'][index] == "ingredient":
            ori_ingredient = sen['tokens'][index]
            # print("ori_ingredient = {}".format(ori_ingredient))
            for ingredient in ingredients:
                if ingredient != ori_ingredient:
                    sen['tokens'][index] = ingredient
                    aug_sens.append(' '.join(sen['tokens']))
                    copy_sen = copy.deepcopy(sen)
                    aug_train_slot.append(copy_sen)

        elif sen['tags'][index] == "dishname":
            ori_dishname = sen['tokens'][index]
            # print("ori_dishname = {}".format(ori_dishname))
            for dishname in dishnames:
                if dishname != ori_dishname:
                    sen['tokens'][index] = dishname
                    aug_sens.append(' '.join(sen['tokens']))
                    copy_sen = copy.deepcopy(sen)
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

# with open(train_save_file, 'w') as f:
#     json.dump(train_schema, f, indent=2)


    


    


# sents = []
# for sample in train_slot:
#     tokens = ""
#     for idx, token in enumerate(sample['tokens']):
#         if idx != len(sample['tokens'])-1:
#             tokens+=token+" "
#         else:
#             tokens+=token
#     sents.append(tokens)
#     # print(tokens)

# with open(train_aug_file, 'w') as f:
#     for idx, sent in enumerate(sents):
#         if idx%2==0:
#             f.write('0\t'+sent+'\n')
#         else:
#             f.write('1\t'+sent+'\n')
