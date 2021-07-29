# Recipe entity recognition task

## Generate recipe entity data
1. generate_data/data_generate.py
    + read the first_turn.csv[split/shuffle] and data/schema/*.json data[format] ->  write data/input/*.json
    + first_turn.csv: record the first turn sentence data
    + data/schema/*.json: record the training data json format
    + data/input/*.json: the training data except the tag not correct
2. data/slot/*.json
    + label the data/input/*.json to data/slot/*.json
    + save the ground truth data
3. generate_data/replace_data_augment.py
    + read data/slot/train.json
    + store the dishname and ingredient list
    + write the augment data to data/slot/train_aug.json
    + 
4. generate_data/generate_ing_dish.py
    + read recipes_with_nutritional_info.json 
    + build dishname_list and ingredient_list
    + save dishname_list and ingredient_list to pickle file

5. run script
```bash=
python generate_data/data_generate.py
python generate_data/replace_data_augment.py
```

## Train bert on recipe entity slot tag
```bash=
python src/preprocess_seq_tag.py datasets
python src/train_seq_tag.py
```
