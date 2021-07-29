import json
import pickle
from tqdm import tqdm


augment_data_path = "../recipes_with_nutritional_info.json"
augment_dishname_save_pickle = "./augment_data/dishname_list.pkl"
augment_ingredient_save_pickle = "./augment_data/ingredient_list.pkl"


with open(augment_data_path) as f:
    augment_data_list = json.load(f)

print("augment data len: {}".format(len(augment_data_list))) # 51235

dishname_list = []
ingredient_list = []

for augment_data in tqdm(augment_data_list):
    # add dishname
    dishname = augment_data['title']
    if dishname not in dishname_list:
        dishname_list.append(dishname)
    
    # add ingredient
    ingredient_lines = augment_data['ingredients']
    for ingredient_line in ingredient_lines:
        for ingredient in ingredient_line['text'].split(','):
            if ingredient.strip() not in ingredient_list:
                ingredient_list.append(ingredient.strip())


print(len(dishname_list)) # 44075
print(len(ingredient_list)) # 839149->624

# store data to pickle file
with open(augment_dishname_save_pickle, 'wb') as f:
    pickle.dump(dishname_list, f)

with open(augment_ingredient_save_pickle, 'wb') as f:
    pickle.dump(ingredient_list, f)



