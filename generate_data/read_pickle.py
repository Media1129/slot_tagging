import pickle




augment_dishname_save_pickle = "./augment_data/dishname_list.pkl"
augment_ingredient_save_pickle = "./augment_data/ingredient_list.pkl"


# load dishname
with open(augment_dishname_save_pickle, 'rb') as f:
    augment_dishname_list = pickle.load(f)

# load ingredient
with open(augment_ingredient_save_pickle, 'rb') as f:
    augment_ingredient_list = pickle.load(f)


ingredient_len_statistic =  [0,0,0,0,0,0,0,0,0,0]
for ingredient in augment_ingredient_list:
    i_len = len(ingredient.split())
    ingredient_len_statistic[i_len]+=1
    # print("len: {} ingredient: {}".format(i_len, ingredient))
print(ingredient_len_statistic) # [1, 372, 160, 52, 16, 11, 5, 6, 1, 0]


dishname_len_statistic = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for dishname in augment_dishname_list:
    i_len = len(dishname.split())

    dishname_len_statistic[i_len]+=1
    # print("len: {} dishname: {}".format(i_len, dishname))
print(dishname_len_statistic) # [0, 476, 7068, 13699, 10248, 6099, 3292, 1610, 863, 392, 214, 66, 28, 14, 4, 0, 1, 0, 0, 1, 0]


