import pickle




augment_dishname_save_pickle = "./augment_data/dishname_list.pkl"
augment_ingredient_save_pickle = "./augment_data/ingredient_list.pkl"


# load dishname
with open(augment_dishname_save_pickle, 'rb') as f:
    augment_dishname_list = pickle.load(f)

# load ingredient
with open(augment_ingredient_save_pickle, 'rb') as f:
    augment_ingredient_list = pickle.load(f)

