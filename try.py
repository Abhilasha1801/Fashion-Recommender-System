import pickle
import numpy as np

feature_list = pickle.load(open('embeddings.pkl', 'rb'))
print(f"Type of feature_list: {type(feature_list)}")
print(f"Length of feature_list: {len(feature_list)}")

if len(feature_list) > 0:
    print(f"Shape of first feature vector: {np.array(feature_list[0]).shape}")
else:
    print("feature_list is empty!")