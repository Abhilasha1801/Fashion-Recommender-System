import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
from tqdm import tqdm

# Load the model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model = Sequential([
    base_model,
    GlobalMaxPooling2D()
])

model.trainable = False

def extract_feature(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    feature = model.predict(preprocessed_img).flatten()
    normalized_feature = feature / norm(feature)
    return normalized_feature

folder_path = "C:/Users/Abhilasha/Downloads/MCA/Python/Fashion Recommender System/images"

feature_list = []
filenames = []

for file in tqdm(os.listdir(folder_path)):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        full_path = os.path.join(folder_path, file)
        feature = extract_feature(full_path, model)
        feature_list.append(feature)
        filenames.append(full_path)

feature_list = np.array(feature_list)

pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

print(f"Saved {len(feature_list)} embeddings.")
