import streamlit as st
import os
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR = pathlib.Path(__file__).parent

feature_list = np.array(pickle.load(open(BASE_DIR / 'embeddings.pkl', 'rb')))
filenames = pickle.load(open(BASE_DIR / 'filenames.pkl','rb'))

def load_model():
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    return tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

model = load_model()

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        uploads_path = BASE_DIR / 'uploads'
        uploads_path.mkdir(exist_ok=True)
        file_path = uploads_path / uploaded_file.name
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# file upload then save
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        display_image = Image.open(file_path)
        st.image(display_image, caption="Uploaded Image", use_column_width=True)

        # Feature extraction
        features = feature_extraction(file_path, model)

        # Get recommendations
        indices = recommend(features, feature_list)

        # Show recommendations
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            col.image(filenames[indices[0][idx]])
    else:
        st.error("Some error occurred in file upload")