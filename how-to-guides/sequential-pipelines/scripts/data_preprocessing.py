from sklearn.datasets import fetch_lfw_people
from utils import *

# Download dataset
dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Preprocess dataset
data_transform = Preprocessing(dataset=dataset)
data_transform.scale_features()
data_transform.describe()
