import cv2
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    return image

def get_dominant_colors(image, k=5):
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(image)
    colors = kmeans.cluster_centers_    
    return colors.astype(int)

def plot_colors(colors):
    plt.figure(figsize=(5, 2), title='Dominant Colors')
    plt.axis("off")
    plt.imshow([colors], aspect='auto')

image = load_image('image.jpg')
dominant_colors = get_dominant_colors(image)
plot_colors(dominant_colors)
plt.show()