from flask import Flask, request, render_template
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dominant_colors = get_dominant_colors(image)
    plot_colors(dominant_colors)
    plt.show()

def get_dominant_colors(image, k=5):
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)
    colors = kmeans.cluster_centers_
    return colors.astype(int)

def plot_colors(colors):
    plt.figure(figsize=(5, 2), title='Dominant Colors')
    plt.axis("off")
    plt.imshow([colors], aspect='auto')

if __name__ == '__main__':
    app.run(debug=True)