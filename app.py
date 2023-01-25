from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import keras.utils as image
import numpy as np
from keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/colon_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


import keras.utils as image
import numpy as np
from keras.models import load_model

from PIL import Image

def model_predict(img_path, model):
    # class label mapping
    labels = {0: 'Type = Cancerous', 1: 'Type = Benign'}

    # Preprocess the new image
    img = image.load_img(img_path, target_size=(224, 224)) # resize the image to the same size as the training images
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # add an extra dimension for the batch size
    img_array /= 255. # normalize the image

    # Make a prediction
    prediction = model.predict(img_array)

    return prediction


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # retrieve the corresponding class label using the index
        class_label = np.argmax(preds)

        # find the index of the highest value in the prediction array
        class_index = np.argmax(preds)

        labels = {0: 'Type = Cancerous', 1: 'Type = Benign'}
        # retrieve the corresponding class label using the index
        class_label = labels[class_index]

        return class_label
    return None


if __name__ == '__main__':
    app.run(debug=True)
