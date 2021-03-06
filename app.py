from flask import Flask
from flask import request
import tensorflow as tf
from tensorflow import keras
import h5py
from keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
from PIL import Image
from flask import render_template
app = Flask(__name__)


@app.route('/', methods=["GET","POST"])
def index():
    if(request.method=='POST'):
        model = keras.models.load_model('asl.h5')
        #model.make_predict_function()
        image_path="test_b.jpeg"
        f = request.files['the_file'].read()
        npimg = np.fromstring(f, np.uint8)
        #f= Image.open(io.BytesIO(f))
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR) # Reads image and returns np.array
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts sinto the corret colorspace (GRAY)
        img = cv2.resize(img, (320, 120)) # Reduce image size so training can be faster
        x = np.array(img, dtype="uint8")
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=4)
        array = model.predict(x)
        alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','space','delete','none']
        indexVal =np.where(array[0] == np.amax(array[0]))
        return (alphabet[indexVal[0][0]])
    return render_template('index.html')
