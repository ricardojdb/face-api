from flask import Flask, request
import numpy as np
import base64
import json
import os
from PIL import Image
import utils

app = Flask(__name__)

face_features = utils.FaceFeatures("models/")

@app.route('/predict/',methods=['GET','POST'])
def predict():

    data = request.get_data()
    output = face_features.model_predict(data)
    
    return output	

if __name__ == "__main__":

	# app.run(debug=True)
	app.run(host='0.0.0.0', port=7000, use_reloader=False)