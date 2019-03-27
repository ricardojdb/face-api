from flask import Flask, request
import numpy as np
import torch

import utils

app = Flask(__name__)

face_recognition = utils.FaceRecognition("models/", "images/")

@app.route('/predict/',methods=['GET','POST'])
def predict():

	data = request.get_data()
	output = face_recognition.model_predict(data)

	return output	

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=7000)