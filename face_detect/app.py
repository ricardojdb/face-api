from flask import Flask, request

import tensorflow as tf
import numpy as np

import utils

app = Flask(__name__)

face_detector = utils.FaceDetector("models/")

@app.route('/predict/',methods=['GET','POST'])
def predict():

	data = request.get_data()
	output = face_detector.get_predict(data)

	return output	

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=7000)