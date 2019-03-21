from models.resnet50_ft_dag import resnet50_ft_dag
from flask import Flask, request
from io import BytesIO
from PIL import Image

import numpy as np
import torchvision
import torch
import base64
import json
import os

import utils

app = Flask(__name__)

model = utils.load_model("models/resnet50_ft_dag.pth")
database = utils.init_dataset("images/", model)

@app.route('/predict/',methods=['GET','POST'])
def predict():

	data = request.get_data()
	output = utils.model_predict(data, model, database)

	return output	

if __name__ == "__main__":

	# app.run(debug=True)
	app.run(host='0.0.0.0', port=7000)