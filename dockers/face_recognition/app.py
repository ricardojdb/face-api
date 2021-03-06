from flask import Flask, request
from flask_cors import CORS
import numpy as np
import torch
import utils
import json

# Initialize the flask app
app = Flask(__name__)
cors = CORS(app, resources={r"/predict": {"origins": "*"}})

# Loads the given model and the image dataset
face_recognition = utils.FaceRecognition("models/", "images/")


@app.route('/update', methods=['POST'])
def update():
    # Obtain the data from the request
    req = json.loads(request.get_json())
    idx = req.get('id')
    name = req.get('name')

    status = face_recognition.update_database(idx, name)

    response = {"success": False}
    if status:
        response["success"] = True

    return json.dumps(response)


@app.route('/predict', methods=['POST'])
def predict():
    # Obtain the data from the request
    data = request.get_data()
    # Runs the model and returns the outputs in a json format
    output = face_recognition.model_predict(data)
    return output

if __name__ == "__main__":
    # Running the Flask app on the url http://0.0.0.0:7000/
    # Use 0.0.0.0 to run in any IP available
    app.run(host='0.0.0.0', port=7000)
