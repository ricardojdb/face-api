from flask import Flask, request
import tensorflow as tf
import numpy as np
import utils

# Initialize the flask app
app = Flask(__name__)

# Loads the given model
face_detector = utils.FaceDetector("models/")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    # Obtain the data from the request
    data = request.args.get('data')
    # Runs the model and returns the outputs in a json format
    output = face_detector.model_predict(data)
    return output

if __name__ == "__main__":
    # Running the Flask app on the url http://0.0.0.0:7000/
    # Use 0.0.0.0 to run in any IP available
    app.run(host='0.0.0.0', port=7000)
