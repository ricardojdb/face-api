from io import BytesIO
from PIL import Image

import tensorflow as tf
import numpy as np

import base64
import json
import os


class FaceDetector(object):
    """
    Initializes and handles de face detection model in tensorflow
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.init_model()

    def decode_img(self, encoded_data):
        """Decodes the encoded data comming from a request.

        Args:
            encoded_data (base64): data comming from the HTTP request.

        Returns:
            array: Data decoded into a usable format.

        """
        return Image.open(BytesIO(base64.b64decode(encoded_data)))

    def init_model(self):
        """Initializes the machine learning model.

        Returns:
            model (object): Loaded pre-trained model used
                to make predictions.

        """
        model_name = "frozen_inference_graph_face.pb"
        return TensoflowFaceDector(os.path.join(self.model_path, model_name))

    def model_predict(self, encoded_data):
        """Decodes and preprocess the data, uses the
        pretrained model to make predictions and
        returns a well formatted json output.

        Args
            encoded_data (base64): data comming from the HTTP request.

        Returns:
            json: A response that contains the output from
                the pre-trained model.
        """
        img = self.decode_img(encoded_data)

        detections = self.model.run(img)

        face_list = []
        for i in range(0, int(detections[-1][0])):
            # extract the confidence (i.e., probability
            # associated with the prediction
            confidence = detections[1][0, i]

            # filter out weak detections by ensuring the `confidence`
            # is greater than the minimum confidence
            if confidence > 0.3:
                # compute the (x, y)-coordinates of
                # the bounding box for the object
                box = detections[0][0, i]
                (ymin, xmin, ymax, xmax) = box

                face_json = {'confidence': float(confidence),
                             'xmin': float(xmin),
                             'ymin': float(ymin),
                             'xmax': float(xmax),
                             'ymax': float(ymax)}

                face_list.append(face_json)

        return json.dumps(face_list)


class TensoflowFaceDector(object):
    def __init__(self, PATH_TO_CKPT):
        """Tensorflow detector
        """

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True

    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """

        image_np = image  # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later
        # in order to prepare the result image with boxes and labels on it.
        # Expand dimensions since the model expects
        # images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')
        # Each box represents a part of the image where
        # a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name(
            'detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name(
            'detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        return (boxes, scores, classes, num_detections)
