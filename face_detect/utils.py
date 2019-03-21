from io import BytesIO
from PIL import Image

import tensorflow as tf
import numpy as np

import base64
import json

def decode_img(data):
    return Image.open(BytesIO(base64.b64decode(data)))

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

        image_np = image #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        return (boxes, scores, classes, num_detections)

def init_model(base_path):
    model_path = "frozen_inference_graph_face.pb"
    return TensoflowFaceDector(base_path+model_path)

def model_predict(data, model):
    
    img = decode_img(data)
    
    detections = model.run(img)    
    
    face_list = []
    for i in range(0, int(detections[-1][0])):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[1][0,i]
        
        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence > 0.3:
            # compute the (x, y)-coordinates of the bounding box for the object
            box =  detections[0][0,i] 
            (ymin, xmin, ymax, xmax) = box

            face_json = {'confidence': float(confidence),
                         'box': [float(xmin), float(ymin), float(xmax), float(ymax)]}
            
            face_list.append(face_json)

    
    return json.dumps(face_list)


    


