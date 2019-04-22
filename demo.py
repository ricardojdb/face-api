from datetime import datetime, timedelta
import numpy as np
import requests
import base64
import cv2
import sys
import os

from utils import utils

def encode_img(img):
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)
    return img_str

table_dict = {}

# Start thread to capture and show the stream.
video_path = 0 
video_capture = utils.WebcamVideoStream(video_path).start()

host = "localhost"

while True:
    # Collect width and height from the stream
    h, w = int(video_capture.h), int(video_capture.w)
    # Read the current frame
    ret, image = video_capture.read()
    
    if not ret: 
        print('No frames has been grabbed') 
        break
    
    img = np.copy(image)
    
    # Call Face detection API
    try:
        detect_req = requests.get(f'http://{host}:7000/predict/', 
            params={"data":encode_img(img)}, timeout=5)
        detections = detect_req.json()
    except:
        detections = []

    data_list = []
    for face in detections:
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = face["confidence"]
        
        # filter out weak detections by ensuring the `confidence` is 
        # greater than the minimum confidence
        if confidence > 0.3:
            # compute the (x, y)-coordinates of the bounding box for the object
            box =  face["box"] * np.array([w, h, w, h])
            (xmin, ymin, xmax, ymax) = box.astype("int")

            # Fix negatives and big numbers
            xmin = max(xmin,0)
            ymin = max(ymin,0)
            xmax = min(xmax,w-1)
            ymax = min(ymax, h-1)

            # Expand the bounding box to capture the entire head not only the face
            # this improve the performance of gender, age and sentiment models
            xmin_wide, ymin_wide, xmax_wide, ymax_wide = utils.get_wide_box(
                w, h, xmin, ymin, xmax, ymax)

            # Extract Face from image with (x,y)-coordinates
            roi_color = img[max(ymin,0):min(ymax,h-1), max(xmin,0):min(xmax,w-1)]
            roi_color_wide = img[ymin_wide:ymax_wide, xmin_wide:xmax_wide]

            if not roi_color.size: continue
            
            # Call Face Features API
            try:
                agen_req = requests.get(f'http://{host}:7002/predict/', 
                    params={"data":encode_img(roi_color_wide)}, timeout=5)
                agen_predict = agen_req.json()
                gender, age = agen_predict["gender"], agen_predict["age"]
            except:
                gender, age = " ", 0
            
            # Call Face Emotion API
            try:
                emot_req = requests.get(f'http://{host}:7001/predict/', 
                    params={"data":encode_img(roi_color)}, timeout=5)            
                scores = emot_req.json()["emotions"]
            except:
                scores = [1,0,0,0,0,0,0,0]            
            
            # Facial Recognition
            try:
                recog_req = requests.get(f'http://{host}:7003/predict/', 
                    params={"data":encode_img(roi_color_wide)}, timeout=5)
                recog = recog_req.json()
                fr_score, label = recog["dist"], recog["label"]
            except:
                fr_score, label = 0, " "
                    
            time_stamp = datetime.now().strftime("%H:%M:%S")

            # Use exponentially weighted average to smooth the 
            # changes in sentiment, age and position.
            if label in table_dict:
                age_ewa = utils.weighted_average(
                    age, table_dict[label][2], beta=0.998)

                scores_ewa = utils.weighted_average(
                    scores, table_dict[label][3:11], beta=0.5)

                box_ewa = utils.weighted_average(
                    [xmin, ymin, xmax, ymax], table_dict[label][-2], beta=0.998)

                table_dict[label] = [label, gender, age_ewa, 
                *scores_ewa, box_ewa, time_stamp]
            else:
                table_dict[label] = [label, gender, age, *scores, 
                [xmin, ymin, xmax, ymax], time_stamp]

            data_list.append(table_dict[label])
            
    # Send outputs to the thread so it can be plotted on the stream.
    video_capture.data_list = data_list
        
    if video_capture.stopped:
        break

cv2.destroyAllWindows()
