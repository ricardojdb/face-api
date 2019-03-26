from threading import Thread
import numpy as np
import traceback
import time
import six
import sys
import cv2
import os

# Set the color for the sentiment bars
colors = {'anger':(0,0,190), 'disgust':(0,184,113),'fear': (98,24,91), 
          'happiness':(8, 154, 255), 'sadness':(231,217,0), 'surprise':(0, 253, 255), 
          'neutral':(200,200,200), 'contempt':(200,200,200)}

classes = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

def get_wide_box(w, h, xmin, ymin, xmax, ymax):
    """
    Expands the boundary face box
    """
    xmin_wide = max(xmin-(xmax-xmin)//4, 0)
    ymin_wide = max(ymin-(ymax-ymin)//4, 0)
    xmax_wide = min(xmax+(xmax-xmin)//6, w-1)
    ymax_wide = min(ymax+(ymax-ymin)//6, h-1)
    return xmin_wide, ymin_wide, xmax_wide, ymax_wide

def weighted_average(Vdw, dw, beta):
    Vdw = np.asarray(Vdw)
    dw = np.asarray(dw)
    return beta * Vdw + (1-beta) * dw

def draw_box(image, label, gender, age, scores, classes, colors, box):
    """ 
    Draws a Bounding Box over a Face.
    Args:
        image (narray): the image containng the face
        label (str): the label that goes on top of the box
        scores (narray): Facial expression prediciton scores 
        [xmin,ymin,xmax,ymax] (int): Bounding box coordinates
    Return:
        result_image (narray): edited image
    """
    xmin, ymin, xmax, ymax = np.array(box, dtype=int)
    
    img = np.copy(image)
    output = np.copy(image)

    fontType = cv2.FONT_HERSHEY_DUPLEX

    fontScale_box = 0.5
    thickness_box = 1
    fontScale = 0.4
    thickness = 1
    
    label = "{} | {}, {}".format(label, gender, int(age))
    
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fontScale_box, thickness_box)
    center = (xmin+5, ymin - text_size[0][1]//2)
    pt2 = (xmin + text_size[0][0] +10, ymin)
    
    box_color = (180,0,0)
    # Rectangle around text
    cv2.rectangle(img, (xmin, ymin - text_size[0][1]*2), pt2, box_color, 2)
    cv2.rectangle(img, (xmin, ymin - text_size[0][1]*2), pt2, box_color, -1)
    # Face rectangle
    cv2.rectangle(img,(xmin,ymin),(xmax,ymax),box_color,2)

    img = cv2.addWeighted(img, alpha, output, 1-alpha,0,output)
    # Draw text
    cv2.putText(img, label, center, cv2.FONT_HERSHEY_DUPLEX, fontScale_box, (255, 255, 255), thickness_box)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for i, score in enumerate(scores):
        bar_label = classes[i] + ' {}%'.format(int(scores[i]*100))
        bar_text_size = cv2.getTextSize(bar_label, fontType, fontScale, thickness)
        # Emotion bars
        bar_x = xmax + 20
        bar_y = ymin + bar_text_size[0][1] + (i*bar_text_size[0][1]*3)
        bar_w = int(70 * score)
        
        bar_xmax = min(bar_x + bar_text_size[0][0], img.shape[1]-1)
        bar_ymax = min(bar_y + bar_text_size[0][1], img.shape[0]-1)

        bar_xmin = min(bar_x, img.shape[1]-1)
        bar_ymin = min(bar_y, img.shape[0]-1)


        font_color = (255,255,255)

        if np.mean(gray[bar_ymin:bar_ymax,bar_xmin:bar_xmax]) > 100: font_color = (0,0,0)

        cv2.putText(img, bar_label, (bar_x, bar_y), fontType, fontScale, font_color, thickness)
        cv2.line(img, (bar_x+5, bar_y+ (bar_text_size[0][1])),(bar_x+5+bar_w, bar_y + (bar_text_size[0][1])), colors[classes[i]],int(fontScale*20))
    
    return img


class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame from the stream
        self.stream = cv2.VideoCapture(src)
        # Change depending on the resolution of the camera
        # self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) 
        # self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.h = self.stream.get(4)
        self.w =self.stream.get(3)
        (self.grabbed, self.frame) = self.stream.read()

        self.data_list = None
        # initialize the variable used to indicate if the thread should be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        self.stopped = False
        self.thread = Thread(target=self.update, name='camera:0', args=())
        self.thread.start()
        return self
 
    def update(self):
        cv2.namedWindow('final_image', cv2.WINDOW_NORMAL)
        # keep looping infinitely until the thread is stopped
        while True:
            try:
                # if the thread indicator variable is set, stop the thread
                if self.stopped:
                    self.stream.release()
                    cv2.destroyAllWindows()
                    return

                # otherwise, read the next frame from the stream
                (self.grabbed, self.frame) = self.stream.read()
                self.h = self.stream.get(4)
                self.w =self.stream.get(3)
                img = np.copy(self.frame)
                
                if not self.grabbed:
                    print('No frames')
                    self.stop()
                    self.stream.release()
                    cv2.destroyAllWindows()
                    return

                if self.data_list != None:
                    for data in self.data_list:
                        scores_dic = {c:s*100 for c,s in zip(classes, data[3:11])}
                        
                        img = draw_box(img, data[0], data[1], data[2], 
                                        data[3:11], classes, colors, data[-2])

                cv2.imshow('final_image',img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stream.release()
                    cv2.destroyAllWindows()
                    self.stop()
                    return
                
            except Exception as e:
                traceback.print_exc(file=sys.stdout)
                return
 
    def read(self):
        # return the frame most recently read
        return (self.grabbed, self.frame)
 
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True