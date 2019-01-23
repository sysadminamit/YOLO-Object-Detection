# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 15:57:13 2019

@author: AMIT CHAKRABORTY
"""

import cv2
import numpy as np
import time

config = 'yolov3.cfg'
weights = 'yolov3.weights'
classes_txt = 'yolov3.txt'
classes = None
#open the text file in read mode where all the classes mentioned that yolo algo can predict
with open(classes_txt, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
#loading the video file
video = cv2.VideoCapture('sealdah1.mp4')
#video = cv2.VideoCapture(0)   #if we want to use primary camera to load video from the loacl machine 
frame_width = int(video.get(3))
frame_height = int(video.get(4))
#saving the output --> for more details plz visit https://docs.opencv.org/3.1.0/dd/d43/tutorial_py_video_display.html
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
record = cv2.VideoWriter('output1.avi',fourcc, 50.0, (frame_width, frame_height))

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    if class_id > 7: # return if object is not vehicle or pedestrian
        return
    if class_id == 0:
        color = (255,255,0) # color for pedestrian
    else:
        color = (0,255,255) # color for vehicle
    label = str(classes[class_id])
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


while True:
    _, frame = video.read()
    start_T = time.time()

    
    Width = frame.shape[1]
    Height = frame.shape[0]
    
    
    scale = 0.00392

    net = cv2.dnn.readNet(weights, config)
    blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    count_vehicles = 0
    count_pedestrians = 0


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height) 
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x,y,w,h = box
        draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        
        if class_ids[i] == 0:
            count_pedestrians += 1
        if 0 < class_ids[i] <= 7:
            count_vehicles += 1

    record.write(frame)
    cv2.putText(frame, 'Detected Vehicles: ' + str(count_vehicles), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
    cv2.putText(frame, 'Detected Pedestrians: ' + str(count_pedestrians), (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)

    
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(5) & 0xFF == 32: # if press SPACE bar
        break
    print(time.time()-start_T)

video.release()
record.release()
cv2.destroyAllWindows()
