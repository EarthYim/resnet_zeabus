#!/usr/bin/env python3

"""
V0.1
WORKS ONLY WITH PYTHON3

-> For Intel NUC !NOT nVidia Jetson

Requirement:
    - OpenCV
    - Fizyr's Keras Implentation of Retinanet -> https://github.com/fizyr/keras-retinanet/
    - Numpy

March 10, 2020
Zeabus Vision
"""

import cv2 as cv
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, Image
from keras_retinanet.utils.image import preprocess_image, resize_image, read_image_bgr
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet import models

CAMERA_TOPIC = "/vision/front/image_rect_color/compressed" #depends on the current camera topic
model = models.load_model("./model/something.h5", backbone_name='resnet50') #path to model

def make_prediction(image):
    
    #set id for labeling
    id = {0:'gate', 1:'flare'}

    draw = image.copy()

    #preprocess image for model
    image = preprocess_image(image)
    (image, scale) = resize_image(image)
    image = np.expand_dims(image, axis=0)

    #make prediction
    (boxes, scores, labels) = model.predict_on_batch(image)

    #filtering and drawing boxes
    box_out = []
    label_out = []
    for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
        
        #threshold score can be optimize for optimum range of results
        if score < 0.3:
            continue

        box /= scale
        box = box.astype("int")
        
        box_out.append(box)
        label_out.append(id[label])
        
        draw_box(draw, box, color=(0,0,255))
        caption = "{} {:.3f}".format(id[label], score) #output label
        draw_caption(draw, box, caption)
    
    return label_out, box_out, draw 


def imageCallback(msg):
    global bgr
    arr = np.fromstring(msg.data, np.uint8)
    bgr = cv.resize(cv.imdecode(arr, 1), (0, 0), fx=1, fy=1)
    label, boxes, img = make_prediction(bgr)
    # box format -> (x, y, x+w, y+h)


def main():
    rospy.init_node('vision_ml', anonymous=True)
    if not rospy.is_shutdown():
        rospy.Subscriber(CAMERA_TOPIC, CompressedImage, imageCallback, queue_size=3)
        rospy.spin()

    else:
        print("Shutting down vision_ml")

if __name__ == '__main__':
    main()

        
