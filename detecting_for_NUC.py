#!/usr/bin/env python3

"""
V0.2
    - Fix ROS messages

WORKS ONLY WITH PYTHON3

-> For Intel NUC !NOT nVidia Jetson

Requirement:
    - OpenCV
    - Fizyr's Keras Implentation of Retinanet -> https://github.com/fizyr/keras-retinanet/
    - Numpy
    - CvBridge build with Python3
March 10, 2020
Zeabus Vision
"""

import cv2 as cv
from cv_bridge import CvBridge
import numpy as np
import rospy
from time import time
from sensor_msgs.msg import CompressedImage, Image
from keras_retinanet.utils.image import preprocess_image, resize_image, read_image_bgr
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet import models

CAMERA_TOPIC = "/vision/front/image_raw/compressed" #depends on the current camera topic
model = models.load_model("../model/m12_99.h5", backbone_name='resnet50') #path to model

#declare global img 
bgr = None

def publish(img, color, subtopic):
    """
        publish picture
    """
    bridge = CvBridge()
    if img is None:
        img = np.zeros((200, 200))
        color = "gray"
    pub = rospy.Publisher('/vision/test/' + str(subtopic), Image, queue_size=10)
    if color == 'gray':
        msg = bridge.cv2_to_imgmsg(img, "mono8")
    elif color == 'bgr':
        msg = bridge.cv2_to_imgmsg(img, "bgr8")
    pub.publish(msg)

def make_prediction(image):
    
    #set id for labeling
    ID = {0:'gate', 1:'flare'}
    print('init fn')
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
        label_out.append(ID[label])
        
        draw_box(draw, box, color=(0,0,255))
        caption = "{} {:.3f}".format(ID[label], score) #output label
        draw_caption(draw, box, caption)
    
    return label_out, box_out, draw 


def imageCallback(msg):
    global bgr
    arr = np.fromstring(msg.data, np.uint8)
    bgr = cv.resize(cv.imdecode(arr, 1), (0, 0), fx=0.3, fy=0.3)
    # box format -> (x, y, x+w, y+h)


def main():
    rospy.init_node('vision_ml', anonymous=True)
    if not rospy.is_shutdown():
        rospy.Subscriber(CAMERA_TOPIC, CompressedImage, imageCallback, queue_size=3)
    
    while not rospy.is_shutdown():
        if bgr is None:
            continue
        checkpt = time()
        label_out, box_out, draw = make_prediction(bgr) # box format -> (x, y, x+w, y+h)
        print(time()-checkpt)
        print(draw.shape)
        cv.imwrite('temp.png', draw)
        publish(draw,'bgr','draw')
    rospy.spin()


if __name__ == '__main__':
    main()

        
