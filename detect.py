#!/usr/bin/env python
import cv2 as cv
import numpy as np
import time
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image
from keras_retinanet.models import load_model
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import resize_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet import models

CAMERA_TOPIC = "/vision/front/image_rect_color/compressed"

def make_prediction(image):
    #load id and model
    id = {0:'gate', 1:'flare'}
    model = models.load_model("./model/c91.h5", backbone_name='resnet50')

    draw = image.copy()

    #preprocess image for model
    image = preprocess_image(image)
    (image, scale) = resize_image(image)
    image = np.expand_dims(image, axis=0)

    #make prediction
    (boxes, scores, labels) = model.predict_on_batch(image)

    #filtering and drawing boxes
    for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
        if score < 0.5:
            continue

        box /= scale
        box = box.astype("int")
        x, y, x_max, y_max = box[0], box[1], box[2], box[3] #output x, y, xmax, ymax
        draw_box(draw, box, color=(0,0,255))
        caption = "{} {:.3f}".format(id[label], score) #output label
        draw_caption(draw, box, caption)

        cv.imwrite("{}.png".format(rospy.get_time()), draw)



def imageCallback(msg):
    cv_image = CvBridge.compressed_imgmsg_to_cv2(msg)
    make_prediction(cv_image)
    

def main():
    rospy.init_node('vision_ml', anonymous=True)
    if not rospy.is_shutdown():
        rospy.Subscriber(CAMERA_TOPIC, CompressedImage, imageCallback, queue_size=3)
        rospy.spin()

    else:
        print("Shutting down ROS Image feature detector module")s

if __name__ == '__main__':
    main()

        