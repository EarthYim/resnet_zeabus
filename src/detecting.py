import cv2 as cv
import numpy as np
from keras_retinanet.models import load_model
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import resize_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet import models

id = {0:'flare_yellow', 1:'gate', 2:'flare_red'}

#MODEL HERE
model = models.load_model("./model/c91.h5", backbone_name='resnet50')

#INPUT HERE
cap = cv.VideoCapture("path")

while True:
    ret, image = cap.read()
    draw = image.copy()

    image = preprocess_image(image)
    (image, scale) = resize_image(image)
    image = np.expand_dims(image, axis=0)

    (boxes, scores, labels) = model.predict_on_batch(image)
    boxes /= scale

    for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
        
        #score needs to be calibrating for finer results 
        #red flare performed the poorest >> less score
        if int(label) == 2 or int(label) == 1 and score < 0:
            continue
        
        #raise score for more accurate labels
        elif score < 0.95:
            continue
        
        #plotting reactangle and label
        box = box.astype("int")
        x, y, x_max, y_max = box[0], box[1], box[2], box[3] #output x, y, xmax, ymax
        draw_box(draw, box, color=(0,0,255))
        caption = "{} {:.3f}".format(id[label], score) #output label
        draw_caption(draw, box, caption)

    cv.imshow('output', draw)
    cv.waitKey(0)

cap.release()
cv.destroyAllWindows()



