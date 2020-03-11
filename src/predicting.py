from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import resize_image
from keras_retinanet import models
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help='path to pre-trained model')
ap.add_argument("-o", "--output", required=True, help="path to directory to store predictions")
args = vars(ap.parse_args())

id = {1:'gate', 0:'flare_yellow', 2:'flare_red'}

LABELS = open('test.csv').read().split("\n")
model = models.load_model(args['model'], backbone_name='resnet50')

for i in range(len(LABELS)):
    file = open(args['output'], 'a')
    lab = LABELS[i].split(',')
    image = read_image_bgr(''.join(['./src/', lab[0].split('/')[-1]]))
    
    #x = int(lab[1])
    #y = int(lab[2])
    #x_max = int(lab[3]) + 1
    #y_max = int(lab[4]) + 1

    #image = image[y:y_max, x:x_max]
    image = preprocess_image(image)
    (image, scale) = resize_image(image)
    image = np.expand_dims(image, axis=0)

    (boxes, scores, labels) = model.predict_on_batch(image)

    for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
        if score < 0.5:
            continue

        box /= scale
        file.write("{}\n".format(",".join([str(lab[0].split('/')[-1]), str(int(box[0])), str(int(box[1])), str(int(box[2])), str(int(box[3])), id[label], str(label), str(score)])))
    
    file.close()





        
