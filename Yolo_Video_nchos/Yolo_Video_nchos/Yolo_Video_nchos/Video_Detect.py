
import cv2
from PIL import Image, ImageDraw, ImageFont
import colorsys
import imghdr
import os
import random
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from utils import *
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from keras_yolo import yolo_head, yolo_boxes_to_corners


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    box_scores =  box_confidence * box_class_probs # (19,19,5,80)
    box_classes = K.argmax(box_scores , axis = -1) #(19,19,5,1)
    box_class_scores = K.max(box_scores , axis =-1) #(19,19,5,1)
    filtering_mask = box_class_scores > threshold
    scores = tf.boolean_mask(box_class_scores ,filtering_mask ) # ( ? 1 )
    boxes = tf.boolean_mask(boxes , filtering_mask) # ( ? 4 )
    classes =  tf.boolean_mask(box_classes ,filtering_mask ) # (? 1)
    return scores, boxes, classes

def iou(box1, box2):
    xi1 = tf.maximum( box1[0] , box2[0] ) # 2
    yi1 = tf.maximum( box1[1] , box2[1] ) # 2
    xi2 = tf.minimum( box1[2] , box2[2] ) # 3
    yi2 = tf.minimum( box1[3] , box2[3] ) # 3
    inter_area = tf.maximum( (xi2 - xi1) , 0) * tf.maximum( ( yi2 - yi1) , 0 )
    box1_area = ( box1[2] - box1[0] ) * ( box1[3] - box1[1] )
    box2_area = ( box2[2] - box2[0] ) * ( box2[3] - box2[1] )
    union_area = box1_area + box2_area  - inter_area
    iou = inter_area / union_area
    return iou

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    nms_indices = tf.image.non_max_suppression( boxes , scores , max_boxes , iou_threshold)
    scores  = tf.gather( scores, nms_indices )
    boxes   = tf.gather( boxes , nms_indices )
    classes = tf.gather( classes , nms_indices )
    return scores, boxes, classes

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs # yolo output is tuple
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    scores, boxes, classes = yolo_filter_boxes(box_confidence , boxes , box_class_probs) # shape is flattened (? , 1 ) ( ? 4) ( ? , 1)
    boxes = scale_boxes(boxes, image_shape)
    scores, boxes, classes = yolo_non_max_suppression( scores, boxes, classes ) # shape is (? , 1 ) ( ? 4) ( ? , 1)
    return scores, boxes, classes

sess = K.get_session()

class_names = read_classes("model_data\\coco_classes.txt")
anchors = read_anchors("model_data\\yolo_anchors.txt")

yolo_model = load_model("model_data\\yolo.h5")

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))





image_shape = None

vcap = cv2.VideoCapture('vtest.avi') 
vwidth = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
vheight = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
image_shape = (vheight , vwidth )
vcap.release()
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)


def predict(sess, image_file):
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))
    out_scores, out_boxes, out_classes =\
    sess.run([scores, boxes, classes] ,\
             feed_dict = {yolo_model.input : image_data })
    #print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    colors = generate_colors(class_names)
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    image.save(os.path.join("out", image_file), quality=90)
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    
    return out_scores, out_boxes, out_classes




def preprocess_Videoimage(img, model_image_size):
    resized_image = cv2.resize( img ,model_image_size , interpolation = cv2.INTER_CUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return img, image_data

def draw_Videoboxes(srcImg, out_scores, out_boxes, out_classes, class_names, colors):
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * srcImg.size[1] + 0.5).astype('int32'))
    thickness = (srcImg.size[0] + srcImg.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(srcImg)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(srcImg.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(srcImg.size[0], np.floor(right + 0.5).astype('int32'))
        #print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

def video_predict(sess):
   
    cap = cv2.VideoCapture('vtest.avi')

    while(cap.isOpened()):

        ret, frame = cap.read()
        img, image_data = preprocess_Videoimage( frame , model_image_size = (608, 608))
        out_scores, out_boxes, out_classes =\
             sess.run([scores, boxes, classes] ,\
             feed_dict = {yolo_model.input : image_data })

        colors = generate_colors(class_names)

        cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)

        draw_Videoboxes(pil_im, out_scores, out_boxes, out_classes, class_names, colors)

        lastimg = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        cv2.imshow('frame',lastimg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

video_predict(sess)





