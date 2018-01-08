import os
__parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import math
import random
import sys
import json 
import numpy as np
import tensorflow as tf
#import cv2

slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

sys.path.append(__parent_dir)

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
#from notebooks import visualization
# Main image processing routine.

VOC_LABELS = {
    0: 'Background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor',
}

def vis_detections(im, classes, dets, scores):
    """Draw detected bounding boxes."""
    im_height=np.shape(im)[0]
    im_width=np.shape(im)[1]
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(0,len(classes)):
        bbox = dets[i, :4]
        cls = classes[i]
        score = scores[i]
            #to pixel range
        bbox[0]=bbox[0]*im_height
        bbox[1]=bbox[1]*im_width
        bbox[2]=bbox[2]*im_height
        bbox[3]=bbox[3]*im_width

        bboxes = np.array(bbox,dtype=int)


        ax.add_patch(
            plt.Rectangle((bboxes[1], bboxes[0]),
                          bboxes[3] - bboxes[1],
                          bboxes[2] - bboxes[0], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[1], bbox[0] - 2,
                '{:s} {:.3f}'.format(str(cls), score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(cls, cls,
    #                                               score),
    #               fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
# config.gpu_options.per_process_gpu_memory_fraction = 0.333

isess = tf.InteractiveSession(config=config)
# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = os.path.join(__parent_dir,'checkpoints','ssd_300_vgg.ckpt')
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

# Test on some demo image and visualize output.
path = os.path.join(__parent_dir,'demo','img')
image_names = sorted(os.listdir(path))

image_current = image_names[3]
# print(image_names,image_current)
img_blob = mpimg.imread( os.path.join(path, image_current))
img_prop = Image.open(os.path.join(path, image_current))



rclasses, rscores, rbboxes =  process_image(img_blob)
classes_names = [VOC_LABELS[k] for k in rclasses.tolist()]
json_obj = {}
json_obj["classes"] = classes_names
json_obj["bboxes"] = rbboxes.tolist()
json_obj["scores"] = rscores.tolist()
json_obj["image_width"] = img_prop.size[0]
json_obj["image_height"] = img_prop.size[1]

print(json.dumps(json_obj))

vis_detections(img_blob,classes_names,rbboxes,rscores)

plt.show()
