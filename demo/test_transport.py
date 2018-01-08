import os
__parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import math
import random

import numpy as np
import tensorflow as tf
#import cv2

slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.append(__parent_dir)

from nets import ssd_vgg_300, ssd_common, np_methods
from nets.ssd_vgg_300 import SSDParams
from preprocessing import ssd_vgg_preprocessing
#from notebooks import visualization
# Main image processing routine.
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
            select_threshold=select_threshold, img_shape=net_shape, num_classes=7, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

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
ssd_net.params = SSDParams(
        img_shape=(300, 300),
        num_classes=6,
        no_annotation_label=21,
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        anchor_size_bounds=[0.15, 0.90],
        # anchor_size_bounds=[0.20, 0.90],
        anchor_sizes=[(21., 45.),
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.),
                      (261., 315.)],
        # anchor_sizes=[(30., 60.),
        #               (60., 111.),
        #               (111., 162.),
        #               (162., 213.),
        #               (213., 264.),
        #               (264., 315.)],
        anchor_ratios=[[2, .5],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5],
                       [2, .5]],
        anchor_steps=[8, 16, 32, 64, 100, 300],
        anchor_offset=0.5,
        normalizations=[20, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = os.path.join(__parent_dir, 'checkpoints', 'transport.ckpt')

isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

# Test on some demo image and visualize output.
path = os.path.join(__parent_dir, 'datasets', 'data\Transport\images')
image_names = sorted(os.listdir(path))

for i,img_name in enumerate(image_names):
    if 'png' not in img_name:
        img = mpimg.imread(os.path.join(path, img_name))
        # if i>50:
        #     print(i,image_names[i+1])
        rclasses, rscores, rbboxes =  process_image(img)

        print(img_name,rclasses,rscores,rbboxes)

# vis_detections(img,rclasses,rbboxes,rscores)
# plt.show()
