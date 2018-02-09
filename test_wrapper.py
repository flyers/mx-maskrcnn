import cv2
import mxnet as mx
import argparse
from rcnn.tools.wrapper import MaskRCNN

parser = argparse.ArgumentParser(description='Test Mask-RCNN network')
parser.add_argument('--name', help='file name', type=str)
parser.add_argument('--model', help='pretrained model', type=str)
args = parser.parse_args()

wrapper = MaskRCNN(network='resnet_fpn',
                   #prefix='model/res50-fpn/blender_v1/alternate/final',
                   prefix=args.model,
                   datatype='Blender_b',
                   epoch=0,
                   ctx=mx.gpu(0))


img_name = args.name

img = cv2.imread(img_name)
# img = cv2.imread('/home/sliay/datasets/realistic/s1/0016.png')
#img = cv2.imread('/home/sliay/Downloads/data/single_object/image0005.jpg')
# img = cv2.imread('/home/sliay/Downloads/data/multi_object/image0016.jpg')
# img = cv2.imread('/home/sliay/Downloads/test/test0127.png')

# See rcnn.tools.wrapper.predict for the meaning of boxes, masks
boxes, masks = wrapper.predict(img, render=True)
