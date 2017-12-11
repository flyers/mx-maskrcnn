import cv2
import mxnet as mx
from rcnn.tools.wrapper import MaskRCNN

wrapper = MaskRCNN(network='resnet_fpn',
                   prefix='model/res50-fpn/blender_v1/alternate/final',
                   epoch=0,
                   ctx=mx.gpu(0))

# img = cv2.imread(
#     '/home/sliay/datasets/ycb_render/21_wo_repeat/00000_Image00000.png')
img = cv2.imread('/home/sliay/datasets/realistic/s1/0016.png')
# img = cv2.imread('/home/sliay/Downloads/test/test0127.png')
wrapper.predict(img)
