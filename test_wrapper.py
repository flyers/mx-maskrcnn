import cv2
import mxnet as mx
from rcnn.tools.wrapper import MaskRCNN

wrapper = MaskRCNN(network='resnet_fpn',
                   #prefix='model/res50-fpn/blender_v1/alternate/final',
                   prefix='model/res50-fpn/blender_b_v1/alternate/final',
                   datatype='Blender_b',
                   epoch=0,
                   ctx=mx.gpu(0))

img = cv2.imread(
    '/home/sliay/datasets/ycb_render/21_wo_repeat/00000_Image00000.png')
# img = cv2.imread('/home/sliay/datasets/realistic/s1/0016.png')
#img = cv2.imread('/home/sliay/Downloads/data/single_object/image0005.jpg')
# img = cv2.imread('/home/sliay/Downloads/data/multi_object/image0016.jpg')
# img = cv2.imread('/home/sliay/Downloads/test/test0127.png')

# See rcnn.tools.wrapper.predict for the meaning of boxes, masks
boxes, masks = wrapper.predict(img, render=True)
