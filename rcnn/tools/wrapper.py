import random
import pprint
import mxnet as mx
import numpy as np
import cv2
import time

from ..config import config, dataset
from ..symbol import *
from ..dataset import *
from ..core.tester import Predictor
from ..utils.load_model import load_param
from ..processing.nms import py_nms_wrapper
from ..processing.bbox_transform import nonlinear_pred, clip_boxes


def check_params(data_shapes, sym, arg_params, aux_params):
    # infer shape
    data_shape_dict = dict(data_shapes)
    arg_shape, _, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))

    # check parameters
    for k in sym.list_arguments():
        if k in data_shape_dict or 'label' in k:
            continue
        assert k in arg_params, k + ' not initialized'
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + \
            str(arg_shape_dict[k]) + ' provided ' + str(arg_params[k].shape)
    for k in sym.list_auxiliary_states():
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + \
            str(aux_shape_dict[k]) + ' provided ' + str(aux_params[k].shape)


class MaskRCNN(object):
    def __init__(self, network, prefix, epoch, datatype='Blender',
                 thresh = 0.1, ctx=mx.gpu(0), image_shape=(3, 480, 640)):
        """
        network(str): 'resnet-fpn'
        prefix(str): model prefix path
        epoch(int): model epoch number
        datatype(str): ['Blender', 'Blender_b'], multi-class vesus two-class
        thresh(float): confidence threshold for filtering
        """
        pprint.pprint(config)
        assert config.TEST.HAS_RPN

        # get symbol and data format
        self.ctx = ctx
        self.im_shape = image_shape
        self.sym = eval('get_' + network + '_mask_test')(
            num_classes=dataset[datatype].NUM_CLASSES, num_anchors=dataset[datatype].NUM_ANCHORS)
        self.provide_data = [("data", (1,) + image_shape), ("im_info", (1, 3))]

        # load model and check model parameters
        arg_params, aux_params = load_param(
            prefix, epoch, convert=True, ctx=ctx, process=True)
        check_params(self.provide_data, self.sym, arg_params, aux_params)

        # decide maximum shape
        data_names = [k[0] for k in self.provide_data]
        label_names = None
        max_data_shape = [self.provide_data[0]]

        # create predictor model
        self.predictor = Predictor(self.sym, data_names, label_names,
                                   context=ctx, max_data_shapes=max_data_shape,
                                   provide_data=self.provide_data, provide_label=None,
                                   arg_params=arg_params, aux_params=aux_params)
        self.nms = py_nms_wrapper(config.TEST.NMS)

        # set dataset classes
        self.classes = dataset[datatype].CLASSES
        self.class_id = dataset[datatype].CLASS_ID
        self.num_classes = dataset[datatype].NUM_CLASSES
        self.thresh = thresh

    def predict(self, img, render=False):
        """
        takes an opencv imread img and predicts the detection and segmentation result
        img(nparray): nparray read by cv2.imread
        render(bool): whether to visualize the detection and segmentation result
        returns:
            all_boxes, all_masks
            all_boxes = [box_class1, box_class2, ...], where box_class is a n by 5 ndarray, each row denotes [x1, y2, x2, y2, score]
            all_masks = [mask_class1, mask_class2, ...], where mask_class is a n by 28 by 28 ndarray, each 28 by 28 mask corresponds to the above bounding box

        """
        t1 = time.time()
        img_ori = img
        im_info = mx.nd.array([[self.im_shape[1], self.im_shape[2], 1]], ctx=self.ctx)
        img = np.transpose(img, (2, 0, 1))
        img = img[np.newaxis, (2, 1, 0)]
        img = mx.nd.array(img, ctx=self.ctx)
        print('data preparing time {}'.format(time.time() - t1))
        t1 = time.time()
        data_batch = mx.io.DataBatch(data=[img, im_info], label=None,
                                     provide_data=self.provide_data, provide_label=None)
        output = self.predictor.predict(data_batch)
        rois = output['rois_output'].asnumpy()[:, 1:]
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
        mask_output = output['mask_prob_output'].asnumpy()

        print('network time {}'.format(time.time() - t1))
        t1 = time.time()

        # post processing
        pred_boxes = nonlinear_pred(rois, bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, self.im_shape[-2:])

        all_boxes = [[[] for _ in xrange(1)]
                     for _ in xrange(self.num_classes)]
        all_masks = [[[] for _ in xrange(1)]
                     for _ in xrange(self.num_classes)]

        label = np.argmax(scores, axis=1)
        label = label[:, np.newaxis]

        for c in self.class_id:
            cls_ind = self.class_id.index(c)
            cls_boxes = pred_boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_masks = mask_output[:, cls_ind, :, :]
            cls_scores = scores[:, cls_ind, np.newaxis]
            keep = np.where((cls_scores >= self.thresh) & (label == cls_ind))[0]
            cls_masks = cls_masks[keep, :, :]
            dets = np.hstack((cls_boxes, cls_scores)
                             ).astype('float32')[keep, :]
            keep = self.nms(dets)
            all_boxes[cls_ind][0] = dets[keep, :]
            all_masks[cls_ind][0] = cls_masks[keep, :]

        boxes_this_image = [all_boxes[cls_ind][0] for cls_ind in range(1, self.num_classes)]
        masks_this_image = [all_masks[cls_ind][0] for cls_ind in range(1, self.num_classes)]

        t1 = time.time()
        # visualize the result
        if render:
            det_map, mask_map = self.visualize(img_ori, boxes_this_image, masks_this_image)
            cv2.imshow('a', det_map)
            cv2.waitKey(0)
            cv2.imshow('b', mask_map)
            cv2.waitKey(0)
            cv2.imwrite('det.png', det_map)
            cv2.imwrite('seg.png', mask_map)

        return boxes_this_image, masks_this_image

    def visualize(self, img, detections, seg_masks):
        mask_map = np.zeros((self.im_shape[1], self.im_shape[2], 3))
        for j in range(len(detections)):
            dets = detections[j]
            masks = seg_masks[j]
            for i in range(len(dets)):
                bbox = dets[i, :4]
                score = dets[i, -1]
                bbox = map(int, bbox)
                mask_image = np.zeros((self.im_shape[1], self.im_shape[2], 3))
                mask = masks[i, :, :]
                mask = cv2.resize(
                    mask, (bbox[2] - bbox[0], (bbox[3] - bbox[1])), interpolation=cv2.INTER_LINEAR)
                threshold = 0.5
                mask[mask > threshold] = 1
                mask[mask <= threshold] = 0
                mask = np.tile(mask, [3, 1, 1])
                mask = np.transpose(mask, (1, 2, 0))
                # show detection result
                color = [random.random(), random.random(), random.random()]
                color = [int(x * 255) for x in color]
                cv2.rectangle(img, tuple(bbox[0:2]),
                              tuple(bbox[2:4]), color, 2)
                cv2.putText(img, str(
                    j+1) + ' %s' % self.classes[j+1], (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
                tmp = np.ones((bbox[3]-bbox[1], bbox[2]-bbox[0]))
                mask_colors = [tmp * x / 255.0 for x in color]
                mask_color = np.dstack(mask_colors)
                mask_color = mask_color * mask
                mask_image[bbox[1]: bbox[3], bbox[0]: bbox[2], :] = mask_color
                mask_map += mask_image
        mask_map /= (mask_map.max()/1.0)
        return img, mask_map
