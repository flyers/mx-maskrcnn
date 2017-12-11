import random
import pprint
import mxnet as mx
import numpy as np
import cv2

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
    def __init__(self, network, prefix, epoch,
                 ctx=mx.gpu(0), image_shape=(3, 480, 640)):
        # print config
        pprint.pprint(config)
        assert config.TEST.HAS_RPN

        # get symbol and data format
        self.ctx = ctx
        self.im_shape = image_shape
        self.sym = eval('get_' + network + '_mask_test')(
            num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
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
        self.classes = dataset.Blender.CLASSES
        self.class_id = dataset.Blender.CLASS_ID
        self.num_classes = dataset.Blender.NUM_CLASSES

    def predict(self, img):
        """
        takes an opencv imread img and predicts the detection and segmentation result
        """
        img_ori = img
        im_info = mx.nd.array([[self.im_shape[1], self.im_shape[2], 1]], ctx=self.ctx)
        img = np.transpose(img, (2, 0, 1))
        img = img[np.newaxis, (2, 1, 0)]
        img = mx.nd.array(img, ctx=self.ctx)
        data_batch = mx.io.DataBatch(data=[img, im_info], label=None,
                                     provide_data=self.provide_data, provide_label=None)
        output = self.predictor.predict(data_batch)
        rois = output['rois_output'].asnumpy()[:, 1:]
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
        mask_output = output['mask_prob_output'].asnumpy()

        # post processing
        pred_boxes = nonlinear_pred(rois, bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, self.im_shape[-2:])

        all_boxes = [[[] for _ in xrange(1)]
                     for _ in xrange(self.num_classes)]
        all_masks = [[[] for _ in xrange(1)]
                     for _ in xrange(self.num_classes)]

        label = np.argmax(scores, axis=1)
        label = label[:, np.newaxis]

        thresh = 0.1
        for c in self.class_id:
            cls_ind = self.class_id.index(c)
            cls_boxes = pred_boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_masks = mask_output[:, cls_ind, :, :]
            cls_scores = scores[:, cls_ind, np.newaxis]
            keep = np.where((cls_scores >= thresh) & (label == cls_ind))[0]
            cls_masks = cls_masks[keep, :, :]
            dets = np.hstack((cls_boxes, cls_scores)
                             ).astype('float32')[keep, :]
            keep = self.nms(dets)
            all_boxes[cls_ind][0] = dets[keep, :]
            all_masks[cls_ind][0] = cls_masks[keep, :]

        boxes_this_image = [[]] + [all_boxes[cls_ind][0]
                                   for cls_ind in range(1, self.num_classes)]
        masks_this_image = [[]] + [all_masks[cls_ind][0]
                                   for cls_ind in range(1, self.num_classes)]

        # visualize the result
        det_map, mask_map = self.visualize(img_ori, boxes_this_image, masks_this_image)
        cv2.imshow('a', det_map)
        cv2.waitKey(0)
        cv2.imshow('b', mask_map)
        cv2.waitKey(0)
        cv2.imwrite('det.png', det_map)
        cv2.imwrite('seg.png', mask_map)

    def visualize(self, img, detections, seg_masks):
        mask_map = np.zeros((self.im_shape[1], self.im_shape[2]))
        for j, labelID in enumerate(self.class_id):
            if labelID == 0:
                continue
            dets = detections[j]
            masks = seg_masks[j]
            for i in range(len(dets)):
                bbox = dets[i, :4]
                score = dets[i, -1]
                bbox = map(int, bbox)
                mask_image = np.zeros((self.im_shape[1], self.im_shape[2]))
                mask = masks[i, :, :]
                mask = cv2.resize(
                    mask, (bbox[2] - bbox[0], (bbox[3] - bbox[1])), interpolation=cv2.INTER_LINEAR)
                mask[mask > 0.5] = j + 1
                mask[mask <= 0.5] = 0
                mask_image[bbox[1]: bbox[3], bbox[0]: bbox[2]] = mask
                mask_map += mask_image
                # show detection result
                color = [random.random(), random.random(), random.random()]
                color = [int(x * 255) for x in color]
                cv2.rectangle(img, tuple(bbox[0:2]),
                              tuple(bbox[2:4]), color, 2)
                cv2.putText(img, str(
                    j) + ' %s' % self.classes[j], (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
        mask_map /= (mask_map.max()/255.0)
        return img, mask_map
