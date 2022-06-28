from easydict import EasyDict

import numpy as np

from scipy.special import softmax

import tensorflow.compat.v1 as tf

import time
import os

from .text_emb_generate import build_text_embedding
from .nms import nms

FLAGS = {
    'prompt_engineering': True,
    'this_is': True,
    
    'temperature': 100.0,
    'use_softmax': False,
}
FLAGS = EasyDict(FLAGS)

ROOT = os.getcwd()

class VILD:
    def __init__(self):
        session = tf.Session(graph=tf.Graph())
        
        saved_model_dir = os.path.join(ROOT,'image_path_v2')
        print(saved_model_dir)

        _ = tf.saved_model.loader.load(session, ['serve'], saved_model_dir)
        self.session = session
        robocup_class = ["A red apple", "orange", "lime", "banana", "onion", "jelly", "cereal box", "potato chip", "instant noodle", "ketchup", "chocolate", "water bottle", "A softdrink bottle", "milk carton", "juice", "coffee cup", "tea", "beer", "table", "chair", "sofa", "fridge", "cabinet"]

        category_name_string = ';'.join(robocup_class)
        category_names = [x.strip() for x in category_name_string.split(';')]
        category_names = ['background'] + category_names
        categories = [{'name': item, 'id': idx+1,} for idx, item in enumerate(category_names)]
        self.category_names = category_names
        robocup_embbed = build_text_embedding(categories,FLAGS=FLAGS)
        self.text_features = robocup_embbed
        
        max_boxes_to_draw = 10
        nms_threshold = 0.8
        min_rpn_score_thresh = 0.9 
        min_box_area = 300
        params = max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area
        self.params = params

    def _detect(self,image_path):
        max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area = self.params

    

        roi_boxes, roi_scores, detection_boxes, scores_unused, box_outputs, detection_masks, visual_features, image_info = self.session.run(
            ['RoiBoxes:0', 'RoiScores:0', '2ndStageBoxes:0', '2ndStageScoresUnused:0', 'BoxOutputs:0', 'MaskOutputs:0', 'VisualFeatOutputs:0', 'ImageInfo:0'],
            feed_dict={'Placeholder:0': [image_path,]})


    
        roi_boxes = np.squeeze(roi_boxes, axis=0)  # squeeze
        # no need to clip the boxes, already done
        roi_scores = np.squeeze(roi_scores, axis=0)

        detection_boxes = np.squeeze(detection_boxes, axis=(0, 2))
        scores_unused = np.squeeze(scores_unused, axis=0)
        box_outputs = np.squeeze(box_outputs, axis=0)
        detection_masks = np.squeeze(detection_masks, axis=0)
        visual_features = np.squeeze(visual_features, axis=0)

        image_info = np.squeeze(image_info, axis=0)  # obtain image info
        image_scale = np.tile(image_info[2:3, :], (1, 2))
        image_height = int(image_info[0, 0])
        image_width = int(image_info[0, 1])

        rescaled_detection_boxes = detection_boxes / image_scale # rescale


        nmsed_indices = nms(
            detection_boxes,
            roi_scores,
            thresh=nms_threshold
            )

        # Compute RPN box size.
        box_sizes = (rescaled_detection_boxes[:, 2] - rescaled_detection_boxes[:, 0]) * (rescaled_detection_boxes[:, 3] - rescaled_detection_boxes[:, 1])

        # Filter out invalid rois (nmsed rois)
        valid_indices = np.where(
            np.logical_and(
            np.isin(np.arange(len(roi_scores), dtype=np.int), nmsed_indices),
            np.logical_and(
                np.logical_not(np.all(roi_boxes == 0., axis=-1)),
                np.logical_and(
                    roi_scores >= min_rpn_score_thresh,
                    box_sizes > min_box_area
                    )
            )    
            )
        )[0]
        # print('number of valid indices', len(valid_indices))

        detection_roi_scores = roi_scores[valid_indices][:max_boxes_to_draw, ...]
        detection_boxes = detection_boxes[valid_indices][:max_boxes_to_draw, ...]
        detection_masks = detection_masks[valid_indices][:max_boxes_to_draw, ...]
        detection_visual_feat = visual_features[valid_indices][:max_boxes_to_draw, ...]
        rescaled_detection_boxes = rescaled_detection_boxes[valid_indices][:max_boxes_to_draw, ...]

        start_time = time.time()
        #################################################################
        # Compute text embeddings and detection scores, and rank results
        # text_features = build_text_embedding(categories)

        
        raw_scores = detection_visual_feat.dot(self.text_features.T)
        if FLAGS.use_softmax:
            scores_all = softmax(FLAGS.temperature * raw_scores, axis=-1)
        else:
            scores_all = raw_scores

        indices = np.argsort(-np.max(scores_all, axis=1))  # Results are ranked by scores
        indices_fg = np.array([i for i in indices if np.argmax(scores_all[i]) != 0])
        #################################################################
        
        n_boxes = rescaled_detection_boxes.shape[0]

        res_bbox = []
        res_label = []
        for anno_idx in indices[0:int(n_boxes)]:
            rpn_score = detection_roi_scores[anno_idx]
            bbox = rescaled_detection_boxes[anno_idx]
            scores = scores_all[anno_idx]
            if np.argmax(scores) == 0:
                continue
                
            res_bbox.append(bbox)
            res_label.append(self.category_names[np.argmax(scores)])
        
        # print("--- %s seconds ---" % (time.time() - start_time))
        return res_label, res_bbox, image_height, image_width

    # def detect_img_seq(self,image_dir):
