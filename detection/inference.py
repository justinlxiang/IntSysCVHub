from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from typing import List
import os
import numpy as np
import logging
from roi import ROI

SCORE_THRESH = 0.5
CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "base_config.yaml"
)
# MODEL_WEIGHTS_FILE = os.path.join('vision', 'model_weights',
#                                   'maskrcnn_0044999.pth')
MODEL_WEIGHTS_FILE = os.path.join("vision", "model_weights", "maskrcnn_Feb_19_2024.pth")


class MaskRCNN():
    def __init__(
        self,
        model_weights_file=MODEL_WEIGHTS_FILE,
        score_thresh=SCORE_THRESH,
        config_file=CONFIG_FILE,
        use_gpu=True,
    ):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_file)
        self.cfg.MODEL.WEIGHTS = model_weights_file
        print(f"Model loaded from {model_weights_file}")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
        if not use_gpu:
            self.cfg.MODEL.DEVICE = "cpu"

        self.predictor = DefaultPredictor(self.cfg)

    def detect(self, image) -> List[ROI]:

        np_img = np.array(image)

        output = None

        try:
            output = self.predictor(np_img)
        except Exception as e:
            print(e)

        rois = []
        for box, mask in zip(
            output["instances"].pred_boxes, output["instances"].pred_masks
        ):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cropped_img = image.crop([x1, y1, x2, y2])
            mask = mask.detach().cpu().numpy()[y1:y2, x1:x2]
            rois.append(ROI(cropped_img, (x1, y1), (x2, y2), mask))

        return rois
    
    
