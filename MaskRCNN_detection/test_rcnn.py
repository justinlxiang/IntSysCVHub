import os
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import numpy as np

SCORE_THRESH = 0.5
CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "base_config.yaml"
)
MODEL_WEIGHTS_FILE = os.path.join("model_weights", "maskrcnn_Feb_19_2024.pth")

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

    def detect(self, image):
        np_img = np.array(image)

        output = None

        try:
            output = self.predictor(np_img)
        except Exception as e:
            print(e)

        return output["instances"].pred_boxes, output["instances"].scores, output["instances"].pred_classes

def crop_and_save_images(input_folder, output_folder, model_weights, config_file):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Initialize the MaskRCNN model
    model = MaskRCNN(model_weights_file=model_weights, config_file=config_file, use_gpu=False)
    
    # Iterate through images in the input folder
    for img_name in sorted(os.listdir(input_folder)):
        img_path = os.path.join(input_folder, img_name)
        if os.path.isfile(img_path):
            # Open image with PIL and detect ROIs
            image = Image.open(img_path)
            boxes, scores, classes = model.detect(image)

            if len(boxes) == 0:
                print(f"No boxes found for {img_name}")
                continue
            
            # Crop and save each ROI as a separate image
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                cropped_img = image.crop([x1, y1, x2, y2])
                cropped_img_path = os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}_crop_{idx}.jpg")
                cropped_img.save(cropped_img_path)
                print(f"Cropped image saved to {cropped_img_path}")

if __name__ == "__main__":
    input_folder = "./2025_targets"
    output_folder = "./Detected_Images"
    config_file = "./MaskRCNN_detection/base_config.yaml"
    model_weights = "./output/model_final.pth"
    
    crop_and_save_images(input_folder, output_folder, model_weights, config_file)
