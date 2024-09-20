import os
import cv2
from PIL import Image
from inference import MaskRCNN

def crop_and_save_images(input_folder, output_folder, model_weights):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Initialize the MaskRCNN model
    model = MaskRCNN(model_weights_file=model_weights, use_gpu=False)
    
    # Iterate through images in the input folder
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        if os.path.isfile(img_path):
            # Open image with PIL and detect ROIs
            image = Image.open(img_path)
            rois = model.detect(image)
            
            # Crop and save each ROI as a separate image
            for idx, roi in enumerate(rois):
                cropped_img = roi.image
                cropped_img_path = os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}_crop_{idx}.jpg")
                cropped_img.save(cropped_img_path)
                print(f"Cropped image saved to {cropped_img_path}")

if __name__ == "__main__":
    input_folder = "./MaskRCNN_Test_images"
    output_folder = "./Detected_Images"
    model_weights = "./model_weights/maskrcnn_Feb_19_2024.pth"
    
    crop_and_save_images(input_folder, output_folder, model_weights)
