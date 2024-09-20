import csv
from PIL import Image, ImageOps

def get_image_orientation(img_path):
    with Image.open(img_path) as img:
        exif_data = img._getexif()
        if exif_data is not None:
            orientation = exif_data.get(0x0112)
            return orientation
        else:
            return None
        

def manually_correct_orientation(img):
    exif = img._getexif()
    if not exif:
        return img

    orientation_key = 274  # decimal value of 0x0112
    if orientation_key in exif:
        orientation = exif[orientation_key]

        if orientation == 3:
            img = img.rotate(180, expand=True)
        elif orientation == 6:
            img = img.rotate(270, expand=True)
        elif orientation == 8:
            img = img.rotate(90, expand=True)

    return img
    
        
def crop_images_from_annotations(csv_file_path, output_folder):
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_path = row['img_name']
            
            top_left_x = int(float(row['top_left_x'])) 
            top_left_y = int(float(row['top_left_y']))
            width = int(float(row['width']))
            height = int(float(row['height']))
            
            try:
                with Image.open("real_runway_imgs/" + img_path) as img:
                    print(f"Original dimensions for {img_path}: {img.size}")  # Debug original size
                    
                    if "DSC" in img_path:
                        img = manually_correct_orientation(img)
                    
                    print(f"Corrected dimensions for {img_path}: {img.size}")  # Debug corrected size
                    print()
                    
                    # Calculate bottom right x and y coordinates
                    bottom_right_x = top_left_x + width 
                    bottom_right_y = top_left_y + height
                    
                    cropped_img = img.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
                    
                    # Save the cropped image
                    output_path = f"{output_folder}/{img_path.split('/')[-1].split('.')[0]}_cropped{str(reader.line_num)}.jpeg"
                    cropped_img.save(output_path)      
                    print(f"Cropped image saved to {output_path}")
            except FileNotFoundError:
                print(f"File {img_path} not found.")
            except Exception as e:
                print(f"Error processing file {img_path}: {e}")

# Example usage
csv_file_path = 'annotations.csv'

output_folder = 'cropped_data'
crop_images_from_annotations(csv_file_path, output_folder)
