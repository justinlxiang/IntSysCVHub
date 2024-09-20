import os
import shutil
from sklearn.model_selection import train_test_split
import csv

def separate_images(attribute):
    # Define paths
    source_folder = 'cropped_data'
    base_destination_folder = 'real_' + (attribute)
    train_folder = os.path.join(base_destination_folder, 'train')
    test_folder = os.path.join(base_destination_folder, 'test')
    
    # Create base directories if they don't exist
    for folder in [train_folder, test_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # Read annotations
    annotations = {}
    with open('annotations.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_name = row['img_name'].split('/')[-1].split('.')[0] + '_cropped' + str(reader.line_num) + '.jpeg'
            shape = row['shape'].strip('"')
            color = row['shape_color'].strip('"')
            annotations[img_name] = {'shape': shape, 'color': color}
    
    # Split data into train and test (80% train, 20% test)
    files = os.listdir(source_folder)
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    
    # Separate images based on attribute and copy to respective folders
    for img_name, info in annotations.items():
        source_path = os.path.join(source_folder, img_name)
        if attribute == 'shape':
            class_folder = info['shape']
        elif attribute == 'color':
            class_folder = info['color']
        else:
            raise ValueError("Attribute must be either 'shape' or 'color'")
        
        if img_name in train_files:
            dest_path = os.path.join(train_folder, class_folder)
        elif img_name in test_files:
            dest_path = os.path.join(test_folder, class_folder)
        else:
            continue  # Skip if file is not in annotations
        
        # Create class folder if it doesn't exist
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        
        shutil.copy(source_path, os.path.join(dest_path, img_name))
        print(f"Copying {img_name} to {dest_path}")

# Call the function for both attributes
separate_images('shape')
separate_images('color')

