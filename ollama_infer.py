import ollama
import pandas as pd
from PIL import Image

# Read the CSV file
df = pd.read_csv('annotations2025.csv')

correct = 0
incorrect = 0

# Iterate through each row/image in the dataset
for index, row in df.iterrows():
    image_path = "2025_targets/" + row['img_name']
    
    # Get bounding box coordinates from row
    x = int(row['top_left_x'])
    y = int(row['top_left_y'])
    w = int(row['width']) 
    h = int(row['height'])

    # Crop the image to the target area
    img = Image.open(image_path)
    cropped = img.crop((x, y, x+w, y+h))
    cropped.save('temp.jpg')

    response = ollama.chat(
        model='llama3.2-vision:11b',
        messages=[{
            'role': 'user',
            'content': 'What single digit number between 0 and 9 is on this target? The line next to the number is the bottom of the number which may be rotated. The answer will never be two digits. 6 and 9 look very similar so use the line to orient it. If there is no number, respond with -1',
            'images': ['temp.jpg']
        }]
    )

    prediction = response['message']['content']
    print(f"Image {index}: Predicted number = {prediction}, Ground truth = {row['classification']}")
    print()

print(f"Correct: {correct}, Incorrect: {incorrect}")