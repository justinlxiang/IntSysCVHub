from together import Together
import pandas as pd
import base64
import io
from PIL import Image
import cv2
import numpy as np
from rotate_target import rotate_image

client = Together()

# Read the CSV file
df = pd.read_csv('annotations2025.csv')

correct = 0
incorrect = 0

# Load reference images 0-9 and encode them
reference_encodings = {}
for i in range(10):
    ref_img = Image.open(f'{i}.png')
    buffered = io.BytesIO()
    ref_img.save(buffered, format="JPEG")
    ref_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    reference_encodings[i] = ref_base64


# Iterate through each row/image in the dataset
for index, row in df.iterrows():
    image_path = "2025_targets/" + row['img_name']  
    with open(image_path, "rb") as image_file:
        # if row['classification'] != 10:
            # continue
        # Get bounding box coordinates from row
        x = int(row['top_left_x'])
        y = int(row['top_left_y']) 
        w = int(row['width'])
        h = int(row['height'])

        img = Image.open(image_path)
        cropped = img.crop((x, y, x+w, y+h))

        
        # Convert PIL Image to OpenCV format
        cv2_image = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
        cv2_image = rotate_image(cv2_image)
        cv2.imwrite('temp.jpg', cv2_image)

        # Convert OpenCV image back to PIL for base64 encoding
        pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        # model="Qwen/Qwen2-VL-72B-Instruct",
        messages=[
            {"role": "system", 
             "content": """You are a helpful assistant that can identify if this target has a number on it.
                          The answer will be yes or no."""},
            {"role": "user", 
             "content": [
                {  "type": "text", "text": "Does this target have a number on it? The number should be written over the white background. Output 'yes' or 'no' at the end of your response."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ]},
        ]
    )
    has_number = response.choices[0].message.content

    print(has_number)

    if has_number == 'No.' or 'no' in has_number.lower():
        print(f"Image {index}: No number")
        print(f"Ground truth = {row['classification']}")
        print('--------------------------------')
        if row['classification'] == 10:
            correct += 1
        else:
            incorrect += 1
        continue
    else:
        print(f"Image {index}: Has number")
        

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        # model="Qwen/Qwen2-VL-72B-Instruct",
        messages=[
            {"role": "system", 
             "content": """You are a helpful assistant that can identify the number on a target.
                          The answer will never be two digits."""},
            {"role": "user", 
             "content": [
                {"type": "text", "text": """What single digit number {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} do you see on this target? The number will be written above the black line. Read the number as written without rotating the image. If you think the number is 6 or 9, double check your answer. Write the answer at the end of your response in the form 'The number is <number>.'"""},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ]},
        ]
    )
    prediction = response.choices[0].message.content.strip()
    
    print(f"Image {index}: Predicted number = {prediction}")
    print(f"Ground truth = {row['classification']}")
    print('--------------------------------')

    if str(row['classification']) in prediction:
        correct += 1
    else:
        incorrect += 1

print(f"Correct: {correct}, Incorrect: {incorrect}")