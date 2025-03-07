from together import Together
import pandas as pd
import base64
import io
from PIL import Image
import cv2
import numpy as np
from rotate_target import rotate_image
import time
import concurrent.futures

client = Together()

# Read the CSV file
df = pd.read_csv('annotations2025.csv')

correct = 0
incorrect = 0

time_start = time.time()

# Iterate through each row/image in the dataset
for index, row in df.iterrows():
    if row['classification'] != 6 and row['classification'] != 9:
        continue
    image_path = "2025_targets/" + row['img_name']  
    with open(image_path, "rb") as image_file:
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

    # Create both requests to run in parallel
    has_number_request = {
        "model": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        "messages": [
            {"role": "system", 
             "content": """You are a helpful assistant that can identify if this target has a number on it.
                          """},
            {"role": "user", 
             "content": [
                {  "type": "text", "text": "Does this target have a number on it? If it looks like the letter O, it is the number 0. The number should be written over the white background. Output your answer in the form 'There is no number' or 'There is a number'."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ]},
        ]
    }
    
    number_identification_request = {
        "model": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        "messages": [
            {"role": "system", 
             "content": """You are a helpful assistant that can identify the number on a target.
                          The answer will never be two digits. The black line is not a 1.""",
            },
            {"role": "user", 
             "content": [
                {"type": "text", "text": """What single digit number 0-9 do you see on this image above the black line? Read the number as written without rotating the image. If you think the number is 6 or 9, double check your answer by reading without rotating the image. Return the number in the form 'The number is <number>.'"""},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ]},
        ]
    }
    
    # Run both requests in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        has_number_future = executor.submit(client.chat.completions.create, **has_number_request)
        number_id_future = executor.submit(client.chat.completions.create, **number_identification_request)
        
        # Get results
        has_number_response = has_number_future.result()
        # Check if the image has no number before waiting for number identification
        has_number = has_number_response.choices[0].message.content
        
        if 'There is no number' in has_number.lower():
            # If no number is detected, we don't need to wait for the number identification result
            number_id_response = None
        else:
            # Only wait for the number identification result if there's a number
            number_id_response = number_id_future.result()
    
    has_number = has_number_response.choices[0].message.content
    prediction = ""
    if number_id_response:
        prediction = number_id_response.choices[0].message.content.strip()
    
    print(has_number)
    
    if has_number == 'No.' or 'no' in has_number.lower():
        if row['classification'] == 10:
            correct += 1
        else:
            incorrect += 1
            print(f"Image {index}: No number")
            print(f"Ground truth = {row['classification']}")
            print('--------------------------------')
    else:
        # print(f"Image {index}: Has number")
        # print(f"Image {index}: Predicted number = {prediction}")
        # print(f"Ground truth = {row['classification']}")
        # print('--------------------------------')
        
        if str(row['classification']) in prediction:
            correct += 1
        else:
            incorrect += 1
            print(f"Image {index}: Has number")
            print(f"Image {index}: Predicted number = {prediction}")
            print(f"Ground truth = {row['classification']}")
            print('--------------------------------')

print(f"Correct: {correct}, Incorrect: {incorrect}")

time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")
print(f"Time per image: {(time_end - time_start) / len(df)} seconds")