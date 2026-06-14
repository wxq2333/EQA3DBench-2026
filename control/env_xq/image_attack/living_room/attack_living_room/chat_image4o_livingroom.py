import os
import json
import base64
import requests
from tqdm import tqdm
import urllib3

# Disable warnings for insecure requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# OpenAI API Key
api_key = "sk-SvjwIvynl7OhO8MjfhZZPmwfj0Ns6YhFekiKGp5NMIWvnlpK"

# Encode image to Base64 format
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Process images and get answers
def process_images(images, question, headers):
    image_b64_list = [encode_image(image) for image in images]
    
    messages = [
        {
            "role": "system", 
            "content": "You are a friendly home assistant robot in a living room. Your task is to understand the images presented to you and provide clear, helpful answers to any questions."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
                        As a home assistant robot, you are responsible for interpreting the images provided.
                        Your evaluation should focus on:
                        - Understanding the context and details of the images.
                        - Offering practical insights or answers based on what you see.
                        - Don't give me anything that doesn't appear in the images.
                        - The pictures I give you are from different locations in a room.
                        - I will provide you with images from different angles around a center to help you understand the environment.s
                        Please respond to the following question using the information from the images:
                    """
                },
                {
                    "type": "text",
                    "text": question
                },
            ]
        },
    ]

    # Add each image to the messages
    for image_b64 in image_b64_list:
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}"
            }
        })

    payload = {
        "model": "gpt-4o-mini",
        "messages": messages
    }

    try:
        response = requests.post("https://xiaoai.plus/v1/chat/completions", headers=headers, json=payload, verify=False)
        response.raise_for_status()  # Check response status
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]  # Extract answer
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        return None

# Save question and answer to JSON file
def save_to_json(question, answer, output_file):
    entry = {"question": question, "answer": answer}
    if os.path.exists(output_file):
        with open(output_file, "r") as file:
            data = json.load(file)
    else:
        data = []

    data.append(entry)
    with open(output_file, "w") as file:
        json.dump(data, file, indent=4)

# Main program
def main(image_directory, question):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
  
    # Collect all image paths
    image_paths = image_directory.split(',')  # Convert to list
    # Call API to process images and question
    answer = process_images(image_paths, question, headers)
    if answer:
        print("GPT-4o的回答:", answer)
        # Save question and answer to JSON
        save_to_json(question, answer, "output.json")
    else:
        print("未能获取回答。")

image_directory = "/home/ubuntu/桌面/control/env_xq/image_attack/living_room/images_livingroom/frame_000000.jpg,/home/ubuntu/桌面/control/env_xq/image_attack/living_room/images_livingroom/frame_000001.jpg,/home/ubuntu/桌面/control/env_xq/image_attack/living_room/images_livingroom/frame_000002.jpg"  # Add more images as needed
question = "Clean up the trash from the bin"

# Run main program
if __name__ == "__main__":
    main(image_directory, question)
                                            