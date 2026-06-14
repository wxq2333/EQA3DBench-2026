import torch
import clip
from PIL import Image
import numpy as np
import sys
import os
sys.path.append('/home/ubuntu/桌面/control/env_xq/estool')
from es import OpenES
import yaml
import subprocess

# 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 加载物品位置信息
with open("/home/ubuntu/桌面/control/env_xq/object_config.yaml", 'r') as f:
    data = yaml.safe_load(f)
    objects = data['objects']

# 标记化文本描述
text = clip.tokenize([" a Microwave on the table "]).to(device)

# 定义从模拟器获取渲染的场景图像的函数
def get_scene_images():
    # 调用 camera.py 获取图像
    result = subprocess.run(["python", "/home/ubuntu/桌面/control/env_xq/camera.py"], capture_output=True)
    
    if result.returncode == 0:
        # 假设 camera.py 会保存三张图片，分别是 scene_image_0.png, scene_image_2.png, scene_image_4.png
        image_paths = [
            "/home/ubuntu/桌面/control/env_xq/observations/scene_image_0.png",
            "/home/ubuntu/桌面/control/env_xq/observations/scene_image_2.png",
            "/home/ubuntu/桌面/control/env_xq/observations/scene_image_4.png"
        ]
        
        images = []
        for image_path in image_paths:
            try:
                # 加载图片
                image = np.array(Image.open(image_path))
                images.append(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {str(e)}")
                return None  # 如果有任何一张图片加载失败，返回 None
        return images  # 返回三张图片的数组
    else:
        print("Error running camera.py:", result.stderr.decode())
        return None

# 定义适应度函数
images = get_scene_images()
if images is not None:
    similarities = []
    for image in images:
        # 图像预处理
        image_tensor = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        
        # 编码图像和文本
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text)
        
        # 计算余弦相似度
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).squeeze().item()
        similarities.append(similarity)
    
    # 计算三张图片的平均相似度
    average_similarity = np.mean(similarities)
    print("Average similarity between images and text is:", average_similarity)
else:
    print("Failed to retrieve scene images.")
