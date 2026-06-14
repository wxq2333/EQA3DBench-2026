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
import time, math, gc, pickle, random
from datetime import datetime
from operator import itemgetter
import warnings
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 修改处1：需要优化的物品列表，只需在此处修改所需优化的物品名称
object_names_to_update = [
    "glass_cup",
    "basic_bowl",
    "microwave_oven"# 示例添加第二个物品
]

with open("/home/ubuntu/桌面/control/env_xq/object_config.yaml", 'r') as f:
    data = yaml.safe_load(f)
    objects = data['objects']

# 标记化文本描述
text = clip.tokenize([" Based on the images provided, the microwave appears to have a metallic exterior. It looks sleek and reflective, which is typical for metallic appliances. "]).to(device)

# ES 参数设置
num_objects = len(object_names_to_update)
num_params = num_objects * 6  # 每个物品6个参数
npop = 50
sigma = 0.1
alpha = 0.1

es = OpenES(num_params=num_params,
            sigma_init=sigma,
            popsize=npop,
            learning_rate=alpha,
            rank_fitness=False)

# 假定所有物品共享同样的位置和旋转范围，可根据实际情况调整
position_ranges = (-1.8, -1.1, 1.65, -3.8, -3.5)  # (x_min, x_max, y_fixed, z_min, z_max)-1.8,-1.1 -3.8,-3.5
rotation_range = (-180, 180)  # (roll_min, roll_max)

def normalize_single_obj_params(params):
    x_min, x_max, y_fixed, z_min, z_max = position_ranges
    roll_min, roll_max = rotation_range
    
    x_norm = (params[0] - x_min) / (x_max - x_min)
    z_norm = (params[2] - z_min) / (z_max - z_min)
    y = y_fixed

    roll_norm = (params[3] + 180) / 360
    pitch_norm = (params[4] + 180) / 360
    yaw_norm = (params[5] + 180) / 360

    return [x_norm, y, z_norm, roll_norm, pitch_norm, yaw_norm]

def denormalize_single_obj_params(normalized_params):
    x_min, x_max, y_fixed, z_min, z_max = position_ranges
    roll_min, roll_max = rotation_range

    x = normalized_params[0] * (x_max - x_min) + x_min
    z = normalized_params[2] * (z_max - z_min) + z_min
    y = y_fixed

    x = np.clip(x, x_min, x_max)
    z = np.clip(z, z_min, z_max)

    roll = normalized_params[3] * 360 - 180
    pitch = normalized_params[4] * 360 - 180
    yaw = normalized_params[5] * 360 - 180

    roll = ((roll + 180) % 360) - 180
    pitch = ((pitch + 180) % 360) - 180
    yaw = ((yaw + 180) % 360) - 180

    return [x, y, z, roll, pitch, yaw]

def update_yaml_file_multiple(obj_names, positions_and_rotations_list):
    # obj_names与positions_and_rotations_list一一对应
    for obj_name, pos_rot in zip(obj_names, positions_and_rotations_list):
        obj_index = next(i for i, obj in enumerate(objects) if obj['name'] == obj_name)
        objects[obj_index]['position'] = [float(x) for x in pos_rot[:3]]
        objects[obj_index]['rotation'] = [float(r) for r in pos_rot[3:]]

    with open("/home/ubuntu/桌面/control/env_xq/object_config.yaml", 'w') as f:
        yaml.dump({'objects': objects}, f, default_flow_style=False, allow_unicode=True)
    print("Updated YAML with new positions and rotations for all specified objects.")

def log_positions_and_rotations(obj_names, positions_and_rotations_list, generation):
    log_file_path = "/home/ubuntu/桌面/control/env_xq/image_attack/positions_log.txt"
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Generation {generation+1}:\n")
        for obj_name, pos_rot in zip(obj_names, positions_and_rotations_list):
            position = pos_rot[:3]
            rotation = pos_rot[3:]
            log_file.write(f"Object: {obj_name}\n")
            log_file.write(f"Position: {position}\n")
            log_file.write(f"Rotation: {rotation}\n\n")

def set_object_positions_and_rotations(obj_names, positions_and_rotations_list):
    update_yaml_file_multiple(obj_names, positions_and_rotations_list)

def get_scene_images():
    result = subprocess.run(["python", "/home/ubuntu/桌面/control/env_xq/image_attack/generate_three_images.py"], capture_output=True)
    if result.returncode == 0:
        image_paths = [
            "/home/ubuntu/桌面/control/env_xq/observations/scene_image_18.png",
            "/home/ubuntu/桌面/control/env_xq/observations/scene_image_30.png",
            "/home/ubuntu/桌面/control/env_xq/observations/scene_image_37.png"
        ]
        
        images = []
        for image_path in image_paths:
            try:
                image = np.array(Image.open(image_path))
                images.append(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {str(e)}")
                return None  
        return images
    else:
        print("Error running generate_three_images.py:", result.stderr.decode())
        return None

def fitness_func(normalized_params):
    # 将 normalized_params 分割为多个物品的参数
    positions_and_rotations_list = []
    for i in range(num_objects):
        start = i * 6
        end = start + 6
        single_obj_params = normalized_params[start:end]
        denorm_params = denormalize_single_obj_params(single_obj_params)
        positions_and_rotations_list.append(denorm_params)

    # 设置物品位置和旋转
    set_object_positions_and_rotations(object_names_to_update, positions_and_rotations_list)

    # 获取图像
    images = get_scene_images()
    if images is None:
        return float('inf')
    
    similarities = []
    for image in images:
        image_tensor = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).squeeze().item()
        similarities.append(similarity)

    average_similarity = np.mean(similarities)
    return -average_similarity

num_generations = 10
for generation in range(num_generations):
    normalized_solutions = es.ask()
    fitness_values = np.array([fitness_func(solution) for solution in normalized_solutions])
    es.tell(fitness_values)

    best_solution = es.result()[0]
    best_fitness = es.result()[1]

    # 将 best_solution 同样解析成多物品的参数，用于日志记录
    best_positions_and_rotations_list = []
    for i in range(num_objects):
        start = i * 6
        end = start + 6
        single_obj_params = best_solution[start:end]
        denorm_params = denormalize_single_obj_params(single_obj_params)
        best_positions_and_rotations_list.append(denorm_params)

    log_positions_and_rotations(object_names_to_update, best_positions_and_rotations_list, generation)
    print(f"Generation {generation+1}, Best fitness: {best_fitness}")

# 最终最优解处理
optimized_positions_and_rotations_list = []
final_best_solution = es.result()[0]
for i in range(num_objects):
    start = i * 6
    end = start + 6
    single_obj_params = final_best_solution[start:end]
    denorm_params = denormalize_single_obj_params(single_obj_params)
    optimized_positions_and_rotations_list.append(denorm_params)

print(f"Optimized positions and rotations: {optimized_positions_and_rotations_list}")
set_object_positions_and_rotations(object_names_to_update, optimized_positions_and_rotations_list)
final_images = get_scene_images()

if final_images is not None and len(final_images) > 0:
    final_image = final_images[0]
    final_pil_image = Image.fromarray(final_image)
    final_pil_image.save("optimized_scene_image.png")
    print("Saved final optimized scene image as optimized_scene_image.png")
else:
    print("Failed to retrieve final images.")
