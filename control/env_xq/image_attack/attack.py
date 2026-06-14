#对单张图片实现成功的攻击
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
text = clip.tokenize(["a Microwave on the table"]).to(device)

# 设置 OpenES
num_params = 6  # Microwave 的 (x, y, z) 位置和 (roll, pitch, yaw) 旋转
npop = 50  # 种群大小
sigma = 0.1 # 变异系数（标准差）
alpha = 0.1  # 学习率
es = OpenES(num_params=num_params,
            sigma_init=sigma,
            popsize=npop,
            learning_rate=alpha,
            rank_fitness=False)

# 针对 Microwave 设置优化范围（这里仍然用示意的物品名称和范围）
# 请根据实际物体名称和期望范围进行修改
position_ranges = (-1.8, -1.1, 1.65, -3.8, -3.5)  # (x_min, x_max, y_fixed, z_min, z_max)
rotation_range = (-180, 180)  # 旋转范围

def normalize_params(positions_and_rotations):
    x_min, x_max, y_fixed, z_min, z_max = position_ranges
    roll_min, roll_max = rotation_range
    
    # 归一化位置
    x_norm = (positions_and_rotations[0] - x_min) / (x_max - x_min)
    z_norm = (positions_and_rotations[2] - z_min) / (z_max - z_min)
    y = y_fixed

    # 归一化旋转 (roll, pitch, yaw)
    roll_norm = (positions_and_rotations[3] + 180) / 360
    pitch_norm = (positions_and_rotations[4] + 180) / 360
    yaw_norm = (positions_and_rotations[5] + 180) / 360

    return [x_norm, y, z_norm, roll_norm, pitch_norm, yaw_norm]

def denormalize_params(normalized_positions_and_rotations):
    x_min, x_max, y_fixed, z_min, z_max = position_ranges
    roll_min, roll_max = rotation_range

    # 反归一化位置
    x = normalized_positions_and_rotations[0] * (x_max - x_min) + x_min
    z = normalized_positions_and_rotations[2] * (z_max - z_min) + z_min
    y = y_fixed

    # 将位置值限制在范围内
    x = np.clip(x, x_min, x_max)
    z = np.clip(z, z_min, z_max)

    # 反归一化旋转 (roll, pitch, yaw)
    roll = normalized_positions_and_rotations[3] * 360 - 180
    pitch = normalized_positions_and_rotations[4] * 360 - 180
    yaw = normalized_positions_and_rotations[5] * 360 - 180

    # 将旋转值限制在 [-180, 180]
    roll = ((roll + 180) % 360) - 180
    pitch = ((pitch + 180) % 360) - 180
    yaw = ((yaw + 180) % 360) - 180

    return [x, y, z, roll, pitch, yaw]

# 更新 YAML 文件中的指定物品位置和旋转
def update_yaml_file(positions_and_rotations):
    # 假设 YAML 文件中包含名为 'cooking_pan_with_black_and_white_spotty_coating' 的对象
    cooking_pan_with_black_and_white_spotty_coating_index = next(i for i, obj in enumerate(objects) if obj['name'] == 'cooking_pan_with_black_and_white_spotty_coating')
    
    # 取出位置和旋转部分
    position = positions_and_rotations[:3]
    rotation = positions_and_rotations[3:]
    
    # 使用索引访问 objects 列表来修改相应物品的 position 和 rotation
    objects[cooking_pan_with_black_and_white_spotty_coating_index]['position'] = [float(x) for x in position]
    objects[cooking_pan_with_black_and_white_spotty_coating_index]['rotation'] = [float(r) for r in rotation]
    
    # 将更新后的对象写入 YAML 文件
    with open("/home/ubuntu/桌面/control/env_xq/object_config.yaml", 'w') as f:
        yaml.dump({'objects': objects}, f, default_flow_style=False, allow_unicode=True)
    print("Updated YAML file with new positions and rotations for cooking_pan_with_black_and_white_spotty_coating.")

def log_positions_and_rotations(positions_and_rotations, generation):
    log_file_path = "/home/ubuntu/桌面/control/env_xq/image_attack/positions_log.txt"
    with open(log_file_path, 'a') as log_file:
        position = positions_and_rotations[:3]
        rotation = positions_and_rotations[3:]
        log_file.write(f"Generation {generation+1}:\n")
        log_file.write(f"Position: {position}\n")
        log_file.write(f"Rotation: {rotation}\n\n")

def set_object_positions_and_rotations(positions_and_rotations):
    # 更新 YAML 文件
    update_yaml_file(positions_and_rotations)

# 从模拟器获取三张场景图像
def get_scene_images():
    # 假设 generate_three_images.py 能根据 object_config.yaml 的修改生成3张场景图片
    result = subprocess.run(["python", "/home/ubuntu/桌面/control/env_xq/image_attack/generate_three_images.py"], capture_output=True)
    
    if result.returncode == 0:
        image_paths = [
            "/home/ubuntu/桌面/control/env_xq/image_attack/generated_images/frame_000000.jpg",
            "/home/ubuntu/桌面/control/env_xq/image_attack/generated_images/frame_000001.jpg",
            "/home/ubuntu/桌面/control/env_xq/image_attack/generated_images/frame_000002.jpg"
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

def fitness_func(normalized_positions_and_rotations):
    # 反归一化参数
    positions_and_rotations = denormalize_params(normalized_positions_and_rotations)
    
    # 设置物品位置和旋转
    set_object_positions_and_rotations(positions_and_rotations)

    # 从模拟器获取图像
    images = get_scene_images()
    if images is None:
        return float('inf')  # 获取图像失败则返回无穷大适应度
    
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
    # 适应度为相似度的负值（我们想最大化相似度，故最小化负相似度）
    return -average_similarity

# 运行 OpenES 优化
num_generations = 100
for generation in range(num_generations):
    normalized_solutions = es.ask()
    fitness_values = np.array([fitness_func(solution) for solution in normalized_solutions])
    es.tell(fitness_values)

    best_solution = es.result()[0]
    best_fitness = es.result()[1]
    best_solution_denormalized = denormalize_params(best_solution)
    log_positions_and_rotations(best_solution_denormalized, generation)

    print(f"Generation {generation+1}, Best fitness: {best_fitness}")

# 优化完成后，获取最优解
optimized_positions_and_rotations = es.result()[0]
optimized_positions_and_rotations_denormalized = denormalize_params(optimized_positions_and_rotations)
print(f"Optimized positions and rotations: {optimized_positions_and_rotations_denormalized}")

# 设置物体到最优解并获取最终的场景图像
set_object_positions_and_rotations(optimized_positions_and_rotations_denormalized)
final_images = get_scene_images()

if final_images is not None and len(final_images) > 0:
    final_image = final_images[0]
    final_pil_image = Image.fromarray(final_image)
    final_pil_image.save("optimized_scene_image.png")
    print("Saved final optimized scene image as optimized_scene_image.png")
else:
    print("Failed to retrieve final images.")
