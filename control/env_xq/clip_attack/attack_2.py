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
text = clip.tokenize(["a computer on the table"]).to(device)

# 设置 OpenES
num_params = 6  # Microwave 的 (x, y, z) 位置和 (roll, pitch, yaw) 旋转
npop = 50  # 种类大小
sigma = 0.1 # 变异系数（标准差）
alpha = 0.1  # 学习率
es = OpenES(num_params=num_params,
            sigma_init=sigma,
            popsize=npop,
            learning_rate=alpha,
            rank_fitness=False)

# 针对 Microwave 设置优化范围
position_ranges = (-1.2, 1.1, 1.1, 15.1, 16.1) # (x_min, x_max, y_fixed, z_min, z_max)
rotation_range = (-180, 180)  # 旋转范围，均为 [-180, 180]

# 对参数进行归一化
def normalize_params(positions_and_rotations):
    x_min, x_max, y_fixed, z_min, z_max = position_ranges
    roll_min, roll_max = rotation_range
    
    # 归一化位置
    x_norm = (positions_and_rotations[0] - x_min) / (x_max - x_min)
    z_norm = (positions_and_rotations[2] - z_min) / (z_max - z_min)
    # 固定 y 值，不归一化
    y = y_fixed

    # 归一化旋转 (roll, pitch, yaw)
    roll_norm = (positions_and_rotations[3] + 180) / 360
    pitch_norm = (positions_and_rotations[4] + 180) / 360
    yaw_norm = (positions_and_rotations[5] + 180) / 360

    # 返回归一化后的值
    return [x_norm, y, z_norm, roll_norm, pitch_norm, yaw_norm]

# 反归一化参数
def denormalize_params(normalized_positions_and_rotations):
    x_min, x_max, y_fixed, z_min, z_max = position_ranges
    roll_min, roll_max = rotation_range

    # 反归一化位置
    x = normalized_positions_and_rotations[0] * (x_max - x_min) + x_min
    z = normalized_positions_and_rotations[2] * (z_max - z_min) + z_min
    # 固定 y 值
    y = y_fixed

    # 将位置值限制在范围内
    x = np.clip(x, x_min, x_max)
    z = np.clip(z, z_min, z_max)

    # 反归一化旋转 (roll, pitch, yaw)
    roll = normalized_positions_and_rotations[3] * 360 - 180
    pitch = normalized_positions_and_rotations[4] * 360 - 180
    yaw = normalized_positions_and_rotations[5] * 360 - 180

    # 将旋转值限制在 [-180, 180] 范围内
    roll = ((roll + 180) % 360) - 180
    pitch = ((pitch + 180) % 360) - 180
    yaw = ((yaw + 180) % 360) - 180

    # 返回反归一化后的值
    return [x, y, z, roll, pitch, yaw]

# 更新 YAML 文件中的 Microwave 位置和旋转
def update_yaml_file(positions_and_rotations):
    microwave_index = next(i for i, obj in enumerate(objects) if obj['name'] == 'computer')
    
    # 取出位置和旋转部分
    position = positions_and_rotations[:3]
    rotation = positions_and_rotations[3:]
    
    # 更新 YAML 文件中的 Microwave 的位置和旋转
    objects[microwave_index]['position'] = [float(x) for x in position]
    objects[microwave_index]['rotation'] = [float(r) for r in rotation]
    
    # 将更新后的对象写入 YAML 文件
    with open("/home/ubuntu/桌面/control/env_xq/object_config.yaml", 'w') as f:
        yaml.dump({'objects': objects}, f, default_flow_style=False, allow_unicode=True)
    print("Updated YAML file with new positions and rotations for Microwave.")

# 记录位置和旋转到文件
def log_positions_and_rotations(positions_and_rotations, generation):
    log_file_path = "/home/ubuntu/桌面/control/env_xq/clip_attack/positions_log.txt"
    with open(log_file_path, 'a') as log_file:
        position = positions_and_rotations[:3]
        rotation = positions_and_rotations[3:]
        log_file.write(f"Generation {generation+1}:\n")
        log_file.write(f"Position: {position}\n")
        log_file.write(f"Rotation: {rotation}\n")
        log_file.write("\n")


# 定义函数以更新 Microwave 的位置和旋转
def set_object_positions_and_rotations(positions_and_rotations):
    # 更新 YAML 文件中的 Microwave 位置和旋转
    update_yaml_file(positions_and_rotations)
    microwave_index = next(i for i, obj in enumerate(objects) if obj['name'] == 'computer')

# 定义从模拟器获取渲染的场景图像的函数
def get_scene_image():
    # 调用 camera.py 获取图像
    result = subprocess.run(["python", "/home/ubuntu/桌面/control/env_xq/camera.py"], capture_output=True)
    
    if result.returncode == 0:
        # 假设 camera.py 将图像保存为 scene_image_2.png
        return np.array(Image.open("/home/ubuntu/桌面/control/env_xq/observations/scene_image_2.png"))
    else:
        print("Error running camera.py:", result.stderr.decode())
        return None

# 定义适应度函数
def fitness_func(normalized_positions_and_rotations):
    # 反归一化参数
    positions_and_rotations = denormalize_params(normalized_positions_and_rotations)
    
    # 设置 Microwave 的位置和旋转
    set_object_positions_and_rotations(positions_and_rotations)

    # 从模拟器获取渲染的场景图像
    image = get_scene_image()
    if image is None:
        return float('inf')  # 如果获取图像失败，返回无穷大作为适应度值
    
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
    print("similarity is", similarity)
    # 返回适应度值（希望最小化的目标）
    return -similarity

# 运行 OpenES 优化
num_generations = 100  # 最大代数
#sigma_decay_rate = 0.99  # 每代减少的比例
for generation in range(num_generations):
    # 生成归一化的解
    normalized_solutions = es.ask()
    # 计算适应度值
    fitness_values = np.array([fitness_func(solution) for solution in normalized_solutions])
    # 更新 OpenES 的内部状态
    es.tell(fitness_values)
    
    # 逐步递减 sigma
    #es.sigma *= sigma_decay_rate

    # 获取当前最优解（归一化后的）
    best_solution = es.result()[0]
    # 反归一化最优解
    best_solution_denormalized = denormalize_params(best_solution)
    # 记录最优解的位置和旋转
    log_positions_and_rotations(best_solution_denormalized, generation)

    # 打印当前最优解的相似度
    best_fitness = es.result()[1]
    print(f"Generation {generation+1}, Best fitness: {best_fitness}")
    
    # 如果希望在一定阀值内停止，可以添加如下判断
#    if best_fitness > -0.01:  # 设定的相似度阀值
#        break

# 获取优化后的位置参数和旋转参数
optimized_positions_and_rotations = es.result()[0]
optimized_positions_and_rotations_denormalized = denormalize_params(optimized_positions_and_rotations)
print(f"Optimized positions and rotations: {optimized_positions_and_rotations_denormalized}")

# 设置 Microwave 到最优解位置和旋转并获取优化后的场景图像
set_object_positions_and_rotations(optimized_positions_and_rotations_denormalized)
final_image = get_scene_image()

# 保存最终的优化图像
if final_image is not None:
    final_pil_image = Image.fromarray(final_image)
    final_pil_image.save("optimized_scene_image.png")