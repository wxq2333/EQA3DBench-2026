import torch
import clip
from PIL import Image
import numpy as np
import sys
import concurrent.futures
import os
import logging
import shutil
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

import yaml
import subprocess
import time, gc, random
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ===================== 全局结果根目录配置 =====================
ROOT_OUTPUT_DIR = "/home/ubuntu/桌面/control/env_xq/image_attack/bath_room/bayes_optim_all_results"
os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)

# 固定路径（原始工程不动）
sys.path.append('/home/ubuntu/桌面/control/env_xq/estool')
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

object_names_to_update = [
    "soap_dispenser",
    "hair_dryer",
    "soap_dish" ,
    "soap_dispenser",
    "toothbrush",
    "toothpaste_tube"
]

yaml_path = "/home/ubuntu/桌面/control/env_xq/image_attack/bath_room/attack_bathroom/object_config_bathroom.yaml"
base_yaml_path = "/home/ubuntu/桌面/control/env_xq/image_attack/bath_room/attack_bathroom/object_config_bathroom_base.yaml"

text_descriptions = [
   "The soap is located on the countertop. Specifically, it appears to be a liquid soap dispenser, sitting near a set of towels and other items. ",
   "The soap dish appears to contain soap"
]

position_ranges = (11.6, 12, 1.0, -10.9, -7.9)
rotation_range = (-180, 180)

# ===================== 归一化/反归一化（不变） =====================
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

def update_yaml_file_multiple(obj_names, positions_and_rotations_list, current_objects):
    for obj_name, pos_rot in zip(obj_names, positions_and_rotations_list):
        obj_index = next((i for i, obj in enumerate(current_objects) if obj['name'] == obj_name), None)
        if obj_index is not None:
            current_objects[obj_index]['position'] = [float(x) for x in pos_rot[:3]]
            current_objects[obj_index]['rotation'] = [float(r) for r in pos_rot[3:]]
        else:
            logging.warning(f"Object '{obj_name}' not found in YAML configuration.")
    with open(yaml_path, 'w') as f:
        yaml.dump({'objects': current_objects}, f, default_flow_style=False, allow_unicode=True)

def set_object_positions_and_rotations(obj_names, positions_and_rotations_list, current_objects):
    update_yaml_file_multiple(obj_names, positions_and_rotations_list, current_objects)

def get_scene_images(max_retries=3, retry_delay=2):
    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(
                ["python", "/home/ubuntu/桌面/control/env_xq/image_attack/bath_room/attack_bathroom/generate_three_images_bathroom.py"],
                capture_output=True, text=True, check=True
            )
            image_paths = [
                "/home/ubuntu/桌面/control/env_xq/image_attack/bath_room/images_bathroom/frame_000000.jpg",
                "/home/ubuntu/桌面/control/env_xq/image_attack/bath_room/images_bathroom/frame_000001.jpg",
                "/home/ubuntu/桌面/control/env_xq/image_attack/bath_room/images_bathroom/frame_000002.jpg"
            ]
            images = []
            for image_path in image_paths:
                if not os.path.exists(image_path): raise FileNotFoundError(image_path)
                img = Image.open(image_path).convert('RGB')
                img_np = np.array(img)
                if img_np.shape != (512,512,3): raise ValueError("shape error")
                images.append(img_np)
            return images
        except Exception as e:
            if attempt < max_retries: time.sleep(retry_delay)
    return None

def batch_calculate_similarity(images, text_features):
    try:
        img_tensors = torch.stack([preprocess(Image.fromarray(img)) for img in images]).to(device)
        img_tensors = img_tensors.float()
        with torch.no_grad():
            img_feat = model.encode_image(img_tensors)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ text_features.T).squeeze().cpu().numpy()
        return sim
    except:
        return np.array([0.0]*len(images))

def fitness_func(normalized_params, text_features, current_objects):
    try:
        pos_list = []
        for i in range(len(object_names_to_update)):
            seg = normalized_params[i*6:(i+1)*6]
            pos_list.append(denormalize_single_obj_params(seg))
        set_object_positions_and_rotations(object_names_to_update, pos_list, current_objects)
        imgs = get_scene_images()
        if imgs is None: return 1e6
        sims = batch_calculate_similarity(imgs, text_features)
        if not np.all(np.isfinite(sims)): return 1e6
        return -np.mean(sims)
    except:
        return 1e6

# ===================== 日志+结果全部存独立文件夹 =====================
def log_positions_and_rotations_save_newfolder(save_subdir, obj_names, pos_list, generation, text_desc):
    log_dir = os.path.join(save_subdir, "positions_logs")
    os.makedirs(log_dir, exist_ok=True)
    sanitized = "_".join(text_desc.split()[:5])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    log_path = os.path.join(log_dir, f"pos_log_gen{generation+1}_{sanitized}_{ts}.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Generation {generation+1}\n")
        for name, pr in zip(obj_names, pos_list):
            f.write(f"Obj:{name} Pos:{pr[:3]} Rot:{pr[3:]}\n")

# ===================== 贝叶斯优化主逻辑（修复参数命名报错） =====================
def optimize_for_text(text_description, index):
    # 1. 创建本条文本专属结果文件夹
    ts_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_name = "_".join(text_description.split()[:5])
    sub_save_dir = os.path.join(ROOT_OUTPUT_DIR, f"text_{index+1}_{sanitized_name}_{ts_folder}")
    os.makedirs(sub_save_dir, exist_ok=True)

    # 2. 日志重定向到当前独立文件夹
    log_file_path = os.path.join(sub_save_dir, "optimization_running_log.log")
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file_path, encoding='utf-8'), logging.StreamHandler(sys.stdout)]
    )

    logging.info(f"=== Start Bayesian Opt for text {index+1} ===")
    logging.info(f"All results save to: {sub_save_dir}")

    # 加载基准yaml
    with open(base_yaml_path, 'r') as f:
        current_objects = yaml.safe_load(f)['objects']

    num_params = len(object_names_to_update)*6
    popsize = 50
    num_generations = 20

    # CLIP文本特征
    text = clip.tokenize([text_description]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)

    # ===================== 修复：给每个参数添加名称（解决报错核心） =====================
    search_space = [Real(0.0, 1.0, name=f'param_{i}') for i in range(num_params)]

    # 目标函数
    @use_named_args(search_space)
    def objective(**kwargs):
        params = np.array([kwargs[f"param_{i}"] for i in range(num_params)])
        return fitness_func(params, text_feat, current_objects)

    # 回调函数
    def callback(res):
        gen = len(res.func_vals)
        best_sol = res.x
        best_fit = res.fun
        logging.info(f"Gen{gen} BestFitness:{best_fit}")

        best_pos = []
        for i in range(len(object_names_to_update)):
            seg = best_sol[i*6:(i+1)*6]
            best_pos.append(denormalize_single_obj_params(seg))
        log_positions_and_rotations_save_newfolder(sub_save_dir, object_names_to_update, best_pos, gen-1, text_description)
        set_object_positions_and_rotations(object_names_to_update, best_pos, current_objects)
        torch.cuda.empty_cache()
        gc.collect()

    # 执行贝叶斯优化
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=num_generations,
        n_points=popsize,
        random_state=42,
        callback=callback,
        verbose=True
    )

    # 最终最优yaml保存到独立文件夹
    final_yaml_save = os.path.join(sub_save_dir, f"final_best_config.yaml")
    with open(final_yaml_save, 'w', encoding='utf-8') as f:
        f.write(f"# Text:{text_description}\n")
        yaml.dump({"objects":current_objects}, f, default_flow_style=False, allow_unicode=True)
    logging.info(f"Final best YAML saved: {final_yaml_save}")

    # 恢复原始主yaml
    shutil.copy(base_yaml_path, yaml_path)
    logging.info("Restore base yaml done.")

def main():
    for idx, desc in enumerate(text_descriptions):
        optimize_for_text(desc, idx)
    logging.info("All Bayesian optimization finished!")

if __name__ == "__main__":
    main()
