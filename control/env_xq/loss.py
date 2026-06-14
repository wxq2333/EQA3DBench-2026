import yaml
import numpy as np
import random
import subprocess
import time
import pandas as pd
import os

def compute_loss(score_file_path):
    # 从 CSV 文件中读取相似度得分，指定没有列名
    scores_df = pd.read_csv(score_file_path, header=None)
    
    # 获取相似度得分的平均值
    mean_score = scores_df[0].mean()  # 0 表示第一列
    return mean_score

def modify_object_config(file_path, object_name, new_position=None, new_rotation=None):
    # 读取现有的YAML文件
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    # 检查数据格式是否正确
    if not isinstance(data, dict) or 'objects' not in data:
        raise ValueError("The YAML file format is incorrect. Expected a dictionary with a key 'objects'.")

    # 获取物品列表
    objects = data['objects']

    # 遍历找到要修改的物品
    for obj in objects:
        if obj['name'] == object_name:
            # 修改位置
            if new_position:
                obj['position'] = new_position
            # 修改旋转
            if new_rotation is not None:
                obj['rotation'] = new_rotation

    # 将修改后的数据写回文件
    with open(file_path, 'w') as file:
        yaml.dump(data, file)


def run_mp3d_view():
    # 调用mp3d_view.py脚本来更新环境
    subprocess.run(["python3", "mp3d_view.py"])

def run_seg():
    # 调用seg.py脚本来计算相似度分数
    subprocess.run(["python3", "seg.py"])

def save_iteration_result(output_dir, iteration, loss, position, rotation):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    # 保存结果到CSV文件
    output_path = os.path.join(output_dir, "optimization_results.csv")
    # 如果文件不存在，创建并写入表头
    if not os.path.exists(output_path):
        with open(output_path, 'w') as file:
            file.write("Iteration,Loss,Position,Rotation\n")
    # 追加写入每次迭代结果
    with open(output_path, 'a') as file:
        file.write(f"{iteration},{loss},{position},{rotation}\n")
    print(f"Iteration {iteration} result saved to {output_path}")

def optimize_object_position(file_path, object_name, score_file_path, output_dir, iterations=100):
    # 初始化最优的损失值为一个较大数
    best_loss = float('inf')
    best_position = None
    best_rotation = None

    # 初始随机位置和旋转（使用三个角度）
    current_position = [8.8, 3.0, 11]  # 初始值
    current_rotation = [30, 30, 30]  # 初始旋转角度：X, Y, Z 轴

    for i in range(iterations):
        # 随机生成新位置
        new_position = [current_position[0] + random.uniform(-1, 1),
                        current_position[1]+ random.uniform(-1, 1),
                        current_position[2] + random.uniform(-1, 1)]
        
        # 随机生成每个轴的旋转角度
        new_rotation = [current_rotation[0] + random.uniform(-5, 5),  # X轴旋转
                        current_rotation[1] + random.uniform(-5, 5),  # Y轴旋转
                        current_rotation[2] + random.uniform(-5, 5)]  # Z轴旋转
        
        # 更新配置文件
        modify_object_config(file_path, object_name, new_position, new_rotation)

        # 更新环境并保存新图像
        run_mp3d_view()
        time.sleep(1)  # 等待图像保存完成

        # 计算新的相似度分数
        run_seg()
        time.sleep(1)  # 等待分数计算完成

        # 读取并计算新的损失值
        current_loss = compute_loss(score_file_path)

        print(f"Iteration {i+1}: Loss = {current_loss:.4f}, Position = {new_position}, Rotation = {new_rotation}")

        # 保存每次迭代的结果
        save_iteration_result(output_dir, i+1, current_loss, new_position, new_rotation)

        # 如果新的损失值小于当前最优的损失值，则更新最优值
        if current_loss < best_loss:
            best_loss = current_loss
            best_position = new_position
            best_rotation = new_rotation

    # 最优配置保存
    print(f"Best Loss: {best_loss:.4f} at Position: {best_position}, Rotation: {best_rotation}")
    modify_object_config(file_path, object_name, best_position, best_rotation)

def main():
    # 示例使用
    file_path = '/home/ubuntu/桌面/control/env_xq/object_config.yaml'  # 物品配置文件路径
    score_file_path = '/home/ubuntu/桌面/control/env_xq/score_data/scores_matrix.csv'  # 相似度得分CSV文件路径
    output_dir = '/home/ubuntu/桌面/control/env_xq/optimization_results'  # 存储优化结果的输出文件夹路径
    object_name = 'Dining_Table'  # 需要优化位置的物品名称

    # 执行优化过程
    optimize_object_position(file_path, object_name, score_file_path, output_dir,iterations=1000)

if __name__ == "__main__":
    main()