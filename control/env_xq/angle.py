import numpy as np
import math
from scipy.spatial.transform import Rotation as R

def calculate_positions_and_rotations(a, radius_list, num_views):
    positions = []
    rotations = []
    euler_angles = []  # 用于存储欧拉角结果

    for radius in radius_list:
        for i in range(num_views):
            angle = (2 * math.pi / num_views) * i

            # 计算新的坐标
            pos_x = a[0] + radius * math.cos(angle)
            pos_z = a[2] + radius * math.sin(angle)
            agent_position = [pos_x, a[1], pos_z]  # 保持 Y 坐标不变

            # 计算代理指向目标的方向向量
            direction_to_center = np.array(a) - np.array(agent_position)
            direction_to_center[1] = 0  # 忽略 Y 轴分量
            direction_to_center /= np.linalg.norm(direction_to_center)  # 归一化

            # 代理的默认前向向量（负 Z 轴）
            agent_forward = np.array([0, 0, -1])

            # 计算点积和叉积
            dot_product = np.dot(agent_forward, direction_to_center)
            cross_product = np.cross(agent_forward, direction_to_center)

            # 计算需要的 yaw 角度
            yaw = math.atan2(cross_product[1], dot_product)
            pitch = 0  # 无需俯仰旋转
            roll = 0  # 无需翻滚旋转

            # 存储欧拉角（转换为度数）
            euler_angles.append([math.degrees(yaw), math.degrees(pitch), math.degrees(roll)])

            # 使用 scipy 进行欧拉角到四元数的转换
            rotation_quaternion = R.from_euler('yxz', [yaw, pitch, roll], degrees=False).as_quat()
            # scipy 返回的四元数顺序为 [x, y, z, w]，需要调整为 [w, x, y, z]
            rotation_quaternion = np.array([rotation_quaternion[3], rotation_quaternion[0], rotation_quaternion[1], rotation_quaternion[2]])

            positions.append(agent_position)
            rotations.append(rotation_quaternion)

    return positions, euler_angles, rotations

# 输入点 a 的坐标
a_point = [14, -1.3, 11]  # 示例坐标
radius_list = [1]  # 半径列表
num_views = 3  # 每个半径的视图数量

positions, euler_angles, rotations = calculate_positions_and_rotations(a_point, radius_list, num_views)

# 输出结果
for i in range(len(positions)):
    print(f"Position: {positions[i]}, Euler Angles (degrees): {euler_angles[i]}, Rotation (Quaternion): {rotations[i]}")
