import numpy as np
import math
import habitat_sim
import magnum as mn
import cv2
import yaml
import os
from habitat_sim.utils.common import quat_rotate_vector, quat_from_angle_axis, quat_from_coeffs
from scipy.spatial.transform import Rotation as R

class HabitatSimEnv:
    def __init__(self, scene_path):
        # 初始化仿真器配置
        self.sim_cfg = habitat_sim.SimulatorConfiguration()
        self.sim_cfg.scene_id = scene_path
        self.sim_cfg.enable_physics = True  # 启用物理引擎

        # 创建相机传感器配置
        self.sensor_specs = []
        color_sensor_spec = habitat_sim.CameraSensorSpec()
        color_sensor_spec.uuid = "color_sensor"
        color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_spec.resolution = [512, 512]
        color_sensor_spec.position = [0.0, 1.5, 0.0]
        self.sensor_specs.append(color_sensor_spec)

        # 创建代理配置
        self.agent_cfg = habitat_sim.agent.AgentConfiguration()
        self.agent_cfg.sensor_specifications = self.sensor_specs

        # 创建仿真器
        cfg = habitat_sim.Configuration(self.sim_cfg, [self.agent_cfg])
        self.sim = habitat_sim.Simulator(cfg)

        # 初始化代理
        self.agent = self.sim.initialize_agent(0)

        # 保存图像计数器
        self.image_count = 0

    def set_agent_initial_position(self, position, rotation_quaternion, euler_angle):
        """
        设置智能体的初始位置和旋转
        """
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(position)  # 设置初始位置

        # 将四元数转换为 [x, y, z, w] 的数组
        rotation_quaternion_coeffs = rotation_quaternion  # [x, y, z, w]

        # 设置代理的旋转
        agent_state.rotation = rotation_quaternion_coeffs

        # 应用状态
        self.agent.set_state(agent_state)

        # 打印旋转分量和欧拉角
        print(f"Initial agent position set to: {agent_state.position}, rotation: {agent_state.rotation}, Euler Angles: {euler_angle}")

    def save_observation(self):
        """
        保存当前观察结果图像
        """
        observations = self.sim.get_sensor_observations()
        rgb_img = observations["color_sensor"]  # 获取传感器的数据
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)  # 转换为OpenCV使用的BGR格式

        # 创建保存路径
        save_path = f"/home/ubuntu/桌面/control/env_xq/observations/scene_image_{self.image_count}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 保存图像
        cv2.imwrite(save_path, rgb_img)
        print(f"Image saved to {save_path}")

        # 更新图像计数器
        self.image_count += 1

    def display_observations(self, positions, rotations, euler_angles):
        """
        显示每个位置的观察结果并保存图像
        """
        for position, rotation, euler_angle in zip(positions, rotations, euler_angles):
            self.set_agent_initial_position(position, rotation, euler_angle)

            # 等待仿真稳定
            self.step_simulation()

            # 获取当前观察结果
            observations = self.sim.get_sensor_observations()
            rgb_img = observations["color_sensor"]  # 获取传感器的数据
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)  # 转换为OpenCV使用的BGR格式

            # 显示图像
            cv2.imshow("Scene", rgb_img)
            cv2.waitKey(1)  # 短暂等待，使窗口能正确显示

            # 保存图像
            self.save_observation()

        cv2.destroyAllWindows()

    def step_simulation(self):
        """
        仿真器步进
        """
        self.sim.step_physics(1.0 / 60.0)

    def load_objects_from_file(self, config_file):
        """
        从 YAML 文件加载物品，并设置位置、旋转和缩放
        """
        # 获取物体模板和刚体物体管理器
        obj_templates_mgr = self.sim.get_object_template_manager()
        rigid_obj_mgr = self.sim.get_rigid_object_manager()

        # 读取 YAML 配置文件
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)

        for obj in data["objects"]:
            # 获取物品信息
            obj_path = obj["path"]
            position = obj["position"]
            rotation_angles = obj["rotation"]

            # 加载物品的配置文件，返回模板 ID 列表
            loaded_template_ids = obj_templates_mgr.load_configs(obj_path)

            if not loaded_template_ids:
                print(f"Error: Failed to load object template for {obj['name']}.")
                continue
            else:
                print(f"Object template loaded successfully for {obj['name']}.")

            # 获取模板 ID
            template_id = loaded_template_ids[0]

            # 使用模板 ID 获取物品模板
            obj_template = obj_templates_mgr.get_template_by_id(template_id)

            # 如果配置中有 'scale' 字段，设置物品的缩放比例
            if "scale" in obj:
                scale_factors = obj["scale"]
                obj_template.scale = mn.Vector3(scale_factors)
                # 重新注册修改后的模板
                obj_templates_mgr.register_template(obj_template, force_registration=True)
                print(f"Object '{obj['name']}' scale set to: {scale_factors}")

            # 使用模板 ID 添加物品到场景中
            object_instance = rigid_obj_mgr.add_object_by_template_id(template_id)

            if object_instance is not None:
                # 设置物体的位置
                object_instance.translation = np.array(position)

                # 使用欧拉角设置物体的旋转
                rotation_x = quat_from_angle_axis(np.deg2rad(rotation_angles[0]), np.array([1, 0, 0]))
                rotation_y = quat_from_angle_axis(np.deg2rad(rotation_angles[1]), np.array([0, 1, 0]))
                rotation_z = quat_from_angle_axis(np.deg2rad(rotation_angles[2]), np.array([0, 0, 1]))
                # 组合旋转（顺序：Z * Y * X）
                combined_rotation = rotation_z * rotation_y * rotation_x

                # 将 Python 四元数转换为 Magnum 四元数
                rotation_quaternion = mn.Quaternion(combined_rotation.imag, combined_rotation.real)

                # 设置物体的旋转
                object_instance.rotation = rotation_quaternion
                print(f"Object '{obj['name']}' placed at position: {position} with rotation: {rotation_angles} degrees")
            else:
                print(f"Error: Object '{obj['name']}' could not be added to the scene.")

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
            rotation_quaternion = R.from_euler('yxz', [yaw, pitch, roll], degrees=False).as_quat()  # [x, y, z, w]

            positions.append(agent_position)
            rotations.append(rotation_quaternion)  # 保持为 [x, y, z, w]

    return positions, euler_angles, rotations

# 使用示例
scene_path = "/home/ubuntu/桌面/control/test_data/mp3d/v1/tasks/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb"
env = HabitatSimEnv(scene_path)

# 输入点 a 的坐标
a_point = [-0.54669417, 0, 14.629016  ]
radius_list = [1.6]  # 半径列表
num_views = 9  # 每个半径的视图数量
positions, euler_angles, rotations = calculate_positions_and_rotations(a_point, radius_list, num_views)

# 从 YAML 文件加载物品
config_file = "/home/ubuntu/桌面/control/env_xq/object_config.yaml"
env.load_objects_from_file(config_file)

# 显示观察结果并保存图像
env.display_observations(positions, rotations, euler_angles)
