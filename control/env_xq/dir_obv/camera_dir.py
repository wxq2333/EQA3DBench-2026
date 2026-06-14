import numpy as np
import math
import habitat_sim
import magnum as mn
import cv2
import yaml
import os
from habitat_sim.utils.common import quat_from_coeffs
from scipy.spatial.transform import Rotation as R

class HabitatSimEnv:
    def __init__(self, scene_path, object_config_path):
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

        # 物体配置文件路径
        self.object_config_path = object_config_path

        # 加载物体到环境中
        self.load_objects_from_file()

    def set_agent_initial_position(self, position, rotation_quaternion):
        """
        设置智能体的初始位置和旋转
        """
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(position)
        agent_state.rotation = quat_from_coeffs(rotation_quaternion)
        self.agent.set_state(agent_state)
        print(f"Agent position set to: {position}, rotation: {rotation_quaternion}")

    def save_observation(self):
        """
        保存当前观察结果图像
        """
        observations = self.sim.get_sensor_observations()
        rgb_img = observations["color_sensor"]
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        # 指定保存路径
        save_path = f"/home/ubuntu/桌面/control/env_xq/dir_obv/dir_obv_img/scene_image_{self.image_count}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, rgb_img)
        print(f"Image saved to {save_path}")

        # 更新图像计数器
        self.image_count += 1

    def display_observations(self, positions, rotations):
        """
        显示指定位置的观察结果并保存图像
        """
        for position, rotation in zip(positions, rotations):
            self.set_agent_initial_position(position, rotation)
            self.sim.step_physics(1.0 / 60.0)  # 等待仿真稳定
            self.save_observation()

        print("All images captured.")

    def load_objects_from_file(self):
        """
        从 YAML 文件加载物品，并设置位置和旋转
        """
        obj_templates_mgr = self.sim.get_object_template_manager()
        rigid_obj_mgr = self.sim.get_rigid_object_manager()

        with open(self.object_config_path, 'r') as f:
            data = yaml.safe_load(f)

        for obj in data["objects"]:
            obj_path = obj["path"]
            position = obj["position"]
            rotation_angles = obj["rotation"]

            loaded_template_ids = obj_templates_mgr.load_configs(obj_path)
            if not loaded_template_ids:
                print(f"Error: Failed to load object template for {obj['name']}.")
                continue

            template_id = loaded_template_ids[0]
            obj_template = obj_templates_mgr.get_template_by_id(template_id)

            if "scale" in obj:
                scale_factors = obj["scale"]
                obj_template.scale = mn.Vector3(scale_factors)
                obj_templates_mgr.register_template(obj_template, force_registration=True)
                print(f"Object '{obj['name']}' scale set to: {scale_factors}")

            object_instance = rigid_obj_mgr.add_object_by_template_id(template_id)

            if object_instance is not None:
                object_instance.translation = np.array(position)

                # 将欧拉角转换为四元数
                rotation_quaternion = R.from_euler('xyz', rotation_angles, degrees=True).as_quat()
                object_instance.rotation = mn.Quaternion(mn.Vector3(rotation_quaternion[:3]), rotation_quaternion[3])

                print(f"Object '{obj['name']}' placed at position: {position} with rotation: {rotation_angles}")
            else:
                print(f"Error: Object '{obj['name']}' could not be added to the scene.")

# 使用示例
scene_path = "/home/ubuntu/桌面/control/test_data/mp3d/v1/tasks/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb"
object_config_path = "/home/ubuntu/桌面/control/env_xq/object_config.yaml"
env = HabitatSimEnv(scene_path, object_config_path)

# 指定智能体位置和旋转
positions = [
    [-0.7397326, 0, 16.194292],
    [-0.0397326, 0, 16.194292],
    [0.9602675, 0, 16.194292]
]
rotations = [
    [0, -0.70710688829422, 0, -0.707106709480286],
    [0, -0.70710688829422, 0, -0.707106709480286],
    [0, -0.70710688829422, 0, -0.707106709480286]
]

# 捕获并保存图像
env.display_observations(positions, rotations)
