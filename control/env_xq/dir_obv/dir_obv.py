import numpy as np
import habitat_sim
import magnum as mn
import cv2
import os
from habitat_sim.utils.common import quat_from_coeffs

class HabitatSimEnv:
    def __init__(self, scene_path, save_dir):
        # 初始化仿真器配置
        self.sim_cfg = habitat_sim.SimulatorConfiguration()
        self.sim_cfg.scene_id = scene_path
        self.sim_cfg.enable_physics = True

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

        # 指定保存路径
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)  # 确保路径存在

    def set_agent_position_and_rotation(self, position, rotation_quaternion):
        """
        设置智能体的初始位置和旋转
        """
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(position)
        agent_state.rotation = quat_from_coeffs(rotation_quaternion)  # [x, y, z, w]

        self.agent.set_state(agent_state)

    def save_observation(self):
        """
        保存当前观察结果图像
        """
        observations = self.sim.get_sensor_observations()
        rgb_img = observations["color_sensor"]
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        # 创建保存路径
        save_path = os.path.join(self.save_dir, f"scene_image_{self.image_count}.png")
        cv2.imwrite(save_path, rgb_img)
        print(f"Image saved to {save_path}")

        # 更新图像计数器
        self.image_count += 1

    def capture_images(self, positions, rotations):
        """
        在给定的位置和旋转捕获图像
        """
        for position, rotation in zip(positions, rotations):
            self.set_agent_position_and_rotation(position, rotation)
            self.sim.step_physics(1.0 / 60.0)
            self.save_observation()

# 使用示例
scene_path = "/home/ubuntu/桌面/control/test_data/mp3d/v1/tasks/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb"
save_dir = "/home/ubuntu/桌面/control/env_xq/dir_obv/dir_obv_img"  # 指定保存路径
env = HabitatSimEnv(scene_path, save_dir)

# 位置和旋转设置
positions = [
    [-0.7397326, 0, 16.194292],
    [-0.0397326, 0, 16.194292],
    [0.9602675, 0, 16.194292]
]
rotations = [
    [0, -0.70710688829422, 0,-0.707106709480286],
    [0, -0.70710688829422, 0,-0.707106709480286],
    [ 0, -0.70710688829422, 0,-0.707106709480286]
]

# 捕获并保存图像
env.capture_images(positions, rotations)
