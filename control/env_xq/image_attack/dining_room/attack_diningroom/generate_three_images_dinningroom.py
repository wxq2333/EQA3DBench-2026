import habitat_sim
import numpy as np
import cv2
import os  # 用于创建目录和检查路径
import yaml  # 用于读取YAML文件
import magnum as mn
from habitat_sim.utils.common import quat_from_angle_axis

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

        # 输出图像保存路径
        self.output_dir = "/home/ubuntu/桌面/control/env_xq/image_attack/dining_room/generated_images"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created directory for images: {self.output_dir}")

    def set_agent_initial_position(self, position, rotation_quaternion):
        """
        设置智能体的初始位置和旋转
        """
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(position)  # 设置初始位置
        agent_state.rotation = np.array(rotation_quaternion)  # [w, x, y, z]

        self.agent.set_state(agent_state)
        print(f"Agent position set to: {agent_state.position}, rotation: {agent_state.rotation}")

    def capture_and_save_observation(self):
        """
        从当前场景获取传感器观察数据并保存单张图像，然后计数器自增。
        """
        # 从仿真器获取传感器观察数据
        observations = self.sim.get_sensor_observations()

        # 获取颜色传感器的图像 (H x W x 4) RGBA
        color_obs = observations["color_sensor"]
        # 转换为BGR
        color_image = cv2.cvtColor(color_obs, cv2.COLOR_RGBA2BGR)

        # 保存图像到指定路径
        image_filename = f"frame_{self.image_count:06d}.jpg"
        image_path = os.path.join(self.output_dir, image_filename)
        cv2.imwrite(image_path, color_image)
        print(f"Saved image: {image_path}")
        self.image_count += 1

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
                # 组合旋转（Z * Y * X）
                combined_rotation = rotation_z * rotation_y * rotation_x

                # 将 Python 四元数转换为 Magnum 四元数
                rotation_quaternion = mn.Quaternion(combined_rotation.imag, combined_rotation.real)

                # 设置物体的旋转
                object_instance.rotation = rotation_quaternion
                print(f"Object '{obj['name']}' placed at position: {position} with rotation: {rotation_angles} degrees")
            else:
                print(f"Error: Object '{obj['name']}' could not be added to the scene.")


# 使用示例
scene_path = "/home/ubuntu/桌面/control/test_data/mp3d/v1/tasks/mp3d/8WUmhLawc2A/8WUmhLawc2A.glb"
env = HabitatSimEnv(scene_path)

# 从 YAML 文件加载物品
config_file = "/home/ubuntu/桌面/control/env_xq/image_attack/dining_room/attack_diningroom/object_config_diningroom.yaml"
env.load_objects_from_file(config_file)

# 定义三个位置和对应的旋转四元数
positions_and_rotations = [
    ([0.00955534 ,-0.3   ,     -3.7763028], [0,  -0.939692676067352, 0, -0.342020064592361]),
    ([-0.20560469 ,-0.3    ,    -3.5730941 ], [0,-0.866025499598342, 0, -0.499999926845038]),
    ([-0.55938154 ,-0.3    ,    -3.3888922 ], [0, -0.866025531397977, 0, -0.499999943053257])
]

# 对每个位置设置智能体，并抓取图像保存
for pos, rot in positions_and_rotations:
    env.set_agent_initial_position(pos, rot)
    env.capture_and_save_observation()

print("All images have been captured and saved.")
