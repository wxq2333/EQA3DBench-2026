import habitat_sim
import numpy as np
import magnum as mn
import cv2
import yaml  # 用于读取YAML文件
import os  # 用于创建目录和检查路径
from habitat_sim.utils.common import quat_rotate_vector, quat_from_angle_axis


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

    def set_agent_initial_position(self, position, rotation_quaternion):
        """
        设置智能体的初始位置和旋转
        """
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(position)  # 设置初始位置

        # 设置初始旋转为指定的四元数
        agent_state.rotation = np.array(rotation_quaternion)  # [w, x, y, z]

        self.agent.set_state(agent_state)
        print(f"Initial agent position set to: {agent_state.position}, rotation: {agent_state.rotation}")

    def display_observations(self):
        """
        显示观察结果并处理键盘事件
        """
        while True:
            observations = self.sim.get_sensor_observations()
            rgb_img = observations["color_sensor"]  # 获取传感器的数据
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)  # 转换为OpenCV使用的BGR格式

            cv2.imshow("Scene", rgb_img)  # 显示图像
            #cv2.waitKey(1)  # 短暂等待，使窗口能正确显示

            key = cv2.waitKey(10)
            if key == ord('w'):
                self.move_agent("w")
            elif key == ord('s'):
                self.move_agent("s")
            elif key == ord('a'):
                self.move_agent("a")
            elif key == ord('d'):
                self.move_agent("d")
            elif key == 27 or key == ord('q'):  # 按ESC或'q'键退出
                break

        cv2.destroyAllWindows()
        
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


    def move_agent(self, action):
        """
        根据用户输入移动智能体并保存图像
        """
        agent_state = self.agent.get_state()
        move_amount = 0.1  # 每次移动的距离
        rotate_amount = np.deg2rad(10)  # 定义旋转角度，每次旋转10度（弧度制）
        forward_dir = quat_rotate_vector(agent_state.rotation, np.array([0, 0, -1]))

        if action == "w":  # 向前移动
            agent_state.position += forward_dir * move_amount
        elif action == "s":  # 向后移动
            agent_state.position -= forward_dir * move_amount
        elif action == "a":  # 向左旋转
            rotation = quat_from_angle_axis(rotate_amount, np.array([0, 1, 0]))
            agent_state.rotation = rotation * agent_state.rotation
        elif action == "d":  # 向右旋转
            rotation = quat_from_angle_axis(-rotate_amount, np.array([0, 1, 0]))
            agent_state.rotation = rotation * agent_state.rotation

        self.agent.set_state(agent_state)
        print(f"Agent position: {agent_state.position}, rotation: {agent_state.rotation}")

        # 保存图像
        self.save_observation()
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

    def step_simulation(self):
        """
        仿真器步进
        """
        self.sim.step_physics(1.0 / 60.0)




# 使用示例
scene_path = "/home/ubuntu/桌面/control/test_data/mp3d/v1/tasks/mp3d/8WUmhLawc2A/8WUmhLawc2A.glb"
env = HabitatSimEnv(scene_path)

# 设置智能体的初始位置和旋转
#env.set_agent_initial_position([12.5, -1.3, 13.1], [0,0.258819050917469, 0, 0.96592588229042])
#env.set_agent_initial_position( [-2.0306475 ,0.4,  -3.0146806], [0, -0.573576331138611, 0, 0.819152295589447])#(x,y,z,w)
env.set_agent_initial_position([0.00955534 ,-0.3   ,     -3.7763028], [0,  -0.939692676067352, 0, -0.342020064592361])

# 从 YAML 文件加载物品
config_file = "/home/ubuntu/桌面/control/env_xq/image_attack/dining_room/attack_diningroom/object_config_diningroom.yaml"
#config_file = "/home/ubuntu/桌面/control/env_xq/object_config.yaml"

env.load_objects_from_file(config_file)

# 显示观察结果并处理键盘事件
env.display_observations()

'''

    def display_observations(self):
        """
        显示观察结果并保存图像，然后自动退出
        """
        # 获取当前观察结果
        observations = self.sim.get_sensor_observations()
        rgb_img = observations["color_sensor"]  # 获取传感器的数据
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)  # 转换为OpenCV使用的BGR格式

        # 显示图像
        cv2.imshow("Scene", rgb_img)
        cv2.waitKey(1)  # 短暂等待，使窗口能正确显示

        # 保存图像
        self.save_observation()

        # 自动关闭窗口并退出
        cv2.destroyAllWindows()
'''

