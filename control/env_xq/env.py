import habitat_sim
import habitat_sim.physics
import yaml
import numpy as np
import cv2
import magnum as mn

class Env():
    ''' Implements of habitat navigation environments '''
    def __init__(self, config_file='/home/ubuntu/桌面/control/env_xq/config_habitatsim.yaml', obj_config_file='/home/ubuntu/桌面/control/env_xq/place_obj.yaml'):
        # Load configuration from file
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

        # Load object configuration from file
        with open(obj_config_file, 'r') as file:
            self.obj_config = yaml.safe_load(file)

        # Setup simulator configuration
        self.sim_cfg = habitat_sim.SimulatorConfiguration()
        self.sim_cfg.gpu_device_id = 0

        # Set scene ID and dataset configuration
        if self.config['env']['sim_name'] == 'mp3d':
            self.sim_cfg.scene_id = self.config['env']['test_scene']
            self.sim_cfg.scene_dataset_config_file = self.config['env']['mp3d_scene_dataset']
        else:
            self.sim_cfg.scene_id = self.config['env']['test_scene']
            self.sim_cfg.scene_dataset_config_file = self.config['env']['mp3d_scene_dataset']
        
        self.sim_cfg.enable_physics = self.config['env']['enable_physics']
        
        # Setup sensor specifications
        self.sensor_specs = []
        self.setup_sensors()
        
        # Setup agent configuration
        self.agent_cfg = habitat_sim.agent.AgentConfiguration()
        self.agent_cfg.sensor_specifications = self.sensor_specs
        
        # Create simulator
        self.cfg = habitat_sim.Configuration(self.sim_cfg, [self.agent_cfg])
        self.sim = habitat_sim.Simulator(self.cfg)
        
        # Seed and initialize agent
        self.sim.seed(self.config['env']['seed'])
        self.agent = self.sim.initialize_agent(self.config['env']["default_agent"])
        
        # Set initial agent state
        self.agent_state = habitat_sim.AgentState()
        self.agent_state.position = np.array([-4.4, 0.0, 1.8])  # Set initial position
        self.agent.set_state(self.agent_state)
        
        # Print initial agent state
        self.print_initial_state()
        
        # Add objects to the scene
        self.PlaceObjfromYaml(obj_config_file)

    def setup_sensors(self):
        if self.config['env']['rgb_sensor']:
            color_sensor_spec = habitat_sim.CameraSensorSpec()
            color_sensor_spec.uuid = "color_sensor"
            color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            color_sensor_spec.resolution = [self.config['env']['obs_height'], self.config['env']['obs_width']]
            color_sensor_spec.position = [0.0, self.config['env']['sensor_height'], 0.0]
            color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            self.sensor_specs.append(color_sensor_spec)

        if self.config['env']['depth_sensor']:
            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth_sensor"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = [self.config['env']['obs_height'], self.config['env']['obs_width']]
            depth_sensor_spec.position = [0.0, self.config['env']['sensor_height'], 0.0]
            depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            self.sensor_specs.append(depth_sensor_spec)

        if self.config['env']['semantic_sensor']:
            semantic_sensor_spec = habitat_sim.CameraSensorSpec()
            semantic_sensor_spec.uuid = "semantic_sensor"
            semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            semantic_sensor_spec.resolution = [self.config['env']['obs_height'], self.config['env']['obs_width']]
            semantic_sensor_spec.position = [0.0, self.config['env']['sensor_height'], 0.0]
            semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            self.sensor_specs.append(semantic_sensor_spec)

    def print_initial_state(self):
        self.agent_state = self.agent.get_state()
        print("Initial agent state: position", self.agent_state.position, "rotation", self.agent_state.rotation)

    def PlaceSingleObj(self, loc=None, rot=None, config=None):
        # place object
        prim_templates_mgr = self.sim.get_asset_template_manager()
        obj_templates_mgr = self.sim.get_object_template_manager()
        rigid_obj_mgr = self.sim.get_rigid_object_manager()

        sphere_template_id = obj_templates_mgr.load_configs(str(config))[0]
        # add a sphere to the scene, returns the object
        sphere_obj = rigid_obj_mgr.add_object_by_template_id(sphere_template_id)
        # move sphere
        sphere_obj.translation = loc
        
        rotation_x = mn.Quaternion.rotation(mn.Deg(rot[0]), [-1.0, 0.0, 0.0])
        rotation_y = mn.Quaternion.rotation(mn.Deg(rot[1]), [0.0, -1.0, 0.0])
        rotation_z = mn.Quaternion.rotation(mn.Deg(rot[2]), [0.0, 0.0, -1.0])
        sphere_obj.rotation = rotation_x*rotation_y*rotation_z
        
    def PlaceObjfromYaml(self, yaml_config='/home/ubuntu/桌面/control/test_xq/place_object.yaml'):
        with open(yaml_config, 'r') as file:
            obj_config = yaml.safe_load(file)
        
        for key, value in obj_config.items():
            self.PlaceSingleObj(loc=value['loc'], rot=value['rot'], config=value['conf'])
        
    def display_observations(self):
        while True:
            observations = self.sim.get_sensor_observations()

            # Display RGB image
            rgb_obs = observations["color_sensor"]
            rgb_img = cv2.cvtColor(rgb_obs, cv2.COLOR_RGB2BGR)

            # Handle key events for controlling the agent
            key = cv2.waitKey(1)
            if key == ord('w'):
                self.move_agent("forward")
            elif key == ord('s'):
                self.move_agent("backward")
            elif key == ord('a'):
                self.move_agent("left")
            elif key == ord('d'):
                self.move_agent("right")
            elif key == 27 or key == ord('q'):  # ESC key or 'q' to exit
                break
            
            cv2.imshow("RGB", rgb_img)

        cv2.destroyAllWindows()

    def move_agent(self, direction):
        agent_state = self.agent.get_state()
        move_amount = 0.1
        if direction == "forward":
            agent_state.position += np.array([0, 0, -move_amount])
        elif direction == "backward":
            agent_state.position += np.array([0, 0, move_amount])
        elif direction == "left":
            agent_state.position += np.array([-move_amount, 0, 0])
        elif direction == "right":
            agent_state.position += np.array([move_amount, 0, 0])
        self.agent.set_state(agent_state)
        self.print_agent_state()  # Print agent state after moving

    def print_agent_state(self):
        agent_state = self.agent.get_state()
        print(f"Agent position: {agent_state.position}, rotation: {agent_state.rotation}")

if __name__ == "__main__":
    env = Env('/home/ubuntu/桌面/control/env_xq/config_habitatsim.yaml', '/home/ubuntu/桌面/control/env_xq/place_obj.yaml')
    env.display_observations()
