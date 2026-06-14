#将下载的glb文件生成对应的json文件
#将整个文件夹下的glb都生成对应的json文件
import os
import json

def generate_config_files(directory):
    # 获取文件夹的绝对路径
    abs_directory = os.path.abspath(directory)
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(abs_directory):
        if filename.lower().endswith('.glb'):
            # 获取文件的完整路径
            file_path = os.path.join(abs_directory, filename)
            
            # 构建配置字典
            config = {
                "render_asset": file_path,
                "collision_asset": file_path,
                "scale": [0.1, 0.1, 0.1],
                "requires_lighting": True
            }
            
            # 构建配置文件的名称
            base_name = os.path.splitext(filename)[0]
            config_filename = f"{base_name}.object_config.json"
            config_path = os.path.join(abs_directory, config_filename)
            
            # 将配置字典写入JSON文件
            with open(config_path, 'w', encoding='utf-8') as json_file:
                json.dump(config, json_file, indent=4, ensure_ascii=False)
            
            print(f"生成配置文件: {config_path}")

if __name__ == "__main__":
    # 指定包含.glb文件的文件夹路径
    target_directory = "/home/ubuntu/桌面/control/env_xq/image_attack/bath_room/objects"
    
    generate_config_files(target_directory)
