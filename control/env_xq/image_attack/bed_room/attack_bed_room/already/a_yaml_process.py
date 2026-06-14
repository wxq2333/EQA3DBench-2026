import os
import json
import base64
import time
import requests
import urllib3
import habitat_sim
import numpy as np
import cv2
import yaml
import magnum as mn
from habitat_sim.utils.common import quat_from_angle_axis
import glob
import pandas as pd

# ==========================================
# ⚙️ 用户配置区
# ==========================================

YAML_FOLDER = "/home/ubuntu/桌面/control/env_xq/image_attack/bed_room/attack_bed_room/already"
EXCEL_FILE = "/home/ubuntu/桌面/datasets/data-qwen_quec.xlsx"
SCENE_PATH = "/home/ubuntu/桌面/control/test_data/mp3d/v1/tasks/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb"
OUTPUT_ROOT = "/home/ubuntu/桌面/control/env_xq/image_attack/bed_room/attack_results_batch"

# 【关键配置】
# 1. 回答写入的主列名
ANSWER_COLUMN_NAME = "Qwen_Result"  
# 2. 位置信息写入的列名 (会自动创建，格式：主列名 + "_Location")
LOCATION_COLUMN_NAME = f"{ANSWER_COLUMN_NAME}_Location" 

API_CONFIGS = [
    {
        "name": "Qwen_VL_Max",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "key": "sk-0a46c0017e094c3ea61c1f15dbdc5bbc", 
        "model": "qwen-vl-max"
    }
]

CAMERA_POSES = [
    ([8.102168, 3.1, 2.7943587], [0, -0.70710688829422, 0, -0.707106709480286]),
    ([7.0052066, 3.1, 1.8813932], [0, -0.70710688829422, 0, -0.707106709480286]),
    ([5.9238224, 3.1, 1.0652195], [0, -0.906307876110077, 0, -0.422618120908737])
]

# ==========================================
# 🛠️ 核心功能
# ==========================================

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def extract_question_from_yaml(yaml_path):
    if not os.path.exists(yaml_path): return None
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i > 10: break
                line = line.strip()
                if line.startswith("# Question:"):
                    return line.split("# Question:", 1)[1].strip()
                elif not line.startswith("#") and line != "":
                    break
        return None
    except: return None

def find_question_in_excel(df, question_text):
    if not question_text:
        return None, None
    
    for col in df.columns:
        series = df[col].astype(str).str.strip()
        
        # 1. 精确匹配
        mask = series == question_text
        if mask.any():
            idx = mask.idxmax()
            return idx, col
        
        # 2. 包含匹配 (仅当问题长度>5时启用，防止误判)
        if len(question_text) > 5:
            mask = series.str.contains(question_text, na=False, regex=False)
            if mask.any():
                idx = mask.idxmax()
                return idx, col
                
    return None, None

class HabitatRenderer:
    def __init__(self, scene_path):
        self.scene_path = scene_path
        self.sim = None
        self.agent = None
        self.obj_templates_mgr = None
        self.rigid_obj_mgr = None

    def initialize(self):
        if self.sim: return True
        try:
            sim_cfg = habitat_sim.SimulatorConfiguration()
            sim_cfg.scene_id = self.scene_path
            sim_cfg.enable_physics = True
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = "color_sensor"
            sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            sensor_spec.resolution = [512, 512]
            sensor_spec.position = [0.0, 1.5, 0.0]
            agent_cfg = habitat_sim.agent.AgentConfiguration()
            agent_cfg.sensor_specifications = [sensor_spec]
            cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
            self.sim = habitat_sim.Simulator(cfg)
            self.agent = self.sim.initialize_agent(0)
            self.obj_templates_mgr = self.sim.get_object_template_manager()
            self.rigid_obj_mgr = self.sim.get_rigid_object_manager()
            return True
        except Exception as e:
            print(f"❌ Sim Init Failed: {e}")
            return False

    def load_yaml_objects(self, yaml_path):
        if not os.path.exists(yaml_path): return False
        try:
            with open(yaml_path, 'r') as f: data = yaml.safe_load(f)
            if "objects" not in data or not data["objects"]: return False
            count = 0
            for obj in data["objects"]:
                try:
                    ids = self.obj_templates_mgr.load_configs(obj["path"])
                    if not ids: continue
                    tid = ids[0]
                    tmpl = self.obj_templates_mgr.get_template_by_id(tid)
                    if "scale" in obj:
                        tmpl.scale = mn.Vector3(obj["scale"])
                        self.obj_templates_mgr.register_template(tmpl, force_registration=True)
                    inst = self.rigid_obj_mgr.add_object_by_template_id(tid)
                    if inst:
                        inst.translation = np.array(obj["position"])
                        rx = quat_from_angle_axis(np.deg2rad(obj["rotation"][0]), np.array([1,0,0]))
                        ry = quat_from_angle_axis(np.deg2rad(obj["rotation"][1]), np.array([0,1,0]))
                        rz = quat_from_angle_axis(np.deg2rad(obj["rotation"][2]), np.array([0,0,1]))
                        inst.rotation = mn.Quaternion((rz*ry*rx).imag, (rz*ry*rx).real)
                        count += 1
                except: pass
            return count > 0
        except: return False

    def render_views(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        paths = []
        try:
            for i, (pos, rot) in enumerate(CAMERA_POSES):
                state = habitat_sim.AgentState()
                state.position = np.array(pos)
                state.rotation = np.array(rot)
                self.agent.set_state(state)
                obs = self.sim.get_sensor_observations()
                img = cv2.cvtColor(obs["color_sensor"], cv2.COLOR_RGBA2BGR)
                fpath = os.path.join(save_dir, f"frame_{i:06d}.jpg")
                cv2.imwrite(fpath, img)
                paths.append(fpath)
            return paths
        except: return []

    def close(self):
        if self.sim: self.sim.close(); self.sim = None

class APIClient:
    def __init__(self, config):
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config['key']}"}
        self.url = config["url"]
        self.model = config["model"]
        self.name = config["name"]

    def query(self, image_paths, question):
        messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        for path in image_paths:
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            except: continue
        payload = {"model": self.model, "messages": messages, "temperature": 0.0}
        try:
            resp = requests.post(self.url, headers=self.headers, json=payload, verify=False, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Error: {str(e)}]"

# ==========================================
# 🏃 主流程 (增强版：记录详细位置)
# ==========================================

def main():
    print("🌟 批量攻击评估 (带位置记录版) 🌟")
    
    if not os.path.exists(EXCEL_FILE):
        print(f"❌ Excel 文件不存在: {EXCEL_FILE}")
        return
    
    # 1. 读取 Excel
    try:
        df = pd.read_excel(EXCEL_FILE, engine='openpyxl', header=0, dtype=str)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 2. 数据清洗
    df.columns = df.columns.str.strip()
    df = df.reset_index(drop=True)
    
    # 3. 检查/创建结果列 和 位置信息列
    if ANSWER_COLUMN_NAME not in df.columns:
        print(f"⚠️ 结果列 '{ANSWER_COLUMN_NAME}' 不存在，已自动创建。")
        df[ANSWER_COLUMN_NAME] = ""
    
    if LOCATION_COLUMN_NAME not in df.columns:
        print(f"⚠️ 位置列 '{LOCATION_COLUMN_NAME}' 不存在，已自动创建。")
        df[LOCATION_COLUMN_NAME] = ""
    
    print(f"✅ 加载成功。共 {len(df)} 行。")
    print(f"   回答将写入列：'{ANSWER_COLUMN_NAME}'")
    print(f"   位置将写入列：'{LOCATION_COLUMN_NAME}'")

    yaml_files = sorted(glob.glob(os.path.join(YAML_FOLDER, "*.yaml")))
    if not yaml_files:
        print("⚠️ 未找到 YAML 文件")
        return

    success_count = 0
    
    for i, yaml_file in enumerate(yaml_files):
        yaml_basename = os.path.basename(yaml_file).replace(".yaml", "")
        print(f"\n[{i+1}/{len(yaml_files)}] 处理: {yaml_basename}")
        
        # A. 提取问题
        question = extract_question_from_yaml(yaml_file)
        if not question:
            print("⚠️ 跳过：未提取到问题")
            continue
        
        # B. 查找问题
        row_idx, found_col = find_question_in_excel(df, question)
        
        if row_idx is None:
            print(f"⚠️ 跳过：在全表中未找到问题 '{question[:30]}...'")
            continue
        
        # C. 计算目标行 (问题行 + 3)
        target_row_idx = row_idx + 3
        
        # D. 渲染 & 推理
        save_dir = os.path.join(OUTPUT_ROOT, yaml_basename)
        renderer = HabitatRenderer(SCENE_PATH)
        answer = ""
        
        try:
            if not renderer.initialize(): raise Exception("Sim Fail")
            renderer.load_yaml_objects(yaml_file)
            imgs = renderer.render_views(save_dir)
            if not imgs: raise Exception("Render Fail")
            
            client = APIClient(API_CONFIGS[0])
            print(f"🤖 调用 {client.name}...")
            answer = client.query(imgs, question)
        except Exception as e:
            print(f"❌ 出错: {e}")
            answer = f"[Error: {e}]"
        finally:
            renderer.close()
        
        # E. 准备写入数据
        
        # 1. 确保 DataFrame 足够长
        if target_row_idx >= len(df):
            needed_rows = target_row_idx - len(df) + 1
            new_rows = pd.DataFrame([{}]*needed_rows, columns=df.columns)
            df = pd.concat([df, new_rows], ignore_index=True)
            print(f"   ⚠️ 表格已自动扩展 {needed_rows} 行，当前总行数：{len(df)}")
        
        # 2. 构建位置信息字符串
        # Excel 行号 = Pandas Index + 2 (1行表头 + 1行偏移)
        excel_row_num = target_row_idx + 2
        location_info = f"✅ 已写入\n来源问题行: Excel第 {row_idx+2} 行 (列: {found_col})\n目标写入行: Excel第 {excel_row_num} 行 (列: {ANSWER_COLUMN_NAME})\n时间: {time.strftime('%H:%M:%S')}"
        
        # 3. 执行写入
        df.at[target_row_idx, ANSWER_COLUMN_NAME] = answer
        df.at[target_row_idx, LOCATION_COLUMN_NAME] = location_info
        
        print(f"✍️ 写入成功:")
        print(f"   📍 目标位置 -> Excel 第 {excel_row_num} 行, 列 '{ANSWER_COLUMN_NAME}'")
        print(f"   📝 位置记录 -> Excel 第 {excel_row_num} 行, 列 '{LOCATION_COLUMN_NAME}'")
        
        success_count += 1
        
        # F. 实时保存
        try:
            df.to_excel(EXCEL_FILE, index=False, engine='openpyxl')
            print(f"💾 文件已保存 (请关闭 Excel 以避免锁定)")
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            print("   💡 提示：请确保 Excel 文件已完全关闭！")
            # 即使保存失败也继续，但标记警告
            time.sleep(5)

        time.sleep(1)

    print("\n" + "="*60)
    print(f"🎉 完成！成功处理: {success_count}")
    print(f"📊 结果分布：")
    print(f"   - 回答内容在列：[{ANSWER_COLUMN_NAME}]")
    print(f"   - 位置详情在列：[{LOCATION_COLUMN_NAME}]")
    print(f"📝 最终文件：{EXCEL_FILE}")
    print("="*60)

if __name__ == "__main__":
    main()