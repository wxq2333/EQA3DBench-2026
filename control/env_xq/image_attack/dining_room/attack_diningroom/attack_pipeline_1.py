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

YAML_FOLDER = "/home/ubuntu/桌面/control/env_xq/image_attack/dining_room/alread"
EXCEL_FILE = "/home/ubuntu/桌面/dataset_new/data-gemini-2.5-pro_quec.xlsx"
SCENE_PATH = "/home/ubuntu/桌面/control/test_data/mp3d/v1/tasks/mp3d/8WUmhLawc2A/8WUmhLawc2A.glb"
OUTPUT_ROOT = "/home/ubuntu/桌面/control/env_xq/image_attack/diningroom_room/attack_results_batch"

# 【关键配置】
# 现在不需要指定列名了，脚本会自动识别问题所在的列并写入该列下方
OFFSET_ROWS = 3  # 答案写在问题下方的第几行 (例如 3 表示中间隔 2 行空行)

API_CONFIGS = [
    {
        "name": "gemini-2.5-pro",
        "url": "https://api.wenwen-ai.com/v1/chat/completions",
        "key": "sk-sZwZhBKNWns1Xi2W48MU43rulcXnMRTc8cTCMwyDYcmAijjM",
        "model": "gemini-2.5-pro"
    }
]
# API_CONFIGS = [
#     {
#         "name": "gemini-2.5-pro",
#         "url": "https://tb.api.mkeai.com/v1/chat/completions",
#         "key": "sk-YrS3MnJJWd661l1EsUmvR2RxAQFP4hScbtaZmWAC3ho0vxtv", 
#         "model": "gemini-2.5-pro"
#     }
# ]
# API_CONFIGS = [
#     {
#         "name": "claude-haiku-4-5-20251001-thinking",
#         "url": "https://api.wenwen-ai.com/v1/chat/completions",
#         "key": "sk-sZwZhBKNWns1Xi2W48MU43rulcXnMRTc8cTCMwyDYcmAijjM", 
#         "model": "claude-haiku-4-5-20251001-thinking"
#     }
# ]


CAMERA_POSES = [
    ([0.00955534 ,-0.3   ,     -3.7763028], [0,  -0.939692676067352, 0, -0.342020064592361]),
    ([-0.20560469 ,-0.3    ,    -3.5730941 ], [0,-0.866025499598342, 0, -0.499999926845038]),
    ([-0.55938154 ,-0.3    ,    -3.3888922 ], [0, -0.866025531397977, 0, -0.499999943053257])
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
    """
    在全表范围内搜索问题文本。
    返回: (row_index, col_name) 或 (None, None)
    """
    if not question_text:
        return None, None
    
    for col in df.columns:
        series = df[col].astype(str).str.strip()
        
        # 1. 精确匹配 (优先)
        mask = series == question_text
        if mask.any():
            idx = mask.idxmax()
            return idx, col
        
        # 2. 包含匹配 (防止 Excel 中有额外标点，仅当问题长度>5时启用)
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
# 🏃 主流程 (同列写入版)
# ==========================================

def main():
    print("🌟 批量攻击评估 (同列写入版) 🌟")
    
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
    
    print(f"✅ 加载成功。共 {len(df)} 行，{len(df.columns)} 列。")
    print(f"   策略：找到问题 -> 在【同一列】向下偏移 {OFFSET_ROWS} 行 -> 写入答案")
    print(f"   不会创建新列。")

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
        
        # B. 查找问题 (返回行索引 和 列名)
        row_idx, found_col = find_question_in_excel(df, question)
        
        if row_idx is None:
            print(f"⚠️ 跳过：在全表中未找到问题 '{question[:30]}...'")
            continue
        
        print(f"📍 定位成功 -> 行索引: {row_idx} (Excel 第 {row_idx+2} 行), 所在列: '{found_col}'")
        
        # C. 计算目标行 (问题行 + 偏移量)
        target_row_idx = row_idx + OFFSET_ROWS
        print(f"🎯 目标写入位置 -> 同一列 '{found_col}', 行索引: {target_row_idx} (Excel 第 {target_row_idx+2} 行)")
        
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
        
        # E. 准备写入数据 (同列写入)
        
        # 1. 确保 DataFrame 足够长
        if target_row_idx >= len(df):
            needed_rows = target_row_idx - len(df) + 1
            new_rows = pd.DataFrame([{}]*needed_rows, columns=df.columns)
            df = pd.concat([df, new_rows], ignore_index=True)
            print(f"   ⚠️ 表格已自动扩展 {needed_rows} 行，当前总行数：{len(df)}")
        
        # 2. 【核心修改】直接写入找到的那一列 (found_col)
        df.at[target_row_idx, found_col] = answer
        
        print(f"✍️ 写入成功:")
        print(f"   📝 内容预览: {answer[:50]}...")
        
        success_count += 1
        
        # F. 实时保存
        try:
            df.to_excel(EXCEL_FILE, index=False, engine='openpyxl')
            print(f"💾 文件已保存 (请关闭 Excel 以避免锁定)")
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            print("   💡 提示：请确保 Excel 文件已完全关闭！")
            time.sleep(5)

        time.sleep(1)

    print("\n" + "="*60)
    print(f"🎉 完成！成功处理: {success_count}")
    print(f"📊 结果已直接填入问题所在的列下方 {OFFSET_ROWS} 行处。")
    print(f"📝 最终文件：{EXCEL_FILE}")
    print("="*60)

if __name__ == "__main__":
    main()