import torch
import clip
from PIL import Image
import numpy as np
import cv2
import os
#本代码用于计算每个像素点的相似度的分数
# 多个模板
multiple_templates = [
    "There is {} in the scene.",
    "There is the {} in the scene.",
    "a photo of {} in the scene.",
    "a photo of the {} in the scene.",
    # ...省略了其他模板
]

# 初始化CLIP模型
def initialize_clip(clip_model_name="ViT-B/32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model_name, device=device)
    return model, preprocess, device

# 获取每个像素的相似度得分
def get_lseg_score(
    clip_model,
    landmarks: list,
    lseg_map: np.array,
    clip_feat_dim: int,
    use_multiple_templates: bool = False,
    avg_mode: int = 0,
    add_other=True,
):
    landmarks_other = landmarks
    if add_other and landmarks_other[-1] != "other":
        landmarks_other = landmarks + ["other"]

    if use_multiple_templates:
        mul_tmp = multiple_templates.copy()
        multi_temp_landmarks_other = [x.format(lm) for lm in landmarks_other for x in mul_tmp]
        text_feats = get_text_feats(multi_temp_landmarks_other, clip_model, clip_feat_dim)

        # 平均特征
        if avg_mode == 0:
            text_feats = text_feats.reshape((-1, len(mul_tmp), text_feats.shape[-1]))
            text_feats = np.mean(text_feats, axis=1)

        map_feats = lseg_map.reshape((-1, lseg_map.shape[-1]))

        scores_list = map_feats @ text_feats.T

        # 平均得分
        if avg_mode == 1:
            scores_list = scores_list.reshape((-1, len(landmarks_other), len(mul_tmp)))
            scores_list = np.mean(scores_list, axis=2)
    else:
        text_feats = get_text_feats(landmarks_other, clip_model, clip_feat_dim)
        map_feats = lseg_map.reshape((-1, lseg_map.shape[-1]))
        scores_list = map_feats @ text_feats.T

    return scores_list

# 计算文本特征
def get_text_feats(in_text, clip_model, clip_feat_dim, batch_size=64):
    if torch.cuda.is_available():
        text_tokens = clip.tokenize(in_text).cuda()
    elif torch.backends.mps.is_available():
        text_tokens = clip.tokenize(in_text).to("mps")
    else:
        text_tokens = clip.tokenize(in_text)
    text_id = 0
    text_feats = np.zeros((len(in_text), clip_feat_dim), dtype=np.float32)
    while text_id < len(text_tokens):  # 分批次进行推理
        batch_size = min(len(in_text) - text_id, batch_size)
        text_batch = text_tokens[text_id : text_id + batch_size]
        with torch.no_grad():
            batch_feats = clip_model.encode_text(text_batch).float()
        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        batch_feats = np.float32(batch_feats.cpu())
        text_feats[text_id : text_id + batch_size, :] = batch_feats
        text_id += batch_size
    return text_feats

# 将输出保存到指定文件夹
def save_output_as_csv(scores_mat, output_dir, output_filename):
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录（如果不存在）
    output_path = os.path.join(output_dir, output_filename)
    np.savetxt(output_path, scores_mat, delimiter=",")  # 将矩阵保存为CSV文件
    print(f"Scores matrix saved to {output_path}")

# 主函数
def main():
    # 直接在代码中指定图像路径和类别
    img_path = "/home/ubuntu/桌面/control/env_xq/observations/scene_image_0.png"  # 替换为实际图像路径
    categories = ["table"]  # 示例类别
    
    # 加载CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_version = "ViT-B/32"
    clip_feat_dim = {
        "RN50": 1024,
        "RN101": 512,
        "RN50x4": 640,
        "RN50x16": 768,
        "RN50x64": 1024,
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14": 768,
    }[clip_version]
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load(clip_version)
    clip_model.to(device).eval()

    # 加载图像
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 从LSeg模型获取特征映射（假设你有一个已经训练好的LSeg模型）
    lseg_map = np.random.rand(rgb.shape[0], rgb.shape[1], clip_feat_dim)  # 示例：随机生成一个特征映射

    # 计算类别相似度得分
    scores_mat = get_lseg_score(clip_model, categories, lseg_map, clip_feat_dim, use_multiple_templates=True, add_other=False)
    print("Scores matrix for each pixel and category:", scores_mat)

    output_dir = "/home/ubuntu/桌面/control/env_xq/score_data"  # 替换为实际的输出文件夹路径
    output_filename = "scores_matrix.csv"  # 输出文件名

# 保存输出
    save_output_as_csv(scores_mat, output_dir, output_filename)

if __name__ == "__main__":
    main()
