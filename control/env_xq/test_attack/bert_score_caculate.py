import logging
from bert_score import score
from typing import List, Tuple
import pandas as pd

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 默认使用英文BERT模型
DEFAULT_MODEL = "bert-base-uncased"

def get_bert_score(sentence1: str, sentence2: str, model_type: str = DEFAULT_MODEL) -> float:
    """
    计算两个英文句子之间的BERTScore相似度
    
    参数:
        sentence1: 第一个句子
        sentence2: 第二个句子
        model_type: 使用的BERT模型类型（默认: bert-base-uncased）
        
    返回:
        float: BERTScore F1分数 (0-1)
    """
    try:
        precision, recall, f1 = score(
            [sentence1],
            [sentence2],
            lang="en",  # 使用英文
            model_type=model_type,
            verbose=False,
        )
        return f1.item()
    except Exception as e:
        logger.error(f"BERTScore计算失败: {e}")
        return 0.0

def batch_bert_score(
    text_pairs: List[Tuple[str, str]], 
    model_type: str = DEFAULT_MODEL,
    batch_size: int = 64,
) -> List[float]:
    """
    批量计算多个英文文本对的BERTScore
    
    参数:
        text_pairs: (文本1, 文本2) 元组列表
        model_type: BERT模型类型
        batch_size: 批处理大小
        
    返回:
        相似度分数列表
    """
    if not text_pairs:
        return []

    # 分批处理
    batches = [text_pairs[i:i + batch_size] for i in range(0, len(text_pairs), batch_size)]
    all_scores = []

    for batch in batches:
        try:
            cands = [pair[0] for pair in batch]
            refs = [pair[1] for pair in batch]

            _, _, f1 = score(
                cands,
                refs,
                lang="en",
                model_type=model_type,
                verbose=False,
            )
            all_scores.extend(f1.tolist())
        except Exception as e:
            logger.error(f"处理批次时出错: {e}")
            all_scores.extend([0.0] * len(batch))

    return all_scores

def calculate_similarity_df(
    df: pd.DataFrame,
    text1_col: str,
    text2_col: str,
    output_col: str = "bert_score",
    model_type: str = DEFAULT_MODEL,
) -> pd.DataFrame:
    """
    为DataFrame中的英文文本对计算BERTScore
    
    参数:
        df: 包含文本对的DataFrame
        text1_col: 第一文本列名
        text2_col: 第二文本列名
        output_col: 输出列名
        model_type: BERT模型类型
        
    返回:
        添加了相似度分数的DataFrame
    """
    if text1_col not in df.columns or text2_col not in df.columns:
        raise ValueError("指定的文本列不存在")

    text_pairs = list(zip(df[text1_col], df[text2_col]))
    scores = batch_bert_score(text_pairs, model_type=model_type)
    df[output_col] = scores
    return df

if __name__ == "__main__":
    # 示例用法
    pairs = [
        (
            "The cat sits on the mat.",
            "A cat is sitting on the mat.",
        ),
        (
            "The quick brown fox jumps over the lazy dog.",
            "A fast brown fox leaps over a sleepy dog.",
        ),
    ]

    scores = batch_bert_score(pairs)
    for (t1, t2), score in zip(pairs, scores):
        print(f"'{t1}' 与 '{t2}' 的相似度: {score:.4f}")