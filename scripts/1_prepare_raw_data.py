"""
从 DianJin-R1-Data 和自定义数据中准备原始数据集
输出：data/raw/raw.jsonl
"""
import json
import os
import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import DATA_CONFIG

MAX_DOWNLOAD_SAMPLES = 400  # 只下载前400条数据

def download_dianjin_data():
    """
    下载 DianJin-R1-Data 数据集（只下载前400条）
    面试点：数据来源的多样性
    """
    try:
        # 方法1：从 ModelScope 下载（国内速度快）
        from modelscope.msdatasets import MsDataset
        ds = MsDataset.load(
            'tongyi_dianjin/DianJin-R1-Data',
            split=f'train[:{MAX_DOWNLOAD_SAMPLES}]'
        )
        print(f"   ✓ 从 ModelScope 下载成功")
        return list(ds)
    except Exception as e:
        print(f"   ModelScope 下载失败: {e}")

    try:
        # 方法2：从 HuggingFace 下载
        from datasets import load_dataset
        ds = load_dataset(
            "DianJin/DianJin-R1-Data",
            split=f"train[:{MAX_DOWNLOAD_SAMPLES}]"
        )
        print(f"   ✓ 从 HuggingFace 下载成功")
        return list(ds)
    except Exception as e:
        print(f"   HuggingFace 下载失败: {e}")

    # 方法3：手动下载提示
    print("⚠️  无法自动下载 DianJin-R1-Data 数据集")
    print("请手动下载：")
    print("  1. 访问 https://huggingface.co/datasets/DianJin/DianJin-R1-Data")
    print("  2. 下载数据集并放到 data/raw/dianjin-r1-data.jsonl")

    # 尝试读取本地文件
    local_path = "data/raw/dianjin-r1-data.jsonl"
    if os.path.exists(local_path):
        with open(local_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
            return data[:MAX_DOWNLOAD_SAMPLES]

    return []

def classify_question_type(question, answer=""):
    """
    根据问题内容分类
    面试点：数据分类的启发式规则
    """
    question_lower = question.lower()

    # 金融计算题特征
    calc_keywords = ["计算", "增长率", "收益率", "市盈率", "多少", "百分之",
                     "同比", "环比", "利润", "营收", "股价"]
    if any(kw in question for kw in calc_keywords):
        return "financial_calculation"

    # 概念问答题特征
    concept_keywords = ["什么是", "定义", "概念", "含义", "解释"]
    if any(kw in question for kw in concept_keywords):
        return "concept_qa"

    # 风险分析题特征
    risk_keywords = ["风险", "影响", "后果", "预测", "趋势"]
    if any(kw in question for kw in risk_keywords):
        return "risk_analysis"

    # 默认为业务推理
    return "business_reasoning"

def create_custom_data():
    """
    创建自定义数据
    面试点：展示数据构造能力
    """
    custom_samples = [
        {
            "question": "某公司2023年营收1000万元，同比增长25%，请计算2022年营收。",
            "gold_answer": "800",
            "type": "financial_calculation",
            "source": "custom",
            "explanation": "2023年营收 / (1 + 增长率) = 1000 / 1.25 = 800万元"
        },
        {
            "question": "什么是资产负债率？如何计算？",
            "gold_answer": "资产负债率 = (总负债 / 总资产) × 100%，用于衡量企业长期偿债能力。",
            "type": "concept_qa",
            "source": "custom"
        },
        {
            "question": "某股票当前价格50元，市盈率20，请计算该公司每股收益。",
            "gold_answer": "2.5",
            "type": "financial_calculation",
            "source": "custom",
            "explanation": "每股收益 = 股价 / 市盈率 = 50 / 20 = 2.5元"
        },
        {
            "question": "央行提高利率对股市有什么影响？",
            "gold_answer": "央行提高利率通常会导致股市下跌，因为：1）融资成本上升，企业盈利下降；2）债券等固定收益产品吸引力增加；3）市场流动性收紧。",
            "type": "business_reasoning",
            "source": "custom"
        },
        {
            "question": "什么是流动比率？正常范围是多少？",
            "gold_answer": "流动比率 = 流动资产 / 流动负债，正常范围为1.5-2.0，用于衡量企业短期偿债能力。",
            "type": "concept_qa",
            "source": "custom"
        },
    ]

    return custom_samples

def stratified_sampling(data_source, target_count, type_distribution):
    """
    分层抽样
    面试点：如何保证数据分布合理
    """
    # 按类型分组
    type_buckets = {}
    for item in data_source:
        qtype = item.get("type")
        if qtype not in type_buckets:
            type_buckets[qtype] = []
        type_buckets[qtype].append(item)

    # 按比例抽样
    sampled_data = []
    for qtype, ratio in type_distribution.items():
        target_n = int(target_count * ratio)
        bucket = type_buckets.get(qtype, [])

        if len(bucket) >= target_n:
            sampled_data.extend(bucket[:target_n])
        else:
            # 不够就全部加入
            sampled_data.extend(bucket)
            print(f"⚠️  {qtype} 只有 {len(bucket)} 条，少于目标 {target_n} 条")

    return sampled_data

def prepare_raw_data():
    """主流程"""
    print("="*60)
    print("阶段1：准备原始数据")
    print("="*60)

    # 创建目录
    os.makedirs("data/raw", exist_ok=True)

    all_data = []

    # 1. 加载 DianJin-R1-Data
    print(f"\n1. 加载 DianJin-R1-Data 数据集（限制 {MAX_DOWNLOAD_SAMPLES} 条）...")
    dianjin_data = download_dianjin_data()

    if dianjin_data:
        print(f"   ✓ 加载成功：{len(dianjin_data)} 条")

        # 转换格式并分类
        for idx, item in enumerate(dianjin_data):
            # DianJin-R1-Data 格式: {instruction, output}
            question = item.get("instruction", "")
            answer = item.get("output", "")

            if question and answer:
                qtype = classify_question_type(question, answer)
                all_data.append({
                    "id": f"dianjin_{idx}",
                    "question": question,
                    "gold_answer": answer,
                    "type": qtype,
                    "source": "dianjin-r1-data"
                })
    else:
        print("   ✗ 加载失败，将仅使用自定义数据")

    # 2. 添加自定义数据
    print("\n2. 添加自定义数据...")
    custom_data = create_custom_data()
    for idx, item in enumerate(custom_data):
        item["id"] = f"custom_{idx}"
        all_data.append(item)
    print(f"   ✓ 添加 {len(custom_data)} 条自定义数据")

    # 3. 分层抽样（如果数据量超过目标）
    if len(all_data) > DATA_CONFIG.total_samples:
        print(f"\n3. 分层抽样至 {DATA_CONFIG.total_samples} 条...")
        all_data = stratified_sampling(
            all_data,
            DATA_CONFIG.total_samples,
            DATA_CONFIG.type_distribution
        )

    # 4. 统计信息
    type_counts = {}
    for item in all_data:
        qtype = item["type"]
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

    print("\n" + "="*60)
    print("数据资产统计")
    print("="*60)
    print(f"总数: {len(all_data)}")
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype:25s}: {count:3d} ({count/len(all_data)*100:5.1f}%)")
    print("="*60)

    # 5. 保存
    output_path = DATA_CONFIG.raw_data_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n✅ 原始数据已保存至: {output_path}")

    return len(all_data)

if __name__ == "__main__":
    count = prepare_raw_data()
    print(f"\n下一步：运行 python scripts/2_distill_data.py")
