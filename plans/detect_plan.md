# 金融推理模型后训练完整执行计划（面试导向版）

> **目标**：一天内完全学会"数据构建 → SFT → RL(GRPO) → 评测 → 部署"全链路
> **约束**：本地 4070 8GB + AutoDL 5090 32GB，预算 100 元
> **核心**：本地构建所有脚本，服务器纯执行训练，准备面试材料

---

## 📋 目录

1. [总体架构](#总体架构)
2. [阶段0：项目初始化](#阶段0项目初始化)
3. [阶段1：数据构建流水线](#阶段1数据构建流水线)
4. [阶段2：训练脚本开发](#阶段2训练脚本开发)
5. [阶段3：服务器训练执行](#阶段3服务器训练执行)
6. [阶段4：评测与部署](#阶段4评测与部署)
7. [阶段5：面试准备](#阶段5面试准备)
8. [常见问题排查](#常见问题排查)
9. [面试问答手册](#面试问答手册)

---

## 总体架构

### 核心思路

基于 Fin-R1 论文的工程实践：

```
原始数据 → 教师蒸馏 → 双重过滤 → 训练数据资产
                                      ↓
                              ┌───────┴───────┐
                              ↓               ↓
                            SFT            GRPO
                         (学格式)        (提准确率)
                              ↓               ↓
                              └───────┬───────┘
                                      ↓
                              评测 → 部署 → 面试
```

### 关键设计决策（面试要点）

| 问题 | 决策 | 原因 |
|------|------|------|
| 为什么要数据蒸馏？ | 用强模型生成 CoT | 小模型缺乏推理能力 |
| 为什么双重过滤？ | 答案正确性 + 推理质量 | 答案是硬约束，推理是质量保证 |
| 为什么先 SFT？ | 学习输出格式 | RL 需要稳定的格式作为基础 |
| 为什么用 GRPO？ | 可验证奖励 + 稳定训练 | 金融场景需要客观标准 |
| 为什么用 vLLM？ | 高性能推理 | PagedAttention + Continuous batching |

---

## 阶段0：项目初始化

### 创建项目结构

```bash
cd c:\gitclones\FTModel

# 创建目录
mkdir -p data/{raw,processed} scripts ckpts reports configs logs

# 创建 .gitignore
cat > .gitignore << 'EOF'
# 模型权重
ckpts/
*.bin
*.safetensors

# 数据文件
data/
!data/.gitkeep

# 日志
logs/
*.log

# Python
__pycache__/
*.pyc
.env

# IDE
.vscode/
.idea/
EOF

# 创建 README
cat > README.md << 'EOF'
# 金融推理模型后训练项目

基于 Fin-R1 论文的完整后训练链路实现。

## 快速开始

1. 本地数据准备：`python scripts/1_prepare_raw_data.py`
2. 服务器训练：见 `完整执行计划_面试导向版.md`

## 项目结构

- `scripts/`: 所有训练和评测脚本
- `data/`: 数据资产（原始、蒸馏、过滤后）
- `ckpts/`: 模型检查点
- `reports/`: 评测报告
- `configs/`: 配置文件
EOF
```

### 创建依赖文件

```bash
cat > requirements.txt << 'EOF'
# 核心框架
torch==2.1.0
transformers==4.45.0
trl==0.12.0
peft==0.13.0

# 量化和加速
bitsandbytes==0.44.0
accelerate==0.34.0

# 部署
vllm==0.6.3

# 数据处理
datasets==2.20.0
pandas==2.1.0

# API 调用
openai==1.35.0
requests==2.31.0

# 工具
tqdm==4.66.0
wandb  # 可选，用于训练监控

# Qwen 特定
sentencepiece==0.2.0
protobuf==4.25.0
EOF
```

### 创建配置文件

```python
# configs/config.py
"""
全局配置文件
面试点：配置管理的最佳实践
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """模型配置"""
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    model_max_length: int = 2048

@dataclass
class LoRAConfig:
    """LoRA 配置"""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = None

    def __post_init__(self):
        if self.target_modules is None:
            # Qwen 系列的 LoRA 目标模块
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class SFTConfig:
    """SFT 训练配置"""
    output_dir: str = "ckpts/sft_lora"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    bf16: bool = True
    optim: str = "paged_adamw_8bit"

@dataclass
class GRPOConfig:
    """GRPO 训练配置"""
    output_dir: str = "ckpts/grpo_lora"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-6
    num_sample_generations: int = 4
    response_length: int = 512
    temperature: float = 0.7
    kl_coef: float = 0.05
    format_reward_weight: float = 0.3
    accuracy_reward_weight: float = 0.7

@dataclass
class APIConfig:
    """API 配置"""
    provider: str = "deepseek"  # deepseek, qwen, openai
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None

    def __post_init__(self):
        # 从环境变量读取
        if self.api_key is None:
            self.api_key = os.getenv("API_KEY")

        # 根据 provider 设置默认值
        if self.provider == "deepseek":
            self.base_url = self.base_url or "https://api.deepseek.com/v1"
            self.model_name = self.model_name or "deepseek-chat"
        elif self.provider == "qwen":
            self.base_url = self.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            self.model_name = self.model_name or "qwen-max"

@dataclass
class DataConfig:
    """数据配置"""
    raw_data_path: str = "data/raw/raw.jsonl"
    distilled_data_path: str = "data/processed/distilled.jsonl"
    sft_data_path: str = "data/processed/sft.jsonl"
    rl_data_path: str = "data/processed/rl.jsonl"
    test_data_path: str = "data/processed/test.jsonl"

    # 数据规模
    total_samples: int = 500
    test_ratio: float = 0.2

    # 分层抽样配置
    type_distribution: dict = None

    def __post_init__(self):
        if self.type_distribution is None:
            self.type_distribution = {
                "financial_calculation": 0.4,
                "business_reasoning": 0.3,
                "concept_qa": 0.2,
                "risk_analysis": 0.1
            }

# 全局配置实例
MODEL_CONFIG = ModelConfig()
LORA_CONFIG = LoRAConfig()
SFT_CONFIG = SFTConfig()
GRPO_CONFIG = GRPOConfig()
API_CONFIG = APIConfig()
DATA_CONFIG = DataConfig()
```

---

## 阶段1：数据构建流水线

### 1.1 准备原始数据

```python
# scripts/1_prepare_raw_data.py
"""
从 qwen-dianjin 和自定义数据中准备原始数据集
输出：data/raw/raw.jsonl
"""
import json
import os
import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import DATA_CONFIG

def download_qwen_dianjin():
    """
    下载 qwen-dianjin 数据集
    面试点：数据来源的多样性
    """
    try:
        # 方法1：从 ModelScope 下载（国内速度快）
        from modelscope.msdatasets import MsDataset
        ds = MsDataset.load('qwen/qwen-dianjin', split='train')
        return list(ds)
    except:
        pass

    try:
        # 方法2：从 HuggingFace 下载
        from datasets import load_dataset
        ds = load_dataset("Qwen/Qwen-Dianjin", split="train")
        return list(ds)
    except:
        pass

    # 方法3：手动下载提示
    print("⚠️  无法自动下载 qwen-dianjin 数据集")
    print("请手动下载：")
    print("  1. 访问 https://github.com/QwenLM/Qwen-Dianjin")
    print("  2. 下载数据集并放到 data/raw/qwen-dianjin.jsonl")

    # 尝试读取本地文件
    local_path = "data/raw/qwen-dianjin.jsonl"
    if os.path.exists(local_path):
        with open(local_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

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

    # 1. 加载 qwen-dianjin
    print("\n1. 加载 qwen-dianjin 数据集...")
    qwen_data = download_qwen_dianjin()

    if qwen_data:
        print(f"   ✓ 加载成功：{len(qwen_data)} 条")

        # 转换格式并分类
        for idx, item in enumerate(qwen_data[:400]):  # 只取前400条
            # qwen-dianjin 的数据格式可能是 {input, target} 或 {question, answer}
            question = item.get("input") or item.get("question", "")
            answer = item.get("target") or item.get("answer", "")

            if question and answer:
                qtype = classify_question_type(question, answer)
                all_data.append({
                    "id": f"qwen_{idx}",
                    "question": question,
                    "gold_answer": answer,
                    "type": qtype,
                    "source": "qwen-dianjin"
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
```

### 1.2 数据蒸馏

```python
# scripts/2_distill_data.py
"""
用教师模型生成 CoT
输出：data/processed/distilled.jsonl
"""
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import DATA_CONFIG, API_CONFIG

def get_api_client():
    """
    获取 API 客户端
    面试点：API 调用的通用封装
    """
    from openai import OpenAI

    if not API_CONFIG.api_key:
        print("⚠️  未设置 API_KEY")
        print("请设置环境变量：export API_KEY=your_api_key")
        print("或在 configs/config.py 中配置")
        sys.exit(1)

    client = OpenAI(
        api_key=API_CONFIG.api_key,
        base_url=API_CONFIG.base_url
    )

    return client

def create_distillation_prompt(question, qtype):
    """
    构造蒸馏 prompt
    面试点：prompt 工程的重要性
    """
    system_prompt = """你是一个金融领域专家。请用以下严格格式回答问题：

<think>
[详细的推理过程，必须包含3-5个清晰的推理步骤]
</think>
<answer>
[最终答案，简洁明确]
</answer>

要求：
1. <think> 部分必须展示完整推理逻辑：
   - 金融计算题：写出公式、代入数值、计算过程
   - 概念题：定义 → 组成要素 → 计算方法/应用场景
   - 分析题：现象 → 原因分析 → 影响/结论
2. 推理步骤用"首先"、"其次"、"然后"、"因此"等连接词
3. <answer> 部分只包含最终答案：
   - 数值题：直接给出数字（不要单位）
   - 概念/分析题：1-2句话的简洁答案
4. 严格遵守标签格式，不要有任何多余内容"""

    user_prompt = f"问题：{question}"

    return system_prompt, user_prompt

def call_teacher_model(client, question, qtype, max_retries=3):
    """
    调用教师模型
    面试点：错误处理和重试机制
    """
    system_prompt, user_prompt = create_distillation_prompt(question, qtype)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=API_CONFIG.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.6,
                max_tokens=1024
            )

            output = response.choices[0].message.content
            return output, None

        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                print(f"   ⚠️  重试 {attempt+1}/{max_retries}（等待 {wait_time}s）: {error_msg[:50]}")
                time.sleep(wait_time)
            else:
                return None, error_msg

    return None, "Max retries exceeded"

def distill_data():
    """主流程"""
    print("="*60)
    print("阶段2：数据蒸馏（Teacher 生成 CoT）")
    print("="*60)

    # 加载原始数据
    print(f"\n加载原始数据: {DATA_CONFIG.raw_data_path}")
    with open(DATA_CONFIG.raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]
    print(f"✓ 加载 {len(raw_data)} 条")

    # 初始化 API
    print(f"\n初始化 API: {API_CONFIG.provider}")
    client = get_api_client()
    print(f"✓ 使用模型: {API_CONFIG.model_name}")

    # 蒸馏
    distilled_data = []
    failed_items = []

    print(f"\n开始蒸馏...")
    for item in tqdm(raw_data, desc="蒸馏进度"):
        teacher_output, error = call_teacher_model(
            client,
            item["question"],
            item["type"]
        )

        if teacher_output:
            distilled_data.append({
                **item,
                "teacher_output": teacher_output
            })
        else:
            failed_items.append({
                "id": item["id"],
                "error": error
            })

        # 每10条保存一次（防止中断丢失）
        if len(distilled_data) % 10 == 0:
            temp_path = DATA_CONFIG.distilled_data_path + ".tmp"
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with open(temp_path, 'w', encoding='utf-8') as f:
                for d in distilled_data:
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')

    # 保存最终结果
    output_path = DATA_CONFIG.distilled_data_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in distilled_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 保存失败记录
    if failed_items:
        fail_path = "data/processed/distill_failures.jsonl"
        with open(fail_path, 'w', encoding='utf-8') as f:
            for item in failed_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 统计
    print("\n" + "="*60)
    print("蒸馏结果")
    print("="*60)
    print(f"总数: {len(raw_data)}")
    print(f"成功: {len(distilled_data)} ({len(distilled_data)/len(raw_data)*100:.1f}%)")
    print(f"失败: {len(failed_items)} ({len(failed_items)/len(raw_data)*100:.1f}%)")
    print("="*60)

    print(f"\n✅ 蒸馏数据已保存至: {output_path}")

    # 展示一个样例
    if distilled_data:
        print("\n" + "="*60)
        print("样例展示")
        print("="*60)
        sample = distilled_data[0]
        print(f"问题: {sample['question']}")
        print(f"\n教师输出:\n{sample['teacher_output']}")
        print("="*60)

    return len(distilled_data)

if __name__ == "__main__":
    count = distill_data()
    print(f"\n下一步：运行 python scripts/3_filter_data.py")
```

### 1.3 双重过滤

```python
# scripts/3_filter_data.py
"""
双重过滤：答案正确性 + 推理质量
输出：data/processed/sft.jsonl, data/processed/rl.jsonl
"""
import json
import os
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import DATA_CONFIG

class DataFilter:
    """
    数据过滤器
    面试点：Fin-R1 的过滤策略
    """

    def __init__(self):
        self.stats = {
            "total": 0,
            "format_ok": 0,
            "answer_correct": 0,
            "reasoning_good": 0,
            "final_pass": 0,
            "filter_reasons": {}
        }

    def extract_answer(self, text):
        """提取 <answer> 中的内容"""
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if not match:
            return None

        answer = match.group(1).strip()

        # 处理 \boxed{} 格式
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer)
        if boxed_match:
            return boxed_match.group(1).strip()

        return answer

    def extract_think(self, text):
        """提取 <think> 中的内容"""
        match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        return match.group(1).strip() if match else None

    # ========== 第一层：格式检查 ==========

    def check_format(self, text):
        """检查格式是否符合要求"""
        think_count = text.count("<think>")
        answer_count = text.count("<answer>")

        if think_count != 1 or answer_count != 1:
            return False, "标签数量不对"

        # 检查标签顺序
        think_pos = text.find("<think>")
        answer_pos = text.find("<answer>")

        if think_pos > answer_pos:
            return False, "标签顺序错误"

        return True, "OK"

    # ========== 第二层：答案正确性 ==========

    def check_answer_math(self, extracted, gold):
        """数学题答案检查"""
        try:
            # 提取数字
            extracted_num = float(re.sub(r'[^\d.-]', '', str(extracted)))
            gold_num = float(re.sub(r'[^\d.-]', '', str(gold)))

            # 相对误差或绝对误差
            if abs(gold_num) > 1:
                # 相对误差
                return abs(extracted_num - gold_num) / abs(gold_num) < 0.01
            else:
                # 绝对误差
                return abs(extracted_num - gold_num) < 0.01
        except:
            return False

    def check_answer_qa(self, extracted, gold):
        """QA 题答案检查（关键词匹配）"""
        gold_lower = str(gold).lower()
        extracted_lower = str(extracted).lower()

        # 直接包含
        if gold_lower in extracted_lower:
            return True

        # 关键词重叠
        gold_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', gold_lower))
        extracted_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', extracted_lower))

        if not gold_keywords:
            return False

        overlap_ratio = len(gold_keywords & extracted_keywords) / len(gold_keywords)
        return overlap_ratio > 0.6

    def check_answer_correctness(self, item):
        """答案正确性总入口"""
        extracted = self.extract_answer(item["teacher_output"])

        if not extracted:
            return False, "无法提取答案"

        gold = item["gold_answer"]
        qtype = item["type"]

        if qtype == "financial_calculation":
            is_correct = self.check_answer_math(extracted, gold)
        else:
            is_correct = self.check_answer_qa(extracted, gold)

        return is_correct, "OK" if is_correct else "答案不正确"

    # ========== 第三层：推理质量 ==========

    def check_reasoning_quality(self, item):
        """
        推理质量检查
        面试点：Fin-R1 用 7 个维度，我们简化为 4 个核心维度
        """
        think_text = self.extract_think(item["teacher_output"])

        if not think_text:
            return False, "缺少推理过程"

        # 维度1：长度合理性
        if len(think_text) < 50:
            return False, "推理过程过短"
        if len(think_text) > 2000:
            return False, "推理过程过长"

        # 维度2：逻辑连接词
        reasoning_keywords = [
            "首先", "其次", "然后", "接着", "最后",
            "因此", "所以", "由于", "根据", "可得",
            "计算", "推导", "分析", "得出", "综上"
        ]
        keyword_count = sum(1 for kw in reasoning_keywords if kw in think_text)
        if keyword_count < 2:
            return False, "缺少逻辑连接词"

        # 维度3：步骤清晰性（金融计算题需要有计算过程）
        if item["type"] == "financial_calculation":
            # 检查是否有数学符号
            if not re.search(r'[=\+\-\*/÷×()]', think_text):
                return False, "计算题缺少计算过程"

        # 维度4：重复内容检查
        sentences = [s.strip() for s in re.split(r'[。！？\n]', think_text) if s.strip()]
        if len(sentences) > 3:
            unique_ratio = len(set(sentences)) / len(sentences)
            if unique_ratio < 0.7:
                return False, "存在大量重复内容"

        return True, "OK"

    # ========== 主流程 ==========

    def filter_item(self, item):
        """过滤单条数据"""
        # 第一层：格式
        format_ok, format_reason = self.check_format(item["teacher_output"])
        if not format_ok:
            self.stats["filter_reasons"][format_reason] = \
                self.stats["filter_reasons"].get(format_reason, 0) + 1
            return False

        self.stats["format_ok"] += 1

        # 第二层：答案
        answer_ok, answer_reason = self.check_answer_correctness(item)
        if not answer_ok:
            self.stats["filter_reasons"][answer_reason] = \
                self.stats["filter_reasons"].get(answer_reason, 0) + 1
            return False

        self.stats["answer_correct"] += 1

        # 第三层：推理质量
        quality_ok, quality_reason = self.check_reasoning_quality(item)
        if not quality_ok:
            self.stats["filter_reasons"][quality_reason] = \
                self.stats["filter_reasons"].get(quality_reason, 0) + 1
            return False

        self.stats["reasoning_good"] += 1
        self.stats["final_pass"] += 1

        return True

    def filter_all(self, data):
        """过滤所有数据"""
        self.stats["total"] = len(data)

        sft_data = []
        rl_data = []

        for item in data:
            if self.filter_item(item):
                # SFT 数据
                sft_data.append({
                    "id": item["id"],
                    "prompt": item["question"],
                    "response": item["teacher_output"],
                    "type": item["type"]
                })

                # RL 数据
                rl_data.append({
                    "id": item["id"],
                    "prompt": item["question"],
                    "gold_answer": item["gold_answer"],
                    "type": item["type"]
                })

        return sft_data, rl_data

    def print_report(self):
        """打印过滤报告"""
        print("\n" + "="*60)
        print("数据过滤报告")
        print("="*60)
        print(f"{'指标':<20s} {'数量':>10s} {'占比':>10s}")
        print("-"*60)
        print(f"{'原始样本':<20s} {self.stats['total']:>10d} {100.0:>9.1f}%")
        print(f"{'格式正确':<20s} {self.stats['format_ok']:>10d} {self.stats['format_ok']/self.stats['total']*100:>9.1f}%")
        print(f"{'答案正确':<20s} {self.stats['answer_correct']:>10d} {self.stats['answer_correct']/self.stats['total']*100:>9.1f}%")
        print(f"{'推理合格':<20s} {self.stats['reasoning_good']:>10d} {self.stats['reasoning_good']/self.stats['total']*100:>9.1f}%")
        print(f"{'最终通过':<20s} {self.stats['final_pass']:>10d} {self.stats['final_pass']/self.stats['total']*100:>9.1f}%")
        print("="*60)

        if self.stats["filter_reasons"]:
            print("\n过滤原因分布:")
            for reason, count in sorted(self.stats["filter_reasons"].items(),
                                       key=lambda x: -x[1]):
                print(f"  {reason:<30s}: {count:>5d}")

        print("="*60)

def split_train_test(sft_data, rl_data, test_ratio=0.2):
    """
    切分训练集和测试集
    面试点：如何避免数据泄漏
    """
    import random
    random.seed(42)

    # 按 ID 对齐
    assert len(sft_data) == len(rl_data)
    assert all(sft["id"] == rl["id"] for sft, rl in zip(sft_data, rl_data))

    # 打乱
    indices = list(range(len(sft_data)))
    random.shuffle(indices)

    # 切分
    test_size = int(len(sft_data) * test_ratio)
    test_indices = set(indices[:test_size])

    sft_train = [sft_data[i] for i in range(len(sft_data)) if i not in test_indices]
    sft_test = [sft_data[i] for i in range(len(sft_data)) if i in test_indices]

    rl_train = [rl_data[i] for i in range(len(rl_data)) if i not in test_indices]
    rl_test = [rl_data[i] for i in range(len(rl_data)) if i in test_indices]

    return sft_train, sft_test, rl_train, rl_test

def filter_data():
    """主流程"""
    print("="*60)
    print("阶段3：双重过滤（答案 + 推理质量）")
    print("="*60)

    # 加载蒸馏数据
    print(f"\n加载蒸馏数据: {DATA_CONFIG.distilled_data_path}")
    with open(DATA_CONFIG.distilled_data_path, 'r', encoding='utf-8') as f:
        distilled_data = [json.loads(line) for line in f]
    print(f"✓ 加载 {len(distilled_data)} 条")

    # 过滤
    print("\n执行过滤...")
    filter_obj = DataFilter()
    sft_data, rl_data = filter_obj.filter_all(distilled_data)

    # 打印报告
    filter_obj.print_report()

    # 切分训练/测试集
    print("\n切分训练/测试集...")
    sft_train, sft_test, rl_train, rl_test = split_train_test(
        sft_data, rl_data, DATA_CONFIG.test_ratio
    )

    print(f"SFT 训练集: {len(sft_train)} 条")
    print(f"SFT 测试集: {len(sft_test)} 条")
    print(f"RL 训练集: {len(rl_train)} 条")
    print(f"RL 测试集: {len(rl_test)} 条")

    # 保存
    os.makedirs("data/processed", exist_ok=True)

    def save_jsonl(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    save_jsonl(sft_train, DATA_CONFIG.sft_data_path)
    save_jsonl(rl_train, DATA_CONFIG.rl_data_path)
    save_jsonl(sft_test + rl_test, DATA_CONFIG.test_data_path)  # 测试集合并

    # 保存统计
    with open("data/processed/filter_stats.json", 'w', encoding='utf-8') as f:
        json.dump(filter_obj.stats, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 数据已保存:")
    print(f"   SFT 训练: {DATA_CONFIG.sft_data_path}")
    print(f"   RL 训练: {DATA_CONFIG.rl_data_path}")
    print(f"   测试集: {DATA_CONFIG.test_data_path}")

    return len(sft_train), len(rl_train), len(sft_test)

if __name__ == "__main__":
    sft_count, rl_count, test_count = filter_data()
    print(f"\n✅ 数据构建完成！")
    print(f"   最终数据资产：SFT {sft_count} 条 + RL {rl_count} 条 + 测试 {test_count} 条")
    print(f"\n下一步：上传到服务器并开始训练")
```

---

## 阶段2：训练脚本开发

由于脚本较长，完整代码已在前面给出。这里总结关键点：

### 2.1 SFT 训练（scripts/4_train_sft.py）

**关键修正**：
1. ✅ 使用 `BitsAndBytesConfig` 进行 4bit 量化
2. ✅ 使用 `prepare_model_for_kbit_training`
3. ✅ 数据格式使用 Qwen 的 chat template
4. ✅ 使用 `SFTTrainer` 的 `dataset_text_field` 参数

### 2.2 GRPO 训练（scripts/5_train_grpo.py）

**关键修正**：
1. ⚠️ **重要**：TRL 的 `GRPOTrainer` reward 函数签名需要检查
2. ⚠️ **重要**：需要确认是否需要 `AutoModelForCausalLMWithValueHead`
3. ✅ 数据格式需要包含 `query` 字段
4. ✅ reward 函数返回 `List[float]`

**修正后的 reward 函数签名**：

```python
def combined_reward(
    samples: List[str],          # 生成的文本
    prompts: List[str],          # 输入的 prompt
    outputs: List[str],          # 完整输出（可选）
    **kwargs                     # 额外参数（gold_answer, type 等）
) -> List[float]:
    """组合奖励函数"""
    # 从 kwargs 提取元数据
    gold_answers = kwargs.get("gold_answers", [])
    types = kwargs.get("types", [])

    # 计算奖励
    format_rewards = compute_format_reward(samples)
    accuracy_rewards = compute_accuracy_reward(samples, gold_answers, types)

    # 加权组合
    combined = [
        0.3 * f + 0.7 * a
        for f, a in zip(format_rewards, accuracy_rewards)
    ]

    return combined
```

---

## 阶段3：服务器训练执行

### 时间和预算修正

**AutoDL 5090 32GB 实际价格**：约 3-4 元/小时（按地区和机型不同）

| 阶段 | 时间 | 费用（按 3.5 元/小时） |
|------|------|----------------------|
| 环境准备 | 0.5h | ~2元 |
| SFT 训练 | 1.5h | ~5元 |
| GRPO 训练 | 2.5h | ~9元 |
| 评测 | 0.5h | ~2元 |
| 部署测试 | 0.5h | ~2元 |
| **总计** | **5.5h** | **~20元** |

**剩余预算**：80 元，足够多次实验和调优。

### 服务器执行流程

```bash
# 1. 开机（选择 PyTorch 2.1 + CUDA 12.1 镜像）
# 2. 上传代码和数据
scp -r FTModel root@your_server_ip:/root/

# 3. 安装依赖
cd /root/FTModel
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 执行训练
python scripts/4_train_sft.py 2>&1 | tee logs/sft_train.log
python scripts/5_train_grpo.py 2>&1 | tee logs/grpo_train.log

# 5. 评测
python scripts/6_evaluate.py 2>&1 | tee logs/eval.log

# 6. 部署
python scripts/7_deploy.py --action merge
python scripts/7_deploy.py --action serve &
sleep 60
python scripts/7_deploy.py --action test
```

---

## 阶段4：评测与部署

（代码已在前面给出，这里不重复）

---

## 阶段5：面试准备

### 5.1 消融实验

创建 `scripts/8_ablation_study.py`（代码已给出）

### 5.2 面试展示材料清单

1. **项目概述 PPT**（3-5 页）：
   - 第1页：项目背景（Fin-R1 论文启发）
   - 第2页：技术架构图
   - 第3页：数据构建流程（双重过滤）
   - 第4页：训练策略（SFT + GRPO）
   - 第5页：实验结果（消融实验对比）

2. **代码仓库**：GitHub/Gitee 公开仓库

3. **技术文档**：
   - README.md（快速开始）
   - 本文档（完整执行计划）
   - API 文档（如果有）

4. **实验报告**：
   - 数据过滤报告（filter_stats.json）
   - 训练曲线图（loss, reward, KL）
   - 评测结果（eval_*.json）
   - 消融实验对比表

---

## 常见问题排查

### 问题1：API 调用失败

**症状**：`2_distill_data.py` 报错 `API key invalid`

**解决**：
```bash
# 设置环境变量
export API_KEY="your_deepseek_or_qwen_api_key"

# 或在 configs/config.py 中直接配置
API_CONFIG.api_key = "sk-xxxxx"
```

### 问题2：显存不足（OOM）

**症状**：训练时报 `CUDA out of memory`

**解决**：
```python
# 在 configs/config.py 中调整参数
SFT_CONFIG.per_device_train_batch_size = 1  # 减小 batch size
SFT_CONFIG.gradient_accumulation_steps = 16  # 增大梯度累积
MODEL_CONFIG.model_max_length = 1024  # 减小序列长度
```

### 问题3：GRPO reward 函数报错

**症状**：`reward_function returned wrong type`

**解决**：确保 reward 函数返回 `List[float]`：
```python
def my_reward(samples, **kwargs):
    rewards = [...]  # 计算奖励
    return rewards  # 必须是 list，不是 tensor
```

### 问题4：vLLM 启动失败

**症状**：`ImportError: No module named 'vllm'`

**解决**：
```bash
# 检查 CUDA 版本
nvcc --version

# 根据 CUDA 版本安装（示例：CUDA 12.1）
pip install vllm==0.6.3

# 如果仍失败，使用 transformers 推理
python scripts/6_evaluate.py --use_transformers
```

### 问题5：模型输出不包含标签

**症状**：训练后模型不输出 `<think><answer>`

**排查**：
1. 检查 SFT 数据格式是否正确
2. 增加训练步数（至少 500 步）
3. 检查 chat template 是否正确应用
4. 查看训练日志中的生成样例

---

## 面试问答手册

### 技术深度类

**Q1: 为什么需要双重过滤？单一打分不行吗？**

A: 双重过滤是 Fin-R1 的核心设计：
1. **答案正确性**是硬约束 - 金融场景不能容忍错误答案，必须先保证答案对
2. **推理质量**是软约束 - 在答案对的基础上，筛选推理过程清晰、逻辑连贯的样本
3. 单一打分会混淆两个维度，导致"答案错但推理像模像样"的样本通过

**Q2: SFT 和 GRPO 的目标函数分别是什么？**

A:
- **SFT 目标函数**：`max_θ E_{(x,y)~D}[log P_θ(y|x)]`
  - 最大化训练数据的似然，本质是行为克隆
  - 让模型学会"先推理再作答"的输出结构

- **GRPO 目标函数**：`max_θ E_x[E_{y~π_θ}[R(x,y)]] - β·KL(π_θ || π_ref)`
  - 最大化奖励期望，同时用 KL 散度约束模型不要偏离 SFT 初始化太远
  - β 是权衡系数，防止 reward hacking

**Q3: GRPO 和 PPO 有什么区别？**

A:
- **PPO**: 用全局 baseline（通常是 value network 的输出）计算优势函数
- **GRPO**: 用同一个 prompt 的多个采样的**组内相对表现**计算优势
  - 优势：不需要独立训练 value network，更稳定
  - 原理：`A(y) = R(y) - mean(R(y_1, ..., y_k))`，其中 y_1...y_k 是同一 prompt 的多个采样

**Q4: 为什么格式奖励和准确性奖励要分开？**

A:
1. **解耦关注点**：格式保证可解释性，准确性保证业务价值
2. **不同难度**：格式奖励容易学（规则明确），准确性奖励难学（需要推理能力）
3. **可调权重**：根据业务需求调整，比如监管场景提高格式权重，业务场景提高准确性权重
4. **Fin-R1 的实验**：单独用格式奖励会导致"格式对但胡说八道"

### 工程实践类

**Q5: 如果数据蒸馏成本太高怎么办？**

A:
1. **短期方案**：
   - 用开源 CoT 数据集（如 MetaMath, OpenO1）改造
   - 用更便宜的 API（DeepSeek 0.14 元/M tokens）
   - 减少数据量（100-200 条就能跑通链路）

2. **长期方案**：
   - 自己训练一个 7B 教师模型（一次投入，反复使用）
   - 用蒸馏后的模型做自我迭代（self-training）
   - 人工编写高质量 CoT 模板

**Q6: 如何扩展到生产环境？**

A:
1. **数据规模**：扩展到 10 万+ 条，覆盖更多金融场景
2. **模型选型**：用 7B/14B 基座，性能更好
3. **评测体系**：
   - 离线评测：多个金融基准测试
   - 在线评测：A/B 测试、用户满意度
   - 人工抽样：定期人工审核生成质量
4. **部署优化**：
   - 多卡并行（tensor parallelism）
   - 负载均衡（多实例 + Nginx）
   - 监控告警（Prometheus + Grafana）
5. **持续迭代**：收集线上反馈，持续优化数据和模型

**Q7: 如何避免数据泄漏？**

A:
1. **严格切分**：训练/测试集按 ID 或时间切分，不能随机打乱后切分
2. **独立测试集**：测试集不参与任何训练过程（包括超参调优）
3. **定期更新**：定期更新测试集，避免模型过拟合旧测试集
4. **隐私保护**：如果用真实业务数据，需要脱敏处理

### 业务理解类

**Q8: 为什么金融场景需要 CoT？**

A:
1. **监管要求**：金融模型需要可解释性，监管部门要求"黑盒"模型给出推理依据
2. **用户信任**：用户看到计算步骤更容易信任结果（尤其是金额计算）
3. **可调试性**：出错时可以定位到具体推理环节，快速修复
4. **可验证性**：人工审核时可以检查推理逻辑，而不仅仅是最终答案

**Q9: 如何评估模型在真实业务中的表现？**

A:
1. **离线指标**：
   - 格式正确率（可解释性）
   - 答案准确率（业务价值）
   - 按任务类型细分（发现短板）

2. **在线指标**：
   - 用户满意度（显式反馈：点赞/点踩）
   - 业务转化率（用户是否采纳模型建议）
   - 错误率（人工审核发现的错误比例）

3. **A/B 测试**：
   - 对比新模型 vs 旧模型
   - 对比模型 vs 人工专家
   - 对比不同 prompt 策略

---

## 总结：交付物检查清单

- [ ] 代码仓库（GitHub/Gitee）
- [ ] 完整数据流水线脚本（1-3）
- [ ] 训练脚本（4-5）
- [ ] 评测和部署脚本（6-7）
- [ ] 消融实验脚本（8）
- [ ] 配置文件（configs/config.py）
- [ ] 依赖文件（requirements.txt）
- [ ] 项目文档（README + 本文档）
- [ ] 实验报告：
  - [ ] 数据过滤报告（filter_stats.json）
  - [ ] 训练日志（logs/*.log）
  - [ ] 评测结果（reports/eval_*.json）
  - [ ] 消融实验（reports/ablation_summary.json）
- [ ] 面试材料：
  - [ ] 项目概述 PPT（3-5 页）
  - [ ] 技术问答准备（本文档第9节）
  - [ ] Demo 视频/截图（可选）

---

## 立即开始

```bash
# 1. 设置环境变量
export API_KEY="your_api_key_here"

# 2. 创建项目结构
cd c:\gitclones\FTModel
mkdir -p data/{raw,processed} scripts ckpts reports configs logs

# 3. 复制本文档中的所有脚本到对应目录

# 4. 本地执行数据构建（3-4 小时）
python scripts/1_prepare_raw_data.py
python scripts/2_distill_data.py
python scripts/3_filter_data.py

# 5. 上传到 AutoDL 并执行训练（5-6 小时）
# （见"阶段3：服务器训练执行"）

# 6. 整理面试材料（1 小时）
```

**预计总时间**：10-12 小时
**预计总费用**：API 费 5 元 + 服务器 20 元 = **25 元**

**剩余预算充裕**，可以多次实验和调优！

---

**Good Luck! 🚀**
