# 金融推理模型后训练项目

基于 **Fin-R1 论文**的完整后训练链路实现，涵盖数据构建、SFT、GRPO 强化学习、评测与部署全流程。

## 项目背景

本项目实现了一个金融领域的推理增强模型，核心思路：

```text
原始数据 → 教师蒸馏 → 双重过滤 → 训练数据资产
                                     ↓
                             ┌───────┴───────┐
                             ↓               ↓
                           SFT            GRPO
                        (学格式)        (提准确率)
                             ↓               ↓
                             └───────┬───────┘
                                     ↓
                             评测 → 部署
```

### 关键技术决策

| 问题                 | 决策                   | 原因                           |
| -------------------- | ---------------------- | ------------------------------ |
| 为什么要数据蒸馏？   | 用强模型生成 CoT       | 小模型缺乏推理能力             |
| 为什么双重过滤？     | 答案正确性 + 推理质量  | 答案是硬约束，推理是质量保证   |
| 为什么先 SFT？       | 学习输出格式           | RL 需要稳定的格式作为基础      |
| 为什么用 GRPO？      | 可验证奖励 + 稳定训练  | 金融场景需要客观标准           |
| 为什么用 vLLM？      | 高性能推理             | PagedAttention + Continuous batching |

---

## 项目结构

```text
FTModel/
├── .gitignore                      # Git 忽略配置
├── README.md                       # 项目说明（本文件）
├── requirements.txt                # Python 依赖
├── detect_plan.md                  # 详细执行计划文档
│
├── configs/                        # 配置文件目录
│   ├── __init__.py
│   └── config.py                   # 全局配置（模型、LoRA、训练参数）
│
├── scripts/                        # 脚本目录
│   ├── __init__.py
│   ├── 1_prepare_raw_data.py       # 阶段1: 准备原始数据
│   ├── 2_distill_data.py           # 阶段2: 数据蒸馏（教师模型生成CoT）
│   ├── 3_filter_data.py            # 阶段3: 双重过滤（答案+推理质量）
│   ├── 4_train_sft.py              # 阶段4: SFT 监督微调
│   ├── 5_train_grpo.py             # 阶段5: GRPO 强化学习
│   ├── 6_evaluate.py               # 阶段6: 模型评测
│   ├── 7_deploy.py                 # 阶段7: 模型部署
│   └── 8_ablation_study.py         # 阶段8: 消融实验
│
├── data/                           # 数据目录
│   ├── raw/                        # 原始数据
│   │   └── raw.jsonl               # 原始问答数据
│   └── processed/                  # 处理后数据
│       ├── distilled.jsonl         # 蒸馏后数据（含CoT）
│       ├── sft.jsonl               # SFT 训练数据
│       ├── rl.jsonl                # RL 训练数据
│       ├── test.jsonl              # 测试数据
│       └── filter_stats.json       # 过滤统计
│
├── ckpts/                          # 模型检查点
│   ├── sft_lora/                   # SFT LoRA 权重
│   ├── grpo_lora/                  # GRPO LoRA 权重
│   └── merged_model/               # 合并后完整模型
│
├── reports/                        # 评测报告
│   ├── eval_*.json                 # 评测摘要
│   ├── eval_details_*.jsonl        # 评测详情
│   └── ablation_summary_*.json     # 消融实验报告
│
└── logs/                           # 训练日志
    ├── sft_train.log
    └── grpo_train.log
```

---

## 环境配置

### 硬件要求

| 阶段           | 最低配置         | 推荐配置                 |
| -------------- | ---------------- | ------------------------ |
| 数据准备       | CPU + 8GB RAM    | -                        |
| 数据蒸馏       | CPU（调用API）   | -                        |
| SFT/GRPO 训练  | 16GB VRAM        | 24GB+ VRAM (5090/A100)   |
| 评测/部署      | 8GB VRAM         | 16GB+ VRAM               |

### 安装依赖

```bash
# 创建虚拟环境（推荐）
conda create -n ftmodel python=3.10
conda activate ftmodel

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 如果在 AutoDL 等云平台，可能需要额外安装
pip install flash-attn --no-build-isolation
```

### 配置 API 密钥

数据蒸馏需要调用教师模型 API（DeepSeek 或 Qwen）：

```bash
# 方法1: 环境变量
export API_KEY="your_api_key_here"

# 方法2: 在 configs/config.py 中直接配置
API_CONFIG.api_key = "sk-xxxxx"
```

---

## 执行流程

### 完整流程图

```text
┌─────────────────────────────────────────────────────────────────┐
│                     本地执行（数据准备）                          │
├─────────────────────────────────────────────────────────────────┤
│  1. 准备原始数据    →  2. 数据蒸馏    →  3. 双重过滤             │
│  (1_prepare_raw)      (2_distill)       (3_filter)              │
│       ↓                    ↓                 ↓                   │
│   raw.jsonl          distilled.jsonl    sft.jsonl + rl.jsonl    │
└─────────────────────────────────────────────────────────────────┘
                              ↓ 上传到服务器
┌─────────────────────────────────────────────────────────────────┐
│                    服务器执行（训练）                             │
├─────────────────────────────────────────────────────────────────┤
│  4. SFT 训练        →  5. GRPO 训练                              │
│  (4_train_sft)         (5_train_grpo)                           │
│       ↓                      ↓                                   │
│  ckpts/sft_lora/       ckpts/grpo_lora/                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    评测与部署                                    │
├─────────────────────────────────────────────────────────────────┤
│  6. 评测            →  7. 部署        →  8. 消融实验             │
│  (6_evaluate)          (7_deploy)        (8_ablation)           │
│       ↓                    ↓                  ↓                  │
│  reports/eval_*.json   vLLM API 服务    reports/ablation_*.json │
└─────────────────────────────────────────────────────────────────┘
```

### 阶段1: 准备原始数据

```bash
python scripts/1_prepare_raw_data.py
```

**功能**：

- 下载 qwen-dianjin 数据集（或使用本地数据）
- 添加自定义金融问答数据
- 自动分类题目类型（计算题/概念题/分析题/推理题）
- 分层抽样保证数据分布

**输出**：`data/raw/raw.jsonl`

### 阶段2: 数据蒸馏

```bash
python scripts/2_distill_data.py
```

**功能**：

- 调用教师模型（DeepSeek/Qwen）生成 Chain-of-Thought
- 强制输出 `<think>...</think><answer>...</answer>` 格式
- 自动重试失败请求（指数退避）
- 增量保存防止中断丢失

**输出**：`data/processed/distilled.jsonl`

### 阶段3: 双重过滤

```bash
python scripts/3_filter_data.py
```

**功能**：

- **第一层**：格式检查（标签完整性和顺序）
- **第二层**：答案正确性（数值匹配/关键词匹配）
- **第三层**：推理质量（长度/逻辑词/计算过程/重复检测）
- 自动切分训练集/测试集（8:2）

**输出**：

- `data/processed/sft.jsonl` - SFT 训练数据
- `data/processed/rl.jsonl` - RL 训练数据
- `data/processed/test.jsonl` - 测试数据
- `data/processed/filter_stats.json` - 过滤统计

### 阶段4: SFT 训练

```bash
python scripts/4_train_sft.py
```

**功能**：

- 4bit 量化加载 Qwen2.5-3B-Instruct
- LoRA 微调（r=16, alpha=32）
- Qwen chat template 格式化
- 梯度检查点节省显存

**核心参数**（可在 `configs/config.py` 调整）：

```python
SFT_CONFIG.num_train_epochs = 2
SFT_CONFIG.per_device_train_batch_size = 2
SFT_CONFIG.gradient_accumulation_steps = 8
SFT_CONFIG.learning_rate = 2e-4
```

**输出**：`ckpts/sft_lora/`

### 阶段5: GRPO 训练

```bash
python scripts/5_train_grpo.py
```

**功能**：

- 基于 SFT 模型继续训练
- 格式奖励（0.3权重）：检查 `<think><answer>` 标签
- 准确性奖励（0.7权重）：答案匹配
- KL 散度约束防止偏离

**核心参数**：

```python
GRPO_CONFIG.num_sample_generations = 4  # 每个 prompt 采样数
GRPO_CONFIG.temperature = 0.7
GRPO_CONFIG.kl_coef = 0.05
GRPO_CONFIG.format_reward_weight = 0.3
GRPO_CONFIG.accuracy_reward_weight = 0.7
```

**输出**：`ckpts/grpo_lora/`

### 阶段6: 评测

```bash
# 评测 GRPO 模型（默认）
python scripts/6_evaluate.py

# 评测 SFT 模型
python scripts/6_evaluate.py --eval_sft

# 使用 vLLM 加速（需要先合并模型）
python scripts/6_evaluate.py --use_vllm --model_path ckpts/merged_model
```

**评测指标**：

- 格式正确率：`<think><answer>` 结构完整
- 答案正确率：数值题精确匹配，QA题关键词匹配
- 按题目类型细分统计

**输出**：`reports/eval_*.json`

### 阶段7: 部署

```bash
# 1. 合并 LoRA 权重
python scripts/7_deploy.py --action merge

# 2. 启动 vLLM 服务
python scripts/7_deploy.py --action serve

# 3. 测试服务
python scripts/7_deploy.py --action test

# 或使用简易服务（不需要 vLLM）
python scripts/7_deploy.py --action simple
```

**API 调用示例**：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ckpts/merged_model",
    "messages": [{"role": "user", "content": "某公司2023年营收1000万元，同比增长25%，请计算2022年营收。"}]
  }'
```

### 阶段8: 消融实验

```bash
python scripts/8_ablation_study.py
```

**对比实验**：

1. Base model only（无训练）
2. SFT only（仅 SFT）
3. SFT + GRPO（完整流程）

**输出**：`reports/ablation_summary_*.json`

---

## 数据格式说明

### 原始数据 (raw.jsonl)

```json
{
  "id": "qwen_0",
  "question": "某公司2023年营收1000万元，同比增长25%，请计算2022年营收。",
  "gold_answer": "800",
  "type": "financial_calculation",
  "source": "qwen-dianjin"
}
```

### 蒸馏数据 (distilled.jsonl)

```json
{
  "id": "qwen_0",
  "question": "...",
  "gold_answer": "800",
  "type": "financial_calculation",
  "teacher_output": "<think>\n首先，设2022年营收为X...\n</think>\n<answer>\n800\n</answer>"
}
```

### SFT 数据 (sft.jsonl)

```json
{
  "id": "qwen_0",
  "prompt": "某公司2023年营收1000万元，同比增长25%，请计算2022年营收。",
  "response": "<think>...</think><answer>800</answer>",
  "type": "financial_calculation"
}
```

### RL 数据 (rl.jsonl)

```json
{
  "id": "qwen_0",
  "prompt": "某公司2023年营收1000万元，同比增长25%，请计算2022年营收。",
  "gold_answer": "800",
  "type": "financial_calculation"
}
```

---

## 配置说明

主要配置在 `configs/config.py`：

```python
# 模型配置
MODEL_CONFIG.base_model = "Qwen/Qwen2.5-3B-Instruct"
MODEL_CONFIG.model_max_length = 2048

# LoRA 配置
LORA_CONFIG.r = 16
LORA_CONFIG.lora_alpha = 32
LORA_CONFIG.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", ...]

# SFT 配置
SFT_CONFIG.num_train_epochs = 2
SFT_CONFIG.learning_rate = 2e-4

# GRPO 配置
GRPO_CONFIG.format_reward_weight = 0.3
GRPO_CONFIG.accuracy_reward_weight = 0.7

# API 配置（数据蒸馏用）
API_CONFIG.provider = "deepseek"  # 或 "qwen"
```

---

## 常见问题

### Q1: API 调用失败

```bash
# 检查 API_KEY 是否设置
echo $API_KEY

# 或直接在配置文件中设置
# configs/config.py: API_CONFIG.api_key = "sk-xxxxx"
```

### Q2: 显存不足 (OOM)

```python
# 在 configs/config.py 中调整
SFT_CONFIG.per_device_train_batch_size = 1  # 减小
SFT_CONFIG.gradient_accumulation_steps = 16  # 增大
MODEL_CONFIG.model_max_length = 1024  # 减小
```

### Q3: vLLM 启动失败

```bash
# 检查是否是 LoRA adapter
# vLLM 需要合并后的完整模型
python scripts/7_deploy.py --action merge
python scripts/7_deploy.py --action serve --model_path ckpts/merged_model
```

### Q4: 模型不输出正确格式

- 检查 SFT 数据格式是否正确
- 增加训练步数（至少 500 步）
- 检查 chat template 是否正确应用

---

## 参考资料

- [Fin-R1 论文](https://arxiv.org/abs/xxx)
- [TRL 文档](https://huggingface.co/docs/trl)
- [vLLM 文档](https://docs.vllm.ai)
- [PEFT 文档](https://huggingface.co/docs/peft)

---

## License

MIT License
