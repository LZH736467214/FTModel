# 金融推理模型后训练项目

基于 **Fin-R1 论文**的完整后训练链路实现，涵盖数据构建、SFT、GRPO 强化学习、评测与部署全流程。

## ✨ 项目特点

- 🏗️ **完整工程链路**：从数据构建到模型部署的全流程实现
- 🧠 **三模型架构**：Teacher（蒸馏）+ Judge（评分）+ Base（训练）分工明确
- 🚀 **本地部署优先**：所有模型本地运行，无需调用外部 API
- 📊 **Wandb 集成**：可视化监控训练过程
- 🔬 **消融实验**：科学验证每个组件的价值

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

### 模型架构

本项目采用三模型架构设计，完全本地部署：

| 角色        | 模型                        | 用途                      | 显存需求    |
| ----------- | --------------------------- | ------------------------- | ----------- |
| **Teacher** | DeepSeek-R1-Distill-Qwen-7B | 蒸馏阶段生成 CoT          | ~8GB (4bit) |
| **Judge**   | Qwen2.5-7B-Instruct         | 过滤、GRPO 评分、评测判分 | ~8GB (4bit) |
| **Base**    | Qwen2.5-1.5B-Instruct       | 待训练的目标小模型        | ~4GB (4bit) |

### 关键技术决策

| 问题               | 决策                  | 原因                                 |
| ------------------ | --------------------- | ------------------------------------ |
| 为什么要数据蒸馏？ | 用强模型生成 CoT      | 小模型缺乏推理能力                   |
| 为什么双重过滤？   | 规则过滤 + Judge 评估 | 答案是硬约束，推理质量需 AI 评估     |
| 为什么先 SFT？     | 学习输出格式          | RL 需要稳定的格式作为基础            |
| 为什么用 GRPO？    | 可验证奖励 + 稳定训练 | 金融场景需要客观标准                 |
| 为什么用 vLLM？    | 高性能推理            | PagedAttention + Continuous batching |

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
├── scripts/                        # 脚本目录（按执行顺序编号）
│   ├── __init__.py
│   ├── 1_prepare_raw_data.py       # 阶段1: 准备原始数据（DianJin-R1-Data + 自定义）
│   ├── 2_distill_data.py           # 阶段2: Teacher 模型蒸馏生成 CoT
│   ├── 3_filter_data.py            # 阶段3: 规则 + Judge 双重过滤
│   ├── 4_train_sft.py              # 阶段4: SFT 监督微调（学格式）
│   ├── 5_train_grpo.py             # 阶段5: GRPO 强化学习（提准确率）
│   ├── 6_evaluate.py               # 阶段6: Base 推理 + Judge 判分
│   ├── 7_deploy.py                 # 阶段7: LoRA 合并 + vLLM 部署
│   └── 8_ablation_study.py         # 阶段8: 消融实验对比
│
├── docs/                           # 文档目录
│   └── wandb_integration.md        # Wandb 集成说明
│
├── dataraw/                        # 原始数据目录
│   └── raw.jsonl                   # 原始问答数据
│
├── dataprocessed/                  # 处理后数据目录
│   ├── distilled.jsonl             # 蒸馏后数据（含 CoT）
│   ├── sft.jsonl                   # SFT 训练数据
│   ├── rl.jsonl                    # RL 训练数据
│   ├── test.jsonl                  # 测试数据
│   └── filter_stats.json           # 过滤统计
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

| 阶段                | 最低配置      | 推荐配置               |
| ------------------- | ------------- | ---------------------- |
| 数据准备            | CPU + 8GB RAM | -                      |
| 数据蒸馏（Teacher） | 8GB VRAM      | 16GB+ VRAM             |
| 双重过滤（Judge）   | 8GB VRAM      | 16GB+ VRAM             |
| SFT/GRPO 训练       | 16GB VRAM     | 24GB+ VRAM (5090/A100) |
| 评测/部署           | 8GB VRAM      | 16GB+ VRAM             |

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

### 配置 Wandb（可选）

项目集成了 Wandb 用于训练监控，首次使用需登录：

```bash
# 安装 wandb
pip install wandb

# 登录（从 https://wandb.ai/authorize 获取 API Key）
wandb login
```

配置位于 `configs/config.py` 的 `WandbConfig`，可禁用：

```python
WANDB_CONFIG.enabled = False  # 禁用 wandb
```

详见 [docs/wandb_integration.md](docs/wandb_integration.md)

---

## 执行流程

### 完整流程图

```text
┌───────────────────────────────────────────────────────────────────────┐
│                     数据准备阶段                                       │
├───────────────────────────────────────────────────────────────────────┤
│  1. 准备原始数据      →  2. Teacher蒸馏   →  3. 规则+Judge过滤        │
│  (1_prepare_raw_data)   (2_distill_data)    (3_filter_data)          │
│       ↓                       ↓                    ↓                  │
│   raw.jsonl             distilled.jsonl     sft.jsonl + rl.jsonl     │
│                         (Teacher生成CoT)    (Judge评估推理质量)       │
└───────────────────────────────────────────────────────────────────────┘
                                ↓
┌───────────────────────────────────────────────────────────────────────┐
│                     模型训练阶段                                       │
├───────────────────────────────────────────────────────────────────────┤
│  4. SFT 训练          →  5. GRPO 训练（含 Judge 评分）                 │
│  (4_train_sft)           (5_train_grpo)                              │
│       ↓                        ↓                                      │
│  ckpts/sft_lora/         ckpts/grpo_lora/                            │
│  (Base学习格式)           (Base提升准确率)                            │
└───────────────────────────────────────────────────────────────────────┘
                                ↓
┌───────────────────────────────────────────────────────────────────────┐
│                     评测与部署阶段                                     │
├───────────────────────────────────────────────────────────────────────┤
│  6. 评测(Judge判分)  →  7. 部署(vLLM)  →  8. 消融实验                  │
│  (6_evaluate)           (7_deploy)        (8_ablation_study)         │
│       ↓                      ↓                   ↓                    │
│  reports/eval_*.json    OpenAI API服务    reports/ablation_*.json    │
└───────────────────────────────────────────────────────────────────────┘
```

### 阶段1: 准备原始数据

```bash
python scripts/1_prepare_raw_data.py
```

**功能**：

- 下载 DianJin-R1-Data 数据集（支持 ModelScope / HuggingFace）
- 添加自定义金融问答数据
- 自动分类题目类型（计算题/概念题/分析题/推理题）
- 分层抽样保证数据分布

**输出**：`dataraw/raw.jsonl`

### 阶段2: 数据蒸馏

```bash
python scripts/2_distill_data.py
```

**功能**：

- 使用本地 **Teacher 模型**（DeepSeek-R1-Distill-Qwen-7B）生成 Chain-of-Thought
- 强制输出 `<think>...</think><answer>...</answer>` 格式
- 4bit 量化加载，节省显存
- 增量保存防止中断丢失

**输出**：`dataprocessed/distilled.jsonl`

### 阶段3: 双重过滤

```bash
python scripts/3_filter_data.py
```

**功能**：

- **第一层（规则过滤）**：
  - 格式检查（标签完整性和顺序）
  - 答案正确性（数值匹配/关键词匹配）
- **第二层（Judge 过滤）**：
  - 使用 **Judge 模型**（Qwen2.5-7B-Instruct）评估推理质量
  - 多维度打分：逻辑清晰度、专业准确性、完整性、简洁性
- 自动切分训练集/测试集（8:2）

**输出**：

- `dataprocessed/sft.jsonl` - SFT 训练数据
- `dataprocessed/rl.jsonl` - RL 训练数据
- `dataprocessed/test.jsonl` - 测试数据
- `dataprocessed/filter_stats.json` - 过滤统计

### 阶段4: SFT 训练

```bash
python scripts/4_train_sft.py
```

**功能**：

- 4bit 量化加载 **Base 模型**（Qwen2.5-1.5B-Instruct）
- LoRA 微调（r=16, alpha=32）
- Qwen chat template 格式化
- 梯度检查点节省显存
- Wandb 实时监控（可选）

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
# 完整模式（使用 Judge 模型评分）
python scripts/5_train_grpo.py

# 简化模式（仅规则评分）
python scripts/5_train_grpo.py --no-judge
```

**功能**：

- 基于 SFT 模型继续训练
- **格式奖励**（0.3权重）：规则检查 `<think><answer>` 标签
- **准确性奖励**（0.7权重）：
  - 规则匹配：数值/关键词匹配
  - Judge 评分：**Judge 模型**综合评估（可选）
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
# 评测 GRPO 模型（默认）+ Judge 判分
python scripts/6_evaluate.py

# 评测 SFT 模型
python scripts/6_evaluate.py --eval_sft

# 仅规则评测（不使用 Judge）
python scripts/6_evaluate.py --no-judge

# 使用 vLLM 加速（需要先合并模型）
python scripts/6_evaluate.py --use_vllm --model_path ckpts/merged_model
```

**评测指标**：

- 格式正确率：`<think><answer>` 结构完整
- 答案正确率：数值题精确匹配，QA题关键词匹配
- Judge 判分：推理质量综合评估
- 按题目类型细分统计

**输出**：`reports/eval_*.json`

### 阶段7: 部署

```bash
# 1. 合并 LoRA 权重到 Base 模型
python scripts/7_deploy.py --action merge

# 2. 启动 vLLM OpenAI-compatible API 服务
python scripts/7_deploy.py --action serve

# 3. 测试服务
python scripts/7_deploy.py --action test

# 或使用简易 Flask 服务（不需要 vLLM）
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

1. **Base model only**（无训练）- Qwen2.5-1.5B-Instruct 原始能力
2. **SFT only**（仅 SFT）- 学习输出格式后的效果
3. **SFT + GRPO**（完整流程）- 强化学习优化后的效果

**输出**：`reports/ablation_summary_*.json`

---

## 数据格式说明

### 原始数据 (raw.jsonl)

```json
{
  "id": "dianjin_0",
  "question": "某公司2023年营收1000万元，同比增长25%，请计算2022年营收。",
  "gold_answer": "800",
  "type": "financial_calculation",
  "source": "dianjin-r1-data"
}
```

### 蒸馏数据 (distilled.jsonl)

```json
{
  "id": "dianjin_0",
  "question": "...",
  "gold_answer": "800",
  "type": "financial_calculation",
  "teacher_output": "<think>\n首先，设2022年营收为X...\n</think>\n<answer>\n800\n</answer>"
}
```

### SFT 数据 (sft.jsonl)

```json
{
  "id": "dianjin_0",
  "prompt": "某公司2023年营收1000万元，同比增长25%，请计算2022年营收。",
  "response": "<think>...</think><answer>800</answer>",
  "type": "financial_calculation"
}
```

### RL 数据 (rl.jsonl)

```json
{
  "id": "dianjin_0",
  "prompt": "某公司2023年营收1000万元，同比增长25%，请计算2022年营收。",
  "gold_answer": "800",
  "type": "financial_calculation"
}
```

---

## 配置说明

主要配置在 `configs/config.py`：

```python
# ========== 三模型架构配置 ==========
# Teacher 模型（用于蒸馏）
TEACHER_CONFIG.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Judge 模型（用于过滤、评分、评测）
JUDGE_CONFIG.model_name = "Qwen/Qwen2.5-7B-Instruct"

# Base 模型（待训练的目标小模型）
MODEL_CONFIG.base_model = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_CONFIG.model_max_length = 2048

# ========== LoRA 配置 ==========
LORA_CONFIG.r = 16
LORA_CONFIG.lora_alpha = 32
LORA_CONFIG.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", ...]

# ========== SFT 配置 ==========
SFT_CONFIG.num_train_epochs = 2
SFT_CONFIG.learning_rate = 2e-4
SFT_CONFIG.per_device_train_batch_size = 2
SFT_CONFIG.gradient_accumulation_steps = 8

# ========== GRPO 配置 ==========
GRPO_CONFIG.format_reward_weight = 0.3
GRPO_CONFIG.accuracy_reward_weight = 0.7
GRPO_CONFIG.kl_coef = 0.05

# ========== Wandb 配置 ==========
WANDB_CONFIG.enabled = True
WANDB_CONFIG.project = "FTModel-Training"
```

---

## 常见问题

### Q1: 显存不足 (OOM)

```python
# 在 configs/config.py 中调整
SFT_CONFIG.per_device_train_batch_size = 1  # 减小
SFT_CONFIG.gradient_accumulation_steps = 16  # 增大
MODEL_CONFIG.model_max_length = 1024  # 减小
```

### Q2: Teacher/Judge 模型加载失败

```bash
# 检查模型是否下载完成
# 可以手动下载到本地，然后修改 config.py 中的模型路径
TEACHER_CONFIG.model_name = "/path/to/local/DeepSeek-R1-Distill-Qwen-7B"
JUDGE_CONFIG.model_name = "/path/to/local/Qwen2.5-7B-Instruct"
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

### Q5: 如何跳过 Judge 模型评分？

```bash
# 过滤阶段：仅使用规则过滤
python scripts/3_filter_data.py --no-judge

# GRPO 阶段：仅使用规则奖励
python scripts/5_train_grpo.py --no-judge

# 评测阶段：仅使用规则评测
python scripts/6_evaluate.py --no-judge
```

---

## 面试要点

本项目涵盖以下面试高频考点：

### 数据工程

- **数据蒸馏**：为什么用 Teacher 模型生成 CoT？小模型缺乏推理能力
- **双重过滤**：规则保证格式，Judge 保证推理质量
- **分层抽样**：保证数据分布合理

### 模型训练

- **LoRA 微调**：参数高效，只训练约 0.1% 的参数
- **4bit 量化**：NF4 + 双重量化，显存占用降低 4 倍
- **GRPO vs PPO**：GRPO 用组内相对表现计算优势，无需 value network

### 奖励设计

- **格式奖励**：保证可解释性
- **准确性奖励**：保证业务价值
- **KL 约束**：防止 reward hacking

### 部署优化

- **vLLM**：PagedAttention + Continuous batching
- **模型合并**：部署时减少推理开销

---

## 参考资料

- [Fin-R1 论文](https://arxiv.org/abs/xxx)
- [TRL 文档](https://huggingface.co/docs/trl)
- [vLLM 文档](https://docs.vllm.ai)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [DeepSeek-R1 模型](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- [Qwen2.5 模型系列](https://huggingface.co/Qwen)
- [DianJin-R1-Data 数据集](https://huggingface.co/datasets/DianJin/DianJin-R1-Data)

---

## License

MIT License
