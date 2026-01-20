# Wandb 集成说明

本项目已集成 [Weights & Biases (wandb)](https://wandb.ai/) 用于训练和评测过程的可视化监控。

## 安装 wandb

```bash
pip install wandb
```

## 初次使用配置

首次使用需要登录 wandb：

```bash
wandb login
```

然后输入你的 API Key（从 https://wandb.ai/authorize 获取）。

## 配置说明

在 [configs/config.py](../configs/config.py) 中有 `WandbConfig` 类：

```python
@dataclass
class WandbConfig:
    """Wandb 配置"""
    project: str = "FTModel-Training"  # wandb 项目名称
    entity: Optional[str] = None  # wandb 团队名称（可选）
    enabled: bool = True  # 是否启用 wandb

    # 运行名称（自动生成时间戳）
    run_name_sft: str = "sft-train"
    run_name_grpo: str = "grpo-train"
    run_name_eval: str = "evaluation"
    run_name_ablation: str = "ablation-study"

    # 日志配置
    log_model: bool = False  # 是否上传模型到 wandb（通常很大，不建议）
    log_interval: int = 10  # 日志记录间隔
```

### 修改配置

可以直接在 `configs/config.py` 中修改：

```python
# 禁用 wandb
WANDB_CONFIG.enabled = False

# 修改项目名称
WANDB_CONFIG.project = "my-custom-project"

# 设置团队名称
WANDB_CONFIG.entity = "my-team-name"
```

## 监控的训练阶段

### 1. T3 - SFT 训练 (scripts/4_train_sft.py)

运行 SFT 训练时会自动记录：

```bash
python scripts/4_train_sft.py
```

**记录指标：**
- 训练损失 (loss)
- 学习率 (learning_rate)
- 梯度范数 (grad_norm)
- 训练步数 (step)
- Epoch 进度

**配置信息：**
- 模型名称
- LoRA 参数 (r, alpha, dropout)
- 训练超参数 (batch_size, learning_rate, epochs)

### 2. T4 - GRPO 训练 (scripts/5_train_grpo.py)

运行 GRPO 训练时会自动记录：

```bash
python scripts/5_train_grpo.py
# 或不使用 Judge 模型评分
python scripts/5_train_grpo.py --no-judge
```

**记录指标：**
- 训练损失 (loss)
- 奖励分数 (reward)
- KL 散度 (kl_divergence)
- 格式奖励 (format_reward)
- 准确性奖励 (accuracy_reward)
- 学习率 (learning_rate)

**配置信息：**
- 是否使用 Judge 模型评分
- GRPO 超参数 (num_generations, temperature, kl_coef)
- 奖励权重 (format_reward_weight, accuracy_reward_weight)

### 3. T5 - 评测 (scripts/6_evaluate.py)

运行评测时会记录评测结果：

```bash
python scripts/6_evaluate.py
# 或评测 SFT 模型
python scripts/6_evaluate.py --eval_sft
# 或不使用 Judge 模型
python scripts/6_evaluate.py --no-judge
```

**记录指标：**
- 格式正确率 (format_accuracy)
- 答案正确率 (answer_accuracy)
- 格式正确数 (format_correct)
- 答案正确数 (answer_correct)
- 平均 Judge 分数 (avg_judge_score，仅在使用 Judge 时)

**按题目类型记录：**
- `{qtype}/format_accuracy`
- `{qtype}/answer_accuracy`
- `{qtype}/total`

**Artifacts：**
- 评测摘要报告 (eval_*.json)
- 详细结果文件 (eval_details_*.jsonl)

### 4. 消融实验 (scripts/8_ablation_study.py)

运行消融实验时会记录对比结果：

```bash
python scripts/8_ablation_study.py
```

**记录指标（按模型）：**
- `base_model/format_accuracy` - 基座模型格式正确率
- `base_model/answer_accuracy` - 基座模型答案正确率
- `sft_only/format_accuracy` - SFT 模型格式正确率
- `sft_only/answer_accuracy` - SFT 模型答案正确率
- `sft_grpo/format_accuracy` - GRPO 模型格式正确率
- `sft_grpo/answer_accuracy` - GRPO 模型答案正确率

**按题目类型记录：**
- `{model}/{qtype}/format_accuracy`
- `{model}/{qtype}/answer_accuracy`

**对比表格：**
- `ablation_comparison` - 包含所有模型对比的表格

**Artifacts：**
- 消融实验报告 (ablation_summary_*.json)

## 查看结果

1. 在终端运行训练/评测后，会输出 wandb 链接
2. 点击链接或访问 https://wandb.ai/
3. 在项目 "FTModel-Training" 中查看所有运行记录
4. 可以对比不同运行、查看指标曲线、下载 artifacts

## 禁用 wandb

如果不想使用 wandb，有两种方式：

### 方式1：修改配置文件（推荐）

在 `configs/config.py` 中设置：

```python
WANDB_CONFIG.enabled = False
```

### 方式2：临时禁用

设置环境变量：

```bash
# Linux/Mac
export WANDB_MODE=disabled

# Windows
set WANDB_MODE=disabled
```

## 高级用法

### 自定义项目名称

```python
# 在 configs/config.py 中
WANDB_CONFIG.project = "FinTech-LLM-Training"
WANDB_CONFIG.entity = "my-company"  # 如果有团队账号
```

### 查看具体某次运行

每次运行都会生成唯一的 run_name，格式为：
- SFT: `sft-train-20260120-143022`
- GRPO: `grpo-train-20260120-150045`
- 评测: `evaluation-grpo_lora-20260120-160030`
- 消融: `ablation-study-20260120_170015`

### 对比不同配置

在 wandb 网页界面中：
1. 选择多个运行
2. 点击 "Compare"
3. 查看指标对比、参数对比

### 下载评测报告

在 wandb Artifacts 中可以找到：
- 评测摘要 JSON 文件
- 评测详情 JSONL 文件
- 消融实验报告

## 常见问题

### Q: wandb 未安装怎么办？

A: 所有脚本都会自动检测 wandb 是否安装，如果未安装会打印警告并继续运行，不影响训练和评测。

### Q: 不想上传大文件怎么办？

A: 默认不会上传模型文件（`log_model=False`），只会上传：
- 训练指标（数值）
- 评测报告（JSON/JSONL，通常几 MB）

### Q: 如何在服务器上使用 wandb？

A: 使用 API Key 方式：

```bash
# 设置 API Key 环境变量
export WANDB_API_KEY="your-api-key-here"

# 然后正常运行训练
python scripts/4_train_sft.py
```

### Q: 可以使用本地 wandb 服务器吗？

A: 可以，设置环境变量：

```bash
export WANDB_BASE_URL="http://your-wandb-server:8080"
```

## 最佳实践

1. **命名规范**：在 `WandbConfig` 中使用有意义的项目名称
2. **标签使用**：每次运行都会自动添加标签（如 "sft", "grpo", "evaluation"）
3. **定期清理**：删除不需要的运行记录以节省空间
4. **团队协作**：设置 `entity` 参数以便团队成员共享结果
5. **实验对比**：使用 wandb Sweep 功能进行超参数搜索

## 示例工作流

```bash
# 1. 准备数据
python scripts/1_prepare_raw_data.py
python scripts/2_distill_data.py
python scripts/3_filter_data.py

# 2. SFT 训练（wandb 自动记录）
python scripts/4_train_sft.py
# 在 wandb 中查看 sft-train-* 运行

# 3. GRPO 训练（wandb 自动记录）
python scripts/5_train_grpo.py
# 在 wandb 中查看 grpo-train-* 运行

# 4. 评测（wandb 自动记录）
python scripts/6_evaluate.py
# 在 wandb 中查看 evaluation-* 运行

# 5. 消融实验（wandb 自动记录）
python scripts/8_ablation_study.py
# 在 wandb 中查看 ablation-study-* 运行

# 6. 在 wandb 网页中对比所有运行
```

## 参考资料

- [Wandb 官方文档](https://docs.wandb.ai/)
- [Transformers + Wandb 集成](https://docs.wandb.ai/guides/integrations/huggingface)
- [TRL + Wandb](https://huggingface.co/docs/trl/main/en/logging)
