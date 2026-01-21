# 服务器部署准备工作详解 (Preparation Guide)

本文档详细描述在租用 GPU 服务器（如 AutoDL, 阿里云 PAI 等）进行最终训练之前，需要在本地完成的准备工作，以确保租用服务器后能最省钱、最高效地完成任务。

## 📋 1. 代码安全性与依赖检查

在上传代码前，必须确保本地代码逻辑无误且依赖完整，避免在服务器上烧钱调试语法错误。

### 1.1 语法静态检查

在本地运行以下命令，确保所有脚本没有基础语法错误：

```powershell
# 运行 py_compile 检查所有脚本
python -m py_compile scripts/1_prepare_raw_data.py
python -m py_compile scripts/2_distill_data.py
python -m py_compile scripts/3_filter_data.py
python -m py_compile scripts/4_train_sft.py
python -m py_compile scripts/5_train_grpo.py
python -m py_compile scripts/6_evaluate.py
python -m py_compile scripts/7_deploy.py
python -m py_compile configs/config.py
```

### 1.2 依赖文件复核

确保 `requirements.txt` 包含了所有服务器需要的包（有些包本地可能已安装但未写入）。
检查 `requirements.txt` 内容是否包含：

- `trl`, `peft`, `transformers`, `torch`, `accelerate`, `bitsandbytes`
- `vllm` (用于部署，注意版本兼容性，推荐 0.6.3+)
- `wandb` (用于监控)
- `modelscope` (若使用魔搭社区下载模型)

## 🛠️ 2. 自动化执行脚本准备

为了最大化利用服务器时间，建议准备 Shell 脚本一键运行。

### 2.1 全流程启动脚本 (`scripts/run_pipeline.sh`)

在 `scripts/` 目录下创建此文件，用于服务器一键执行：

```bash
#!/bin/bash
set -e  #遇到错误立即停止

# 0. 环境准备
echo ">>> [Phase 0] 安装依赖..."
pip install -r requirements.txt

# 1. 准备数据 (通常这一步很快，服务器上也跑一遍以防万一)
echo ">>> [Phase 1] 准备原始数据..."
python scripts/1_prepare_raw_data.py

# 2. 数据蒸馏 (耗时较长)
echo ">>> [Phase 2] 开始数据蒸馏 (Teacher Model)..."
python scripts/2_distill_data.py

# 3. 数据过滤
echo ">>> [Phase 3] 数据过滤与评分..."
python scripts/3_filter_data.py

# 4. SFT 训练
echo ">>> [Phase 4] SFT 训练启动..."
python scripts/4_train_sft.py

# 5. GRPO 训练
echo ">>> [Phase 5] GRPO 强化学习启动..."
python scripts/5_train_grpo.py

# 6. 评测
echo ">>> [Phase 6] 最终模型评测..."
python scripts/6_evaluate.py

echo ">>> ✅ 全流程执行完毕！"
```

### 2.2 环境自检脚本 (`scripts/check_env.py`)

用于开机后第一时间确认显卡状态是否符合预期：

```python
import torch
import os

print("="*60)
print("环境自检报告")
print("="*60)

# CUDA 检查
if torch.cuda.is_available():
    print(f"✅ CUDA Available: True")
    print(f"✅ GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {prop.name}")
        print(f"   Memory: {prop.total_memory / 1024**3:.2f} GB")
else:
    print("❌ CUDA Not Available!")

# 库版本检查
import transformers
import peft
import trl
print(f"✅ Transformers: {transformers.__version__}")
print(f"✅ PEFT: {peft.__version__}")
print(f"✅ TRL: {trl.__version__}")
```

## 💾 3. 数据与版本控制策略

### 3.1 Git 忽略策略

服务器上只需要代码，**不需要**本地产生的临时大文件。确保 `.gitignore` 包含：

```gitignore
# 数据文件 (在服务器重新生成或单独上传)
data/
!data/raw/raw.jsonl  # 如果本地生成好了原始数据，可以保留这个
ckpts/               # 模型权重不上传
logs/                # 日志文件
__pycache__/
*.pyc
.env                 # 密钥文件绝对不能上传
```

### 3.2 数据上传策略

- **方案 A (推荐)**: 代码推送到 GitHub，数据生成脚本在服务器运行。
  - 优点：传输快，纯净。
- **方案 B**: 本地运行 `1_prepare_raw_data.py` 生成好 `raw.jsonl`，连同代码一起上传。
  - 优点：确保数据源完全一致，不用担心服务器下载数据集网络问题。

## ⚠️ 4. 显存风险预案 (OOM Plan B)

虽然目标是 5090 (32GB)，但 GRPO 阶段同时加载 Base Model (1.5B) + Judge Model (7B) 仍然有显存压力。如果遇到 OOM (Out of Memory)，请按以下清单调整 `configs/config.py`：

**调整优先级 (从上到下)：**

1.  **Reduce Batch Size**: `GRPOConfig.per_device_train_batch_size` 设为 `1`。
2.  **Gradient Accumulation**: 增加 `gradient_accumulation_steps` (如 4 -> 8) 以保持总批次大小不变。
3.  **Quantization**: 确保 Judge Model 和 Teacher Model 强制使用 `load_in_4bit=True`。
4.  **Reduce Context**: `max_new_tokens` 或 `response_length` 适当减小 (如 1024 -> 512)。
5.  **Sample Generations**: 减少 `num_sample_generations` (如 4 -> 2)。

## 🚀 5. 创建服务器后的 "黄金5分钟" 操作流

1.  **连接终端**: SSH 连接。
2.  **克隆代码**: `git clone https://github.com/yourusername/FTModel.git`
3.  **开启 Screen/Tmux**: 防止网络断开导致训练中断。
    - `screen -S train`
4.  **环境安装**: `pip install -r requirements.txt`
5.  **运行自检**: `python scripts/check_env.py`
6.  **启动 WanDB (可选)**: `wandb login`
7.  **一键起飞**: `bash scripts/run_pipeline.sh`

---

**准备完成标志：**

- [ ] 本地 `py_compile` 全通过。
- [ ] `requirements.txt` 已确认。
- [ ] 代码已 Push 到 GitHub 私有仓库。
- [ ] 拥有一份 `run_pipeline.sh`。
