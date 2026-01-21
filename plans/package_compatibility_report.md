# RTX 5090 包兼容性检查报告

生成时间: 2026-01-21

## 更新后的依赖包版本

### 核心框架
- torch: 2.1.0 → **2.7.0** (支持 CUDA 12.x，RTX 5090 优化)
- transformers: 4.45.0 → **4.48.0**
- trl: 0.12.0 → **0.14.0**
- peft: 0.13.0 → **0.15.0**

### 量化和加速
- bitsandbytes: 0.44.0 → **0.45.1**
- accelerate: 0.34.0 → **1.2.1**

### 部署
- vllm: 0.6.3 → **0.6.6.post1**

### 数据处理
- datasets: 2.20.0 → **3.3.1** ⚠️ **主要版本升级**

### 其他工具
- pandas: 2.1.0 → **2.2.3**
- openai: 1.35.0 → **1.61.2**
- requests: 2.31.0 → **2.32.3**
- tqdm: 4.66.0 → **4.67.1**
- wandb: (未指定) → **>=0.19.1**
- sentencepiece: 0.2.0 (保持不变)
- protobuf: 4.25.0 → **5.29.3**

---

## 脚本导入分析

### 1. scripts/1_prepare_raw_data.py
**导入的包:**
- `datasets` (load_dataset)
- `json`, `os`, `sys`, `pathlib`

**潜在问题:**
- ⚠️ `datasets` 从 2.20.0 升级到 3.3.1 (主要版本升级)
- `load_dataset` API 在 3.x 中可能有变化

**建议:**
- 测试 `load_dataset` 函数是否正常工作
- 检查 split 参数语法是否变化

---

### 2. scripts/2_distill_data.py
**导入的包:**
- `transformers` (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
- `torch`
- `json`, `os`, `sys`, `pathlib`, `tqdm`

**潜在问题:**
- ✓ 所有导入应该向后兼容
- BitsAndBytesConfig 在新版 transformers 中保持稳定

---

### 3. scripts/3_filter_data.py
**导入的包:**
- `transformers` (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
- `torch`
- `json`, `os`, `sys`, `re`, `pathlib`, `tqdm`

**潜在问题:**
- ✓ 所有导入应该向后兼容

---

### 4. scripts/4_train_sft.py ⚠️ **关键检查**
**导入的包:**
- `torch`
- `datasets` (Dataset)
- `transformers` (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments)
- `peft` (LoraConfig, get_peft_model, prepare_model_for_kbit_training)
- `trl` (SFTTrainer, SFTConfig)

**潜在问题:**
- ⚠️ **trl 0.14.0**: `SFTConfig` 和 `SFTTrainer` API 可能有变化
  - 第 205-226 行: `SFTConfig` 参数是否仍然有效
  - 第 229-234 行: `SFTTrainer` 初始化参数
- ⚠️ **datasets 3.3.1**: `Dataset.from_list()` 是否保持兼容

**需要验证的代码位置:**
```python
# 第 205 行
training_args = SFTConfig(
    output_dir=SFT_CONFIG.output_dir,
    # ... 其他参数
    dataset_text_field="text",  # 检查此参数是否仍然有效
)

# 第 229 行
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,  # 注意：可能需要改为 processing_class
)
```

---

### 5. scripts/5_train_grpo.py ⚠️ **关键检查**
**导入的包:**
- `torch`
- `datasets` (Dataset)
- `transformers` (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
- `peft` (PeftModel)
- `trl` (GRPOConfig, GRPOTrainer)

**潜在问题:**
- ⚠️ **trl 0.14.0**: `GRPOConfig` 和 `GRPOTrainer` API 可能有变化
  - 第 525-543 行: `GRPOConfig` 参数是否仍然有效
  - 第 552-558 行: `GRPOTrainer` 初始化参数
  - **第 556 行**: `processing_class=tokenizer` 参数名称是否正确

**需要验证的代码位置:**
```python
# 第 525 行
grpo_config = GRPOConfig(
    output_dir=GRPO_CONFIG.output_dir,
    num_generations=GRPO_CONFIG.num_sample_generations,  # 检查参数名
    max_completion_length=GRPO_CONFIG.response_length,  # 检查参数名
    # ...
)

# 第 552 行
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=formatted_dataset,
    processing_class=tokenizer,  # ⚠️ 新版本可能是 tokenizer 而非 processing_class
    reward_funcs=reward_fn,
)
```

---

### 6. scripts/6_evaluate.py
**导入的包:**
- `torch`
- `transformers` (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
- `peft` (PeftModel)
- `vllm` (LLM, SamplingParams)
- `tqdm`, `json`, `os`, `sys`, `re`, `pathlib`, `datetime`

**潜在问题:**
- ⚠️ **vllm 0.6.6.post1**: API 可能有小的变化
- 第 325 行和第 397 行使用 vllm，需要测试

---

### 7. scripts/7_deploy.py
**导入的包:**
- `torch`
- `transformers` (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
- `peft` (PeftModel)
- `vllm` (需要时导入)
- `subprocess`, `requests`, `time`

**潜在问题:**
- ⚠️ **vllm 0.6.6.post1**: 启动命令可能有变化
- 第 124-132 行: vLLM 启动参数需要验证

---

## 关键兼容性问题总结

### 🔴 **高优先级** - 需要立即检查

1. **trl SFTTrainer 参数变化 (scripts/4_train_sft.py)**
   - 检查 `tokenizer` 参数是否应该改为 `processing_class`
   - 检查 `SFTConfig` 的 `dataset_text_field` 参数是否仍然有效

2. **trl GRPOTrainer 参数变化 (scripts/5_train_grpo.py)**
   - 验证 `processing_class=tokenizer` 是否正确
   - 验证 `GRPOConfig` 的参数名称 (`num_generations`, `max_completion_length`)

3. **datasets 主要版本升级 (scripts/1_prepare_raw_data.py)**
   - 测试 `load_dataset` 的 split 语法
   - 测试 `Dataset.from_list()` 是否正常工作

### 🟡 **中优先级** - 建议测试

4. **vllm API 变化 (scripts/6_evaluate.py, scripts/7_deploy.py)**
   - 测试 vLLM 的 LLM 初始化参数
   - 测试 vLLM 启动服务的命令行参数

5. **transformers BitsAndBytesConfig (所有训练脚本)**
   - 验证量化配置参数是否兼容

### ✅ **低优先级** - 应该兼容

6. **其他包**: pandas, openai, requests, tqdm, wandb, protobuf
   - 这些包的升级应该向后兼容

---

## 推荐的验证步骤

### 第一步：安装新版本依赖
```bash
pip install -r requirements.txt
```

### 第二步：运行简单的导入测试
```bash
python -c "from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer; print('TRL import OK')"
python -c "from datasets import Dataset, load_dataset; print('Datasets import OK')"
python -c "from vllm import LLM, SamplingParams; print('vLLM import OK')"
```

### 第三步：检查 TRL 参数兼容性
创建测试脚本检查 `SFTTrainer` 和 `GRPOTrainer` 的参数签名：
```python
import inspect
from trl import SFTTrainer, GRPOTrainer

# 检查 SFTTrainer 参数
print("SFTTrainer parameters:")
print(inspect.signature(SFTTrainer.__init__))

# 检查 GRPOTrainer 参数
print("\nGRPOTrainer parameters:")
print(inspect.signature(GRPOTrainer.__init__))
```

### 第四步：运行单元测试
在运行完整训练前，测试每个脚本的核心功能。

---

## 可能需要的代码修改

### 如果 SFTTrainer 参数变化：
```python
# 旧版本 (可能)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,  # 旧参数名
)

# 新版本 (可能需要)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,  # 新参数名
)
```

### 如果 GRPOTrainer 参数变化：
```python
# 当前代码
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=formatted_dataset,
    processing_class=tokenizer,  # 检查这个参数名
    reward_funcs=reward_fn,
)
```

---

## 结论

总体来说，大部分依赖包的升级应该是向后兼容的。主要需要关注的是：

1. **datasets 3.x 的主要版本升级** - 需要测试
2. **trl 0.14.0 的 API 变化** - 需要验证参数名称
3. **vllm 的小版本升级** - 需要测试服务启动

建议在完整训练前：
1. 先进行导入测试
2. 检查 API 参数签名
3. 运行小规模测试确保兼容性
4. 再进行完整训练

如果遇到兼容性问题，可以考虑：
- 降级某些包到中间版本
- 修改代码以适配新 API
- 查看对应包的 changelog 和 migration guide
