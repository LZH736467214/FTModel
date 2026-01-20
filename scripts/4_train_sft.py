"""
SFT 训练脚本
核心技术点：
1. 4bit 量化加载 (BitsAndBytesConfig)
2. LoRA 微调 (peft)
3. Qwen chat template 数据格式化
4. TRL SFTTrainer

面试点：
- 为什么要 4bit 量化？节省显存，让 3B 模型在 8GB 显卡上跑起来
- 为什么用 LoRA？参数高效微调，只训练约 0.1% 的参数
- 为什么用 SFTTrainer？TRL 库封装好的监督微调训练器
"""
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import MODEL_CONFIG, LORA_CONFIG, SFT_CONFIG, DATA_CONFIG


def load_model_and_tokenizer():
    """
    加载 4bit 量化模型和分词器
    面试点：BitsAndBytesConfig 的各项配置含义
    """
    print("=" * 60)
    print("加载模型和分词器")
    print("=" * 60)

    # 4bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                    # 使用 4bit 量化
        bnb_4bit_quant_type="nf4",            # 使用 NormalFloat4 量化类型
        bnb_4bit_compute_dtype=torch.bfloat16, # 计算时使用 bf16
        bnb_4bit_use_double_quant=True,       # 双重量化，进一步节省显存
    )

    print(f"加载基座模型: {MODEL_CONFIG.base_model}")

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # 为 kbit 训练准备模型
    model = prepare_model_for_kbit_training(model)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIG.base_model,
        trust_remote_code=True,
        padding_side="right",  # SFT 需要右填充
    )

    # 设置 pad_token（Qwen 可能没有设置）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"✓ 模型加载完成，参数量: {model.num_parameters() / 1e9:.2f}B")
    print(f"✓ 词表大小: {len(tokenizer)}")

    return model, tokenizer


def apply_lora(model):
    """
    应用 LoRA 配置
    面试点：LoRA 的原理和超参选择
    """
    print("\n" + "=" * 60)
    print("应用 LoRA")
    print("=" * 60)

    lora_config = LoraConfig(
        r=LORA_CONFIG.r,                        # LoRA 秩
        lora_alpha=LORA_CONFIG.lora_alpha,      # LoRA alpha（缩放系数）
        lora_dropout=LORA_CONFIG.lora_dropout,  # Dropout 比例
        target_modules=LORA_CONFIG.target_modules,  # 目标模块
        bias="none",                            # 不训练 bias
        task_type="CAUSAL_LM",                  # 因果语言模型任务
    )

    model = get_peft_model(model, lora_config)

    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 可训练参数: {trainable_params:,} ({trainable_params / all_params * 100:.2f}%)")
    print(f"✓ 总参数: {all_params:,}")

    return model


def load_sft_data(tokenizer):
    """
    加载并格式化 SFT 数据
    面试点：如何使用 chat template
    """
    print("\n" + "=" * 60)
    print("加载 SFT 数据")
    print("=" * 60)

    # 加载数据
    print(f"读取数据: {DATA_CONFIG.sft_data_path}")
    with open(DATA_CONFIG.sft_data_path, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]

    print(f"✓ 加载 {len(raw_data)} 条训练数据")

    # 格式化为 Qwen chat 格式
    def format_example(example):
        """使用 Qwen 的 chat template 格式化"""
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]}
        ]
        # 使用 tokenizer 的 chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}

    # 转换数据
    formatted_data = [format_example(item) for item in raw_data]

    # 创建 Dataset
    dataset = Dataset.from_list(formatted_data)

    # 展示样例
    print("\n样例数据:")
    print("-" * 40)
    print(formatted_data[0]["text"][:500] + "...")
    print("-" * 40)

    return dataset


def train_sft(model, tokenizer, dataset):
    """
    执行 SFT 训练
    面试点：SFTTrainer 的配置和使用
    """
    print("\n" + "=" * 60)
    print("开始 SFT 训练")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(SFT_CONFIG.output_dir, exist_ok=True)

    # 训练配置
    training_args = SFTConfig(
        output_dir=SFT_CONFIG.output_dir,
        num_train_epochs=SFT_CONFIG.num_train_epochs,
        per_device_train_batch_size=SFT_CONFIG.per_device_train_batch_size,
        gradient_accumulation_steps=SFT_CONFIG.gradient_accumulation_steps,
        learning_rate=SFT_CONFIG.learning_rate,
        lr_scheduler_type=SFT_CONFIG.lr_scheduler_type,
        warmup_ratio=SFT_CONFIG.warmup_ratio,
        logging_steps=SFT_CONFIG.logging_steps,
        save_steps=SFT_CONFIG.save_steps,
        save_total_limit=SFT_CONFIG.save_total_limit,
        bf16=SFT_CONFIG.bf16,
        optim=SFT_CONFIG.optim,
        max_seq_length=MODEL_CONFIG.model_max_length,
        dataset_text_field="text",  # 指定文本字段
        report_to="none",  # 不使用 wandb（可改为 "wandb"）
        gradient_checkpointing=True,  # 节省显存
        # 其他优化
        remove_unused_columns=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )

    # 创建训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # 打印训练配置
    print(f"训练配置:")
    print(f"  - Epochs: {SFT_CONFIG.num_train_epochs}")
    print(f"  - Batch size: {SFT_CONFIG.per_device_train_batch_size}")
    print(f"  - Gradient accumulation: {SFT_CONFIG.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {SFT_CONFIG.per_device_train_batch_size * SFT_CONFIG.gradient_accumulation_steps}")
    print(f"  - Learning rate: {SFT_CONFIG.learning_rate}")
    print(f"  - Max sequence length: {MODEL_CONFIG.model_max_length}")

    # 开始训练
    print("\n开始训练...")
    trainer.train()

    # 保存最终模型
    print("\n保存模型...")
    trainer.save_model(SFT_CONFIG.output_dir)
    tokenizer.save_pretrained(SFT_CONFIG.output_dir)

    print(f"\n✅ SFT 训练完成！")
    print(f"   模型已保存至: {SFT_CONFIG.output_dir}")

    return trainer


def test_generation(model, tokenizer):
    """
    测试模型生成
    面试点：验证训练效果
    """
    print("\n" + "=" * 60)
    print("测试生成")
    print("=" * 60)

    test_prompts = [
        "某公司2023年营收1000万元，同比增长25%，请计算2022年营收。",
        "什么是资产负债率？"
    ]

    model.eval()
    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        print(f"\n问题: {prompt}")
        print(f"回答: {response[:300]}...")
        print("-" * 40)


def main():
    """主函数"""
    print("=" * 60)
    print("金融推理模型 SFT 训练")
    print("=" * 60)

    # 1. 加载模型
    model, tokenizer = load_model_and_tokenizer()

    # 2. 应用 LoRA
    model = apply_lora(model)

    # 3. 加载数据
    dataset = load_sft_data(tokenizer)

    # 4. 训练
    trainer = train_sft(model, tokenizer, dataset)

    # 5. 测试生成
    test_generation(model, tokenizer)

    print("\n" + "=" * 60)
    print("✅ SFT 训练全部完成！")
    print("=" * 60)
    print(f"\n下一步：运行 python scripts/5_train_grpo.py")


if __name__ == "__main__":
    main()
