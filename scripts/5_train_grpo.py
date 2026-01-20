"""
GRPO 训练脚本
核心技术点：
1. 基于 SFT 模型继续训练
2. 格式奖励 + 准确性奖励
3. TRL GRPOTrainer
4. KL 散度约束

面试点：
- GRPO vs PPO：GRPO 用组内相对表现计算优势，不需要 value network
- 为什么双奖励：格式保证可解释性，准确性保证业务价值
- KL 约束的作用：防止模型偏离 SFT 初始化太远，避免 reward hacking
"""
import json
import os
import sys
import re
from pathlib import Path
from typing import List

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import MODEL_CONFIG, LORA_CONFIG, SFT_CONFIG, GRPO_CONFIG, DATA_CONFIG


# ============ 奖励函数 ============

def compute_format_reward(responses: List[str]) -> List[float]:
    """
    格式奖励函数
    面试点：检查 <think> 和 <answer> 标签的完整性和顺序

    奖励标准：
    - 完整标签且顺序正确：1.0
    - 有标签但不完整或顺序错误：0.5
    - 无标签：0.0
    """
    rewards = []
    for response in responses:
        # 检查标签存在性
        has_think_open = "<think>" in response
        has_think_close = "</think>" in response
        has_answer_open = "<answer>" in response
        has_answer_close = "</answer>" in response

        # 完整性检查
        think_complete = has_think_open and has_think_close
        answer_complete = has_answer_open and has_answer_close

        if think_complete and answer_complete:
            # 检查顺序：<think> 应该在 <answer> 之前
            think_pos = response.find("<think>")
            answer_pos = response.find("<answer>")
            if think_pos < answer_pos:
                rewards.append(1.0)
            else:
                rewards.append(0.5)  # 顺序错误
        elif think_complete or answer_complete:
            rewards.append(0.5)  # 部分完整
        else:
            rewards.append(0.0)  # 完全缺失

    return rewards


def extract_answer_from_response(response: str) -> str:
    """从响应中提取答案"""
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        # 处理 \boxed{} 格式
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer)
        if boxed_match:
            return boxed_match.group(1).strip()
        return answer
    return ""


def check_answer_math(extracted: str, gold: str) -> bool:
    """数学题答案检查"""
    try:
        extracted_num = float(re.sub(r'[^\d.-]', '', str(extracted)))
        gold_num = float(re.sub(r'[^\d.-]', '', str(gold)))
        if abs(gold_num) > 1:
            return abs(extracted_num - gold_num) / abs(gold_num) < 0.05
        else:
            return abs(extracted_num - gold_num) < 0.05
    except:
        return False


def check_answer_qa(extracted: str, gold: str) -> bool:
    """QA 题答案检查"""
    gold_lower = str(gold).lower()
    extracted_lower = str(extracted).lower()

    if gold_lower in extracted_lower:
        return True

    gold_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', gold_lower))
    extracted_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', extracted_lower))

    if not gold_keywords:
        return False

    overlap_ratio = len(gold_keywords & extracted_keywords) / len(gold_keywords)
    return overlap_ratio > 0.5


def compute_accuracy_reward(
    responses: List[str],
    gold_answers: List[str],
    types: List[str]
) -> List[float]:
    """
    准确性奖励函数
    面试点：根据题目类型使用不同的评判标准

    奖励标准：
    - 完全正确：1.0
    - 部分正确（QA 题）：0.5
    - 错误：0.0
    """
    rewards = []
    for response, gold, qtype in zip(responses, gold_answers, types):
        extracted = extract_answer_from_response(response)

        if not extracted:
            rewards.append(0.0)
            continue

        if qtype == "financial_calculation":
            # 数值题：严格匹配
            if check_answer_math(extracted, gold):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            # QA 题：关键词匹配
            if check_answer_qa(extracted, gold):
                rewards.append(1.0)
            elif len(set(gold.split()) & set(extracted.split())) > 0:
                rewards.append(0.3)  # 部分相关
            else:
                rewards.append(0.0)

    return rewards


def combined_reward_function(
    completions: List[str],
    gold_answers: List[str],
    types: List[str],
) -> List[float]:
    """
    组合奖励函数
    面试点：Fin-R1 的奖励设计

    组合方式：0.3 * 格式奖励 + 0.7 * 准确性奖励
    """
    format_rewards = compute_format_reward(completions)
    accuracy_rewards = compute_accuracy_reward(completions, gold_answers, types)

    combined = [
        GRPO_CONFIG.format_reward_weight * f + GRPO_CONFIG.accuracy_reward_weight * a
        for f, a in zip(format_rewards, accuracy_rewards)
    ]

    return combined


# ============ 模型加载 ============

def load_sft_model():
    """
    加载 SFT 训练后的模型
    面试点：如何加载 LoRA adapter
    """
    print("=" * 60)
    print("加载 SFT 模型")
    print("=" * 60)

    # 4bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"加载基座模型: {MODEL_CONFIG.base_model}")

    # 加载基座模型
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # 加载 LoRA adapter
    print(f"加载 SFT LoRA: {SFT_CONFIG.output_dir}")
    model = PeftModel.from_pretrained(
        base_model,
        SFT_CONFIG.output_dir,
        is_trainable=True,  # GRPO 需要继续训练
    )

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        SFT_CONFIG.output_dir,
        trust_remote_code=True,
        padding_side="left",  # 生成时需要左填充
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"✓ 模型加载完成")

    return model, tokenizer


def load_rl_data():
    """
    加载 RL 训练数据
    面试点：GRPO 数据格式要求
    """
    print("\n" + "=" * 60)
    print("加载 RL 数据")
    print("=" * 60)

    print(f"读取数据: {DATA_CONFIG.rl_data_path}")
    with open(DATA_CONFIG.rl_data_path, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]

    print(f"✓ 加载 {len(raw_data)} 条训练数据")

    # 转换格式
    formatted_data = []
    for item in raw_data:
        formatted_data.append({
            "prompt": item["prompt"],
            "gold_answer": item["gold_answer"],
            "type": item["type"],
        })

    dataset = Dataset.from_list(formatted_data)

    return dataset


# ============ 自定义 GRPO Trainer ============

class FinanceGRPOTrainer(GRPOTrainer):
    """
    自定义 GRPO Trainer
    重写奖励计算方法以支持金融领域的双重奖励
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_rewards(self, prompts, completions, **kwargs):
        """
        计算奖励
        这里我们使用自定义的组合奖励函数
        """
        # 从 batch 中提取 gold_answer 和 type
        gold_answers = kwargs.get("gold_answers", [""] * len(completions))
        types = kwargs.get("types", ["business_reasoning"] * len(completions))

        rewards = combined_reward_function(completions, gold_answers, types)

        return torch.tensor(rewards, device=self.accelerator.device)


def create_reward_function(dataset):
    """
    创建奖励函数闭包
    用于在 GRPOTrainer 中计算奖励
    """
    # 创建 prompt 到 gold_answer 和 type 的映射
    prompt_to_meta = {}
    for item in dataset:
        prompt_to_meta[item["prompt"]] = {
            "gold_answer": item["gold_answer"],
            "type": item["type"],
        }

    def reward_fn(completions, prompts=None, **kwargs):
        """
        奖励函数
        TRL GRPOTrainer 会调用这个函数
        """
        # 获取元数据
        gold_answers = []
        types = []

        if prompts is not None:
            for prompt in prompts:
                meta = prompt_to_meta.get(prompt, {"gold_answer": "", "type": "business_reasoning"})
                gold_answers.append(meta["gold_answer"])
                types.append(meta["type"])
        else:
            gold_answers = [""] * len(completions)
            types = ["business_reasoning"] * len(completions)

        # 计算组合奖励
        rewards = combined_reward_function(completions, gold_answers, types)

        return rewards

    return reward_fn


def train_grpo(model, tokenizer, dataset):
    """
    执行 GRPO 训练
    面试点：GRPOTrainer 的配置
    """
    print("\n" + "=" * 60)
    print("开始 GRPO 训练")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(GRPO_CONFIG.output_dir, exist_ok=True)

    # 创建奖励函数
    reward_fn = create_reward_function(dataset)

    # GRPO 配置
    grpo_config = GRPOConfig(
        output_dir=GRPO_CONFIG.output_dir,
        num_train_epochs=GRPO_CONFIG.num_train_epochs,
        per_device_train_batch_size=GRPO_CONFIG.per_device_train_batch_size,
        gradient_accumulation_steps=GRPO_CONFIG.gradient_accumulation_steps,
        learning_rate=GRPO_CONFIG.learning_rate,
        # GRPO 特定参数
        num_generations=GRPO_CONFIG.num_sample_generations,  # 每个 prompt 生成多少个样本
        max_completion_length=GRPO_CONFIG.response_length,
        temperature=GRPO_CONFIG.temperature,
        # 其他参数
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,  # 保留额外字段
    )

    # 准备数据集格式（GRPOTrainer 需要 "prompt" 字段）
    def format_prompt(example):
        return {"prompt": example["prompt"]}

    formatted_dataset = dataset.map(format_prompt)

    # 创建训练器
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,  # 奖励函数
    )

    # 打印配置
    print(f"GRPO 训练配置:")
    print(f"  - Epochs: {GRPO_CONFIG.num_train_epochs}")
    print(f"  - Batch size: {GRPO_CONFIG.per_device_train_batch_size}")
    print(f"  - Num generations per prompt: {GRPO_CONFIG.num_sample_generations}")
    print(f"  - Learning rate: {GRPO_CONFIG.learning_rate}")
    print(f"  - Temperature: {GRPO_CONFIG.temperature}")
    print(f"  - Format reward weight: {GRPO_CONFIG.format_reward_weight}")
    print(f"  - Accuracy reward weight: {GRPO_CONFIG.accuracy_reward_weight}")

    # 开始训练
    print("\n开始训练...")
    trainer.train()

    # 保存模型
    print("\n保存模型...")
    trainer.save_model(GRPO_CONFIG.output_dir)
    tokenizer.save_pretrained(GRPO_CONFIG.output_dir)

    print(f"\n✅ GRPO 训练完成！")
    print(f"   模型已保存至: {GRPO_CONFIG.output_dir}")

    return trainer


def test_generation(model, tokenizer):
    """
    测试 GRPO 后的模型生成
    """
    print("\n" + "=" * 60)
    print("测试生成")
    print("=" * 60)

    test_prompts = [
        "某公司2023年营收1000万元，同比增长25%，请计算2022年营收。",
        "什么是流动比率？正常范围是多少？",
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
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # 计算奖励
        format_reward = compute_format_reward([response])[0]

        print(f"\n问题: {prompt}")
        print(f"回答: {response[:400]}...")
        print(f"格式奖励: {format_reward:.2f}")
        print("-" * 40)


def main():
    """主函数"""
    print("=" * 60)
    print("金融推理模型 GRPO 训练")
    print("=" * 60)

    # 检查 SFT 模型是否存在
    if not os.path.exists(SFT_CONFIG.output_dir):
        print(f"❌ 错误：SFT 模型不存在: {SFT_CONFIG.output_dir}")
        print("   请先运行 python scripts/4_train_sft.py")
        sys.exit(1)

    # 1. 加载模型
    model, tokenizer = load_sft_model()

    # 2. 加载数据
    dataset = load_rl_data()

    # 3. GRPO 训练
    trainer = train_grpo(model, tokenizer, dataset)

    # 4. 测试生成
    test_generation(model, tokenizer)

    print("\n" + "=" * 60)
    print("✅ GRPO 训练全部完成！")
    print("=" * 60)
    print(f"\n下一步：运行 python scripts/6_evaluate.py 进行评测")


if __name__ == "__main__":
    main()
