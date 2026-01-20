"""
评测脚本
支持 vLLM 和 transformers 两种推理方式

功能：
1. 加载测试集生成回答
2. 计算格式正确率
3. 计算答案准确率（按任务类型）
4. 生成 JSON 评测报告

面试点：
- 评测指标设计：格式 + 准确率 + 按类型细分
- vLLM vs transformers：vLLM 高吞吐，transformers 兼容性好
"""
import argparse
import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import MODEL_CONFIG, SFT_CONFIG, GRPO_CONFIG, DATA_CONFIG


# ============ 评测指标计算 ============

def extract_answer(text: str) -> str:
    """提取 <answer> 中的内容"""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer)
        if boxed_match:
            return boxed_match.group(1).strip()
        return answer
    return ""


def check_format(response: str) -> bool:
    """检查格式是否正确"""
    has_think = "<think>" in response and "</think>" in response
    has_answer = "<answer>" in response and "</answer>" in response

    if has_think and has_answer:
        think_pos = response.find("<think>")
        answer_pos = response.find("<answer>")
        return think_pos < answer_pos
    return False


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


def check_answer(response: str, gold_answer: str, qtype: str) -> bool:
    """检查答案是否正确"""
    extracted = extract_answer(response)
    if not extracted:
        return False

    if qtype == "financial_calculation":
        return check_answer_math(extracted, gold_answer)
    else:
        return check_answer_qa(extracted, gold_answer)


# ============ 模型加载 ============

def load_model_transformers(model_path: str):
    """
    使用 transformers 加载模型
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"加载模型 (transformers): {model_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 判断是 LoRA adapter 还是完整模型
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        # LoRA adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # 完整模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("✓ 模型加载完成")

    return model, tokenizer


def load_model_vllm(model_path: str):
    """
    使用 vLLM 加载模型
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("❌ vLLM 未安装，请运行: pip install vllm")
        sys.exit(1)

    print(f"加载模型 (vLLM): {model_path}")

    # 判断是否是 LoRA adapter
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        # 需要先合并 LoRA
        print("⚠️  检测到 LoRA adapter，vLLM 需要合并后的模型")
        print("   请先运行: python scripts/7_deploy.py --action merge")
        print("   然后使用合并后的模型路径进行评测")
        sys.exit(1)

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
    )

    print("✓ vLLM 模型加载完成")

    return llm, None


# ============ 推理函数 ============

def generate_transformers(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> List[str]:
    """
    使用 transformers 生成
    """
    responses = []

    for prompt in tqdm(prompts, desc="生成中"):
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
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        responses.append(response)

    return responses


def generate_vllm(
    llm,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> List[str]:
    """
    使用 vLLM 生成
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
    )

    # 格式化 prompt
    formatted_prompts = []
    for prompt in prompts:
        # 简单格式化（vLLM 可能需要不同的格式）
        formatted_prompts.append(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")

    outputs = llm.generate(formatted_prompts, sampling_params)

    responses = [output.outputs[0].text for output in outputs]

    return responses


# ============ 评测主流程 ============

def evaluate(
    model_path: str,
    use_vllm: bool = False,
    output_dir: str = "reports",
):
    """
    执行评测
    """
    print("=" * 60)
    print("金融推理模型评测")
    print("=" * 60)

    # 加载测试数据
    print(f"\n加载测试数据: {DATA_CONFIG.test_data_path}")
    with open(DATA_CONFIG.test_data_path, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    print(f"✓ 加载 {len(test_data)} 条测试数据")

    # 加载模型
    if use_vllm:
        model, tokenizer = load_model_vllm(model_path)
    else:
        model, tokenizer = load_model_transformers(model_path)

    # 提取 prompts
    prompts = [item["prompt"] for item in test_data]

    # 生成回答
    print("\n生成回答...")
    if use_vllm:
        responses = generate_vllm(model, prompts)
    else:
        responses = generate_transformers(model, tokenizer, prompts)

    # 计算评测指标
    print("\n计算评测指标...")

    results = {
        "total": len(test_data),
        "format_correct": 0,
        "answer_correct": 0,
        "by_type": {},
        "details": [],
    }

    for item, response in zip(test_data, responses):
        prompt = item["prompt"]
        gold_answer = item.get("gold_answer", "")
        qtype = item.get("type", "unknown")

        # 格式检查
        format_ok = check_format(response)
        if format_ok:
            results["format_correct"] += 1

        # 答案检查
        answer_ok = check_answer(response, gold_answer, qtype) if gold_answer else False
        if answer_ok:
            results["answer_correct"] += 1

        # 按类型统计
        if qtype not in results["by_type"]:
            results["by_type"][qtype] = {
                "total": 0,
                "format_correct": 0,
                "answer_correct": 0,
            }
        results["by_type"][qtype]["total"] += 1
        if format_ok:
            results["by_type"][qtype]["format_correct"] += 1
        if answer_ok:
            results["by_type"][qtype]["answer_correct"] += 1

        # 保存详情
        results["details"].append({
            "id": item.get("id", ""),
            "prompt": prompt,
            "gold_answer": gold_answer,
            "type": qtype,
            "response": response,
            "format_ok": format_ok,
            "answer_ok": answer_ok,
        })

    # 计算比例
    results["format_accuracy"] = results["format_correct"] / results["total"]
    results["answer_accuracy"] = results["answer_correct"] / results["total"]

    for qtype in results["by_type"]:
        type_results = results["by_type"][qtype]
        type_results["format_accuracy"] = type_results["format_correct"] / type_results["total"]
        type_results["answer_accuracy"] = type_results["answer_correct"] / type_results["total"]

    # 打印报告
    print("\n" + "=" * 60)
    print("评测报告")
    print("=" * 60)
    print(f"测试样本数: {results['total']}")
    print(f"格式正确率: {results['format_accuracy']:.2%} ({results['format_correct']}/{results['total']})")
    print(f"答案正确率: {results['answer_accuracy']:.2%} ({results['answer_correct']}/{results['total']})")
    print("\n按类型统计:")
    for qtype, stats in results["by_type"].items():
        print(f"  {qtype}:")
        print(f"    - 样本数: {stats['total']}")
        print(f"    - 格式正确率: {stats['format_accuracy']:.2%}")
        print(f"    - 答案正确率: {stats['answer_accuracy']:.2%}")
    print("=" * 60)

    # 保存报告
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(model_path)

    # 保存摘要
    summary_path = os.path.join(output_dir, f"eval_{model_name}_{timestamp}.json")
    summary = {
        "model_path": model_path,
        "timestamp": timestamp,
        "total": results["total"],
        "format_accuracy": results["format_accuracy"],
        "answer_accuracy": results["answer_accuracy"],
        "by_type": results["by_type"],
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 保存详情
    details_path = os.path.join(output_dir, f"eval_details_{model_name}_{timestamp}.jsonl")
    with open(details_path, 'w', encoding='utf-8') as f:
        for item in results["details"]:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n✅ 评测完成！")
    print(f"   摘要报告: {summary_path}")
    print(f"   详细结果: {details_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="金融推理模型评测")
    parser.add_argument(
        "--model_path",
        type=str,
        default=GRPO_CONFIG.output_dir,
        help="模型路径（默认使用 GRPO 模型）"
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="使用 vLLM 进行推理（默认使用 transformers）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports",
        help="评测报告输出目录"
    )
    parser.add_argument(
        "--eval_sft",
        action="store_true",
        help="评测 SFT 模型而不是 GRPO 模型"
    )

    args = parser.parse_args()

    # 选择模型
    if args.eval_sft:
        model_path = SFT_CONFIG.output_dir
    else:
        model_path = args.model_path

    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        sys.exit(1)

    # 检查测试数据是否存在
    if not os.path.exists(DATA_CONFIG.test_data_path):
        print(f"❌ 测试数据不存在: {DATA_CONFIG.test_data_path}")
        print("   请先运行数据准备流程")
        sys.exit(1)

    evaluate(
        model_path=model_path,
        use_vllm=args.use_vllm,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
