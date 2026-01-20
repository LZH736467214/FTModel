"""
消融实验脚本
对比不同训练策略的效果

实验设置：
1. Base model only（无训练）
2. SFT only（仅 SFT）
3. SFT + GRPO（完整流程）
4. 不同 reward 权重对比（可选）

面试点：
- 消融实验的重要性：证明每个组件的价值
- 如何设计对比实验：控制变量法
- 如何解读实验结果：关注相对提升
"""
import argparse
import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import MODEL_CONFIG, SFT_CONFIG, GRPO_CONFIG, DATA_CONFIG, WANDB_CONFIG


# ============ 评测函数（复用自 6_evaluate.py）============

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
        return response.find("<think>") < response.find("<answer>")
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
    return len(gold_keywords & extracted_keywords) / len(gold_keywords) > 0.5


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

def load_model(model_path: str, is_lora: bool = False):
    """
    加载模型
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"加载模型: {model_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if is_lora or os.path.exists(os.path.join(model_path, "adapter_config.json")):
        # LoRA adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        # 完整模型或基座模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
) -> List[str]:
    """
    生成回答
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
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        responses.append(response)

    return responses


def evaluate_model(
    model,
    tokenizer,
    test_data: List[Dict],
) -> Dict[str, Any]:
    """
    评估模型
    """
    prompts = [item["prompt"] for item in test_data]
    responses = generate_responses(model, tokenizer, prompts)

    results = {
        "total": len(test_data),
        "format_correct": 0,
        "answer_correct": 0,
        "by_type": {},
    }

    for item, response in zip(test_data, responses):
        gold_answer = item.get("gold_answer", "")
        qtype = item.get("type", "unknown")

        format_ok = check_format(response)
        if format_ok:
            results["format_correct"] += 1

        answer_ok = check_answer(response, gold_answer, qtype) if gold_answer else False
        if answer_ok:
            results["answer_correct"] += 1

        if qtype not in results["by_type"]:
            results["by_type"][qtype] = {"total": 0, "format_correct": 0, "answer_correct": 0}
        results["by_type"][qtype]["total"] += 1
        if format_ok:
            results["by_type"][qtype]["format_correct"] += 1
        if answer_ok:
            results["by_type"][qtype]["answer_correct"] += 1

    results["format_accuracy"] = results["format_correct"] / results["total"]
    results["answer_accuracy"] = results["answer_correct"] / results["total"]

    for qtype in results["by_type"]:
        type_results = results["by_type"][qtype]
        type_results["format_accuracy"] = type_results["format_correct"] / type_results["total"]
        type_results["answer_accuracy"] = type_results["answer_correct"] / type_results["total"]

    return results


# ============ 消融实验 ============

def run_ablation_study(
    test_data: List[Dict],
    output_dir: str = "reports",
):
    """
    执行消融实验
    """
    print("=" * 60)
    print("消融实验")
    print("=" * 60)

    ablation_results = {}

    # 1. 评估基座模型
    print("\n" + "=" * 60)
    print("实验 1: 基座模型 (Base Model)")
    print("=" * 60)

    try:
        model, tokenizer = load_model(MODEL_CONFIG.base_model, is_lora=False)
        results = evaluate_model(model, tokenizer, test_data)
        ablation_results["base_model"] = {
            "name": "Base Model (Qwen2.5-1.5B-Instruct)",
            "format_accuracy": results["format_accuracy"],
            "answer_accuracy": results["answer_accuracy"],
            "by_type": results["by_type"],
        }
        print(f"格式正确率: {results['format_accuracy']:.2%}")
        print(f"答案正确率: {results['answer_accuracy']:.2%}")

        # 释放显存
        del model, tokenizer
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"⚠️  基座模型评估失败: {e}")
        ablation_results["base_model"] = {"error": str(e)}

    # 2. 评估 SFT 模型
    print("\n" + "=" * 60)
    print("实验 2: SFT 模型")
    print("=" * 60)

    if os.path.exists(SFT_CONFIG.output_dir):
        try:
            model, tokenizer = load_model(SFT_CONFIG.output_dir, is_lora=True)
            results = evaluate_model(model, tokenizer, test_data)
            ablation_results["sft_only"] = {
                "name": "SFT Only",
                "format_accuracy": results["format_accuracy"],
                "answer_accuracy": results["answer_accuracy"],
                "by_type": results["by_type"],
            }
            print(f"格式正确率: {results['format_accuracy']:.2%}")
            print(f"答案正确率: {results['answer_accuracy']:.2%}")

            del model, tokenizer
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️  SFT 模型评估失败: {e}")
            ablation_results["sft_only"] = {"error": str(e)}
    else:
        print(f"⚠️  SFT 模型不存在: {SFT_CONFIG.output_dir}")
        ablation_results["sft_only"] = {"error": "Model not found"}

    # 3. 评估 SFT + GRPO 模型
    print("\n" + "=" * 60)
    print("实验 3: SFT + GRPO 模型")
    print("=" * 60)

    if os.path.exists(GRPO_CONFIG.output_dir):
        try:
            model, tokenizer = load_model(GRPO_CONFIG.output_dir, is_lora=True)
            results = evaluate_model(model, tokenizer, test_data)
            ablation_results["sft_grpo"] = {
                "name": "SFT + GRPO",
                "format_accuracy": results["format_accuracy"],
                "answer_accuracy": results["answer_accuracy"],
                "by_type": results["by_type"],
            }
            print(f"格式正确率: {results['format_accuracy']:.2%}")
            print(f"答案正确率: {results['answer_accuracy']:.2%}")

            del model, tokenizer
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️  GRPO 模型评估失败: {e}")
            ablation_results["sft_grpo"] = {"error": str(e)}
    else:
        print(f"⚠️  GRPO 模型不存在: {GRPO_CONFIG.output_dir}")
        ablation_results["sft_grpo"] = {"error": "Model not found"}

    return ablation_results


def generate_report(
    ablation_results: Dict[str, Any],
    output_dir: str = "reports",
):
    """
    生成消融实验报告
    """
    print("\n" + "=" * 60)
    print("消融实验报告")
    print("=" * 60)

    # 打印对比表格
    print("\n模型对比:")
    print("-" * 70)
    print(f"{'模型':<25s} {'格式正确率':>15s} {'答案正确率':>15s} {'提升':>10s}")
    print("-" * 70)

    base_answer_acc = 0
    for key, result in ablation_results.items():
        if "error" in result:
            print(f"{result.get('name', key):<25s} {'N/A':>15s} {'N/A':>15s} {'N/A':>10s}")
        else:
            format_acc = result["format_accuracy"]
            answer_acc = result["answer_accuracy"]

            if key == "base_model":
                base_answer_acc = answer_acc
                improvement = "-"
            else:
                if base_answer_acc > 0:
                    improvement = f"+{(answer_acc - base_answer_acc) / base_answer_acc * 100:.1f}%"
                else:
                    improvement = "N/A"

            print(f"{result['name']:<25s} {format_acc:>14.1%} {answer_acc:>14.1%} {improvement:>10s}")

    print("-" * 70)

    # 按类型对比
    print("\n按题目类型对比答案正确率:")
    print("-" * 70)

    # 收集所有类型
    all_types = set()
    for result in ablation_results.values():
        if "by_type" in result:
            all_types.update(result["by_type"].keys())

    for qtype in sorted(all_types):
        print(f"\n{qtype}:")
        for key, result in ablation_results.items():
            if "error" not in result and qtype in result.get("by_type", {}):
                type_result = result["by_type"][qtype]
                print(f"  {result['name']:<20s}: {type_result['answer_accuracy']:.1%}")

    # 保存报告
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = os.path.join(output_dir, f"ablation_summary_{timestamp}.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "results": ablation_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 报告已保存: {report_path}")

    # 记录到 wandb
    if WANDB_CONFIG.enabled:
        try:
            import wandb

            run_name = f"{WANDB_CONFIG.run_name_ablation}-{timestamp}"
            wandb.init(
                project=WANDB_CONFIG.project,
                entity=WANDB_CONFIG.entity,
                name=run_name,
                config={
                    "experiment_type": "ablation_study",
                    "models_tested": list(ablation_results.keys()),
                },
                tags=["ablation", "comparison"],
            )

            # 记录每个模型的指标
            comparison_data = []
            for key, result in ablation_results.items():
                if "error" not in result:
                    wandb.log({
                        f"{key}/format_accuracy": result["format_accuracy"],
                        f"{key}/answer_accuracy": result["answer_accuracy"],
                    })

                    # 准备对比表格数据
                    comparison_data.append({
                        "model": result["name"],
                        "format_accuracy": result["format_accuracy"],
                        "answer_accuracy": result["answer_accuracy"],
                    })

                    # 记录按类型的结果
                    if "by_type" in result:
                        for qtype, stats in result["by_type"].items():
                            wandb.log({
                                f"{key}/{qtype}/format_accuracy": stats["format_accuracy"],
                                f"{key}/{qtype}/answer_accuracy": stats["answer_accuracy"],
                            })

            # 创建对比表格
            if comparison_data:
                wandb.log({"ablation_comparison": wandb.Table(
                    columns=["model", "format_accuracy", "answer_accuracy"],
                    data=[[d["model"], d["format_accuracy"], d["answer_accuracy"]] for d in comparison_data]
                )})

            # 保存报告为 artifact
            artifact = wandb.Artifact(
                name=f"ablation-report-{timestamp}",
                type="ablation_study",
                description="Ablation study report comparing Base, SFT, and GRPO models",
            )
            artifact.add_file(report_path)
            wandb.log_artifact(artifact)

            print("✓ 消融实验结果已记录到 wandb")
            wandb.finish()
        except ImportError:
            print("⚠️  wandb 未安装，跳过 wandb 日志")
        except Exception as e:
            print(f"⚠️  wandb 记录失败: {e}")

    # 生成结论
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)

    conclusions = []

    if "base_model" in ablation_results and "error" not in ablation_results["base_model"]:
        base_format = ablation_results["base_model"]["format_accuracy"]
        if base_format < 0.3:
            conclusions.append("- 基座模型几乎不输出正确格式，说明格式需要通过训练学习")

    if "sft_only" in ablation_results and "error" not in ablation_results["sft_only"]:
        sft_format = ablation_results["sft_only"]["format_accuracy"]
        if sft_format > 0.8:
            conclusions.append("- SFT 有效地教会了模型输出正确格式（<think><answer>结构）")

    if ("sft_only" in ablation_results and "error" not in ablation_results["sft_only"] and
        "sft_grpo" in ablation_results and "error" not in ablation_results["sft_grpo"]):
        sft_acc = ablation_results["sft_only"]["answer_accuracy"]
        grpo_acc = ablation_results["sft_grpo"]["answer_accuracy"]
        if grpo_acc > sft_acc:
            improvement = (grpo_acc - sft_acc) / sft_acc * 100 if sft_acc > 0 else 0
            conclusions.append(f"- GRPO 在 SFT 基础上进一步提升了答案准确率 (+{improvement:.1f}%)")
        elif grpo_acc < sft_acc:
            conclusions.append("- ⚠️ GRPO 未能提升准确率，可能需要调整奖励函数或训练参数")

    if conclusions:
        for c in conclusions:
            print(c)
    else:
        print("- 数据不足以得出结论，请确保所有模型都已训练完成")

    print("=" * 60)

    return report_path


def main():
    parser = argparse.ArgumentParser(description="消融实验")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports",
        help="报告输出目录"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="测试样本数（默认使用全部）"
    )

    args = parser.parse_args()

    # 检查测试数据
    if not os.path.exists(DATA_CONFIG.test_data_path):
        print(f"❌ 测试数据不存在: {DATA_CONFIG.test_data_path}")
        print("   请先运行数据准备流程")
        sys.exit(1)

    # 加载测试数据
    print(f"加载测试数据: {DATA_CONFIG.test_data_path}")
    with open(DATA_CONFIG.test_data_path, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]

    if args.sample_size and args.sample_size < len(test_data):
        import random
        random.seed(42)
        test_data = random.sample(test_data, args.sample_size)

    print(f"测试样本数: {len(test_data)}")

    # 执行消融实验
    ablation_results = run_ablation_study(test_data, args.output_dir)

    # 生成报告
    generate_report(ablation_results, args.output_dir)


if __name__ == "__main__":
    main()
