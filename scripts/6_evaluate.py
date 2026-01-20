"""
T5 评测阶段：Base 推理 + Judge 判分
- Base 模型：Qwen2.5-1.5B-Instruct（推理）
- Judge 模型：Qwen2.5-7B-Instruct（判分）

功能：
1. 加载测试集，使用 Base 模型生成回答
2. 使用 Judge 模型判分（可选）
3. 计算格式正确率
4. 计算答案准确率（按任务类型）
5. 生成 JSON 评测报告

面试点：
- 评测指标设计：格式 + 准确率 + 按类型细分
- Judge 模型判分：比规则更灵活，能评估推理质量
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
from configs.config import (
    MODEL_CONFIG, SFT_CONFIG, GRPO_CONFIG, DATA_CONFIG,
    JUDGE_CONFIG, LOCAL_INFERENCE_CONFIG, WANDB_CONFIG
)


# ============ Judge 模型 ============

class JudgeModel:
    """
    Judge 模型封装（用于评测判分）
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load(self):
        """加载 Judge 模型"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print("=" * 60)
        print("加载 Judge 模型（用于评测判分）")
        print("=" * 60)
        print(f"模型: {JUDGE_CONFIG.model_name}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=LOCAL_INFERENCE_CONFIG.load_in_4bit,
            bnb_4bit_quant_type=LOCAL_INFERENCE_CONFIG.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=LOCAL_INFERENCE_CONFIG.use_double_quant,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            JUDGE_CONFIG.model_name,
            quantization_config=bnb_config,
            device_map=LOCAL_INFERENCE_CONFIG.device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            JUDGE_CONFIG.model_name,
            trust_remote_code=True,
            padding_side="left",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        print("✓ Judge 模型加载完成")

    def unload(self):
        """释放 GPU 显存"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()

    def judge_response(self, question: str, response: str, gold_answer: str, qtype: str) -> Dict[str, Any]:
        """
        使用 Judge 模型判分
        返回：详细的评估结果
        """
        prompt = f"""请评估以下金融问题回答的质量。

问题：{question}
题目类型：{qtype}
标准答案：{gold_answer}

模型回答：
{response}

请从以下维度评估：
1. 格式规范（是否符合 <think>...</think><answer>...</answer> 格式）
2. 答案正确性（答案是否与标准答案匹配）
3. 推理质量（推理过程是否清晰、完整）

请严格按以下格式输出：
格式正确：是/否
答案正确：是/否
推理质量：优秀/良好/一般/较差
总分：X分（满分10分）
评语：[简短评语]"""

        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=JUDGE_CONFIG.max_new_tokens,
                    temperature=JUDGE_CONFIG.temperature,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            return self._parse_judge_result(response_text)

        except Exception as e:
            return {
                "format_ok": False,
                "answer_ok": False,
                "reasoning_quality": "未知",
                "score": 0,
                "comment": f"Judge 评估失败: {str(e)}"
            }

    def _parse_judge_result(self, response: str) -> Dict[str, Any]:
        """解析 Judge 模型的评估结果"""
        result = {
            "format_ok": False,
            "answer_ok": False,
            "reasoning_quality": "未知",
            "score": 0,
            "comment": ""
        }

        try:
            # 提取格式正确
            format_match = re.search(r'格式正确[：:]\s*(是|否)', response)
            if format_match:
                result["format_ok"] = format_match.group(1) == "是"

            # 提取答案正确
            answer_match = re.search(r'答案正确[：:]\s*(是|否)', response)
            if answer_match:
                result["answer_ok"] = answer_match.group(1) == "是"

            # 提取推理质量
            quality_match = re.search(r'推理质量[：:]\s*(优秀|良好|一般|较差)', response)
            if quality_match:
                result["reasoning_quality"] = quality_match.group(1)

            # 提取总分
            score_match = re.search(r'总分[：:]\s*(\d+)', response)
            if score_match:
                result["score"] = int(score_match.group(1))

            # 提取评语
            comment_match = re.search(r'评语[：:]\s*(.+?)(?:\n|$)', response, re.DOTALL)
            if comment_match:
                result["comment"] = comment_match.group(1).strip()

        except Exception:
            pass

        return result


# ============ 规则评测指标计算 ============

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
    """检查格式是否正确（规则）"""
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


def check_answer_rule(response: str, gold_answer: str, qtype: str) -> bool:
    """使用规则检查答案是否正确"""
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
    使用 transformers 加载 Base 模型
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"加载 Base 模型 (transformers): {model_path}")

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
    print("✓ Base 模型加载完成")

    return model, tokenizer


def load_model_vllm(model_path: str):
    """
    使用 vLLM 加载 Base 模型
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("❌ vLLM 未安装，请运行: pip install vllm")
        sys.exit(1)

    print(f"加载 Base 模型 (vLLM): {model_path}")

    # 判断是否是 LoRA adapter
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("⚠️  检测到 LoRA adapter，vLLM 需要合并后的模型")
        print("   请先运行: python scripts/7_deploy.py --action merge")
        sys.exit(1)

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
    )

    print("✓ Base 模型 (vLLM) 加载完成")

    return llm, None


# ============ 推理函数 ============

def generate_transformers(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> List[str]:
    """使用 transformers 生成"""
    responses = []

    for prompt in tqdm(prompts, desc="Base 模型推理中"):
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
    """使用 vLLM 生成"""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
    )

    # 格式化 prompt
    formatted_prompts = []
    for prompt in prompts:
        formatted_prompts.append(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")

    outputs = llm.generate(formatted_prompts, sampling_params)

    responses = [output.outputs[0].text for output in outputs]

    return responses


# ============ 评测主流程 ============

def evaluate(
    model_path: str,
    use_vllm: bool = False,
    use_judge: bool = True,
    output_dir: str = "reports",
):
    """
    执行评测
    """
    print("=" * 60)
    print("T5 评测阶段：Base 推理 + Judge 判分")
    print("=" * 60)

    # 初始化 wandb
    wandb_run = None
    if WANDB_CONFIG.enabled:
        try:
            import wandb

            model_name = os.path.basename(model_path)
            run_name = f"{WANDB_CONFIG.run_name_eval}-{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb_run = wandb.init(
                project=WANDB_CONFIG.project,
                entity=WANDB_CONFIG.entity,
                name=run_name,
                config={
                    "model_path": model_path,
                    "use_vllm": use_vllm,
                    "use_judge": use_judge,
                },
                tags=["evaluation", "judge-scoring" if use_judge else "rule-scoring"],
            )
            print(f"✓ Wandb 已初始化: {run_name}")
        except ImportError:
            print("⚠️  wandb 未安装，跳过 wandb 日志")
        except Exception as e:
            print(f"⚠️  wandb 初始化失败: {e}")
    else:
        print("ℹ️  Wandb 未启用")

    # 加载测试数据
    print(f"\n加载测试数据: {DATA_CONFIG.test_data_path}")
    with open(DATA_CONFIG.test_data_path, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    print(f"✓ 加载 {len(test_data)} 条测试数据")

    # 加载 Base 模型
    if use_vllm:
        base_model, tokenizer = load_model_vllm(model_path)
    else:
        base_model, tokenizer = load_model_transformers(model_path)

    # 提取 prompts
    prompts = [item["prompt"] for item in test_data]

    # 使用 Base 模型生成回答
    print("\n使用 Base 模型生成回答...")
    if use_vllm:
        responses = generate_vllm(base_model, prompts)
    else:
        responses = generate_transformers(base_model, tokenizer, prompts)

    # 释放 Base 模型显存
    if not use_vllm:
        del base_model
        torch.cuda.empty_cache()

    # 加载 Judge 模型（如果启用）
    judge_model = None
    if use_judge:
        judge_model = JudgeModel()
        judge_model.load()

    # 计算评测指标
    print("\n计算评测指标...")

    results = {
        "total": len(test_data),
        "format_correct": 0,
        "answer_correct": 0,
        "by_type": {},
        "judge_scores": [],
        "details": [],
    }

    for item, response in tqdm(zip(test_data, responses), total=len(test_data), desc="Judge 判分中" if use_judge else "规则评估中"):
        prompt = item["prompt"]
        gold_answer = item.get("gold_answer", "")
        qtype = item.get("type", "unknown")

        # 规则评估（作为基准）
        format_ok_rule = check_format(response)
        answer_ok_rule = check_answer_rule(response, gold_answer, qtype) if gold_answer else False

        # Judge 评估
        if use_judge and judge_model:
            judge_result = judge_model.judge_response(
                question=prompt,
                response=response,
                gold_answer=gold_answer,
                qtype=qtype
            )
            format_ok = judge_result["format_ok"]
            answer_ok = judge_result["answer_ok"]
            judge_score = judge_result["score"]
            judge_comment = judge_result["comment"]
            results["judge_scores"].append(judge_score)
        else:
            format_ok = format_ok_rule
            answer_ok = answer_ok_rule
            judge_score = None
            judge_comment = None

        # 统计
        if format_ok:
            results["format_correct"] += 1
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
        detail = {
            "id": item.get("id", ""),
            "prompt": prompt,
            "gold_answer": gold_answer,
            "type": qtype,
            "response": response,
            "format_ok": format_ok,
            "answer_ok": answer_ok,
            "format_ok_rule": format_ok_rule,
            "answer_ok_rule": answer_ok_rule,
        }
        if use_judge:
            detail["judge_score"] = judge_score
            detail["judge_comment"] = judge_comment
        results["details"].append(detail)

    # 清理 Judge 模型
    if judge_model:
        judge_model.unload()

    # 计算比例
    results["format_accuracy"] = results["format_correct"] / results["total"]
    results["answer_accuracy"] = results["answer_correct"] / results["total"]

    if results["judge_scores"]:
        results["avg_judge_score"] = sum(results["judge_scores"]) / len(results["judge_scores"])

    for qtype in results["by_type"]:
        type_results = results["by_type"][qtype]
        type_results["format_accuracy"] = type_results["format_correct"] / type_results["total"]
        type_results["answer_accuracy"] = type_results["answer_correct"] / type_results["total"]

    # 打印报告
    print("\n" + "=" * 60)
    print("评测报告")
    print("=" * 60)
    print(f"评测方式: {'Judge 模型判分' if use_judge else '规则评估'}")
    print(f"测试样本数: {results['total']}")
    print(f"格式正确率: {results['format_accuracy']:.2%} ({results['format_correct']}/{results['total']})")
    print(f"答案正确率: {results['answer_accuracy']:.2%} ({results['answer_correct']}/{results['total']})")
    if use_judge and results.get("avg_judge_score"):
        print(f"平均 Judge 分数: {results['avg_judge_score']:.2f}/10")
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
        "use_judge": use_judge,
        "total": results["total"],
        "format_accuracy": results["format_accuracy"],
        "answer_accuracy": results["answer_accuracy"],
        "avg_judge_score": results.get("avg_judge_score"),
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

    # 记录到 wandb
    if wandb_run is not None:
        try:
            import wandb

            # 记录总体指标
            wandb.log({
                "format_accuracy": results["format_accuracy"],
                "answer_accuracy": results["answer_accuracy"],
                "format_correct": results["format_correct"],
                "answer_correct": results["answer_correct"],
                "total": results["total"],
            })

            # 记录 Judge 分数
            if use_judge and results.get("avg_judge_score"):
                wandb.log({"avg_judge_score": results["avg_judge_score"]})

            # 记录按类型统计
            for qtype, stats in results["by_type"].items():
                wandb.log({
                    f"{qtype}/format_accuracy": stats["format_accuracy"],
                    f"{qtype}/answer_accuracy": stats["answer_accuracy"],
                    f"{qtype}/total": stats["total"],
                })

            # 保存评测报告为 artifact
            artifact = wandb.Artifact(
                name=f"eval-report-{model_name}",
                type="evaluation",
                description=f"Evaluation report for {model_path}",
            )
            artifact.add_file(summary_path)
            artifact.add_file(details_path)
            wandb.log_artifact(artifact)

            print("✓ 结果已记录到 wandb")
            wandb.finish()
        except Exception as e:
            print(f"⚠️  wandb 记录失败: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="T5 评测阶段：Base 推理 + Judge 判分")
    parser.add_argument(
        "--model_path",
        type=str,
        default=GRPO_CONFIG.output_dir,
        help="Base 模型路径（默认使用 GRPO 模型）"
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="使用 vLLM 进行推理（默认使用 transformers）"
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="不使用 Judge 模型判分（仅使用规则评估）"
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

    use_judge = not args.no_judge

    print("=" * 60)
    print("T5 评测阶段")
    print(f"Base 模型：{model_path}")
    print(f"Judge 模型：{JUDGE_CONFIG.model_name if use_judge else '禁用'}")
    print("=" * 60)

    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"❌ Base 模型不存在: {model_path}")
        sys.exit(1)

    # 检查测试数据是否存在
    if not os.path.exists(DATA_CONFIG.test_data_path):
        print(f"❌ 测试数据不存在: {DATA_CONFIG.test_data_path}")
        print("   请先运行数据准备流程")
        sys.exit(1)

    evaluate(
        model_path=model_path,
        use_vllm=args.use_vllm,
        use_judge=use_judge,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
