"""
T1 蒸馏阶段：用本地 Teacher 模型生成 CoT
Teacher 模型：DeepSeek-R1-Distill-Qwen-7B（本地部署，不调用外部 API）

输出：data/processed/distilled.jsonl
"""
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import DATA_CONFIG, TEACHER_CONFIG, LOCAL_INFERENCE_CONFIG


def load_teacher_model():
    """
    加载本地 Teacher 模型（DeepSeek-R1-Distill-Qwen-7B）
    使用 4bit 量化以节省显存
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("=" * 60)
    print("加载 Teacher 模型")
    print("=" * 60)
    print(f"模型: {TEACHER_CONFIG.model_name}")

    # 4bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOCAL_INFERENCE_CONFIG.load_in_4bit,
        bnb_4bit_quant_type=LOCAL_INFERENCE_CONFIG.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=LOCAL_INFERENCE_CONFIG.use_double_quant,
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_CONFIG.model_name,
        quantization_config=bnb_config,
        device_map=LOCAL_INFERENCE_CONFIG.device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        TEACHER_CONFIG.model_name,
        trust_remote_code=True,
        padding_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    print(f"✓ Teacher 模型加载完成")

    return model, tokenizer


def create_distillation_prompt(question, qtype):
    """
    构造蒸馏 prompt
    面试点：prompt 工程的重要性
    """
    system_prompt = """你是一个金融领域专家。请用以下严格格式回答问题：

<think>
[详细的推理过程，必须包含3-5个清晰的推理步骤]
</think>
<answer>
[最终答案，简洁明确]
</answer>

要求：
1. <think> 部分必须展示完整推理逻辑：
   - 金融计算题：写出公式、代入数值、计算过程
   - 概念题：定义 → 组成要素 → 计算方法/应用场景
   - 分析题：现象 → 原因分析 → 影响/结论
2. 推理步骤用"首先"、"其次"、"然后"、"因此"等连接词
3. <answer> 部分只包含最终答案：
   - 数值题：直接给出数字（不要单位）
   - 概念/分析题：1-2句话的简洁答案
4. 严格遵守标签格式，不要有任何多余内容"""

    user_prompt = f"问题：{question}"

    return system_prompt, user_prompt


def call_teacher_model(model, tokenizer, question, qtype):
    """
    使用本地 Teacher 模型生成 CoT
    """
    system_prompt, user_prompt = create_distillation_prompt(question, qtype)

    # 构造消息
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 使用 chat template 格式化
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=TEACHER_CONFIG.max_new_tokens,
                temperature=TEACHER_CONFIG.temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 解码输出（只取生成的部分）
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return response, None

    except Exception as e:
        return None, str(e)


def distill_data():
    """主流程"""
    print("=" * 60)
    print("T1 蒸馏阶段：Teacher 生成 CoT（本地推理）")
    print("=" * 60)

    # 加载原始数据
    print(f"\n加载原始数据: {DATA_CONFIG.raw_data_path}")
    with open(DATA_CONFIG.raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]
    print(f"✓ 加载 {len(raw_data)} 条")

    # 加载本地 Teacher 模型
    model, tokenizer = load_teacher_model()

    # 蒸馏
    distilled_data = []
    failed_items = []

    print(f"\n开始蒸馏...")
    for item in tqdm(raw_data, desc="蒸馏进度"):
        teacher_output, error = call_teacher_model(
            model,
            tokenizer,
            item["question"],
            item["type"]
        )

        if teacher_output:
            distilled_data.append({
                **item,
                "teacher_output": teacher_output
            })
        else:
            failed_items.append({
                "id": item["id"],
                "error": error
            })

        # 每10条保存一次（防止中断丢失）
        if len(distilled_data) % 10 == 0:
            temp_path = DATA_CONFIG.distilled_data_path + ".tmp"
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with open(temp_path, 'w', encoding='utf-8') as f:
                for d in distilled_data:
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')

    # 保存最终结果
    output_path = DATA_CONFIG.distilled_data_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in distilled_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 保存失败记录
    if failed_items:
        fail_path = "data/processed/distill_failures.jsonl"
        with open(fail_path, 'w', encoding='utf-8') as f:
            for item in failed_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 释放 GPU 显存
    del model
    torch.cuda.empty_cache()

    # 统计
    print("\n" + "=" * 60)
    print("蒸馏结果")
    print("=" * 60)
    print(f"总数: {len(raw_data)}")
    print(f"成功: {len(distilled_data)} ({len(distilled_data)/len(raw_data)*100:.1f}%)")
    print(f"失败: {len(failed_items)} ({len(failed_items)/len(raw_data)*100:.1f}%)")
    print("=" * 60)

    print(f"\n✅ 蒸馏数据已保存至: {output_path}")

    # 展示一个样例
    if distilled_data:
        print("\n" + "=" * 60)
        print("样例展示")
        print("=" * 60)
        sample = distilled_data[0]
        print(f"问题: {sample['question']}")
        print(f"\n教师输出:\n{sample['teacher_output']}")
        print("=" * 60)

    return len(distilled_data)


if __name__ == "__main__":
    count = distill_data()
    print(f"\n下一步：运行 python scripts/3_filter_data.py")
