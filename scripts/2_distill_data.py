"""
用教师模型生成 CoT
输出：data/processed/distilled.jsonl
"""
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import DATA_CONFIG, API_CONFIG

def get_api_client():
    """
    获取 API 客户端
    面试点：API 调用的通用封装
    """
    from openai import OpenAI

    if not API_CONFIG.api_key:
        print("⚠️  未设置 API_KEY")
        print("请设置环境变量：export API_KEY=your_api_key")
        print("或在 configs/config.py 中配置")
        sys.exit(1)

    client = OpenAI(
        api_key=API_CONFIG.api_key,
        base_url=API_CONFIG.base_url
    )

    return client

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

def call_teacher_model(client, question, qtype, max_retries=3):
    """
    调用教师模型
    面试点：错误处理和重试机制
    """
    system_prompt, user_prompt = create_distillation_prompt(question, qtype)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=API_CONFIG.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.6,
                max_tokens=1024
            )

            output = response.choices[0].message.content
            return output, None

        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                print(f"   ⚠️  重试 {attempt+1}/{max_retries}（等待 {wait_time}s）: {error_msg[:50]}")
                time.sleep(wait_time)
            else:
                return None, error_msg

    return None, "Max retries exceeded"

def distill_data():
    """主流程"""
    print("="*60)
    print("阶段2：数据蒸馏（Teacher 生成 CoT）")
    print("="*60)

    # 加载原始数据
    print(f"\n加载原始数据: {DATA_CONFIG.raw_data_path}")
    with open(DATA_CONFIG.raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]
    print(f"✓ 加载 {len(raw_data)} 条")

    # 初始化 API
    print(f"\n初始化 API: {API_CONFIG.provider}")
    client = get_api_client()
    print(f"✓ 使用模型: {API_CONFIG.model_name}")

    # 蒸馏
    distilled_data = []
    failed_items = []

    print(f"\n开始蒸馏...")
    for item in tqdm(raw_data, desc="蒸馏进度"):
        teacher_output, error = call_teacher_model(
            client,
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

    # 统计
    print("\n" + "="*60)
    print("蒸馏结果")
    print("="*60)
    print(f"总数: {len(raw_data)}")
    print(f"成功: {len(distilled_data)} ({len(distilled_data)/len(raw_data)*100:.1f}%)")
    print(f"失败: {len(failed_items)} ({len(failed_items)/len(raw_data)*100:.1f}%)")
    print("="*60)

    print(f"\n✅ 蒸馏数据已保存至: {output_path}")

    # 展示一个样例
    if distilled_data:
        print("\n" + "="*60)
        print("样例展示")
        print("="*60)
        sample = distilled_data[0]
        print(f"问题: {sample['question']}")
        print(f"\n教师输出:\n{sample['teacher_output']}")
        print("="*60)

    return len(distilled_data)

if __name__ == "__main__":
    count = distill_data()
    print(f"\n下一步：运行 python scripts/3_filter_data.py")
