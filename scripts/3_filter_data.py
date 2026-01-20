"""
双重过滤：答案正确性 + 推理质量
输出：data/processed/sft.jsonl, data/processed/rl.jsonl
"""
import json
import os
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import DATA_CONFIG

class DataFilter:
    """
    数据过滤器
    面试点：Fin-R1 的过滤策略
    """

    def __init__(self):
        self.stats = {
            "total": 0,
            "format_ok": 0,
            "answer_correct": 0,
            "reasoning_good": 0,
            "final_pass": 0,
            "filter_reasons": {}
        }

    def extract_answer(self, text):
        """提取 <answer> 中的内容"""
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if not match:
            return None

        answer = match.group(1).strip()

        # 处理 \boxed{} 格式
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer)
        if boxed_match:
            return boxed_match.group(1).strip()

        return answer

    def extract_think(self, text):
        """提取 <think> 中的内容"""
        match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        return match.group(1).strip() if match else None

    # ========== 第一层：格式检查 ==========

    def check_format(self, text):
        """检查格式是否符合要求"""
        think_count = text.count("<think>")
        answer_count = text.count("<answer>")

        if think_count != 1 or answer_count != 1:
            return False, "标签数量不对"

        # 检查标签顺序
        think_pos = text.find("<think>")
        answer_pos = text.find("<answer>")

        if think_pos > answer_pos:
            return False, "标签顺序错误"

        return True, "OK"

    # ========== 第二层：答案正确性 ==========

    def check_answer_math(self, extracted, gold):
        """数学题答案检查"""
        try:
            # 提取数字
            extracted_num = float(re.sub(r'[^\d.-]', '', str(extracted)))
            gold_num = float(re.sub(r'[^\d.-]', '', str(gold)))

            # 相对误差或绝对误差
            if abs(gold_num) > 1:
                # 相对误差
                return abs(extracted_num - gold_num) / abs(gold_num) < 0.01
            else:
                # 绝对误差
                return abs(extracted_num - gold_num) < 0.01
        except:
            return False

    def check_answer_qa(self, extracted, gold):
        """QA 题答案检查（关键词匹配）"""
        gold_lower = str(gold).lower()
        extracted_lower = str(extracted).lower()

        # 直接包含
        if gold_lower in extracted_lower:
            return True

        # 关键词重叠
        gold_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', gold_lower))
        extracted_keywords = set(re.findall(r'[\w\u4e00-\u9fff]+', extracted_lower))

        if not gold_keywords:
            return False

        overlap_ratio = len(gold_keywords & extracted_keywords) / len(gold_keywords)
        return overlap_ratio > 0.6

    def check_answer_correctness(self, item):
        """答案正确性总入口"""
        extracted = self.extract_answer(item["teacher_output"])

        if not extracted:
            return False, "无法提取答案"

        gold = item["gold_answer"]
        qtype = item["type"]

        if qtype == "financial_calculation":
            is_correct = self.check_answer_math(extracted, gold)
        else:
            is_correct = self.check_answer_qa(extracted, gold)

        return is_correct, "OK" if is_correct else "答案不正确"

    # ========== 第三层：推理质量 ==========

    def check_reasoning_quality(self, item):
        """
        推理质量检查
        面试点：Fin-R1 用 7 个维度，我们简化为 4 个核心维度
        """
        think_text = self.extract_think(item["teacher_output"])

        if not think_text:
            return False, "缺少推理过程"

        # 维度1：长度合理性
        if len(think_text) < 50:
            return False, "推理过程过短"
        if len(think_text) > 2000:
            return False, "推理过程过长"

        # 维度2：逻辑连接词
        reasoning_keywords = [
            "首先", "其次", "然后", "接着", "最后",
            "因此", "所以", "由于", "根据", "可得",
            "计算", "推导", "分析", "得出", "综上"
        ]
        keyword_count = sum(1 for kw in reasoning_keywords if kw in think_text)
        if keyword_count < 2:
            return False, "缺少逻辑连接词"

        # 维度3：步骤清晰性（金融计算题需要有计算过程）
        if item["type"] == "financial_calculation":
            # 检查是否有数学符号
            if not re.search(r'[=\+\-\*/÷×()]', think_text):
                return False, "计算题缺少计算过程"

        # 维度4：重复内容检查
        sentences = [s.strip() for s in re.split(r'[。！？\n]', think_text) if s.strip()]
        if len(sentences) > 3:
            unique_ratio = len(set(sentences)) / len(sentences)
            if unique_ratio < 0.7:
                return False, "存在大量重复内容"

        return True, "OK"

    # ========== 主流程 ==========

    def filter_item(self, item):
        """过滤单条数据"""
        # 第一层：格式
        format_ok, format_reason = self.check_format(item["teacher_output"])
        if not format_ok:
            self.stats["filter_reasons"][format_reason] = \
                self.stats["filter_reasons"].get(format_reason, 0) + 1
            return False

        self.stats["format_ok"] += 1

        # 第二层：答案
        answer_ok, answer_reason = self.check_answer_correctness(item)
        if not answer_ok:
            self.stats["filter_reasons"][answer_reason] = \
                self.stats["filter_reasons"].get(answer_reason, 0) + 1
            return False

        self.stats["answer_correct"] += 1

        # 第三层：推理质量
        quality_ok, quality_reason = self.check_reasoning_quality(item)
        if not quality_ok:
            self.stats["filter_reasons"][quality_reason] = \
                self.stats["filter_reasons"].get(quality_reason, 0) + 1
            return False

        self.stats["reasoning_good"] += 1
        self.stats["final_pass"] += 1

        return True

    def filter_all(self, data):
        """过滤所有数据"""
        self.stats["total"] = len(data)

        sft_data = []
        rl_data = []

        for item in data:
            if self.filter_item(item):
                # SFT 数据
                sft_data.append({
                    "id": item["id"],
                    "prompt": item["question"],
                    "response": item["teacher_output"],
                    "type": item["type"]
                })

                # RL 数据
                rl_data.append({
                    "id": item["id"],
                    "prompt": item["question"],
                    "gold_answer": item["gold_answer"],
                    "type": item["type"]
                })

        return sft_data, rl_data

    def print_report(self):
        """打印过滤报告"""
        print("\n" + "="*60)
        print("数据过滤报告")
        print("="*60)
        print(f"{'指标':<20s} {'数量':>10s} {'占比':>10s}")
        print("-"*60)
        print(f"{'原始样本':<20s} {self.stats['total']:>10d} {100.0:>9.1f}%")
        print(f"{'格式正确':<20s} {self.stats['format_ok']:>10d} {self.stats['format_ok']/self.stats['total']*100:>9.1f}%")
        print(f"{'答案正确':<20s} {self.stats['answer_correct']:>10d} {self.stats['answer_correct']/self.stats['total']*100:>9.1f}%")
        print(f"{'推理合格':<20s} {self.stats['reasoning_good']:>10d} {self.stats['reasoning_good']/self.stats['total']*100:>9.1f}%")
        print(f"{'最终通过':<20s} {self.stats['final_pass']:>10d} {self.stats['final_pass']/self.stats['total']*100:>9.1f}%")
        print("="*60)

        if self.stats["filter_reasons"]:
            print("\n过滤原因分布:")
            for reason, count in sorted(self.stats["filter_reasons"].items(),
                                       key=lambda x: -x[1]):
                print(f"  {reason:<30s}: {count:>5d}")

        print("="*60)

def split_train_test(sft_data, rl_data, test_ratio=0.2):
    """
    切分训练集和测试集
    面试点：如何避免数据泄漏
    """
    import random
    random.seed(42)

    # 按 ID 对齐
    assert len(sft_data) == len(rl_data)
    assert all(sft["id"] == rl["id"] for sft, rl in zip(sft_data, rl_data))

    # 打乱
    indices = list(range(len(sft_data)))
    random.shuffle(indices)

    # 切分
    test_size = int(len(sft_data) * test_ratio)
    test_indices = set(indices[:test_size])

    sft_train = [sft_data[i] for i in range(len(sft_data)) if i not in test_indices]
    sft_test = [sft_data[i] for i in range(len(sft_data)) if i in test_indices]

    rl_train = [rl_data[i] for i in range(len(rl_data)) if i not in test_indices]
    rl_test = [rl_data[i] for i in range(len(rl_data)) if i in test_indices]

    return sft_train, sft_test, rl_train, rl_test

def filter_data():
    """主流程"""
    print("="*60)
    print("阶段3：双重过滤（答案 + 推理质量）")
    print("="*60)

    # 加载蒸馏数据
    print(f"\n加载蒸馏数据: {DATA_CONFIG.distilled_data_path}")
    with open(DATA_CONFIG.distilled_data_path, 'r', encoding='utf-8') as f:
        distilled_data = [json.loads(line) for line in f]
    print(f"✓ 加载 {len(distilled_data)} 条")

    # 过滤
    print("\n执行过滤...")
    filter_obj = DataFilter()
    sft_data, rl_data = filter_obj.filter_all(distilled_data)

    # 打印报告
    filter_obj.print_report()

    # 切分训练/测试集
    print("\n切分训练/测试集...")
    sft_train, sft_test, rl_train, rl_test = split_train_test(
        sft_data, rl_data, DATA_CONFIG.test_ratio
    )

    print(f"SFT 训练集: {len(sft_train)} 条")
    print(f"SFT 测试集: {len(sft_test)} 条")
    print(f"RL 训练集: {len(rl_train)} 条")
    print(f"RL 测试集: {len(rl_test)} 条")

    # 保存
    os.makedirs("data/processed", exist_ok=True)

    def save_jsonl(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    save_jsonl(sft_train, DATA_CONFIG.sft_data_path)
    save_jsonl(rl_train, DATA_CONFIG.rl_data_path)
    save_jsonl(sft_test + rl_test, DATA_CONFIG.test_data_path)  # 测试集合并

    # 保存统计
    with open("data/processed/filter_stats.json", 'w', encoding='utf-8') as f:
        json.dump(filter_obj.stats, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 数据已保存:")
    print(f"   SFT 训练: {DATA_CONFIG.sft_data_path}")
    print(f"   RL 训练: {DATA_CONFIG.rl_data_path}")
    print(f"   测试集: {DATA_CONFIG.test_data_path}")

    return len(sft_train), len(rl_train), len(sft_test)

if __name__ == "__main__":
    sft_count, rl_count, test_count = filter_data()
    print(f"\n✅ 数据构建完成！")
    print(f"   最终数据资产：SFT {sft_count} 条 + RL {rl_count} 条 + 测试 {test_count} 条")
    print(f"\n下一步：上传到服务器并开始训练")
