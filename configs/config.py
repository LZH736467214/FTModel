"""
全局配置文件
面试点：配置管理的最佳实践

模型架构（本地部署，不调用外部API）：
- Teacher: DeepSeek-R1-Distill-Qwen-7B（蒸馏阶段生成CoT）
- Judge: Qwen2.5-7B-Instruct（过滤、GRPO评分、评测判分）
- Base: Qwen2.5-1.5B-Instruct（待训练的小模型）
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class TeacherModelConfig:
    """Teacher 模型配置（用于蒸馏）"""
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    model_max_length: int = 2048
    temperature: float = 0.6
    max_new_tokens: int = 1024

@dataclass
class JudgeModelConfig:
    """Judge 模型配置（用于评分和过滤）"""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    model_max_length: int = 2048
    temperature: float = 0.1  # 评分时用较低温度保证一致性
    max_new_tokens: int = 512

@dataclass
class ModelConfig:
    """Base 模型配置（待训练的小模型）"""
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_max_length: int = 2048

@dataclass
class LoRAConfig:
    """LoRA 配置"""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = None

    def __post_init__(self):
        if self.target_modules is None:
            # Qwen 系列的 LoRA 目标模块
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class SFTConfig:
    """SFT 训练配置"""
    output_dir: str = "ckpts/sft_lora"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    bf16: bool = True
    optim: str = "paged_adamw_8bit"

@dataclass
class GRPOConfig:
    """GRPO 训练配置"""
    output_dir: str = "ckpts/grpo_lora"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-6
    num_sample_generations: int = 4
    response_length: int = 512
    temperature: float = 0.7
    kl_coef: float = 0.05
    format_reward_weight: float = 0.3
    accuracy_reward_weight: float = 0.7

@dataclass
class LocalInferenceConfig:
    """本地推理配置"""
    # 量化配置
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True
    # GPU 配置
    device_map: str = "auto"
    gpu_memory_utilization: float = 0.9

@dataclass
class WandbConfig:
    """Wandb 配置"""
    project: str = "FTModel-Training"  # wandb 项目名称
    entity: Optional[str] = None  # wandb 团队名称（可选）
    enabled: bool = True  # 是否启用 wandb

    # 运行名称（自动生成）
    run_name_sft: str = "sft-train"
    run_name_grpo: str = "grpo-train"
    run_name_eval: str = "evaluation"
    run_name_ablation: str = "ablation-study"

    # 日志配置
    log_model: bool = False  # 是否上传模型到 wandb（通常很大，不建议）
    log_interval: int = 10  # 日志记录间隔

@dataclass
class DataConfig:
    """数据配置"""
    raw_data_path: str = "data/raw/raw.jsonl"
    distilled_data_path: str = "data/processed/distilled.jsonl"
    sft_data_path: str = "data/processed/sft.jsonl"
    rl_data_path: str = "data/processed/rl.jsonl"
    test_data_path: str = "data/processed/test.jsonl"

    # 数据规模
    total_samples: int = 500
    test_ratio: float = 0.2

    # 分层抽样配置
    type_distribution: dict = None

    def __post_init__(self):
        if self.type_distribution is None:
            self.type_distribution = {
                "financial_calculation": 0.4,
                "business_reasoning": 0.3,
                "concept_qa": 0.2,
                "risk_analysis": 0.1
            }

# 全局配置实例
TEACHER_CONFIG = TeacherModelConfig()  # Teacher: DeepSeek-R1-Distill-Qwen-7B
JUDGE_CONFIG = JudgeModelConfig()      # Judge: Qwen2.5-7B-Instruct
MODEL_CONFIG = ModelConfig()           # Base: Qwen2.5-1.5B-Instruct
LORA_CONFIG = LoRAConfig()
SFT_CONFIG = SFTConfig()
GRPO_CONFIG = GRPOConfig()
LOCAL_INFERENCE_CONFIG = LocalInferenceConfig()
WANDB_CONFIG = WandbConfig()
DATA_CONFIG = DataConfig()
