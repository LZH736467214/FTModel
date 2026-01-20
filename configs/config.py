"""
全局配置文件
面试点：配置管理的最佳实践
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """模型配置"""
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
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
class APIConfig:
    """API 配置"""
    provider: str = "deepseek"  # deepseek, qwen, openai
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None

    def __post_init__(self):
        # 从环境变量读取
        if self.api_key is None:
            self.api_key = os.getenv("API_KEY")

        # 根据 provider 设置默认值
        if self.provider == "deepseek":
            self.base_url = self.base_url or "https://api.deepseek.com/v1"
            self.model_name = self.model_name or "deepseek-chat"
        elif self.provider == "qwen":
            self.base_url = self.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            self.model_name = self.model_name or "qwen-max"

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
MODEL_CONFIG = ModelConfig()
LORA_CONFIG = LoRAConfig()
SFT_CONFIG = SFTConfig()
GRPO_CONFIG = GRPOConfig()
API_CONFIG = APIConfig()
DATA_CONFIG = DataConfig()
