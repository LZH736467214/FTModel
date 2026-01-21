"""
包兼容性验证脚本
用于检查更新后的依赖包是否与现有代码兼容
"""
import sys
import inspect

def check_imports():
    """检查所有关键包是否能正常导入"""
    print("=" * 60)
    print("步骤1: 检查包导入")
    print("=" * 60)

    packages_to_check = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("peft", "PEFT"),
        ("trl", "TRL"),
        ("datasets", "Datasets"),
        ("bitsandbytes", "BitsAndBytes"),
        ("accelerate", "Accelerate"),
        ("vllm", "vLLM"),
        ("pandas", "Pandas"),
        ("tqdm", "TQDM"),
    ]

    failed = []
    for package, name in packages_to_check:
        try:
            __import__(package)
            print(f"✓ {name:<20s} - 导入成功")
        except ImportError as e:
            print(f"✗ {name:<20s} - 导入失败: {e}")
            failed.append(name)

    if failed:
        print(f"\n⚠️  以下包导入失败: {', '.join(failed)}")
        return False
    else:
        print("\n✅ 所有包导入成功")
        return True


def check_package_versions():
    """检查包版本"""
    print("\n" + "=" * 60)
    print("步骤2: 检查包版本")
    print("=" * 60)

    import torch
    import transformers
    import peft
    import trl
    import datasets
    import bitsandbytes
    import accelerate

    try:
        import vllm
        vllm_version = vllm.__version__
    except:
        vllm_version = "未安装"

    print(f"torch:        {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"peft:         {peft.__version__}")
    print(f"trl:          {trl.__version__}")
    print(f"datasets:     {datasets.__version__}")
    print(f"bitsandbytes: {bitsandbytes.__version__}")
    print(f"accelerate:   {accelerate.__version__}")
    print(f"vllm:         {vllm_version}")

    # 验证关键版本
    errors = []

    if not torch.__version__.startswith("2.7"):
        errors.append(f"⚠️  torch 版本不是 2.7.x: {torch.__version__}")

    if not transformers.__version__.startswith("4.48"):
        errors.append(f"⚠️  transformers 版本不是 4.48.x: {transformers.__version__}")

    if not datasets.__version__.startswith("3."):
        errors.append(f"⚠️  datasets 版本不是 3.x: {datasets.__version__}")

    if errors:
        print("\n版本警告:")
        for error in errors:
            print(error)
    else:
        print("\n✅ 所有关键包版本正确")


def check_trl_api():
    """检查 TRL API 兼容性"""
    print("\n" + "=" * 60)
    print("步骤3: 检查 TRL API 兼容性")
    print("=" * 60)

    try:
        from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer
        print("✓ TRL 类导入成功")

        # 检查 SFTTrainer 参数
        print("\n检查 SFTTrainer 参数...")
        sft_params = inspect.signature(SFTTrainer.__init__).parameters
        param_names = list(sft_params.keys())
        print(f"  参数数量: {len(param_names)}")

        # 检查关键参数
        key_params = ['model', 'args', 'train_dataset']
        has_tokenizer = 'tokenizer' in param_names
        has_processing_class = 'processing_class' in param_names

        print(f"  - 包含 'tokenizer' 参数: {has_tokenizer}")
        print(f"  - 包含 'processing_class' 参数: {has_processing_class}")

        if has_processing_class:
            print("  ℹ️  建议使用 'processing_class' 而非 'tokenizer'")
        elif has_tokenizer:
            print("  ℹ️  使用 'tokenizer' 参数")
        else:
            print("  ⚠️  未找到 tokenizer 相关参数")

        # 检查 GRPOTrainer 参数
        print("\n检查 GRPOTrainer 参数...")
        grpo_params = inspect.signature(GRPOTrainer.__init__).parameters
        param_names = list(grpo_params.keys())
        print(f"  参数数量: {len(param_names)}")

        has_tokenizer = 'tokenizer' in param_names
        has_processing_class = 'processing_class' in param_names
        has_reward_funcs = 'reward_funcs' in param_names

        print(f"  - 包含 'tokenizer' 参数: {has_tokenizer}")
        print(f"  - 包含 'processing_class' 参数: {has_processing_class}")
        print(f"  - 包含 'reward_funcs' 参数: {has_reward_funcs}")

        if has_processing_class:
            print("  ✓ 'processing_class' 参数存在 (当前代码使用此参数)")
        elif has_tokenizer:
            print("  ⚠️  需要将代码中的 'processing_class' 改为 'tokenizer'")

        # 检查 GRPOConfig 参数
        print("\n检查 GRPOConfig 参数...")
        grpo_config_params = inspect.signature(GRPOConfig.__init__).parameters
        param_names = list(grpo_config_params.keys())

        has_num_generations = 'num_generations' in param_names
        has_max_completion_length = 'max_completion_length' in param_names

        print(f"  - 包含 'num_generations' 参数: {has_num_generations}")
        print(f"  - 包含 'max_completion_length' 参数: {has_max_completion_length}")

        if has_num_generations and has_max_completion_length:
            print("  ✓ 关键参数存在")
        else:
            print("  ⚠️  某些参数可能已重命名，请查看文档")

        print("\n✅ TRL API 检查完成")

    except Exception as e:
        print(f"✗ TRL API 检查失败: {e}")
        import traceback
        traceback.print_exc()


def check_datasets_api():
    """检查 datasets API 兼容性"""
    print("\n" + "=" * 60)
    print("步骤4: 检查 Datasets API 兼容性")
    print("=" * 60)

    try:
        from datasets import Dataset, load_dataset

        # 测试 Dataset.from_list
        print("测试 Dataset.from_list()...")
        test_data = [
            {"text": "测试1", "label": 0},
            {"text": "测试2", "label": 1},
        ]
        dataset = Dataset.from_list(test_data)
        print(f"  ✓ Dataset.from_list() 工作正常，创建了 {len(dataset)} 条数据")

        print("\n✅ Datasets API 检查完成")

    except Exception as e:
        print(f"✗ Datasets API 检查失败: {e}")
        import traceback
        traceback.print_exc()


def check_transformers_api():
    """检查 transformers API 兼容性"""
    print("\n" + "=" * 60)
    print("步骤5: 检查 Transformers API 兼容性")
    print("=" * 60)

    try:
        from transformers import BitsAndBytesConfig
        import torch

        # 测试 BitsAndBytesConfig
        print("测试 BitsAndBytesConfig()...")
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("  ✓ BitsAndBytesConfig 初始化成功")

        print("\n✅ Transformers API 检查完成")

    except Exception as e:
        print(f"✗ Transformers API 检查失败: {e}")
        import traceback
        traceback.print_exc()


def check_cuda():
    """检查 CUDA 和 GPU 兼容性"""
    print("\n" + "=" * 60)
    print("步骤6: 检查 CUDA 和 GPU")
    print("=" * 60)

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"CUDA 可用: {cuda_available}")

        if cuda_available:
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 数量: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {gpu_name}")

                # 检查是否是 RTX 5090
                if "5090" in gpu_name:
                    print(f"    ✓ 检测到 RTX 5090")

            # 检查 CUDA 计算能力
            capability = torch.cuda.get_device_capability()
            print(f"CUDA 计算能力: {capability[0]}.{capability[1]}")

            # 检查 bfloat16 支持
            if torch.cuda.is_bf16_supported():
                print("✓ 支持 bfloat16")
            else:
                print("⚠️  不支持 bfloat16")

            print("\n✅ GPU 检查完成")
        else:
            print("⚠️  未检测到 CUDA，将使用 CPU 模式")

    except Exception as e:
        print(f"✗ CUDA 检查失败: {e}")


def main():
    """主函数"""
    print("\n")
    print("=" * 60)
    print("RTX 5090 包兼容性验证脚本")
    print("=" * 60)
    print("\n")

    # 运行所有检查
    check_imports()
    check_package_versions()
    check_trl_api()
    check_datasets_api()
    check_transformers_api()
    check_cuda()

    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)
    print("\n如果发现任何问题，请参考 package_compatibility_report.md")
    print("了解详细的兼容性信息和修复建议。\n")


if __name__ == "__main__":
    main()
