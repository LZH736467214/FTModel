"""
T6 部署阶段：只部署最终 Base 模型
Base 模型：Qwen2.5-1.5B-Instruct（训练后的最终模型）

支持四种操作：
1. merge: 合并 LoRA 权重到 Base 模型
2. serve: 启动 vLLM OpenAI-compatible API 服务
3. test: 测试部署的服务
4. simple: 使用 Flask 启动简易服务（不需要 vLLM）

面试点：
- 为什么要合并 LoRA？部署时减少推理开销
- vLLM 的优势：PagedAttention + Continuous batching
- OpenAI API 兼容：便于集成
- 只部署 Base 模型：小模型部署成本低，推理速度快
"""
import argparse
import json
import os
import sys
import subprocess
import time
import requests
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import MODEL_CONFIG, SFT_CONFIG, GRPO_CONFIG


# ============ 合并 LoRA ============

def merge_lora_weights(
    base_model: str,
    lora_path: str,
    output_path: str,
):
    """
    合并 LoRA 权重到基座模型
    面试点：merge_and_unload 的原理
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("=" * 60)
    print("合并 LoRA 权重")
    print("=" * 60)

    print(f"基座模型: {base_model}")
    print(f"LoRA 路径: {lora_path}")
    print(f"输出路径: {output_path}")

    # 检查 LoRA 是否存在
    if not os.path.exists(lora_path):
        print(f"❌ LoRA adapter 不存在: {lora_path}")
        sys.exit(1)

    # 加载基座模型（不量化，用于合并）
    print("\n加载基座模型...")
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 加载 LoRA
    print("加载 LoRA adapter...")
    model = PeftModel.from_pretrained(base_model_obj, lora_path)

    # 合并权重
    print("合并权重...")
    model = model.merge_and_unload()

    # 保存合并后的模型
    print(f"保存合并后的模型到: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)

    # 保存 tokenizer
    print("保存 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print(f"\n✅ 合并完成！")
    print(f"   合并后模型: {output_path}")

    return output_path


# ============ 启动 vLLM 服务 ============

def start_vllm_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """
    启动 vLLM OpenAI-compatible API 服务
    面试点：vLLM 的高性能推理
    """
    print("=" * 60)
    print("启动 vLLM 服务")
    print("=" * 60)

    print(f"模型路径: {model_path}")
    print(f"服务地址: http://{host}:{port}")

    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        print("   请先运行: python scripts/7_deploy.py --action merge")
        sys.exit(1)

    # 检查是否是 LoRA adapter
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("⚠️  检测到 LoRA adapter")
        print("   vLLM 需要合并后的完整模型")
        print("   请先运行: python scripts/7_deploy.py --action merge")
        sys.exit(1)

    # 构建 vLLM 启动命令
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.9",
    ]

    print(f"\n启动命令: {' '.join(cmd)}")
    print("\n启动服务中...")

    # 启动服务
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    # 等待服务启动
    print("等待服务就绪...")
    for i in range(60):  # 最多等待 60 秒
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=1)
            if response.status_code == 200:
                print(f"\n✅ 服务启动成功！")
                print(f"   API 地址: http://localhost:{port}/v1")
                print(f"   健康检查: http://localhost:{port}/health")
                print(f"\n使用示例:")
                print(f'   curl http://localhost:{port}/v1/chat/completions \\')
                print(f'     -H "Content-Type: application/json" \\')
                print(f'     -d \'{{"model": "{model_path}", "messages": [{{"role": "user", "content": "什么是资产负债率？"}}]}}\'')
                break
        except:
            time.sleep(1)
            print(".", end="", flush=True)
    else:
        print("\n⚠️  服务启动超时，请检查日志")

    # 保持进程运行
    print("\n按 Ctrl+C 停止服务")
    try:
        for line in process.stdout:
            print(line, end="")
    except KeyboardInterrupt:
        print("\n停止服务...")
        process.terminate()

    return process


# ============ 测试服务 ============

def test_server(
    host: str = "localhost",
    port: int = 8000,
    model_name: str = None,
):
    """
    测试部署的服务
    面试点：API 测试的重要性
    """
    print("=" * 60)
    print("测试 vLLM 服务")
    print("=" * 60)

    base_url = f"http://{host}:{port}"

    # 健康检查
    print("\n1. 健康检查...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✓ 服务健康")
        else:
            print(f"   ✗ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ 无法连接服务: {e}")
        return False

    # 获取模型列表
    print("\n2. 获取模型列表...")
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        models = response.json()
        print(f"   可用模型: {[m['id'] for m in models.get('data', [])]}")
        if model_name is None and models.get('data'):
            model_name = models['data'][0]['id']
    except Exception as e:
        print(f"   ✗ 获取模型列表失败: {e}")

    # 测试生成
    print("\n3. 测试生成...")
    test_cases = [
        {
            "prompt": "某公司2023年营收1000万元，同比增长25%，请计算2022年营收。",
            "expected_type": "financial_calculation",
        },
        {
            "prompt": "什么是资产负债率？",
            "expected_type": "concept_qa",
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n   测试 {i}: {case['prompt'][:30]}...")

        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": case["prompt"]}],
                    "max_tokens": 512,
                    "temperature": 0.7,
                },
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # 检查格式
                has_think = "<think>" in content and "</think>" in content
                has_answer = "<answer>" in content and "</answer>" in content

                print(f"   回答: {content[:200]}...")
                print(f"   格式检查: think={has_think}, answer={has_answer}")
            else:
                print(f"   ✗ 请求失败: {response.status_code}")
                print(f"   {response.text}")

        except Exception as e:
            print(f"   ✗ 测试失败: {e}")

    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)

    return True


# ============ 简易推理服务（不需要 vLLM）============

def start_simple_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """
    使用 Flask 启动简易推理服务（不需要 vLLM）
    适用于简单测试
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("❌ Flask 未安装，请运行: pip install flask")
        sys.exit(1)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print("=" * 60)
    print("启动简易推理服务")
    print("=" * 60)

    # 加载模型
    print(f"加载模型: {model_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("✓ 模型加载完成")

    # 创建 Flask 应用
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy"})

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completions():
        data = request.json
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 512)
        temperature = data.get("temperature", 0.7)

        # 格式化输入
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return jsonify({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response
                }
            }]
        })

    print(f"\n启动服务: http://{host}:{port}")
    app.run(host=host, port=port)


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="T6 部署阶段：只部署最终 Base 模型")
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=["merge", "serve", "test", "simple"],
        help="操作类型: merge(合并LoRA), serve(启动vLLM服务), test(测试服务), simple(简易服务)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="模型路径（merge 时为 LoRA 路径，serve 时为合并后模型路径）"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="ckpts/merged_model",
        help="合并后模型输出路径"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务监听地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务监听端口"
    )

    args = parser.parse_args()

    if args.action == "merge":
        # 合并 LoRA
        lora_path = args.model_path or GRPO_CONFIG.output_dir
        merge_lora_weights(
            base_model=MODEL_CONFIG.base_model,
            lora_path=lora_path,
            output_path=args.output_path,
        )

    elif args.action == "serve":
        # 启动 vLLM 服务
        model_path = args.model_path or args.output_path
        start_vllm_server(
            model_path=model_path,
            host=args.host,
            port=args.port,
        )

    elif args.action == "test":
        # 测试服务
        test_server(
            host="localhost",
            port=args.port,
        )

    elif args.action == "simple":
        # 简易服务（不需要 vLLM）
        model_path = args.model_path or GRPO_CONFIG.output_dir
        start_simple_server(
            model_path=model_path,
            host=args.host,
            port=args.port,
        )


if __name__ == "__main__":
    main()
