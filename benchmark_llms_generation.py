
"""
Benchmark Inference Time Across Multiple LLMs:
- Mistral-7B (4-bit AWQ)
- Qwen2-7B (FP16)
- Cohere Command R+ (API)

This version ensures fair benchmarking by separating:
- Model loading time
- Inference-only time
- Optional warm-up runs

Dependencies:
- transformers
- torch
- cohere

To install:
    pip install transformers torch cohere
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import cohere
from getpass import getpass

prompt = "Explain the concept of model quantization in simple terms."
max_tokens = 150

def run_mistral_awq():
    print("=== Mistral-7B (AWQ) ===")
    model_id = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    _ = generator("Warm-up.", max_new_tokens=30, do_sample=False)
    start = time.time()
    output = generator(prompt, max_new_tokens=max_tokens, do_sample=False)
    end = time.time()
    print("Output:\n", output[0]["generated_text"])
    print(f"Inference-only time: {end - start:.2f} seconds\n")

def run_qwen_fp16():
    print("=== Qwen2-7B (FP16) ===")
    model_id = "Qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    _ = generator("Warm-up.", max_new_tokens=30, do_sample=False)
    start = time.time()
    output = generator(prompt, max_new_tokens=max_tokens, do_sample=False)
    end = time.time()
    print("Output:\n", output[0]["generated_text"])
    print(f"Inference-only time: {end - start:.2f} seconds\n")

def run_cohere_command_r(api_key):
    print("=== Cohere Command R+ (Hosted) ===")
    co = cohere.Client(api_key)
    start = time.time()
    response = co.chat(model="command-r-plus", message=prompt)
    end = time.time()
    print("Output:\n", response.text)
    print(f"Inference-only time: {end - start:.2f} seconds\n")

if __name__ == "__main__":
    run_mistral_awq()
    run_qwen_fp16()
    api_key = getpass("Enter your Cohere API key: ")
    run_cohere_command_r(api_key)
