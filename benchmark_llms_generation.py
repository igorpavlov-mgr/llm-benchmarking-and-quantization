
"""
Benchmark Inference Time Across Multiple LLMs:
- Mistral-7B (4-bit AWQ)
- Qwen2-7B (FP16)
- Cohere Command R+ (API)

Dependencies:
- transformers
- torch
- cohere

To install:
    pip install transformers torch cohere
"""

import time
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import torch
import cohere

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
    start = time.time()
    output = generator(prompt, max_new_tokens=max_tokens, do_sample=False)
    end = time.time()
    print("Output:\n", output[0]["generated_text"])
    print(f"\nInference Time: {end - start:.2f} seconds\n")

def run_qwen_fp16():
    print("=== Qwen2-7B (FP16) ===")
    model_id = "Qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    start = time.time()
    output = generator(prompt, max_new_tokens=max_tokens, do_sample=False)
    end = time.time()
    print("Output:\n", output[0]["generated_text"])
    print(f"\nInference Time: {end - start:.2f} seconds\n")

def run_cohere_command_r(api_key):
    print("=== Cohere Command R+ (Hosted) ===")
    co = cohere.Client(api_key)
    start = time.time()
    response = co.chat(model="command-r-plus", message=prompt)
    end = time.time()
    print("Output:\n", response.text)
    print(f"\nInference Time: {end - start:.2f} seconds\n")

if __name__ == "__main__":
    run_mistral_awq()
    run_qwen_fp16()
    api_key = input("Enter your Cohere API key: ")
    run_cohere_command_r(api_key)
