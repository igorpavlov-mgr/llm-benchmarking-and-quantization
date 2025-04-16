# LLM Benchmarking & Quantization Demos

This simple project benchmarks the inference performance of large language models (LLMs) across different deployment strategies and optimization techniques.

It includes:
- Local 4-bit quantized models (Mistral-7B-AWQ)
- Hosted LLM API (Cohere Command R+)
- Transformer-based FP16 inference (Qwen2-7B)
- Post-training dynamic quantization with ONNX (DistilBERT)

# Features

- Compare inference time of hosted vs local LLMs
- Apply dynamic quantization using Hugging Face Optimum
- Benchmark model latency using a consistent prompt
- Compatible with Google Colab and local Python environments

# Benchmark Summary (PDF)

[View Benchmark Report](docs/benchmark-summary.pdf)

This report includes:
- A chart comparing inference latency
- Model-specific comments
- Test methodology and environment

# Installation

Clone the repo and install dependencies:

'''bash
git clone https://github.com/your-username/llm-benchmarking-and-quantization.git
cd llm-benchmarking-and-quantization
pip install -r requirements.txt
'''

# Scripts

# 'benchmark_llms_generation.py'
Benchmarks 3 LLMs:
- Mistral-7B (4-bit AWQ)
- Qwen2-7B (FP16)
- Cohere Command R+ (via API)

Prompts a common question and logs inference latency.

# 'quantize_distilbert_onnx.py'
- Loads DistilBERT classification model (SST-2)
- Converts to ONNX and applies dynamic INT8 quantization
- Tests the quantized model using a simple pipeline

# Requirements

All dependencies are listed in 'requirements.txt', including:
- 'transformers'
- 'torch'
- 'cohere'
- 'optimum'
- 'onnxruntime'

# License

MIT License — for research, learning, and demonstration purposes.
