# LLM Benchmarking & Quantization Demos

This project benchmarks the inference performance of large language models (LLMs) across different deployment strategies and optimization techniques.

It includes:
- Local 4-bit quantized models (Mistral-7B-AWQ)
- Hosted LLM API (Cohere Command R+)
- Transformer-based FP16 inference (Qwen2-7B)
- Post-training dynamic quantization with ONNX (DistilBERT)

---

## Features

- Compare inference time of hosted vs local LLMs
- Apply dynamic quantization using Hugging Face Optimum
- Benchmark **inference-only** latency using a consistent prompt
- Compatible with Google Colab and local Python environments

---

## Benchmark Summary (PDF)

[View Benchmark Report](benchmark-summary.pdf)

This report includes:
- A bar chart comparing **clean inference latency**
- Comments on shard loading, offloading, and quantization
- Methodology used in 'benchmark_llms_generation.py' and 'quantize_distilbert_onnx.py'

> Prompt used for all LLMs:  
> '"Explain the concept of model quantization in simple terms."'

---

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/llm-benchmarking-and-quantization.git
cd llm-benchmarking-and-quantization
pip install -r requirements.txt

## License

MIT License — for research, learning, and demonstration purposes.
