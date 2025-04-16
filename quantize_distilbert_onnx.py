
"""
Quantize DistilBERT for sentiment classification using Hugging Face Optimum + ONNX.

Dependencies:
- transformers
- optimum
- onnxruntime

To install:
    pip install transformers optimum onnxruntime
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoOptimizationConfig
from optimum.exporters.onnx import main_export

def main():
    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    onnx_model_path = "onnx_model"
    main_export(
        model_name_or_path=model_id,
        output=onnx_model_path,
        task="text-classification"
    )

    quantized_model_path = "onnx_model_quantized"
    optimization_config = AutoOptimizationConfig()
    quantized_model = ORTModelForSequenceClassification.from_pretrained(
        onnx_model_path,
        export=False,
        optimization_config=optimization_config,
        save_dir=quantized_model_path
    )

    classifier = pipeline("text-classification", model=quantized_model, tokenizer=tokenizer, device=-1)
    result = classifier("This model has been quantized and runs faster!")
    print("🔎 Quantized Model Output:", result)

if __name__ == "__main__":
    main()
