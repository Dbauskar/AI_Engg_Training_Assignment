"""Download the ONNX models into ./models (idempotent). Usage: uv run python scripts/fetch_models.py [name ...]"""
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = {  # local dir -> (hf repo, files)
    "bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5", ["onnx/model.onnx", "tokenizer.json", "config.json"]),
    "ms-marco-MiniLM-L-6-v2": ("cross-encoder/ms-marco-MiniLM-L-6-v2", ["onnx/model.onnx", "tokenizer.json", "config.json"]),
    "nli-deberta-v3-xsmall": ("cross-encoder/nli-deberta-v3-xsmall", ["onnx/model.onnx", "tokenizer.json", "config.json"]),
    # a DIFFERENT NLI model, used only by the offline evaluation, so faithfulness is not graded by the model that verified the answers
    "nli-MiniLM2-L6-H768": ("cross-encoder/nli-MiniLM2-L6-H768", ["onnx/model.onnx", "tokenizer.json", "config.json"]),
}

if __name__ == "__main__":
    for name in sys.argv[1:] or MODELS:
        repo, files = MODELS[name]
        for f in files:
            print(hf_hub_download(repo, f, local_dir=ROOT / "models" / name))
