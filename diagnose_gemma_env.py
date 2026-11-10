#!/usr/bin/env python3
"""Print the underlying PyTorch/CUDA and Transformers import diagnostics."""
from __future__ import annotations
import importlib.metadata as md
import platform
import traceback

def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "NOT INSTALLED"

print("Python:", platform.python_version())
for name in ("torch", "torchvision", "torchaudio", "transformers", "accelerate"):
    print(f"{name}: {version(name)}")

try:
    import torch
    print("torch.__version__:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("BF16 supported:", torch.cuda.is_bf16_supported())
except Exception:
    print("PyTorch/CUDA diagnostic failed:")
    traceback.print_exc()

try:
    import transformers
    print("transformers.__version__:", transformers.__version__)
    from transformers import AutoProcessor
    print("AutoProcessor import: OK")
    try:
        from transformers import AutoModelForImageTextToText
        print("AutoModelForImageTextToText import: OK")
    except Exception:
        print("AutoModelForImageTextToText import failed:")
        traceback.print_exc()
except Exception:
    print("AutoProcessor import failed; underlying traceback follows:")
    traceback.print_exc()
