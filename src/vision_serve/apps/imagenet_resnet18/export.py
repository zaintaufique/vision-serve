"""Export this app's model to ONNX.

Run: python -m vision_serve.apps.imagenet_resnet18.export
"""

from __future__ import annotations

from pathlib import Path

import torch

from vision_serve.apps.imagenet_resnet18.app import MODEL_PATH, load_torch_model


def export() -> Path:
    model = load_torch_model()
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {MODEL_PATH}...")
    onnx_program = torch.onnx.export(
        model,
        dummy_input,
        dynamo=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: torch.export.Dim("batch_size")}},
        opset_version=18,
    )
    onnx_program.save(str(MODEL_PATH), external_data=False)

    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {MODEL_PATH} ({size_mb:.1f} MB)")
    return MODEL_PATH


if __name__ == "__main__":
    export()
