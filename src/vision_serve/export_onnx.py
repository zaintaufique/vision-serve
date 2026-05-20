"""Export pretrained ResNet18 to ONNX format.

Run with: uv run python -m vision_serve.export_onnx
"""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights, resnet18

# Where to write the ONNX file (relative to the project root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = PROJECT_ROOT / "models" / "resnet18.onnx"


def export() -> Path:
    """Load pretrained ResNet18 and export to ONNX as a single file.

    ResNet18 (~45 MB) is well under the protobuf 2 GB limit, so we save as
    a single self-contained file rather than the multi-file external-data
    format. Single-file deploys are simpler: one S3 object, one Docker
    layer, no risk of missing sidecar weights. For models that approach
    or exceed 2 GB, external-data becomes mandatory.
    """
    print("Loading pretrained ResNet18...")
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {OUTPUT_PATH}...")
    onnx_program = torch.onnx.export(
        model,
        dummy_input,
        dynamo=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_shapes={
            "x": {0: torch.export.Dim("batch_size")},
        },
        opset_version=18,
    )
    # Save as single file (external_data=False).
    onnx_program.save(str(OUTPUT_PATH), external_data=False)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUTPUT_PATH} ({size_mb:.1f} MB)")
    return OUTPUT_PATH


if __name__ == "__main__":
    export()
