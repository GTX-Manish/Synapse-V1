"""Export the trained Synapse V1 checkpoint into a Hugging Face-style bundle.

Reads the training checkpoint (``checkpoints/best_model.pt``) and writes a clean,
framework-agnostic artifact set that can be published to the Hugging Face Hub:

    config.json         architecture hyperparameters
    model.safetensors   model weights in SafeTensors format
    tokenizer.json      character-level tokenizer mappings
    model.py            copy of the model definition (so the loader below works)

Note: Synapse V1 is a plain ``torch.nn.Module`` built from scratch — it is NOT a
``transformers.PreTrainedModel``. The bundle is therefore loaded with the small helper
shown in the repository README / model card, not ``AutoModel.from_pretrained``.

Usage:
    python scripts/export_hf.py --checkpoint checkpoints/best_model.pt --out hf_export
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


CONFIG_KEYS = [
    "vocab_size",
    "max_sequence_length",
    "d_model",
    "num_heads",
    "d_ff",
    "num_layers",
]


def export(checkpoint_path: Path, out_dir: Path, model_source: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) config.json — architecture description (no transformers auto-loading keys).
    config = {key: checkpoint[key] for key in CONFIG_KEYS}
    config.update(
        {
            "model_name": "Synapse-V1",
            "architecture": "decoder-only transformer (built from scratch)",
            "activation": "relu",
            "normalization": "pre-ln (pre-layernorm)",
            "positional_encoding": "learned",
            "tie_word_embeddings": False,
            "dtype": "float32",
        }
    )
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    # 2) model.safetensors — contiguous, CPU, float32 weights.
    state_dict = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in checkpoint["model_state_dict"].items()
    }
    save_file(
        state_dict,
        str(out_dir / "model.safetensors"),
        metadata={"format": "pt", "model_name": "Synapse-V1"},
    )

    # 3) tokenizer.json — character-level tokenizer mappings.
    tokenizer = {
        "type": "character-level",
        "vocab_size": checkpoint["vocab_size"],
        "char_to_id": checkpoint["char_to_id"],
        "id_to_char": {str(k): v for k, v in checkpoint["id_to_char"].items()},
    }
    (out_dir / "tokenizer.json").write_text(
        json.dumps(tokenizer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 4) model.py — copy the definition so the documented loader works.
    shutil.copyfile(model_source, out_dir / "model.py")

    num_params = sum(t.numel() for t in state_dict.values())
    print(f"Exported bundle to {out_dir}/")
    print(f"  config.json        ({len(config)} keys)")
    print(f"  model.safetensors  ({num_params:,} parameters)")
    print(f"  tokenizer.json     (vocab_size={checkpoint['vocab_size']})")
    print(f"  model.py           (copied from {model_source})")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "checkpoints" / "best_model.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "hf_export",
    )
    parser.add_argument(
        "--model-source",
        type=Path,
        default=repo_root / "model.py",
    )
    args = parser.parse_args()

    export(args.checkpoint, args.out, args.model_source)


if __name__ == "__main__":
    main()
