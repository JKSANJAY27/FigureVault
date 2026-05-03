"""
training/finetune.py — Gemma4 E4B fine-tuning script

Uses Unsloth + FastVisionModel to fine-tune Gemma4 E4B on figure+data pairs.
Designed to run on Kaggle T4 GPU or Google Colab.

Usage:
  python training/finetune.py --data-dir training_data/ --output-dir models/finetuned/

NOTE: This script requires GPU and the Unsloth package.
Install on Kaggle/Colab with:
  pip install unsloth[colab-new]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builder for fine-tuning instruction format
# ---------------------------------------------------------------------------

INSTRUCTION_TEMPLATE = """You are a scientific data extraction expert.
Extract ALL numerical data from the provided scientific figure image.
Output a JSON array of data series objects.

Figure Caption: {caption}

Output:"""


def build_finetuning_sample(
    figure_path: str,
    caption: str,
    ground_truth_json: str,
) -> dict:
    """Build one training sample in the Unsloth conversation format.

    Parameters
    ----------
    figure_path : str
        Path to the figure PNG (will be loaded as PIL image).
    caption : str
        Figure caption text.
    ground_truth_json : str
        JSON string of the expected extraction output.

    Returns
    -------
    dict
        Conversation dict compatible with FastVisionModel.get_inputs().
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": figure_path},
                    {"type": "text", "text": INSTRUCTION_TEMPLATE.format(caption=caption)},
                ],
            },
            {
                "role": "assistant",
                "content": ground_truth_json,
            },
        ]
    }


class FigureVaultFinetuner:
    """Fine-tune Gemma4 E4B on figure data extraction pairs.

    Parameters
    ----------
    data_dir : Path
        Directory containing training_data/ with PNG + JSON pairs.
    output_dir : Path
        Where to save the fine-tuned LoRA adapters.
    model_name : str
        HuggingFace model ID for Gemma4.
    max_seq_length : int
        Maximum token sequence length for training.
    lora_rank : int
        LoRA rank (higher = more parameters, more capacity).
    batch_size : int
        Training batch size per GPU.
    epochs : int
        Number of training epochs.
    learning_rate : float
        AdamW learning rate.
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        model_name: str = "unsloth/gemma-3-4b-it",
        max_seq_length: int = 2048,
        lora_rank: int = 16,
        batch_size: int = 2,
        epochs: int = 3,
        learning_rate: float = 2e-4,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.lora_rank = lora_rank
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def load_dataset(self) -> list[dict]:
        """Load all training pairs from data_dir.

        Expects files named ``<stem>.png`` and ``<stem>_meta.json`` where
        the JSON has at least a ``caption`` and ``ground_truth`` key.

        Returns
        -------
        list[dict]
            List of conversation dicts ready for Unsloth training.
        """
        samples: list[dict] = []
        meta_files = list(self.data_dir.rglob("*_meta.json"))
        logger.info("Loading %d training samples from %s", len(meta_files), self.data_dir)

        for meta_file in meta_files:
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                fig_path = meta_file.with_suffix("").with_name(
                    meta_file.stem.replace("_meta", "") + ".png"
                )
                if not fig_path.exists():
                    continue
                caption = meta.get("caption", "")
                gt = meta.get("ground_truth", "[]")
                samples.append(build_finetuning_sample(str(fig_path), caption, gt))
            except Exception as exc:
                logger.warning("Skipping %s: %s", meta_file, exc)

        logger.info("Loaded %d valid training samples", len(samples))
        return samples

    def train(self) -> None:
        """Run the full fine-tuning loop.

        Requires: unsloth, torch, transformers, trl
        Install with: pip install unsloth[colab-new]
        """
        try:
            from unsloth import FastVisionModel
            import torch
        except ImportError:
            raise ImportError(
                "Unsloth is not installed. Run:\n"
                "  pip install unsloth[colab-new]\n"
                "This script requires a GPU environment (Kaggle / Colab)."
            )

        logger.info("Loading model: %s", self.model_name)
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )

        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            lora_dropout=0,
            bias="none",
            random_state=42,
        )

        samples = self.load_dataset()
        if not samples:
            raise ValueError("No training samples found — run data_collector.py first")

        # Convert to HuggingFace Dataset format
        from datasets import Dataset
        dataset = Dataset.from_list(samples)

        from trl import SFTTrainer, SFTConfig
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                per_device_train_batch_size=self.batch_size,
                gradient_accumulation_steps=4,
                warmup_ratio=0.1,
                num_train_epochs=self.epochs,
                learning_rate=self.learning_rate,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=10,
                optim="adamw_8bit",
                output_dir=str(self.output_dir / "checkpoints"),
                report_to="none",
            ),
        )

        logger.info("Starting fine-tuning for %d epochs", self.epochs)
        trainer.train()

        # Save LoRA adapters
        adapter_path = self.output_dir / "lora_adapters"
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))
        logger.info("Saved LoRA adapters to %s", adapter_path)

        # Optionally save merged model
        merged_path = self.output_dir / "merged_model"
        model.save_pretrained_merged(str(merged_path), tokenizer, save_method="merged_16bit")
        logger.info("Saved merged model to %s", merged_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma4 E4B on FigureVault data")
    parser.add_argument("--data-dir", default="training_data", help="Training data directory")
    parser.add_argument("--output-dir", default="models/finetuned", help="Output directory for adapters")
    parser.add_argument("--model", default="unsloth/gemma-3-4b-it", help="Base model name")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    finetuner = FigureVaultFinetuner(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        learning_rate=args.lr,
    )
    finetuner.train()


if __name__ == "__main__":
    main()
