#!/usr/bin/env python
"""LoRA fine-tuning script for structured recommendation explanation tasks."""

import argparse
import json
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class Seq2SeqExample:
    prompt: str
    target: str


class JsonlSeq2SeqDataset(Dataset):
    def __init__(self, path: str):
        self.examples: List[Seq2SeqExample] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_num}: {exc}") from exc
                if "input" not in record or "output" not in record:
                    raise ValueError(
                        f"Line {line_num} must contain 'input' and 'output' fields"
                    )
                prompt = str(record["input"]).strip()
                target = str(record["output"]).strip()
                self.examples.append(Seq2SeqExample(prompt=prompt, target=target))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Seq2SeqExample:
        return self.examples[idx]


class Seq2SeqDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Seq2SeqExample]) -> Dict[str, torch.Tensor]:
        prompts = [example.prompt for example in batch]
        targets = [example.target for example in batch]

        prompt_encodings = self.tokenizer(
            prompts,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        full_texts = [
            f"{prompt}\n{target}" if prompt else target
            for prompt, target in zip(prompts, targets)
        ]
        full_encodings = self.tokenizer(
            full_texts,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = []
        labels = []
        attention_mask = []

        for prompt_ids, full_ids, full_mask in zip(
            prompt_encodings["input_ids"],
            full_encodings["input_ids"],
            full_encodings["attention_mask"],
        ):
            prompt_len = min(len(prompt_ids), len(full_ids))
            label_ids = [-100] * prompt_len + full_ids[prompt_len:]
            label_ids = label_ids[: len(full_ids)]
            input_ids.append(full_ids)
            labels.append(label_ids)
            attention_mask.append(full_mask)

        batch_inputs = self.tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_labels = self.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"]
        batch_labels[batch_labels == self.tokenizer.pad_token_id] = -100
        batch_inputs["labels"] = batch_labels
        return batch_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for explanations")
    parser.add_argument("--base_model", required=True, help="Base model name/path")
    parser.add_argument("--train_file", required=True, help="Path to JSONL train data")
    parser.add_argument("--output_dir", required=True, help="Output directory for adapter")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,v_proj",
        help="Comma-separated list of target modules for LoRA",
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = JsonlSeq2SeqDataset(args.train_file)
    data_collator = Seq2SeqDataCollator(tokenizer, max_length=args.max_length)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
