"""

Module 6: Train QA Models
Fine-tune a QA model on BioASQ dataset using Hugging Face Transformers

"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator
)
import argparse

# -------------------------
# Load data
# -------------------------
def load_bioasq(path):
    """Load BioASQ dataset from JSON file """
    with open(path) as f:
        raw_data = json.load(f)

    samples = []
    for ex in raw_data:
        question = ex["question"].strip()
        context = ex["context"].strip()
        answer_text = ex["answer"].strip()

        answer_start = context.find(answer_text)
        if answer_start == -1:
            continue

        samples.append({
            "id": ex.get("id", ""),
            "question": question,
            "context": context,
            "answers": {
                "text": [answer_text],
                "answer_start": [answer_start]
            }
        })
    return Dataset.from_list(samples)

# -------------------------
# Preprocess function
# -------------------------
def preprocess_function(examples, tokenizer):
    """Tokenize and align labels for QA
    
    Args:
        examples: Batch of examples from dataset
        tokenizer: Hugging Face tokenizer
    Returns:        Tokenized examples with start and end positions
    
    """
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        answer_start = answers["answer_start"][0]
        answer_text = answers["text"][0]

        start_pos = 0
        end_pos = 0

        if answer_start == -1:
            start_pos = end_pos = cls_index
        else:
            for idx, (start, end) in enumerate(offsets):
                if sequence_ids[idx] != 1:
                    continue
                if start <= answer_start < end:
                    start_pos = idx
                if start < answer_start + len(answer_text) <= end:
                    end_pos = idx
                    break
            if start_pos == 0 and end_pos == 0:
                start_pos = end_pos = cls_index

        start_positions.append(start_pos)
        end_positions.append(end_pos)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions

    return tokenized_examples

# -------------------------
# Main training function
# -------------------------
def train(model_name, output_dir, train_file):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    dataset = load_bioasq(train_file)
    print(f" Loaded {len(dataset)} examples from {train_file}")

    tokenized = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=dataset.column_names)

    args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=1,
        fp16=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=default_data_collator
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f" Model saved to {output_dir}")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., deepset/roberta-base-squad2)")
    parser.add_argument("--output", type=str, required=True, help="Path to save fine-tuned model")
    parser.add_argument("--train_file", type=str, default="data/processed/bioasq_train.json")
    args = parser.parse_args()

    train(args.model, args.output, args.train_file)
