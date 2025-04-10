
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig
import pandas as pd
import json


# Load the dataset
dataset_file = "./pair_data.json"
dataset = json.load(open(dataset_file, "r"))
dataset = [{"original": i["huffington_post_style_headline"], "sarcastic": i["onion_style_headline"]} for i in dataset]
# dataset = dataset[:5000]
print("[INFO] Dataset size:", len(dataset))
# exit()


def format_input(data):
    data["input_text"] = (
        "### Instruction:\nRewrite the following headline sarcastically.\n"
        f"### Input:\n{data['original']}\n"
        f"### Response:\n{data['sarcastic']}"
    )
    return data

dataset = [format_input(data) for data in dataset]

model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

def tokenize(example):
    tokens = tokenizer(
        example["input_text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = Dataset.from_pandas(pd.DataFrame(dataset))
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

train_dataset = train_dataset.map(tokenize, batched=True)
eval_dataset = eval_dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="fine_tuned_llama2-original_to_sarcastic",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_dir="logs",
    save_strategy="epoch",
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("fine_tuned_llama2-original_to_sarcastic")
tokenizer.save_pretrained("fine_tuned_llama2-original_to_sarcastic")
