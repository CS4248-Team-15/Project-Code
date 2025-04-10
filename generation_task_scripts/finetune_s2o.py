
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
correct_index_file = "./correct_index.json"
correct_index = json.load(open(correct_index_file, "r"))
dataset = json.load(open(dataset_file, "r"))
dataset = [{"original": i["huffington_post_style_headline"], "sarcastic": i["onion_style_headline"]} for i in dataset]

dataset = [dataset[i] for i in correct_index if i < len(dataset)]
# dataset = dataset[:5000]
print("[INFO] Dataset size:", len(dataset))
# exit()
# 

def format_input(data):
    data["input_text"] = (
        "### Instruction:\nRewrite the following sarcastic headline in a normal, factual tone.\n"
        f"### Input:\n{data['sarcastic']}\n"
        f"### Response:\n{data['original']}"
    )
    return data

dataset = [format_input(data) for data in dataset]


model_name = "meta-llama/Llama-3.2-1B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# lora_config = LoraConfig(
#     r=32,
#     lora_alpha=64,
#     target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type=TaskType.CAUSAL_LM
# )

# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

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
    output_dir="fine_tuned_llama2-sarcastic_to_original",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_dir="logs",
    save_strategy="steps",
    save_steps=100,
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

model.save_pretrained("fine_tuned_llama2-sarcastic_to_original")
tokenizer.save_pretrained("fine_tuned_llama2-sarcastic_to_original")
