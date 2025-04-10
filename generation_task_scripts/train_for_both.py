import json
from filtereddata import headline_pairs
# Load the dataset
# dataset_file = "./pair_data.json"
# correct_index_file = "./correct_index.json"
# correct_index = json.load(open(correct_index_file, "r"))
dataset = headline_pairs
dataset = [{"original": i["huffington_post_style_headline"], "sarcastic": i["onion_style_headline"]} for i in dataset]
# print(len(dataset))

# dataset = [dataset[i] for i in correct_index if i < len(dataset)]
# assert len(dataset) == len(correct_index), "Dataset size does not match correct index size"

train_dataset = dataset[:3200]
_valid_dataset = dataset[3200:4800]

for i in range(len(train_dataset)): # randomly set half of the dataset to be 1, and the other half to be 0
    if i % 2 == 0:
        train_dataset[i]["make_sarcastic"] = 1
    else:
        train_dataset[i]["make_sarcastic"] = 0
        
def format_input(data):
    if data["make_sarcastic"] == 1:
        data["input_text"] = (
            "### Instruction:\nRewrite the following headline sarcastically.\n"
            f"### Input:\n{data['original']}\n"
            f"### Response:\n{data['sarcastic']}"
        )
    else:
        data["input_text"] = (
            "### Instruction:\nRewrite the following sarcastic headline in a normal, factual tone.\n"
            f"### Input:\n{data['sarcastic']}\n"
            f"### Response:\n{data['original']}"
        )
    return data

train_dataset = [format_input(data) for data in train_dataset]
eval_dataset = []
for i in range(len(_valid_dataset)):
    tmp = {"original": _valid_dataset[i]["original"], "sarcastic": _valid_dataset[i]["sarcastic"], "make_sarcastic": 1}
    eval_dataset.append(format_input(tmp))
    tmp = {"original": _valid_dataset[i]["original"], "sarcastic": _valid_dataset[i]["sarcastic"], "make_sarcastic": 0}
    eval_dataset.append(format_input(tmp))

# print(train_dataset[0])
# print(eval_dataset[0])
# exit(0)

from datasets import Dataset
import pandas as pd
train_dataset = train_dataset[:3200]
const_train_dataset = Dataset.from_pandas(pd.DataFrame(train_dataset))
const_eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_dataset))
print(len(const_train_dataset))
print(len(const_eval_dataset))

# print(const_train_dataset[0])
# print(const_eval_dataset[0])
# exit(0)


import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
def finetune_llama(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize(example):
        tokens = tokenizer(
            example["input_text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    
    train_dataset = const_train_dataset.map(tokenize, batched=True)
    eval_dataset = const_eval_dataset.map(tokenize, batched=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    [160, 320, 640, 1600, 3200]
    [5, 10, 20, 50, 100]
    training_args = TrainingArguments(
        output_dir=model_name+"_both_full_finetuned",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_dir="logs",
        save_strategy="steps",
        save_steps=5,
        bf16=True,
        logging_steps=1,
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


if __name__ == "__main__":
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    # model_name = "meta-llama/Llama-3.2-3B-Instruct"
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    finetune_llama(model_name)