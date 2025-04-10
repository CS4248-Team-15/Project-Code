import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel




input_headlines_original = [
    "biden announces new plan to lower prescription drug prices by 2026",
    "climate report shows 2024 was fifth hottest year on record globally",
    "supreme court hears arguments in landmark social media free speech case",
    "new study finds link between daily walking and improved mental health",
    "tech companies face renewed scrutiny over use of ai in hiring practices",
]

input_headlines_sarcastic = [
    "nation unites in hope that someone else will fix everything",
    "local man heroically refreshes email 78 times instead of starting work",
    "study finds 4 out of 5 americans just guess their password every time",
    "scientists warn earth now 3 bad days away from total collapse",
    "new app helps users optimize daily schedule of existential dread",
]


output_headlines = {}

def get_output_from_model(base_model, adapter_path):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    if adapter_path != None:
        model = PeftModel.from_pretrained(model, base_model + "_both_full_finetuned/" + adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model + "_both_full_finetuned/" + adapter_path, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        
        
    model.eval()

    def generate_sarcastic_headline(prompt, max_new_tokens=50):

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.9,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        result = tokenizer.decode(output[0], skip_special_tokens=True)
        response = result.split("### Response:")[-1].strip()
        return response

    if adapter_path == None:
        adapter_path = "base"
    output_headlines[base_model + "-" + adapter_path] = []
    for i in input_headlines_original:
        prompt = (
                "### Instruction:\nRewrite the following headline sarcastically.\n"
                f"### Input:\n{i}\n"
                f"### Response:\n"
            )
        tmp = []
        for _ in range(5): # resample 5 times
            res = generate_sarcastic_headline(prompt)
            tmp.append(res)
        
        output_headlines[base_model + "-" + adapter_path].append(tmp)
        
    
    for i in input_headlines_sarcastic:
        prompt = (
                "### Instruction:\nRewrite the following sarcastic headline in a normal, factual tone.\n"
                f"### Input:\n{i}\n"
                f"### Response:\n"
            )
        tmp = []
        for _ in range(5):
            res = generate_sarcastic_headline(prompt)
            tmp.append(res)
        
        output_headlines[base_model + "-" + adapter_path].append(tmp)


if __name__ == "__main__":
    # base_model = "meta-llama/Llama-3.2-1B-Instruct"
    # adapter_path = "lora-llama2-sarcasm"
    # get_output_from_model(base_model, adapter_path)
    
    base_model = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
    ]
    adapter_path = [
        None,
        "checkpoint-5",
        "checkpoint-10",
        "checkpoint-20",
        "checkpoint-50",
        "checkpoint-100",
    ]
    for i in range(len(base_model)):
        print(base_model[i])
        for j in range(len(adapter_path)):
            get_output_from_model(base_model[i], adapter_path[j])
        
    
    print(output_headlines)
    
    import json
    json.dump(output_headlines, open("output_headlines.json", "w"), indent=4)
