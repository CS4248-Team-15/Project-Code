import openai
from openai import OpenAI
import os
import timeit
import sys
import json
openai.api_key_path = "./src/openaikey/openai.key"
os.environ["OPENAI_API_KEY"]= ""
client = OpenAI()

def chat_complete(messages, max_length, stop, temp=0, stream=False, engine="gpt-4o"):

  try:
    time1 = timeit.default_timer()
    res = client.chat.completions.create(
      model=engine,
      messages=messages,
      max_tokens=max_length,
      temperature=temp,
      stop=stop,
      stream=stream,
      n=1
    )
    time2 = timeit.default_timer()
    timespan = time2 - time1
    # print(res, "?")
    result = {
      "completion_id": res.id,
      "completion_content": res.choices[0].message.content,
      "completion_model": res.model,
      "completion_usage": {"completion_tokens": res.usage.completion_tokens, "prompt_tokens": res.usage.prompt_tokens},
      "timespan": timespan,
      "error_msg": None
    }
  except Exception as e:
    # print("Exception in complete_prompt: ", e, file=sys.stderr)
    # print(e, "???")
    result = {
      "error_msg": str(e)
    }
  return result


import time
import hashlib
import os
_FILECACHE_ENABLED = True
def file_cached(func_id, force_new=False, interval=500, valid_f=None, max_retry=10, retry_f=None):
  cache_folder = "LLM_cache_new"
  if not os.path.exists(cache_folder):
    os.mkdir(cache_folder)
  collision_filepath = cache_folder + "/__collisions.jsonl"
  collision_f = open(collision_filepath, "a")
  def _cache_get(store_key):
    # check if <cache_folder>/<store_key>.json exists
    # if exists, return json.load(<cache_folder>/<store_key>.json)
    cache_path = cache_folder + "/" + store_key + ".json"
    if not os.path.exists(cache_path): raise Exception("Cache file not found: " + cache_path)

    ### copy it to "./LLM_cache" folder
    # os.system(f"cp {cache_path} ./LLM_cache/")

    with open(cache_path, "r") as f:
      return json.load(f)
  def _cache_set(store_key, data):
    # write json.dump(data) to <cache_folder>/<store_key>.json
    cache_path = cache_folder + "/" + store_key + ".json"
    with open(cache_path, "w") as f:
      json.dump(data, f)
    ### copy it to "./LLM_cache" folder
    # os.system(f"cp {cache_path} ./LLM_cache/")
  def _file_cached_deco(func):
    global _FILECACHE_ENABLED, _FILECACHE_COLLISION_DICT
    def wrapper(**kwargs):
      params = kwargs
      param_key = json.dumps(params)
      # get sha256 of param_key
      hash = hashlib.sha256(param_key.encode('utf-8')).hexdigest()
      store_key = f"APICacheV1-{func_id}-" + hash
      failure_key = f"APIFailV1-{func_id}-" + hash
      ret_val = None
      if _FILECACHE_ENABLED and not force_new:
        try:
          result = _cache_get(store_key)
          existing_param_key = json.dumps(result["params"])
          if existing_param_key != param_key:
            to_append = {"fetching": param_key, "existing": existing_param_key}
            collision_f.write(json.dumps(to_append) + "\n", flush=True)
            raise Exception("COLLISION!!! param_key mismatch! store_key=" + store_key)
          ret_val = result["ret_val"]
          if valid_f is not None and not valid_f(ret_val):
            ret_val = None
            raise Exception("INVALID!!! cached result failed validation! store_key=" + store_key)
        except Exception as e:
          # print("[logviz_cached]", e)
          pass
      if ret_val is None:
        retry_count = 0
        while True:
          ret_val = func(**kwargs)
          if valid_f is None:
            break
          if valid_f(ret_val):
            break
          if retry_f is not None and retry_f(ret_val):
            retry_count += 1
            if retry_count >= max_retry:
              failure_msg = "Validation failed after max_retry | " + failure_key
              _cache_set(failure_key, {"_failure_msg": failure_msg, "func_id": func_id, "params": params, "ret_val": ret_val})
              raise Exception(failure_msg)
            time.sleep(interval)
            print("[logviz_cached] Validation failed and retry allowed. Retrying ...")
          else:
            print(ret_val)
            failure_msg = "Not valid & not retryable | " + failure_key
            _cache_set(failure_key, {"_failure_msg": failure_msg, "func_id": func_id, "params": params, "ret_val": ret_val})
            raise Exception(failure_msg)
        if _FILECACHE_ENABLED:
          try:
            _cache_set(store_key, {"func_id": func_id, "params": params, "ret_val": ret_val})
          except Exception as e:
            print("[logviz_cached]", e)
      return ret_val
    return wrapper
  return _file_cached_deco

def _complete_prompt_cached_retry_f(ret_val):
  if ret_val is not None:
    if "error_msg" in ret_val:
      error_msg = ret_val["error_msg"]
      if error_msg is not None:
        if "That model is currently overloaded with other requests." in error_msg:
          return True
  return False

@file_cached("chatgptcomp", interval=10, valid_f=lambda x: "completion_id" in x, retry_f=_complete_prompt_cached_retry_f, max_retry=10)
def complete_prompt_cached(**kwargs):
  return chat_complete(**kwargs)


def turbo_chat_completion(system, chat_history, user, engine, max_length=-1, temp=0):

    gpt_msg = []
    gpt_msg.append({"role": "system", "content": system})
    for item in chat_history: assert len(item) == 2, "chat_history should be a list of (user, assistant) tuples. Get: " + str(item)
    for hist_user, hist_assistant in chat_history:
      gpt_msg.append({"role": "user", "content": hist_user})
      gpt_msg.append({"role": "assistant", "content": hist_assistant})
    gpt_msg.append({"role": "user", "content": user})

    max_length = 8192 # 32k context length

    result = complete_prompt_cached(messages=gpt_msg, max_length=max_length, stop="", temp=temp, stream=False, engine=engine)
    
    if "error_msg" in result and result["error_msg"] is not None:
      return result

    return {
      "completion": result["completion_content"],
      "usage": result["completion_usage"],
      "timespan": result["timespan"],
      "prompt": gpt_msg,
    }

system_prompt_1 = """You are a helpful assistant that judge the the performance of two language models on the task of rewriting Onion-style headlines into Huffington Post-style headlines.

You will be given one input headline in Onion-style and two output headlines in Huffington Post-style from two different models. Your task is to determine which model performed better in this task.

Input headline:
```
Onion-style headline
```

Output headline from model [A]:
```
Huffington Post-style headline
```

Output headline from model [B]:
```
Huffington Post-style headline
```


You should return [A] or [B] depending on which model performed better.
```
[A]
```

You should directly output the result without any other text.
You should always pick [A] or [B], and never say "I don't know" or "I can't tell".
"""

system_prompt_2 = """You are a helpful assistant that judge the the performance of two language models on the task of rewriting Huffington Post-style headlines into Onion-style headlines.

You will be given one input headline in Huffington Post-style and two output headlines in Onion-style from two different models. Your task is to determine which model performed better in this task.

Input headline:
```
Huffington Post-style headline
```

Output headline from model [A]:
```
Onion-style headline
```

Output headline from model [B]:
```
Onion-style headline
```

You should return [A] or [B] depending on which model performed better.

You should output:
```
[A]
```

or 

``` 
[B]
```

You should directly output the result without any other text.
You should always pick [A] or [B], and never say "I don't know" or "I can't tell".
"""

history = [
    # ("```1. nation's dogs vow to keep barking at nothing until humans understand what's really going on\n...```",
    #  "```1. why your dog might be barking at 'nothing' — experts say it could be something more\n...```")
]


input_template = lambda input, o1, o2: f"""
Input headline:
```
{input}
```

Output headline from model [A]:
```
{o1}
```

Output headline from model [B]:
```
{o2}
```

Please output your judgement:
"""

import json
from tqdm import tqdm

input_headlines_original = [
    ## normal to onion
    "biden announces new plan to lower prescription drug prices by 2026",
    "climate report shows 2024 was fifth hottest year on record globally",
    "supreme court hears arguments in landmark social media free speech case",
    "new study finds link between daily walking and improved mental health",
    "tech companies face renewed scrutiny over use of ai in hiring practices",
    
    ## onion to normal
    "nation unites in hope that someone else will fix everything",
    "local man heroically refreshes email 78 times instead of starting work",
    "study finds 4 out of 5 americans just guess their password every time",
    "scientists warn earth now 3 bad days away from total collapse",
    "new app helps users optimize daily schedule of existential dread",
]


import random
if __name__ == "__main__":
    output_headlines_file = "output_headlines.json"
    output_headlines = json.load(open(output_headlines_file, "r"))
    # print(len(output_headlines.keys()))
    scores = {model_name: 0 for model_name in output_headlines}
    win_log = {model_name: {model_name_2:0 for model_name_2 in output_headlines} for model_name in output_headlines}
    compared_set = set()
    total_compare = 0
    for model_name_1 in output_headlines:
        compared_set.add(model_name_1)
        for model_name_2 in output_headlines:
            if model_name_2 in compared_set:
                continue
            
            print(f"[INFO] ##### Comparing {model_name_1} and {model_name_2} #####")
            
            model_1_output = output_headlines[model_name_1]
            model_2_output = output_headlines[model_name_2]
            
            assert len(model_1_output) == len(model_2_output) == 10, "Output lengths should match"
            
            for i in range(len(model_1_output)):
                print(f"[INFO] Comparing for input {i}")
                input_headline = input_headlines_original[i]
                
                # ramdomly sample one output for each model
                model_1_output_headline = random.choice(model_1_output[i])
                model_2_output_headline = random.choice(model_2_output[i])
                
                assert type(model_1_output_headline) == str and type(model_2_output_headline) == str, "Output should be a string"
                
                if i <= 4: # normal to onion
                    system_prompt = system_prompt_2
                else: # onion to normal
                    system_prompt = system_prompt_1
                
                exchange = random.randint(0, 1)
                if exchange == 0:
                    user_prompt = input_template(input_headline, model_1_output_headline, model_2_output_headline)
                else:
                    user_prompt = input_template(input_headline, model_2_output_headline, model_1_output_headline)
                
                # print(f"[INFO] system {system_prompt}")
                # print(f"[INFO] User prompt: {user_prompt}")
                # exit(0)
                # print(f"[INFO] System prompt: {system_prompt}")
                res = turbo_chat_completion(
                    system_prompt, 
                    [],
                    user=user_prompt, 
                    engine="gpt-4o", 
                    max_length=1024, 
                    temp=0
                )
                
                output_lines = res['completion']
                # print(f"[INFO] Output: {output_lines}")
                if output_lines.startswith("```") and output_lines.endswith("```"):
                    output_lines = output_lines[3:-3]
                
                output_lines = output_lines.strip()

                if exchange == 0:
                    if output_lines == "[A]":
                        scores[model_name_1] += 1
                        win_log[model_name_1][model_name_2] += 1
                        win_log[model_name_2][model_name_1] -= 1
                    else:
                        scores[model_name_2] += 1
                        win_log[model_name_2][model_name_1] += 1
                        win_log[model_name_1][model_name_2] -= 1
                else:
                    if output_lines == "[B]":
                        scores[model_name_1] += 1
                        win_log[model_name_1][model_name_2] += 1
                        win_log[model_name_2][model_name_1] -= 1
                    else:
                        scores[model_name_2] += 1
                        win_log[model_name_2][model_name_1] += 1
                        win_log[model_name_1][model_name_2] -= 1
                    
            total_compare += 1
            
    
    
    
    print(total_compare)
    print(scores)
    print(win_log)

    json.dump(scores, open("scores.json", "w"))
    json.dump(win_log, open("win_log.json", "w"))
