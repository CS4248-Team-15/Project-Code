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
  cache_folder = "LLM_cache"
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

system_prompt = """You are a helpful assistant that converts Onion-style headlines into Huffington Post-style headlines.

Your task is to keep the core meaning of each Onion headline while rewriting it to be more appealing, engaging, and realistic — as if published by the Huffington Post.

You will be given a numbered list of Onion-style headlines, formatted like this:
```
1. Onion-style headline 1
2. Onion-style headline 2
...
```

You should return a numbered list of rewritten Huffington Post-style headlines in the same order:
```
1. Huffington Post-style headline 1
2. Huffington Post-style headline 2
...
```

You should directly output the result without any other text.
"""

history = [
    ("```1. nation's dogs vow to keep barking at nothing until humans understand what's really going on\n...```",
     "```1. why your dog might be barking at 'nothing' — experts say it could be something more\n...```")
]

import json
from tqdm import tqdm

def batchify(lst, n):
    """Split list into batches of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
if __name__ == "__main__":
    data_file = "Sarcasm_Headlines_Dataset.json"
    headlines = []
    with open(data_file, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["is_sarcastic"] == 1:
                headlines.append(data["headline"])

    print("Total sarcastic headlines: ", len(headlines))

    all_pairs = []
    batch_size = 100
    for batch in tqdm(list(batchify(headlines, batch_size)), desc="Processing batches"):
        batch_input = "\n".join([f"{i+1}. {h}" for i, h in enumerate(batch)])
        user_prompt = f"```\n{batch_input}\n```"

        res = turbo_chat_completion(
            system_prompt, 
            history,
            user=user_prompt, 
            engine="gpt-4o", 
            max_length=-1, 
            temp=0.7
        )

        output = res['completion']
        assert output.startswith("```") and output.endswith("```"), "Output should be wrapped in triple backticks."
        output_lines = output[3:-3].strip().split("\n")

        output_texts = [line.split(". ", 1)[1] if ". " in line else line for line in output_lines]
        assert len(output_texts) == len(batch), f"Expected {len(batch)} lines, got {len(output_texts)}"

        pairs = [
            {"onion_style_headline": onion, "huffington_post_style_headline": huff}
            for onion, huff in zip(batch, output_texts)
        ]
        all_pairs.extend(pairs)

    with open("onion2huffington_gen.json", "w") as f:
        json.dump(all_pairs, f, indent=4)

    print(f"Done! {len(all_pairs)} headline pairs saved to onion2huffington_gen.json")
