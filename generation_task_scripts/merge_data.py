data_file1 = "huffington2onion_gen.json"
data_file2 = "onion2huffington_gen.json"

import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def tolowercase(data):
    for item in data:
        item['huffington_post_style_headline'] = item['huffington_post_style_headline'].lower()
        item['onion_style_headline'] = item['onion_style_headline'].lower()
    return data

def merge_data(data1, data2):
    merged_data = []
    data1 = tolowercase(data1)
    data2 = tolowercase(data2)
    
    for item in data1:
        merged_data.append(item)
    
    for item in data2:
        merged_data.append(item)
    return merged_data

data1 = load_json(data_file1)
data2 = load_json(data_file2)

merged_data = merge_data(data1, data2)
print(f"Merged data length: {len(merged_data)}")
json.dump(merged_data, open("pair_data.json", "w"), indent=4)
