import os, errno
import json

import datetime
import random

import torch
import pandas as pd
import numpy as np
from transformers.models.bigscience176b import BigScience176BLMHeadModel
from transformers import AutoTokenizer

model_name = "/gpfswork/rech/six/uan68tv/model-conversion/main-gs-47400-transformers-sharded"
output_save_folder = "/gpfswork/rech/six/uan68tv/code/bloom-book/prompts"
N_PROMPTS = 10
MAX_LENGTH = 50

def create_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def get_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained("bigscience/tokenizer")
    model = BigScience176BLMHeadModel.from_pretrained(model_name, use_cache=False, low_cpu_mem_usage=True)
    print("successfully loaded model")
    device_map = {
        0:[0, 1, 2, 3, 4, 5, 6, 7, 8],
        1:[9, 10, 11, 12, 13, 14, 15, 16, 17],
        2:[18, 19, 20, 21, 22, 23, 24, 25, 26],
        3:[27, 28, 29, 30, 31, 32, 33, 34, 35],
        4:[36, 37, 38, 39, 40, 41, 42, 43, 44],
        5:[45, 46, 47, 48, 49, 50, 51, 52, 53],
        6:[54, 55, 56, 57, 58, 59, 60, 61, 62],
        7:[63, 64, 65, 66, 67, 68, 69],
    }
    model.parallelize(device_map)
    model.eval()
    return model, tokenizer

def generate_from_text(model, text, tokenizer, max_length=50, greedy=False, output_json=None):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    if greedy:
        greedy_output = model.generate(input_ids.to('cuda:0'), max_length=max_length)
    else:
        greedy_output = model.generate(input_ids.to('cuda:0'), max_length=max_length, do_sample=True, top_k=0)
    if output_json:
        output_json['inputs'].append(text)
        output_json['outputs'].append(tokenizer.decode(greedy_output[0], skip_special_tokens=True) + "\n")
    return output_json

def get_recent_prompts(path_csv, n_prompts=N_PROMPTS):
    # LINK = "https://docs.google.com/spreadsheets/d/1WzPQ0-1CcQ9ZQPQ7g7L8jYMTThJWpLTnFy19U_7o1Jo/export?format=csv"
    data = pd.read_csv(path_csv)
    data = data[data['Timestamp'] > datetime.datetime.today().strftime('%Y-%m-%d')]
    selected_prompts = np.unique(data["Model Prompt"].values)
    random.shuffle(selected_prompts)
    return selected_prompts[:n_prompts].tolist()

def main():
    model, tokenizer = get_model_and_tokenizer(model_name)
    output_json = {"inputs":[], "outputs":[]}
    prompts = get_recent_prompts(N_PROMPTS)
    # TODO - batch wise - debug with 350M
    for prompt in prompts:
        output_json = generate_from_text(model, prompt, tokenizer, output_json=output_json, max_length=MAX_LENGTH)
    
    output_dir = os.path.join(output_save_folder, "prompts-{}".format(datetime.datetime.today().strftime('%Y-%m-%d')))
    create_dir(output_dir)
    with open(os.path.join(output_dir, "json_output.json"), "w") as f:
        json.dump(output_json, f)

if __name__ == "__main__":
    main()