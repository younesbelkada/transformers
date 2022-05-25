import os, errno

from transformers.models.bigscience176b import BigScience176BLMHeadModel
from transformers import AutoTokenizer

model_name = "/gpfsssd/worksf/projects/rech/six/uan68tv/model-conversion/bloom"
prompts = [
    "What is the name of the planet that the Sun is orbiting?",
    "What is the name of the planet that the Earth is orbiting?",
]

# Batch wise generation not supported yet - waiting for the PR to be merged
MAX_LENGTH = 100

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

def generate_from_text(model, text, tokenizer, max_length=50, greedy=False):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    greedy_output = model.generate(input_ids.to('cuda:0'), max_length=max_length, do_sample=greedy, top_k=0)
    return greedy_output

def main():
    model, tokenizer = get_model_and_tokenizer(model_name)

    for prompt in prompts:
        output = generate_from_text(model, prompt, tokenizer, max_length=MAX_LENGTH, greedy=False)
        print(tokenizer.decode(output, skip_special_tokens=True))

if __name__ == "__main__":
    main()