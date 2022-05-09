import torch
from transformers.models.bigscience176b import BigScience176BLMHeadModel
from transformers import AutoTokenizer, AutoConfig, logging

logging.set_verbosity_debug()

# , logging

# logging.set_verbosity_debug()

# model_name = "/gpfswork/rech/six/uan68tv/model-conversion/tr11e-350M-transformers-sharded"
model_name = "/gpfswork/rech/six/uan68tv/model-conversion/main-gs-47400-transformers-sharded"

config = AutoConfig.from_pretrained(model_name)

model = BigScience176BLMHeadModel.from_pretrained(model_name, use_cache=False, low_cpu_mem_usage=True)

print("successfully loaded model")

input_ids = [[132619,   3478,    368, 109586,  35433,      2,   2175,  23714,  73173, 144252,	 2,   2175,  23714,  73173, 144252,	 2,     77, 132619, 3478,    368]]
input_ids_tensor = torch.LongTensor(input_ids).to("cuda:0")

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
model = model.cuda()
model.eval()

print("successfully parallelized model")

ATTN_MASK = torch.triu(torch.ones(1, 1, 20, 20), diagonal=1).to("cuda:0").to(model.dtype)

input_tensor = torch.LongTensor(EXAMPLE_IDS).to("cuda:0")

logits = model(input_tensor, attention_mask=ATTN_MASK).logits

# tokenizer = AutoTokenizer.from_pretrained("bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles")
# tokenizer = AutoTokenizer.from_pretrained("bigscience/tokenizer")

# input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='pt')

# # generate text until the output length (which includes the context length) reaches 50
# i = 0
# MAX_LEN=50
# while i < MAX_LEN:
#     seq_length = len(input_ids)
#     attention_mask = torch.tril(torch.ones(
#         (1, seq_length, seq_length), device=model.device)).view(
#             1, 1, seq_length, seq_length)
#     logits = model(input_ids, attention_mask=attention_mask)['logits']
#     next_token = logits.argmax(dim=-1).item()
#     input_ids = torch.cat((input_ids, torch.tensor([next_token], device=model.device)))
#     i = i + 1

# greedy_output = model.generate(input_ids.to('cuda:0'), max_length=50)


# print("Output:\n" + 100 * '-')
# print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

