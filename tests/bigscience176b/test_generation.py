from transformers.models.bigscience176b import BigScience176BLMHeadModel
from transformers import AutoTokenizer, AutoConfig

# model_name = "/gpfswork/rech/six/uan68tv/model-conversion/tr11e-350M-transformers-sharded"
model_name = "/gpfswork/rech/six/uan68tv/model-conversion/main-gs-47700-transformers-sharded"

config = AutoConfig.from_pretrained(model_name)

model = BigScience176BLMHeadModel.from_pretrained(model_name, use_cache=False)

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

print("successfully parallelized model")

tokenizer = AutoTokenizer.from_pretrained("bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles")

input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='pt')

# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids.to('cuda:0'), max_length=50)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))