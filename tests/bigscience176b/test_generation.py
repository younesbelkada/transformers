from transformers.models.bigscience176b import BigScience176BLMHeadModel
from transformers import AutoTokenizer, AutoConfig

model_name = "/gpfswork/rech/six/uan68tv/model-conversion/tr11e-350M-transformers-sharded"
config = AutoConfig.from_pretrained(model_name)

model = BigScience176BLMHeadModel.from_pretrained(model_name, use_cache=False, low_cpu_mem_usage=True)

device_map = {0:[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
model.parallelize(device_map)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles")

input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='pt')

# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids, max_length=50)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))