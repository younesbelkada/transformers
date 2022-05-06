import torch
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map, load_sharded_checkpoint_in_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoConfig

# model_name = "/gpfswork/rech/six/uan68tv/model-conversion/tr11e-350M-transformers-sharded"
model_name = "/gpfswork/rech/six/uan68tv/model-conversion/main-gs-47400-transformers-sharded"
tokenizer_name="bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles"
config = AutoConfig.from_pretrained(model_name)

# model = BigScience176BLMHeadModel.from_pretrained(model_name, use_cache=False, low_cpu_mem_usage=True)
with init_empty_weights():
    model = AutoModel.from_config(config)

print("successfully loaded model")

model.tie_weights()
# Determine a device map that maximizes the available GPUs, you can also write your own.
# If using a different model, adjust `T5Block` to the proper class (for instance `"GPTJBlock"`)
# device_map = infer_auto_device_map(model, no_split_module_classes=["BigScience176BBlock"])
# device_map = infer_auto_device_map(model, no_split_module_classes=["BigScience176BBlock", "BigScience176BMLP", "BigScience176BAttention"])
device_map = infer_auto_device_map(model, no_split_module_classes=["BigScience176BBlock", "BigScience176BAttention","BigScience176BMLP"])
# no_split_module_classes=["BigScience176BBlock", "BigScience176BMLP", "BigScience176BAttention"])
device_map = {'word_embeddings': 0, 'word_embeddings_layernorm': 0, 'h.0': 0, 'h.1': 0, 'h.2': 0, 'h.3': 0, 'h.4': 0, 'h.5': 0, 'h.6': 0, 'h.7': 0, 'h.8': 0, 'h.9': 0, 'h.10': 0, 'h.11': 0, 'h.12': 0, 'h.13': 0, 'h.14': 1, 'h.15': 1, 'h.16': 1, 'h.17': 1, 'h.18': 1, 'h.19': 1, 'h.20': 1, 'h.21': 1, 'h.22': 1, 'h.23': 1, 'h.24': 1, 'h.25': 1, 'h.26': 1, 'h.27': 1, 'h.28': 1, 'h.29': 1, 'h.30': 2, 'h.31': 2, 'h.32': 2, 'h.33': 2, 'h.34': 2, 'h.35': 2, 'h.36': 2, 'h.37': 2, 'h.38': 2, 'h.39': 2, 'h.40': 2, 'h.41': 3, 'h.42': 3, 'h.43': 3, 'h.44': 3, 'h.45': 3, 'h.46': 3, 'h.47': 3, 'h.48': 3, 'h.49': 3, 'h.50': 3, 'h.51': 4, 'h.52': 4, 'h.53': 4, 'h.54': 4, 'h.55': 4, 'h.56': 4, 'h.57': 4, 'h.58': 4, 'h.59': 4, 'h.60': 5, 'h.61': 6, 'h.62': 6, 'h.63': 6, 'h.64': 6, 'h.65': 6, 'h.66': 6, 'h.67': 6, 'h.68': 6, 'h.69': 6, 'ln_f': 'cpu'}

print(device_map)
# Load the sharded checkpoint inside the model. This will load each part of the model on the device specified by `device_map`
load_sharded_checkpoint_in_model(model, model_name, device_map=device_map)
# This will make that model that leaves on several different devices just work.
model = dispatch_model(model, device_map)

# The rest should feel familiar:
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
inputs = tokenizer("Task: copy but say the opposite. PSG won its match against Barca.", return_tensors="pt")
inputs = inputs.to(0)
with torch.no_grad():
    output = model.generate(inputs["input_ids"])
tokenizer.decode(output[0].tolist())