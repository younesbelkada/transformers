import torch
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map, load_sharded_checkpoint_in_model
from transformers import AutoConfig, BigScience176BLMHeadModel, AutoTokenizer
from transformers import AutoTokenizer, AutoConfig

# model_name = "/gpfswork/rech/six/uan68tv/model-conversion/tr11e-350M-transformers-sharded"
model_name = "/gpfswork/rech/six/uan68tv/model-conversion/main-gs-47400-transformers-sharded"
tokenizer_name="bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles"
config = AutoConfig.from_pretrained(model_name)

# model = BigScience176BLMHeadModel.from_pretrained(model_name, use_cache=False, low_cpu_mem_usage=True)
with init_empty_weights():
    # model = BigScience176BLMHeadModel.from_config(config)
    model = BigScience176BLMHeadModel.from_pretrained(config)

print("successfully loaded model")

model.tie_weights()
# Determine a device map that maximizes the available GPUs, you can also write your own.
# If using a different model, adjust `T5Block` to the proper class (for instance `"GPTJBlock"`)
# device_map = infer_auto_device_map(model, no_split_module_classes=["BigScience176BBlock"])
# device_map = infer_auto_device_map(model, no_split_module_classes=["BigScience176BBlock", "BigScience176BMLP", "BigScience176BAttention"])
device_map = infer_auto_device_map(model, no_split_module_classes=["BigScience176BBlock", "BigScience176BAttention","BigScience176BMLP"])
# no_split_module_classes=["BigScience176BBlock", "BigScience176BMLP", "BigScience176BAttention"])
device_map = {'transformer.word_embeddings': 0, 'transformer.word_embeddings_layernorm': 0, 'transformer.h.0': 0, 'transformer.h.1': 0, 'transformer.h.2': 0, 'transformer.h.3': 0, 'transformer.h.4': 0, 'transformer.h.5': 0, 'transformer.h.6': 0, 'transformer.h.7': 0, 'transformer.h.8': 0, 'transformer.h.9': 0, 'transformer.h.10': 0, 'transformer.h.11': 0, 'transformer.h.12': 0, 'transformer.h.13': 0, 'transformer.h.14': 1, 'transformer.h.15': 1, 'transformer.h.16': 1, 'transformer.h.17': 1, 'transformer.h.18': 1, 'transformer.h.19': 1, 'transformer.h.20': 1, 'transformer.h.21': 1, 'transformer.h.22': 1, 'transformer.h.23': 1, 'transformer.h.24': 1, 'transformer.h.25': 1, 'transformer.h.26': 1, 'transformer.h.27': 1, 'transformer.h.28': 1, 'transformer.h.29': 1, 'transformer.h.30': 2, 'transformer.h.31': 2, 'transformer.h.32': 2, 'transformer.h.33': 2, 'transformer.h.34': 2, 'transformer.h.35': 2, 'transformer.h.36': 2, 'transformer.h.37': 2, 'transformer.h.38': 2, 'transformer.h.39': 2, 'transformer.h.40': 2, 'transformer.h.41': 3, 'transformer.h.42': 3, 'transformer.h.43': 3, 'transformer.h.44': 3, 'transformer.h.45': 3, 'transformer.h.46': 3, 'transformer.h.47': 3, 'transformer.h.48': 3, 'transformer.h.49': 3, 'transformer.h.50': 3, 'transformer.h.51': 4, 'transformer.h.52': 4, 'transformer.h.53': 4, 'transformer.h.54': 4, 'transformer.h.55': 4, 'transformer.h.56': 4, 'transformer.h.57': 4, 'transformer.h.58': 4, 'transformer.h.59': 4, 'transformer.h.60': 5, 'transformer.h.61': 6, 'transformer.h.62': 6, 'transformer.h.63': 6, 'transformer.h.64': 6, 'transformer.h.65': 6, 'transformer.h.66': 6, 'transformer.h.67': 6, 'transformer.h.68': 6, 'transformer.h.69': 6, 'transformer.ln_f': 6, 'lm_head':0}

print(device_map)
# Load the sharded checkpoint inside the model. This will load each part of the model on the device specified by `device_map`
load_sharded_checkpoint_in_model(model, model_name, device_map=device_map)
# This will make that model that leaves on several different devices just work.
model = dispatch_model(model, device_map)

# The rest should feel familiar:
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
# inputslogits = tokenizer.encode("Hello, my name is Ingrid.", return_tensors="pt").to(0)
# with torch.no_grad():
#     logits = model(inputslogits["input_ids"])
# print(logits)

inputs = tokenizer("Task: copy but say the opposite. PSG won its match against Barca.", return_tensors="pt")
inputs = inputs.to(0)
with torch.no_grad():
    output = model.generate(inputs["input_ids"])
tokenizer.decode(output[0].tolist())