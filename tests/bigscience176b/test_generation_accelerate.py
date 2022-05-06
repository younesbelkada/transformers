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
device_map = infer_auto_device_map(model, no_split_module_classes=["BigScience176BBlock"])
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