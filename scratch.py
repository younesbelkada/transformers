from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-bloom", device_map="auto", torch_dtype="auto")
print(model.lm_head.weight.dtype)

model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-bloom", device_map="auto")
print(model.lm_head.weight.dtype)

model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-bloom")
print(model.lm_head.weight.dtype)

model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-bloom", torch_dtype="auto")
print(model.lm_head.weight.dtype)