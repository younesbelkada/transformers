from huggingface_hub import HfApi

api = HfApi()
d = api.list_datasets(author="bigscience-data", use_auth_token=True)
corpus = [d.id for d in d if "roots_" in d.id]

print(len(corpus))
print(corpus)