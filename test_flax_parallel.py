import re

from flax.core.frozen_dict import freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.experimental import PartitionSpec as P


# Sentinels
_unmatched = object()
# For specifying empty leaf dict `{}`
empty_dict = object()

def _match(qs, ks):
    """Return True if regexes in qs match any window of strings in tuple ks."""
    # compile regexes and force complete match
    qts = tuple(map(lambda x: re.compile(x + "$"), qs))
    for i in range(len(ks) - len(qs) + 1):
        matches = [x.match(y) for x, y in zip(qts, ks[i:])]
        if matches and all(matches):
            return True
    return False

def _replacement_rules(rules):
    def replace(key, val):
        for rule, replacement in rules:
            if _match(rule, key):
                return replacement
        return val
    return replace
def _get_partition_rules():
    return [
        # Embeddings
        (("word_embeddings", "embedding"), P(None, "mp")),
        # Attention
        ((r"self_attention", "query_key_value", "kernel"), P("mp", None)),
        ((r"self_attention", "query_key_value", "bias"), P(None)),
        ((r"self_attention", "dense", "kernel"), P(None, "mp")),
        ((r"self_attention", "dense", "kernel"), P("mp")),
        # FFN
        ((r"mlp", "dense_4h_to_h", "kernel"), P("mp", None)),
        ((r"mlp", "dense_4h_to_h", "bias"), P(None)),
        ((r"mlp", "dense_h_to_4h", "kernel"), P(None, "mp")),
        ((r"mlp", "dense_h_to_4h", "bias"), P("mp")),
        # layer norms
        (("(bias|scale)",), None),
        # projection
        # (("lm_head", "kernel"), P(None, "mp")),
    ]
def set_partitions(in_dict):
    rules = _get_partition_rules()
    replace = _replacement_rules(rules)
    initd = {k: _unmatched for k in flatten_dict(in_dict)}
    result = {k: replace(k, v) for k, v in initd.items()}
    assert _unmatched not in result.values(), "Incomplete partition spec."
    return freeze(unflatten_dict(result))
import numpy as np
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
from jax.experimental import PartitionSpec as P
from jax.experimental import maps
from jax.experimental.pjit import pjit
from transformers import FlaxBloomForCausalLM, BloomConfig
from transformers import AutoTokenizer
ckpt = "bigscience/bloom-6b3"
model, params = FlaxBloomForCausalLM.from_pretrained(ckpt, _do_init=False)
tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=False)

spec = set_partitions(model.params_shape_tree)

shard_params = pjit(
    model.to_bf16,
    in_axis_resources=(spec,),
    out_axis_resources=spec,
)

mesh_shape = (1, 8)
devices = np.array(jax.devices()).reshape(mesh_shape)
# create a mesh and bind names to mesh axses
mesh = maps.Mesh(devices, ("dp", "mp"))
# shard the model params
with mesh:
    params = shard_params(freeze(params))
    
def generate(params, input_ids, attention_mask):
    output_ids = model.generate(input_ids, attention_mask=attention_mask, params=params).sequences
    return output_ids

p_generate = pjit(generate, in_axis_resources=(spec, P("dp"), P("dp")), out_axis_resources=P("dp"))

tokenizer.padding_side = "left"
model.config.max_length = 256
model.config.num_beams = 1
model.config.do_sample = True
model.config.pad_token_id = tokenizer.pad_token_id

prompt = "Reciepe for pasta with coconut:"
inputs = tokenizer([prompt] * 8, return_tensors="jax", padding="max_length", truncation=True, max_length=64) # BS = 


with mesh:
    gen_ids = p_generate(freeze(params), inputs["input_ids"], inputs["attention_mask"])
    
generated_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)