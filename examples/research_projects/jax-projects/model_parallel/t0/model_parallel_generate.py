#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
from jax.experimental import PartitionSpec as P
from jax.experimental import maps
from jax.experimental.pjit import pjit

from transformers import FlaxT5ForConditionalGeneration, T5TokenizerFast

from partitions import set_partitions


# abstract_init: init the model weights only abstractly, eval_shape will return a pytree
# with the structure as weights but without any actual values, this will just contain
# the shape information. This should be used when initializing very large models
# that don't fit on single device and where sharded inittilization is necessary.

# if load_on_cpu is True, params will be a numpy arrays on CPU.
# This should be used for very large model that don't fit on single device
# and where we need to shard the weights.
platform = jax.lib.xla_bridge.get_backend().platform
dtype = jnp.dtype("bfloat16") if platform == "tpu" else jnp.float32
model, params = FlaxT5ForConditionalGeneration.from_pretrained(
    "valhalla/T0pp-flax-test", dtype=dtype, _do_init=False, force_download=False
)
tokenizer = T5TokenizerFast.from_pretrained("bigscience/T0")


# create the partition spec using the rules defined in `partitions.py`
# The `partition.py` file defines the `PyTree` of `ParitionSpec` for the T5 model which describes how the model will be sharded.
# The actual sharding is auto-matically handled by `pjit`. The weights are sharded accross all local devices.
# To adapt the script for other models, we need to also change the `ParitionSpec` accordingly.
# The good thing is this does not require you modfify the modeling code at all!
partition_spec = set_partitions(unfreeze(params))


def to_bf16(t):
    return jax.tree_map(lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, t)


def to_fp32(t):
    return jax.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, t)


shard_params = pjit(
    lambda params: to_bf16(params) if platform == "tpu" else to_fp32(params),
    in_axis_resources=(partition_spec,),
    out_axis_resources=partition_spec,
)

# define the mesh, this can be defined however you like
# NOTE: to be able to shard the model, the sharded dimention needs to be a multiple of devices it'll be sharded on.

# (1, 8) is 8 way tensor parallelism
# change this to (2, 4) and you get 2D parallelism (DP+TP): 2 way data parallel, and 4 way tensor parallel. How cool!
mesh_shape = (1, 8)
devices = np.array(jax.devices()).reshape(mesh_shape)
# create a mesh and bind names to mesh axses
mesh = maps.Mesh(devices, ("dp", "mp"))


# shard the model params
with maps.mesh(mesh.devices, mesh.axis_names):
    params = shard_params(freeze(params))


def generate(params, input_ids, attention_mask):
    output_ids = model.generate(input_ids, attention_mask=attention_mask, params=params).sequences
    return output_ids

p_generate = pjit(generate, in_axis_resources=(partition_spec, P("dp"), P("dp")), out_axis_resources=P("dp"))


# NOTE: Since `pjit` compiles the code, we need to fix the generations params like max_length, BS
# changing this values, will cause re-compilation (which is fast on TPU V3s)
model.config.max_length = 64
model.config.num_beams = 1

prompt = "is this review positive or negative? Review: Best cast iron skillet you will every buy."
inputs = tokenizer([prompt] * 8, return_tensors="jax", padding="max_length", truncation=True, max_length=512) # BS = 8

with maps.mesh(mesh.devices, mesh.axis_names):
    gen_ids = p_generate(freeze(params), inputs["input_ids"], inputs["attention_mask"])

generated_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
print("generated text: ", generated_text)