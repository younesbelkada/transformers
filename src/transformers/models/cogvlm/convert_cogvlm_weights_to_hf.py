# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import argparse

import torch
from huggingface_hub import hf_hub_download

from transformers import (
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    CLIPImageProcessor,
    CogVlmConfig,
    CogVlmForConditionalGeneration,
    LlavaProcessor,
    AutoModelForCausalLM,
)


KEYS_TO_MODIFY_MAPPING = {
    "model.vision.": "model.vision_model.model.",
    "patch_embedding.proj": "embeddings.patch_embedding",
    "patch_embedding.cls_embedding": "embeddings.class_embedding",
    "patch_embedding.position_embedding.weight": "embeddings.position_embedding.weight",
    "model.vision_model.model.linear_proj": "model.vision_model.model.vision_projector"
}

VISION_KEYS_TO_MODIFY_MAPPING = {
    "input_layernorm": "layer_norm1",
    "post_attention_layernorm": "layer_norm2",
    "transformer": "encoder",
    "attention": "self_attn",
    ".dense.": ".out_proj.",
}

def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        qkv_decontiguified = False

        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

            if "vision_model" in key:
                for key_to_modify, new_key in VISION_KEYS_TO_MODIFY_MAPPING.items():
                    if key_to_modify in key:
                        key = key.replace(key_to_modify, new_key)

                if "query_key_value" in key:
                    q_proj, k_proj, v_proj = decontiguify_qkv(value)

                    for name, tensor in zip(("q_proj", "k_proj", "v_proj"), (q_proj, k_proj, v_proj)):
                        new_key = key.replace("query_key_value", name)
                        new_state_dict[new_key] = tensor
                    
                    qkv_decontiguified = True

        # Remove the first dim
        if "class_embedding" in key:
            value = value[0, :]

        if not qkv_decontiguified:
            new_state_dict[key] = value
    return new_state_dict


def decontiguify_qkv(qkv_layer):
    hidden_dim = int(qkv_layer.shape[0] / 3)

    q_proj = qkv_layer[:hidden_dim]
    k_proj = qkv_layer[hidden_dim:2*hidden_dim]
    v_proj = qkv_layer[2*hidden_dim:]
    return q_proj, k_proj, v_proj


def convert_cogvlm_llama_to_hf():
    torch.set_default_dtype(torch.float16)

    config = CogVlmConfig()

    with torch.device("meta"):
        model = CogVlmForConditionalGeneration(config)


    original_model = AutoModelForCausalLM.from_pretrained("THUDM/cogvlm-chat-hf", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
    state_dict = original_model.state_dict()

    state_dict = convert_state_dict_to_hf(state_dict)
    model.load_state_dict(state_dict, strict=True, assign=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_model_id",
        help="Hub location of the text model",
    )
    parser.add_argument(
        "--vision_model_id",
        help="Hub location of the vision model",
    )
    parser.add_argument(
        "--output_hub_path",
        help="Location on the hub of the converted model",
    )
    args = parser.parse_args()
    convert_cogvlm_llama_to_hf()


if __name__ == "__main__":
    main()
