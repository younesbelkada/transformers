# coding=utf-8
# Copyright 2023 Microsoft Research & University of Wisconsin-Madison and the HuggingFace Inc. team. All rights reserved.
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
""" CogVlm model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)

COGVLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "THUDM/cogvlm-base": "https://huggingface.co/THUDM/cogvlm-base/resolve/main/config.json",
}


class CogVlmVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CogVlmForConditionalGeneration`]. It is used to instantiate an
    CogVlm model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the CogVlm-9B.

    e.g. [cogvlm-hf/cogvlm-9b](https://huggingface.co/cogvlm-hf/cogvlm-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`CogVlmVisionConfig`,  *optional*):
            Custom vision config or dict
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Can be any of `LlamaConfig` or `MistralConfig`.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the CLIP backbone.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the CogVlm model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~CogVlmForConditionalGeneration`]

    Example:

    ```python
    >>> from transformers import CogVlmForConditionalGeneration, CogVlmConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a CogVlm cogvlm-1.5-7b style configuration
    >>> configuration = CogVlmConfig(vision_config, text_config)

    >>> # Initializing a model from the cogvlm-1.5-7b style configuration
    >>> model = CogVlmForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "cogvlm"
    is_composition = False

    def __init__(
        self,
        hidden_act="gelu",
        hidden_size=4096,
        initializer_range=0.02,
        intermediate_size=11008,
        max_position_embeddings=2048,
        num_attention_heads=32,
        num_hidden_layers=32,
        vocab_size=32000,
        patch_size=14,
        image_size=490,
        layer_norm_eps=1e-6,
        num_channels=3,
        attention_dropout=0.1,
        projection_hidden_size=4096,
        projection_intermediate_size=11008,
        **kwargs,
    ):
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.patch_size = patch_size
        self.image_size = image_size
        self.layer_norm_eps = layer_norm_eps
        self.num_channels = num_channels
        self.attention_dropout = attention_dropout
        self.projection_hidden_size = projection_hidden_size
        self.projection_intermediate_size = projection_intermediate_size

        super().__init__(**kwargs)

class CogVlmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CogVlmForConditionalGeneration`]. It is used to instantiate an
    CogVlm model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the CogVlm-9B.

    e.g. [cogvlm-hf/cogvlm-9b](https://huggingface.co/cogvlm-hf/cogvlm-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`CogVlmVisionConfig`,  *optional*):
            Custom vision config or dict
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Can be any of `LlamaConfig` or `MistralConfig`.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the CLIP backbone.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the CogVlm model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~CogVlmForConditionalGeneration`]

    Example:

    ```python
    >>> from transformers import CogVlmForConditionalGeneration, CogVlmConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a CogVlm cogvlm-1.5-7b style configuration
    >>> configuration = CogVlmConfig(vision_config, text_config)

    >>> # Initializing a model from the cogvlm-1.5-7b style configuration
    >>> model = CogVlmForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "cogvlm"
    is_composition = False

    def __init__(
        self,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=4096,
        initializer_range=0.02,
        intermediate_size=11008,
        max_position_embeddings=2048,
        num_attention_heads=32,
        num_hidden_layers=32,
        pad_token_id=0,
        rms_norm_eps=1e-05,
        vision_config=None,
        ignore_index=-100,
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        vocab_size=32000,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        self.vision_config = vision_config

        if isinstance(self.vision_config, dict):
            self.vision_config = CogVlmVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = CogVlmVisionConfig(
                intermediate_size=15360,
                hidden_size=1792,
                patch_size=14,
                image_size=490,
                num_hidden_layers=63,
                num_attention_heads=16,
                vocab_size=32000,
                layer_norm_eps=1e-6,
                hidden_act="gelu",
                projection_hidden_size=hidden_size,
                projection_intermediate_size=intermediate_size,
            )
        self.vocab_size = self.vocab_size

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.rms_norm_eps = rms_norm_eps

        super().__init__(**kwargs)
