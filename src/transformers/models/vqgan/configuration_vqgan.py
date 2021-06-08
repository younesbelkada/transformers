# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" VQGAN model configuration """

import copy

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

VQGAN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openai/clip-vit-base-patch32": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json",
    # See all VQGAN models at https://huggingface.co/models?filter=clip
}

class VQGANConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.VQGANModel`. It is used to
    instantiate an VQGAN model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the VQGAN
    `openai/clip-vit-base-patch32 <https://huggingface.co/openai/clip-vit-base-patch32>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (:obj:`int`, `optional`, defaults to 224):
            The size (resolution) of each image.
        patch_size (:obj:`int`, `optional`, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` :obj:`"quick_gelu"` are supported.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (:obj:`float`, `optional`, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.

    Example::

        >>> from transformers import VQGANVisionModel, VQGANVisionConfig

        >>> # Initializing a VQGANVisionModel with openai/clip-vit-base-patch32 style configuration
        >>> configuration = VQGANVisionConfig()

        >>> # Initializing a VQGANVisionModel model from the openai/clip-vit-base-patch32 style configuration
        >>> model = VQGANVisionModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "vqgan_model"

    def __init__(
        self,
        embed_dim=256,
        n_embed=1024,
        double_z=False,
        z_channels=256,
        resolution=256,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=None,  # num_down = len(ch_mult)-1
        num_res_blocks=2,
        attn_resolutions=None,
        dropout=0.0,
        disc_conditional=False,
        disc_in_channels=3,
        disc_start=0,
        disc_weight=0.8,
        disc_num_layers=3,
        codebook_weight=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.double_z = double_z
        self.z_channels = z_channels
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_ch = out_ch
        self.ch = ch
        self.ch_mult = ch_mult if ch_mult is not None else [1,1,2,2,4]
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions if attn_resolutions is not None else [16]
        self.dropout = dropout
        self.disc_conditional = disc_conditional
        self.disc_in_channels = disc_in_channels
        self.disc_start = disc_start
        self.disc_weight = disc_weight
        self.disc_num_layers = disc_num_layers
        self.codebook_weight = codebook_weight
