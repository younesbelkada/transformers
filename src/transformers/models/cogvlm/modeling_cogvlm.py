# coding=utf-8
# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch CogVlm model."""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ... import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_outputs import ModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPooling, BaseModelOutput
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_cogvlm import CogVlmConfig, CogVlmVisionConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "CogVlmConfig"

COGVLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "THUDM/cogvlm-base",
    # See all CogVlm models at https://huggingface.co/models?filter=cogvlm
]


LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


# Copied from transformers.models.clip.modeling_clip.CogVlmVisionVisionEmbeddings with CogVlmVisionVision->CogVlmVision
class CogVlmVisionEmbeddings(nn.Module):
    # Ignore Copy
    def __init__(self, config: CogVlmVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


# Copied from transformers.models.clip.modeling_clip.CogVlmVisionAttention with CogVlmVision->CogVlmVision
class CogVlmVisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


# Copied from transformers.models.clip.modeling_clip.CogVlmVisionMLP with CogVlmVision->CogVlmVision
class CogVlmVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CogVlmVisionMLP with CLIP->CogVlmVision
class CogVlmVisionEncoderLayer(nn.Module):
    def __init__(self, config: CogVlmVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CogVlmVisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CogVlmVisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


@dataclass
# Copied from transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast with Idefics->CogVlm
class CogVlmCausalLMOutputWithPast(ModelOutput):
    """
    Base class for CogVlm causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


COGVLM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`CogVlmConfig`] or [`CogVlmVisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->CogVlmVision
class CogVlmVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CogVlmVisionEncoderLayer`].

    Args:
        config: CogVlmVisionConfig
    """

    def __init__(self, config: CogVlmVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CogVlmVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )



class CogVlmVisionTransformer(nn.Module):
    def __init__(self, config: CogVlmVisionConfig):
        super().__init__()
        self.config = config

        self.embeddings = CogVlmVisionEmbeddings(config)
        self.encoder = CogVlmVisionEncoder(config)

        self.vision_projector = CogVlmVisionProjector(config)

        self.boi = nn.Parameter(torch.zeros(1, 1, config.projection_hidden_size))
        self.eoi = nn.Parameter(torch.zeros(1, 1, config.projection_hidden_size))

    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CogVlmVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    COGVLM_START_DOCSTRING,
)
# Copied from transformers.models.llava.modeling_llava.LlavaPreTrainedModel with Llava->CogVlm,llava->cogvlm
class CogVlmPreTrainedModel(PreTrainedModel):
    config_class = CogVlmConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CogVlmVisionAttention"]
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        # important: this ported version of CogVlm isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/cogvlm should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


@add_start_docstrings(
    """The vision model from CogVlmVision without any head or projection on top.""",
    COGVLM_START_DOCSTRING,
)
class CogVlmVisionModel(CogVlmPreTrainedModel):
    config_class = CogVlmVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["CogVlmVisionEncoderLayer"]

    def __init__(self, config: CogVlmVisionConfig):
        super().__init__(config)
        self.model = CogVlmVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CogVlmVisionModel

        >>> model = CogVlmVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CogVlmVisionProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_proj = nn.Linear(config.hidden_size, config.projection_hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(config.projection_hidden_size)
        self.act1 = nn.GELU()
        self.act2 = nn.functional.silu
        self.dense_h_to_4h = nn.Linear(config.projection_hidden_size, config.projection_intermediate_size, bias=False)
        self.gate_proj = nn.Linear(config.projection_hidden_size, config.projection_intermediate_size, bias=False)
        self.dense_4h_to_h = nn.Linear(config.projection_intermediate_size, config.projection_hidden_size, bias=False)

    def forward(self, x):
        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x)
        return x


class CogVlmRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class CogVlmMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))
        return down_proj


def get_expert_mask(token_type_ids: torch.LongTensor) -> (torch.BoolTensor, torch.BoolTensor):
    vision_token_mask = torch.zeros_like(token_type_ids, dtype=torch.bool)
    vision_token_mask[:, :-1] = (token_type_ids[:, :-1] == VISION_TOKEN_TYPE) & (
        token_type_ids[:, 1:] == VISION_TOKEN_TYPE
    )
    language_token_mask = ~vision_token_mask
    return vision_token_mask, language_token_mask


class CogVlmVisionExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.language_mlp = CogVlmMLP(config)
        self.vision_mlp = CogVlmMLP(config)

    def forward(self, hidden_states: torch.FloatTensor, token_type_ids: torch.LongTensor):
        output = torch.empty(hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device)
        vision_token_mask, language_token_mask = get_expert_mask(token_type_ids)
        output[vision_token_mask] = self.vision_mlp(hidden_states[vision_token_mask])
        output[language_token_mask] = self.language_mlp(hidden_states[language_token_mask])
        return output


def attention_fn(
    query_layer: torch.FloatTensor,
    key_layer: torch.FloatTensor,
    value_layer: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    *,
    scaling_attention_score: bool = True,
    attention_dropout: nn.Module = None,
):
    attention_mask_bool = attention_mask == 0
    is_low_triangle = (attention_mask_bool == torch.ones_like(attention_mask_bool, dtype=torch.float).tril()).all()
    is_full = (attention_mask_bool > 0).all()
    if int(torch.__version__.split(".")[0]) >= 2 and scaling_attention_score and (is_full or is_low_triangle):
        dropout_p = 0.0 if attention_dropout is None or not attention_dropout.training else attention_dropout.p
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, attn_mask=None, dropout_p=dropout_p, is_causal=not is_full
        )
        return context_layer, None
    else:
        if scaling_attention_score:
            query_layer = query_layer / math.sqrt(query_layer.shape[-1])
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores + attention_mask
        attention_scores = nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_layer.dtype)
        if attention_dropout is not None:
            attention_scores = attention_dropout(attention_scores)
        context_layer = torch.matmul(attention_scores, value_layer)
        return context_layer, attention_scores


# source: https://github.com/THUDM/SwissArmyTransformer/blob/1b160fcf42989c1ffcb964587f51241618f39f5b/tests/test_triton_rotary_embedding.py#L25.
class CogVlmRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half, learnable=False, device=torch.device("cpu")):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer("inv_freq", inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            cos_cached = cos_cached.to(x.dtype)
            sin_cached = sin_cached.to(x.dtype)
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


class CogVlmVisionExpertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.rotary_emb = CogVlmRotaryEmbedding(dim=self.head_dim)
        self.vision_expert_query_key_value = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=False)
        self.vision_expert_dense = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.language_expert_query_key_value = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=False)
        self.language_expert_dense = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [batch_size, sequence_length, num_head*head_dim] into a 4D tensor with size
        [batch_size, num_heads, seq_length, head_dim]."""
        new_tensor_shape = tensor.size()[:-1] + (self.num_heads, self.head_dim)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        vision_token_mask, language_token_mask = get_expert_mask(token_type_ids)

        shape = list(hidden_states.shape)
        shape[-1] = shape[-1] * 3
        mixed_raw_layer = torch.empty(shape, dtype=hidden_states.dtype, device=hidden_states.device)
        mixed_raw_layer[vision_token_mask] = self.vision_expert_query_key_value(hidden_states[vision_token_mask])
        mixed_raw_layer[language_token_mask] = self.language_expert_query_key_value(hidden_states[language_token_mask])

        query_states, key_states, value_states = torch.split(mixed_raw_layer, self.hidden_size, dim=-1)
        query_states = self._transpose_for_scores(query_states)
        key_states = self._transpose_for_scores(key_states)
        value_states = self._transpose_for_scores(value_states)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb_index_bhs(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        context_layer, attention_probs = attention_fn(
            query_layer=query_states,
            key_layer=key_states,
            value_layer=value_states,
            attention_mask=attention_mask,
            scaling_attention_score=True,
            attention_dropout=None,
        )
        if context_layer.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {context_layer.size()}"
            )
        context_layer = context_layer.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)

        attn_output = torch.empty(context_layer.shape, dtype=hidden_states.dtype, device=hidden_states.device)
        attn_output[vision_token_mask] = self.vision_expert_dense(context_layer[vision_token_mask])
        attn_output[language_token_mask] = self.language_expert_dense(context_layer[language_token_mask])

        return (
            (attn_output, attention_probs, past_key_value)
            if output_attentions
            else (attn_output, None, past_key_value)
        )


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


def apply_rotary_pos_emb_index_bhs(queries, keys, cos, sin, position_id):
    # batch_size, num_head, seq_len, hidden_size
    cos, sin = (
        nn.functional.embedding(position_id, cos.squeeze(1)).unsqueeze(1),
        nn.functional.embedding(position_id, sin.squeeze(1)).unsqueeze(1),
    )
    queries = (queries * cos) + (rotate_half(queries) * sin)
    keys = (keys * cos) + (rotate_half(keys) * sin)
    return queries, keys


class CogVlmDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = CogVlmVisionExpertAttention(config=config)
        self.mlp = CogVlmVisionExpertMLP(config)
        self.input_layernorm = CogVlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CogVlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, token_type_ids=token_type_ids)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

@add_start_docstrings(
    """
    CogVlm model without any head on top, just outputting raw hidden states.
    """,
    COGVLM_START_DOCSTRING,
)
class CogVlmModel(CogVlmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([CogVlmDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = CogVlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.vision_model = CogVlmVisionModel(config.vision_config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def encode_images(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        images_features = self.vision(pixel_values)
        return images_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: List[List[torch.Tensor]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored,
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import CogVlmProcessor, CogVlmModel
        >>> import torch
        >>> import requests
        >>> from PIL import Image

        >>> processor = CogVlmProcessor.from_pretrained("THUDM/cogvlm-chat-hf")
        >>> model = CogVlmModel.from_pretrained("THUDM/cogvlm-chat-hf")

        >>> # load image
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> query = "Describe this image"

        >>> prompt = f"Question: {query} Answer:"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # forward pass
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> last_hidden_state = outputs.last_hidden_state
        ```
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past_key_values is not None:
            pass  # generate mode with past_key_values. the image features are already mapped
        else:
            # not allow for inputs_embeds, because we want to process image feature
            assert input_ids is not None and inputs_embeds is None, f"{input_ids} {inputs_embeds}"

            if pixel_values is not None:
                # multi-modality
                if len(input_ids) != len(pixel_values):
                    raise ValueError("Make sure to pass as many texts as images")
                
                # prepend the input_ids and token_type_ids with image tokens
                num_vision_tokens = (
                    self.config.vision_config.image_size // self.config.vision_config.patch_size
                ) ** 2 + 2
                batch_size = input_ids.shape[0]

                vision_input_ids = torch.tensor([self.config.bos_token_id] + [self.config.pad_token_id] * num_vision_tokens).repeat(batch_size, 1).to(input_ids.device)
                vision_token_type_ids = torch.tensor([LANGUAGE_TOKEN_TYPE] + [VISION_TOKEN_TYPE] * num_vision_tokens).repeat(batch_size, 1).to(token_type_ids.device)

                print("Shape of vision input ids:", vision_input_ids.shape)
                print("Shape of vision token type ids:", vision_token_type_ids.shape)
                print("Shape of input ids:", input_ids[:,1:].shape)
                print("Shape of token type ids:", token_type_ids[:,1:].shape)

                input_ids = torch.cat([vision_input_ids, input_ids[:,1:]], dim=1)
                token_type_ids = torch.cat([vision_token_type_ids, token_type_ids[:,1:]], dim=1)
                attention_mask = torch.tensor([1] * input_ids.shape[1]).repeat(batch_size, 1).to(input_ids.device)

                print("Shape of input ids after concatenation:", input_ids.shape)
                print("Shape of token type ids after concatenation:", token_type_ids.shape)

                inputs_embeds = self.embed_tokens(input_ids)
                images_features = self.encode_images(pixel_values)
                images_features = images_features.reshape(-1, images_features.shape[-1])
                images_features = images_features.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                inputs_embeds = inputs_embeds.index_put([token_type_ids == VISION_TOKEN_TYPE], images_features)

            else:
                # TODO verify single-modality
                if token_type_ids is None:
                    token_type_ids = (
                        torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device) * LANGUAGE_TOKEN_TYPE
                    )
                assert not (
                    token_type_ids == VISION_TOKEN_TYPE
                ).any(), f"{(token_type_ids == VISION_TOKEN_TYPE).sum()}"
                inputs_embeds = self.embed_tokens(input_ids)

            if position_ids is None:
                print("Position ids are None, setting them:")
                position_ids = build_position_ids(token_type_ids, attention_mask)
            input_ids = None

        return self.llm_forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def llm_forward(
        self,
        input_ids: torch.LongTensor = None,
        token_type_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """largely copy from llama forward and adapt for cogvlm with `token_type_ids`"""

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

COGVLM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CogVlmVisionImageProcessor.__call__`] for details ([]`LlavaProcessor`] uses
            [`CogVlmVisionImageProcessor`] for processing images).
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    """The COGVLM model which consists of a vision backbone and a language model.""",
    COGVLM_START_DOCSTRING,
)
class CogVlmForConditionalGeneration(CogVlmPreTrainedModel):
    def __init__(self, config: CogVlmConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.model = CogVlmModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens.weight = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.weight = new_embeddings

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds


    @add_start_docstrings_to_model_forward(COGVLM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CogVlmCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CogVlmCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CogVlmForConditionalGeneration

        >>> model = CogVlmForConditionalGeneration.from_pretrained("cogvlm-hf/cogvlm-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("cogvlm-hf/cogvlm-1.5-7b-hf")

        >>> prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=text, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "There seems to be a stop sign"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                    )

                image_features = self.multi_modal_projector(selected_image_feature)
                inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, position_ids
                )
                if labels is None:
                    labels = torch.full_like(attention_mask, self.config.ignore_index).to(torch.long)
            else:
                # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
                # generation with cache
                if past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                    # Retrieve the first layer to inspect the logits and mask out the hidden states
                    # that are set to 0
                    first_layer_past_key_value = past_key_values[0][0][:, 0, :, 0]
                    batch_index, non_attended_tokens = torch.where(first_layer_past_key_value == 0)
                    # Get the target length
                    target_seqlen = first_layer_past_key_value.shape[-1] + 1

                    extended_attention_mask = torch.ones(
                        (attention_mask.shape[0], target_seqlen - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )

                    # Zero-out the places where we don't need to attend
                    extended_attention_mask[batch_index, non_attended_tokens] = 0

                    attention_mask = torch.cat((attention_mask, extended_attention_mask), dim=1)
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CogVlmCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, attention_mask=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
