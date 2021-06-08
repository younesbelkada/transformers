# coding=utf-8
# Copyright 2021 The VQGAN Authors and The HuggingFace Team. All rights reserved.
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
""" PyTorch VQGAN model. """


from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import numpy as np

from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_vqgan import VQGANConfig
from .perceptual_loss_vqgan import VQLPIPSWithDiscriminator


logger = logging.get_logger(__name__)


VQGAN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai/vqgan-vit-base-patch32",
    # See all VQGAN models at https://huggingface.co/models?filter=vqgan
]


class VQGANOutput(ModelOutput):
    """
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`return_loss` is :obj:`True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(:obj:`torch.FloatTensor` of shape :obj:`(image_batch_size, text_batch_size)`):
            The scaled dot product scores between :obj:`image_embeds` and :obj:`text_embeds`. This represents the
            image-text similarity scores.
        logits_per_text:(:obj:`torch.FloatTensor` of shape :obj:`(text_batch_size, image_batch_size)`):
            The scaled dot product scores between :obj:`text_embeds` and :obj:`image_embeds`. This represents the
            text-image similarity scores.
        text_embeds(:obj:`torch.FloatTensor` of shape :obj:`(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            :class:`~transformers.VQGANTextModel`.
        image_embeds(:obj:`torch.FloatTensor` of shape :obj:`(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            :class:`~transformers.VQGANVisionModel`.
        text_model_output(:obj:`BaseModelOutputWithPooling`):
            The output of the :class:`~transformers.VQGANTextModel`.
        vision_model_output(:obj:`BaseModelOutputWithPooling`):
            The output of the :class:`~transformers.VQGANVisionModel`.
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class VQGANVectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta):
        super(VQGANVectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        ## could possible replace this here
        # #\start...
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # dtype min encodings: torch.float32
        # min_encodings shape: torch.Size([2048, 512])
        # min_encoding_indices.shape: torch.Size([2048, 1])

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        #.........\end

        # with:
        # .........\start
        #min_encoding_indices = torch.argmin(d, dim=1)
        #z_q = self.embedding(min_encoding_indices)
        # ......\end......... (TODO)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class VQGANUpsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class VQGANDownsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class VQGANResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class VQGANAttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


class VQGANEncoder(nn.Module):
    def __init__(self, config):
        # *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
        #          attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
        #          resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks
        ch = config.ch
        ch_mult = config.ch_mult
        attn_resolutions = config.attn_resolutions
        dropout = config.dropout
        z_channels = config.z_channels
        temb_ch = 0
        double_z = config.double_z
        resolution = config.resolution
        in_channels = config.in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(VQGANResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(VQGANAttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = VQGANDownsample(block_in, with_conv=True)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = VQGANResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = VQGANAttnBlock(block_in)
        self.mid.block_2 = VQGANResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VQGANDecoder(nn.Module):
    def __init__(self, config):
        # *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
        #          attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
        #          resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks
        ch = config.ch
        out_ch = config.out_ch
        ch_mult = config.ch_mult
        attn_resolutions = config.attn_resolutions
        dropout = config.dropout
        z_channels = config.z_channels
        temb_ch = 0
        give_pre_end = False
        resolution = config.resolution
        in_channels = config.in_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        logger.info("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = VQGANResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = VQGANAttnBlock(block_in)
        self.mid.block_2 = VQGANResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(VQGANResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(VQGANAttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = VQGANUpsample(block_in, with_conv=True)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VQGANPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VQGANConfig
    base_model_prefix = "vqgan"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        # factor = self.config.initializer_factor
        # if isinstance(module, VQGANEmbeddings):
        #     factor = self.config.initializer_factor
        #     nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim ** -0.5 * factor)
        #     nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
        #     nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        # elif isinstance(module, VQGANAttention):
        #     factor = self.config.initializer_factor
        #     in_proj_std = (module.embed_dim ** -0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
        #     out_proj_std = (module.embed_dim ** -0.5) * factor
        #     nn.init.normal_(module.q_proj.weight, std=in_proj_std)
        #     nn.init.normal_(module.k_proj.weight, std=in_proj_std)
        #     nn.init.normal_(module.v_proj.weight, std=in_proj_std)
        #     nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        # elif isinstance(module, VQGANMLP):
        #     factor = self.config.initializer_factor
        #     in_proj_std = (
        #         (module.config.hidden_size ** -0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
        #     )
        #     fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
        #     nn.init.normal_(module.fc1.weight, std=fc_std)
        #     nn.init.normal_(module.fc2.weight, std=in_proj_std)
        # elif isinstance(module, VQGANModel):
        #     nn.init.normal_(
        #         module.text_projection.weight,
        #         std=module.text_embed_dim ** -0.5 * self.config.initializer_factor,
        #     )
        #     nn.init.normal_(
        #         module.visual_projection.weight,
        #         std=module.vision_embed_dim ** -0.5 * self.config.initializer_factor,
        #     )

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


VQGAN_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ subclass. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config (:class:`~transformers.VQGANConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

VQGAN_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            :class:`~transformers.VQGANFeatureExtractor`. See :meth:`transformers.VQGANFeatureExtractor.__call__` for
            details.
        return_loss (:obj:`bool`, `optional`):
            Whether or not to return the contrastive loss.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""

@add_start_docstrings(VQGAN_START_DOCSTRING)
class VQGANModel(VQGANPreTrainedModel):
    config_class = VQGANConfig

    def __init__(self, config: VQGANConfig):
        super().__init__(config)

        self.encoder = VQGANEncoder(config)
        self.decoder = VQGANDecoder(config)
        # self.loss = VQLPIPSWithDiscriminator(**config)
        self.quantize = VQGANVectorQuantizer(config.n_embed, config.embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(config.z_channels, config.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(config.embed_dim, config.z_channels, 1)

        self.init_weights()

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec


    @add_start_docstrings_to_model_forward(VQGAN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=VQGANOutput, config_class=VQGANConfig)
    def forward(
        self,
        pixel_values=None,
        return_loss=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        quant, diff, _ = self.encode(pixel_values)
        dec = self.decode(quant)

        loss = None
        if return_loss:
            if self.optimizer_idx == 0:
                # autoencode
                aeloss, log_dict_ae = self.loss(diff, pixel_values, dec, self.optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                loss = aeloss

            if self.optimizer_idx == 1:
                # discriminator
                discloss, log_dict_disc = self.loss(diff, pixel_values, dec, self.optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                loss = discloss

        if not return_dict:
            output = (dec, diff)
            return ((loss,) + output) if loss is not None else output

        return VQGANOutput(
            loss=loss,
            dec=dec,
            diff=diff,
        )
