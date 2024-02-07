# coding=utf-8
# Copyright 2023 Microsoft, clefourrier The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch MassFormer model."""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import Iterable, Iterator, List, Optional, Tuple, Union
from transformers import GraphormerConfig, GraphormerModel

# from ...activations import ACT2FN
# from ...modeling_outputs import (
#     BaseModelOutputWithNoAttention,
#     SequenceClassifierOutput,
# )
# from ...modeling_utils import PreTrainedModel
# from ...utils import logging
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithNoAttention, SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from .configuration_massformer import MassFormerConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "clefourrier/graphormer-base-pcqm4mv2"
_CONFIG_FOR_DOC = "MassFormerConfig"


MASSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "clefourrier/graphormer-base-pcqm4mv2",
    # See all MassFormer models at https://huggingface.co/models?filter=transformer
]

def mask_prediction_by_mass(raw_prediction, prec_mass_idx, prec_mass_offset, mask_value=0.):
    # adapted from NEIMS
    # raw_prediction is [B,D], prec_mass_idx is [B]

    max_idx = raw_prediction.shape[1]
    assert th.all(prec_mass_idx < max_idx)
    idx = th.arange(max_idx, device=prec_mass_idx.device)
    mask = (
        idx.unsqueeze(0) <= (
            prec_mass_idx.unsqueeze(1) +
            prec_mass_offset)).float()
    return mask * raw_prediction + (1. - mask) * mask_value


def reverse_prediction(raw_prediction, prec_mass_idx, prec_mass_offset):
    # adapted from NEIMS
    # raw_prediction is [B,D], prec_mass_idx is [B]

    batch_size = raw_prediction.shape[0]
    max_idx = raw_prediction.shape[1]
    assert th.all(prec_mass_idx < max_idx)
    rev_prediction = th.flip(raw_prediction, dims=(1,))
    # convention is to shift right, so we express as negative to go left
    offset_idx = th.minimum(
        max_idx * th.ones_like(prec_mass_idx),
        prec_mass_idx + prec_mass_offset + 1)
    shifts = - (max_idx - offset_idx)
    gather_idx = th.arange(
        max_idx,
        device=raw_prediction.device).unsqueeze(0).expand(
        batch_size,
        max_idx)
    gather_idx = (gather_idx - shifts.unsqueeze(1)) % max_idx
    offset_rev_prediction = th.gather(rev_prediction, 1, gather_idx)
    # you could mask_prediction_by_mass here but it's unnecessary
    return offset_rev_prediction


def quant_noise(module: nn.Module, p: float, block_size: int):
    """
    From:
    https://github.com/facebookresearch/fairseq/blob/dd0079bde7f678b0cd0715cbd0ae68d661b7226d/fairseq/modules/quant_noise.py

    Wraps modules and applies quantization noise to the weights for subsequent quantization with Iterative Product
    Quantization as described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights, see "And the Bit Goes Down:
          Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper which consists in randomly dropping
          blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    if not isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
        raise NotImplementedError("Module unsupported for quant_noise.")

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        if module.weight.size(1) % block_size != 0:
            raise AssertionError("Input features must be a multiple of block sizes")

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            if module.in_channels % block_size != 0:
                raise AssertionError("Input channels must be a multiple of block sizes")
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            if k % block_size != 0:
                raise AssertionError("Kernel size must be a multiple of block size")

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = th.zeros(in_features // block_size * out_features, device=weight.device)
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = th.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = th.zeros(weight.size(0), weight.size(1), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])

            # scale weights and apply mask
            mask = mask.to(th.bool)  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module


class LayerDropModuleList(nn.ModuleList):
    """
    From:
    https://github.com/facebookresearch/fairseq/blob/dd0079bde7f678b0cd0715cbd0ae68d661b7226d/fairseq/modules/layer_drop.py
    A LayerDrop implementation based on [`th.nn.ModuleList`]. LayerDrop as described in
    https://arxiv.org/abs/1909.11556.

    We refresh the choice of which layers to drop every time we iterate over the LayerDropModuleList instance. During
    evaluation we always iterate over all layers.

    Usage:

    ```python
    layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
    for layer in layers:  # this might iterate over layers 1 and 3
        x = layer(x)
    for layer in layers:  # this might iterate over all layers
        x = layer(x)
    for layer in layers:  # this might not iterate over any layers
        x = layer(x)
    ```

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add
    """

    def __init__(self, p: float, modules: Optional[Iterable[nn.Module]] = None):
        super().__init__(modules)
        self.p = p

    def __iter__(self) -> Iterator[nn.Module]:
        dropout_probs = th.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p):
                yield m


class MassFormerGraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(self, config: MassFormerConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_atoms = config.num_atoms

        self.atom_encoder = nn.Embedding(config.num_atoms + 1, config.hidden_size, padding_idx=config.pad_token_id)
        self.in_degree_encoder = nn.Embedding(
            config.num_in_degree, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.out_degree_encoder = nn.Embedding(
            config.num_out_degree, config.hidden_size, padding_idx=config.pad_token_id
        )

        self.graph_token = nn.Embedding(1, config.hidden_size)

    def forward(
        self,
        input_nodes: th.LongTensor,
        in_degree: th.LongTensor,
        out_degree: th.LongTensor,
    ) -> th.Tensor:
        n_graph, n_node = input_nodes.size()[:2]

        node_feature = (  # node feature + graph token
            self.atom_encoder(input_nodes).sum(dim=-2)  # [n_graph, n_node, n_hidden]
            + self.in_degree_encoder(in_degree)
            + self.out_degree_encoder(out_degree)
        )

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        graph_node_feature = th.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature


class MassFormerGraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(self, config: MassFormerConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.multi_hop_max_dist = config.multi_hop_max_dist

        # We do not change edge feature embedding learning, as edge embeddings are represented as a combination of the original features
        # + shortest path
        self.edge_encoder = nn.Embedding(config.num_edges + 1, config.num_attention_heads, padding_idx=0)

        self.edge_type = config.edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(
                config.num_edge_dis * config.num_attention_heads * config.num_attention_heads,
                1,
            )

        self.spatial_pos_encoder = nn.Embedding(config.num_spatial, config.num_attention_heads, padding_idx=0)

        self.graph_token_virtual_distance = nn.Embedding(1, config.num_attention_heads)

    def forward(
        self,
        input_nodes: th.LongTensor,
        attn_bias: th.Tensor,
        spatial_pos: th.LongTensor,
        input_edges: th.LongTensor,
        attn_edge_type: th.LongTensor,
    ) -> th.Tensor:
        n_graph, n_node = input_nodes.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == "multi_hop":
            spatial_pos_ = spatial_pos.clone()

            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, input_nodes > 1 to input_nodes - 1
            spatial_pos_ = th.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                input_edges = input_edges[:, :, :, : self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]

            input_edges = self.edge_encoder(input_edges).mean(-2)
            max_dist = input_edges.size(-2)
            edge_input_flat = input_edges.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_input_flat = th.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[:max_dist, :, :],
            )
            input_edges = edge_input_flat.reshape(max_dist, n_graph, n_node, n_node, self.num_heads).permute(
                1, 2, 3, 0, 4
            )
            input_edges = (input_edges.sum(-2) / (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            input_edges = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + input_edges
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias


class MassFormerMultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, config: MassFormerConfig):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.kdim = config.kdim if config.kdim is not None else config.embedding_dim
        self.vdim = config.vdim if config.vdim is not None else config.embedding_dim
        self.qkv_same_dim = self.kdim == config.embedding_dim and self.vdim == config.embedding_dim

        self.num_heads = config.num_attention_heads
        self.attention_dropout_module = th.nn.Dropout(p=config.attention_dropout, inplace=False)

        self.head_dim = config.embedding_dim // config.num_attention_heads
        if not (self.head_dim * config.num_attention_heads == self.embedding_dim):
            raise AssertionError("The embedding_dim must be divisible by num_heads.")
        self.scaling = self.head_dim**-0.5

        self.self_attention = True  # config.self_attention
        if not (self.self_attention):
            raise NotImplementedError("The MassFormer model only supports self attention for now.")
        if self.self_attention and not self.qkv_same_dim:
            raise AssertionError("Self-attention requires query, key and value to be of the same size.")

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, config.embedding_dim, bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, config.embedding_dim, bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )
        self.q_proj = quant_noise(
            nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )

        self.out_proj = quant_noise(
            nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )

        self.onnx_trace = False

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: th.LongTensor,
        key: Optional[th.Tensor],
        value: Optional[th.Tensor],
        attn_bias: Optional[th.Tensor],
        key_padding_mask: Optional[th.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[th.Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """
        Args:
            key_padding_mask (Bytetorch.Tensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (Bytetorch.Tensor, optional): typically used to
                implement causal attention, where the mask prevents the attention from looking forward in time
                (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default: return the average attention weights over all
                heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embedding_dim = query.size()
        src_len = tgt_len
        if not (embedding_dim == self.embedding_dim):
            raise AssertionError(
                f"The query embedding dimension {embedding_dim} is not equal to the expected embedding_dim"
                f" {self.embedding_dim}."
            )
        if not (list(query.size()) == [tgt_len, bsz, embedding_dim]):
            raise AssertionError("Query size incorrect in MassFormer, compared to model dimensions.")

        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not th.jit.is_scripting():
                if (key_bsz != bsz) or (value is None) or not (src_len, bsz == value.shape[:2]):
                    raise AssertionError(
                        "The batch shape does not match the key or value shapes provided to the attention."
                    )

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if (k is None) or not (k.size(1) == src_len):
            raise AssertionError("The shape of the key generated in the attention is incorrect")

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            if key_padding_mask.size(0) != bsz or key_padding_mask.size(1) != src_len:
                raise AssertionError(
                    "The shape of the generated padding mask for the key does not match expected dimensions."
                )
        attn_weights = th.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        if list(attn_weights.size()) != [bsz * self.num_heads, tgt_len, src_len]:
            raise AssertionError("The attention weights generated do not match the expected dimensions.")

        if attn_bias is not None:
            attn_weights += attn_bias.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(th.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = th.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.attention_dropout_module(attn_weights)

        if v is None:
            raise AssertionError("No value generated")
        attn = th.bmm(attn_probs, v)
        if list(attn.size()) != [bsz * self.num_heads, tgt_len, self.head_dim]:
            raise AssertionError("The attention generated do not match the expected dimensions.")

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embedding_dim)
        attn: th.Tensor = self.out_proj(attn)

        attn_weights = None
        if need_weights:
            attn_weights = attn_weights_float.contiguous().view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    def apply_sparse_mask(self, attn_weights: th.Tensor, tgt_len: int, src_len: int, bsz: int) -> th.Tensor:
        return attn_weights


class MassFormerGraphEncoderLayer(nn.Module):
    def __init__(self, config: MassFormerConfig) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = config.embedding_dim
        self.num_attention_heads = config.num_attention_heads
        self.q_noise = config.q_noise
        self.qn_block_size = config.qn_block_size
        self.pre_layernorm = config.pre_layernorm

        self.dropout_module = th.nn.Dropout(p=config.dropout, inplace=False)

        self.activation_dropout_module = th.nn.Dropout(p=config.activation_dropout, inplace=False)

        # Initialize blocks
        self.activation_fn = ACT2FN[config.activation_fn]
        self.self_attn = MassFormerMultiheadAttention(config)

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.fc1 = self.build_fc(
            self.embedding_dim,
            config.ffn_embedding_dim,
            q_noise=config.q_noise,
            qn_block_size=config.qn_block_size,
        )
        self.fc2 = self.build_fc(
            config.ffn_embedding_dim,
            self.embedding_dim,
            q_noise=config.q_noise,
            qn_block_size=config.qn_block_size,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def build_fc(
        self, input_dim: int, output_dim: int, q_noise: float, qn_block_size: int
    ) -> Union[nn.Module, nn.Linear, nn.Embedding, nn.Conv2d]:
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def forward(
        self,
        input_nodes: th.Tensor,
        self_attn_bias: Optional[th.Tensor] = None,
        self_attn_mask: Optional[th.Tensor] = None,
        self_attn_padding_mask: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """
        nn.LayerNorm is applied either before or after the self-attention/ffn modules similar to the original
        Transformer implementation.
        """
        residual = input_nodes
        if self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)

        input_nodes, attn = self.self_attn(
            query=input_nodes,
            key=input_nodes,
            value=input_nodes,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = residual + input_nodes
        if not self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)

        residual = input_nodes
        if self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)
        input_nodes = self.activation_fn(self.fc1(input_nodes))
        input_nodes = self.activation_dropout_module(input_nodes)
        input_nodes = self.fc2(input_nodes)
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = residual + input_nodes
        if not self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)

        return input_nodes, attn


class MassFormerGraphEncoder(nn.Module):
    def __init__(self, config: MassFormerConfig):
        super().__init__()

        self.dropout_module = th.nn.Dropout(p=config.dropout, inplace=False)
        self.layerdrop = config.layerdrop
        self.embedding_dim = config.embedding_dim
        self.apply_massformer_init = config.apply_massformer_init
        self.traceable = config.traceable

        self.graph_node_feature = MassFormerGraphNodeFeature(config)
        self.graph_attn_bias = MassFormerGraphAttnBias(config)

        self.embed_scale = config.embed_scale

        if config.q_noise > 0:
            self.quant_noise = quant_noise(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                config.q_noise,
                config.qn_block_size,
            )
        else:
            self.quant_noise = None

        if config.encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
        else:
            self.emb_layer_norm = None

        if config.pre_layernorm:
            self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([MassFormerGraphEncoderLayer(config) for _ in range(config.num_hidden_layers)])

        # Apply initialization of model params after building the model
        if config.freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")

        for layer in range(config.num_trans_layers_to_freeze):
            m = self.layers[layer]
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(
        self,
        input_nodes: th.LongTensor,
        input_edges: th.LongTensor,
        attn_bias: th.Tensor,
        in_degree: th.LongTensor,
        out_degree: th.LongTensor,
        spatial_pos: th.LongTensor,
        attn_edge_type: th.LongTensor,
        perturb=None,
        last_state_only: bool = False,
        token_embeddings: Optional[th.Tensor] = None,
        attn_mask: Optional[th.Tensor] = None,
    ) -> Tuple[Union[th.Tensor, List[th.LongTensor]], th.Tensor]:
        # compute padding mask. This is needed for multi-head attention
        data_x = input_nodes
        n_graph, n_node = data_x.size()[:2]
        padding_mask = (data_x[:, :, 0]).eq(0)
        padding_mask_cls = th.zeros(n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        padding_mask = th.cat((padding_mask_cls, padding_mask), dim=1)

        attn_bias = self.graph_attn_bias(input_nodes, attn_bias, spatial_pos, input_edges, attn_edge_type)

        if token_embeddings is not None:
            input_nodes = token_embeddings
        else:
            input_nodes = self.graph_node_feature(input_nodes, in_degree, out_degree)

        if perturb is not None:
            input_nodes[:, 1:, :] += perturb

        if self.embed_scale is not None:
            input_nodes = input_nodes * self.embed_scale

        if self.quant_noise is not None:
            input_nodes = self.quant_noise(input_nodes)

        if self.emb_layer_norm is not None:
            input_nodes = self.emb_layer_norm(input_nodes)

        input_nodes = self.dropout_module(input_nodes)

        input_nodes = input_nodes.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(input_nodes)

        for layer in self.layers:
            input_nodes, _ = layer(
                input_nodes,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
            )
            if not last_state_only:
                inner_states.append(input_nodes)

        graph_rep = input_nodes[0, :, :]

        if last_state_only:
            inner_states = [input_nodes]

        if self.traceable:
            return th.stack(inner_states), graph_rep
        else:
            return inner_states, graph_rep


class MassFormerDecoderHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        """num_classes should be 1 for regression, or the number of classes for classification"""
        self.lm_output_learned_bias = nn.Parameter(th.zeros(1))
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        self.num_classes = num_classes

    def forward(self, input_nodes: th.Tensor, **unused) -> th.Tensor:
        input_nodes = self.classifier(input_nodes)
        input_nodes = input_nodes + self.lm_output_learned_bias
        return input_nodes


class MassFormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MassFormerConfig
    base_model_prefix = "massformer"
    main_input_name_nodes = "input_nodes"
    main_input_name_edges = "input_edges"

    def normal_(self, data: th.Tensor):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    def init_massformer_params(self, module: Union[nn.Linear, nn.Embedding, MassFormerMultiheadAttention]):
        """
        Initialize the weights specific to the MassFormer Model.
        """
        if isinstance(module, nn.Linear):
            self.normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            self.normal_(module.weight.data)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, MassFormerMultiheadAttention):
            self.normal_(module.q_proj.weight.data)
            self.normal_(module.k_proj.weight.data)
            self.normal_(module.v_proj.weight.data)

    def _init_weights(
        self,
        module: Union[
            nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm, MassFormerMultiheadAttention, MassFormerGraphEncoder
        ],
    ):
        """
        Initialize the weights
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # We might be missing part of the Linear init, dependant on the layer num
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, MassFormerMultiheadAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.reset_parameters()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, MassFormerGraphEncoder):
            if module.apply_massformer_init:
                module.apply(self.init_massformer_params)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class MassFormerModel(MassFormerPreTrainedModel):
    """The MassFormer model is a graph-encoder model.

    It goes from a graph to its representation. If you want to use the model for a downstream classification task, use
    MassFormerForGraphClassification instead. For any other downstream task, feel free to add a new class, or combine
    this model with a downstream model of your choice, following the example in MassFormerForGraphClassification.
    """

    def __init__(self, config: MassFormerConfig):
        super().__init__(config)
        self.max_nodes = config.max_nodes

        self.graph_encoder = MassFormerGraphEncoder(config)

        self.share_input_output_embed = config.share_input_output_embed
        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = not getattr(config, "remove_head", False)

        self.lm_head_transform_weight = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.activation_fn = ACT2FN[config.activation_fn]
        self.layer_norm = nn.LayerNorm(config.embedding_dim)

        self.post_init()

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(th.zeros(1))

    def forward(
        self,
        input_nodes: th.LongTensor,
        input_edges: th.LongTensor,
        attn_bias: th.Tensor,
        in_degree: th.LongTensor,
        out_degree: th.LongTensor,
        spatial_pos: th.LongTensor,
        attn_edge_type: th.LongTensor,
        perturb: Optional[th.FloatTensor] = None,
        masked_tokens: None = None,
        return_dict: Optional[bool] = None,
        **unused,
    ) -> Union[Tuple[th.LongTensor], BaseModelOutputWithNoAttention]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inner_states, graph_rep = self.graph_encoder(
            input_nodes, input_edges, attn_bias, in_degree, out_degree, spatial_pos, attn_edge_type, perturb=perturb
        )

        # last inner state, then revert Batch and Graph len
        input_nodes = inner_states[-1].transpose(0, 1)

        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        input_nodes = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(input_nodes)))

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(self.graph_encoder.embed_tokens, "weight"):
            input_nodes = th.nn.functional.linear(input_nodes, self.graph_encoder.embed_tokens.weight)

        if not return_dict:
            return tuple(x for x in [input_nodes, inner_states] if x is not None)
        return BaseModelOutputWithNoAttention(last_hidden_state=input_nodes, hidden_states=inner_states)

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes


class MassFormerForGraphClassification(MassFormerPreTrainedModel):
    """
    This model can be used for graph-level classification or regression tasks.

    It can be trained on
    - regression (by setting config.num_classes to 1); there should be one float-type label per graph
    - one task classification (by setting config.num_classes to the number of classes); there should be one integer
      label per graph
    - binary multi-task classification (by setting config.num_classes to the number of labels); there should be a list
      of integer labels for each graph.
    """

    def __init__(self, config: MassFormerConfig):
        super().__init__(config)
        self.encoder = MassFormerModel(config)
        self.embedding_dim = config.embedding_dim
        self.num_classes = config.num_classes
        self.classifier = MassFormerDecoderHead(self.embedding_dim, self.num_classes)
        self.is_encoder_decoder = True

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_nodes: th.LongTensor,
        input_edges: th.LongTensor,
        attn_bias: th.Tensor,
        in_degree: th.LongTensor,
        out_degree: th.LongTensor,
        spatial_pos: th.LongTensor,
        attn_edge_type: th.LongTensor,
        labels: Optional[th.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **unused,
    ) -> Union[Tuple[th.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_nodes,
            input_edges,
            attn_bias,
            in_degree,
            out_degree,
            spatial_pos,
            attn_edge_type,
            return_dict=True,
        )
        outputs, hidden_states = encoder_outputs["last_hidden_state"], encoder_outputs["hidden_states"]

        head_outputs = self.classifier(outputs)
        logits = head_outputs[:, 0, :].contiguous()

        loss = None
        if labels is not None:
            mask = ~th.isnan(labels)

            if self.num_classes == 1:  # regression
                loss_fct = MSELoss()
                loss = loss_fct(logits[mask].squeeze(), labels[mask].squeeze().float())
            elif self.num_classes > 1 and len(labels.shape) == 1:  # One task classification
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits[mask].view(-1, self.num_classes), labels[mask].view(-1))
            else:  # Binary multi-task classification
                loss_fct = BCEWithLogitsLoss(reduction="sum")
                loss = loss_fct(logits[mask], labels[mask])

        if not return_dict:
            return tuple(x for x in [loss, logits, hidden_states] if x is not None)
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=hidden_states, attentions=None)

class MassFormerLinearBlock(nn.Module):

    def __init__(self, in_feats, out_feats, dropout=0.1):
        super(MassFormerLinearBlock, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.bn(self.dropout(F.relu(self.linear(x))))


class NeimsBlock(nn.Module):
    """ from the NEIMS paper (uses LeakyReLU instead of ReLU) """

    def __init__(self, in_dim, out_dim, dropout):

        super(NeimsBlock, self).__init__()
        bottleneck_factor = 0.5
        bottleneck_size = int(round(bottleneck_factor * out_dim))
        self.in_batch_norm = nn.BatchNorm1d(in_dim)
        self.in_activation = nn.LeakyReLU()
        self.in_linear = nn.Linear(in_dim, bottleneck_size)
        self.out_batch_norm = nn.BatchNorm1d(bottleneck_size)
        self.out_linear = nn.Linear(bottleneck_size, out_dim)
        self.out_activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        h = x
        h = self.in_batch_norm(h)
        h = self.in_activation(h)
        h = self.dropout(h)
        h = self.in_linear(h)
        h = self.out_batch_norm(h)
        h = self.out_activation(h)
        h = self.out_linear(h)
        return h
    

class MLPModule(nn.Module):

    # TODO: fill in the rest of the arguments
    def __init__(self, g_dim, m_dim, o_dim, ff_h_dim, embed_linear, embed_agg, ff_layer_type, dropout, bidirectional_prediction, output_activation, output_normalization, ff_num_layers, gate_prediction, ff_skip):

        super(MLPModule, self).__init__()

        self.g_dim = g_dim # size of input graph embedding
        self.m_dim = m_dim # size of input metadata
        self.o_dim = o_dim # size of output
        self.ff_h_dim = ff_h_dim or g_dim # size of hidden layer in feedforward network
        self.embed_linear = embed_linear or False # whether to project inputs to ff_h_dim
        self.embed_agg =  embed_agg or "avg" # aggregation method for embedding
        self.ff_layer_type = ff_layer_type or "neims" # type of feedforward layer
        self.dropout = dropout or 0.1 # dropout rate
        self.bidirectional_prediction = bidirectional_prediction or False # whether to predict forward and reverse
        self.output_activation = output_activation or "relu" # activation function for output
        self.output_normalization = output_normalization or "l1" # normalization function for output
        self.ff_num_layers = ff_num_layers or 4 # number of feedforward layers
        self.gate_prediction = gate_prediction # whether to gate the output
        self.ff_skip = ff_skip# whether to skip connections in feedforward layers
        self.gt_gate_prediction = False # whether to gate the output with ground truth

        if self.embed_linear:
            # project each input to ff_h_dim
            self.g_embed_layer = nn.Linear(self.g_dim, self.ff_h_dim)
            self.m_embed_layer = nn.Linear(self.m_dim, self.ff_h_dim)
            self.g_embed_dim = self.ff_h_dim
            self.m_embed_dim = self.ff_h_dim
        else:
            # don't modify the inputs
            self.g_embed_layer = nn.Identity()
            self.m_embed_layer = nn.Identity()
            self.g_embed_dim = self.g_dim
            self.m_embed_dim = self.m_dim
        if self.embed_agg == "concat":
            self.embed_agg_fn = lambda x: th.cat(x, dim=1)
            self.embed_dim = self.g_embed_dim + self.m_embed_dim
        elif self.embed_agg == "add":
            assert self.g_embed_dim == self.m_embed_dim
            self.embed_agg_fn = lambda x: sum(x)
            self.embed_dim = self.g_embed_dim
        elif self.embed_agg == "avg":
            print(f"self.g_embed_dim: {self.g_embed_dim}")
            print(f"self.m_embed_dim: {self.m_embed_dim}")
            assert self.g_embed_dim == self.m_embed_dim
            self.embed_agg_fn = lambda x: sum(x) / len(x)
            self.embed_dim = self.g_embed_dim
        else:
            raise ValueError("invalid agg_embed")
        self.ff_layers = nn.ModuleList([])
        self.out_modules = []
        if self.ff_layer_type == "standard":
            ff_layer = MassFormerLinearBlock
        else:
            assert self.ff_layer_type == "neims", self.ff_layer_type
            ff_layer = NeimsBlock
        self.ff_layers.append(nn.Linear(self.embed_dim, self.ff_h_dim))
        self.out_modules.extend(["ff_layers"])
        for i in range(self.ff_num_layers):
            self.ff_layers.append(
                ff_layer(
                    self.ff_h_dim,
                    self.ff_h_dim,
                    self.dropout))
        if self.bidirectional_prediction:
            # assumes gating, mass masking
            self.forw_out_layer = nn.Linear(self.ff_h_dim, self.o_dim)
            self.rev_out_layer = nn.Linear(self.ff_h_dim, self.o_dim)
            self.out_gate = nn.Sequential(
                *[nn.Linear(self.ff_h_dim, self.o_dim), nn.Sigmoid()])
        else:
            self.out_layer = nn.Linear(self.ff_h_dim, self.o_dim)
            if self.gate_prediction:
                self.out_gate = nn.Sequential(
                    *[nn.Linear(self.ff_h_dim, self.o_dim), nn.Sigmoid()])
        # output activation
        if self.output_activation == "relu":
            self.output_activation_fn = F.relu
        elif self.output_activation == "sp":
            self.output_activation_fn = F.softplus
        elif self.output_activation == "sm":
            # you shouldn't gate with sm
            assert not self.bidirectional_prediction
            assert not self.gate_prediction
            self.output_activation_fn = lambda x: F.softmax(x, dim=1)
        else:
            raise ValueError(
                f"invalid output_activation: {self.output_activation}")
        # output normalization
        if self.output_normalization == "l1":
            self.output_normalization_fn = lambda x: F.normalize(x, p=1, dim=1)
        elif self.output_normalization == "l2":
            self.output_normalization_fn = lambda x: F.normalize(x, p=2, dim=1)
        elif self.output_normalization == "none":
            self.output_normalization_fn = lambda x: x
        else:
            raise ValueError(
                f"invalid output_normalization: {self.output_normalization}")

    def forward(self, data):
        
        input_embeds = []
        # add the graph embedding
        g_embed = self.g_embed_layer(data["graph_embed"])
        input_embeds.append(g_embed)
        # add the metadata embedding
        m_embed = self.m_embed_layer(data["spec_meta"])
        input_embeds.append(m_embed)
        # aggregate
        fh = self.embed_agg_fn(input_embeds)
        # apply feedforward layers
        fh = self.ff_layers[0](fh)
        for ff_layer in self.ff_layers[1:]:
            if self.ff_skip:
                fh = fh + ff_layer(fh)
            else:
                fh = ff_layer(fh)
        if self.bidirectional_prediction:
            ff = self.forw_out_layer(fh)
            fr = reverse_prediction(
                self.rev_out_layer(fh),
                data["prec_mz_idx"],
                self.prec_mass_offset)
            fg = self.out_gate(fh)
            fo = ff * fg + fr * (1. - fg)
            fo = mask_prediction_by_mass(
                fo, data["prec_mz_idx"], self.prec_mass_offset)
        else:
            # apply output layer
            fo = self.out_layer(fh)
            # apply gating
            if self.gate_prediction:
                    fg = self.out_gate(fh)
                    fo = fg * fo
        # apply output activation
        fo = self.output_activation_fn(fo)
        # apply gt gating
        if self.gt_gate_prediction:
            # binarize gt spec
            gt_fo = (data["spec"] > 0.).float()
            # map binary to [1-gt_gate_val,gt_gate_val]
            assert self.gt_gate_val > 0.5
            gt_fo = gt_fo * (2 * self.gt_gate_val - 1.) + \
                (1. - self.gt_gate_val)
            # multiply
            fo = gt_fo * fo
        # apply normalization
        fo = self.output_normalization_fn(fo)
        # package
        output_d = {"pred":fo}
        return output_d
    

class MassFormerModel(nn.Module):

    def __init__(self, graphormer_config):
        super(MassFormerModel, self).__init__()
        self.graphormer_module = GraphormerModel(config=graphormer_config)
        self.mlp_module = MLPModule(
            g_dim=768,
            m_dim=10,  # Adjust the dimension to match tensor a
            o_dim=1000,
            ff_h_dim=1000,
            embed_linear=False,
            embed_agg="concat",
            ff_layer_type="neims", 
            dropout=0.1,
            bidirectional_prediction=False,
            output_activation="relu",
            output_normalization="l1",
            ff_num_layers=4,
            gate_prediction=False,
            ff_skip=True
        )


    def get_graph_data(self, graph_entry):
        print(f"graph_entry keys = {list(graph_entry.keys())}")

        # this list of arguments is basically what preprocess_item produces
        gf_keys = ['input_nodes', 'attn_bias', 'attn_edge_type', 'spatial_pos', 'in_degree', 'out_degree', 'input_edges', 'labels']
        gf_item = {}
        shapes = {}

        for k in gf_keys:
            # note: some keys are optional, so we need to check for them
            if k in graph_entry:
                gf_item[k] = graph_entry[k]

                if isinstance(gf_item[k], th.Tensor):  # Check if it is a tensor
                    shapes[k] = gf_item[k].shape # Store the shape for later
            else:
                # helpful debugging message
                print(f"Warning: {k} not found in item")

        # After the loop, print all shapes
        print(f"---------------- Shape of gf_items ----------------")
        for k, shape in shapes.items():
            print(f"Shape of {k}: {shape}")    
        return gf_item
        

    def get_spec_data(self, spec_entry):
        print(f"spec_entry keys = {list(spec_entry.keys())}")
        # remove gf-related keys
        gf_related_keys = ['spatial_pos', 'labels', 'y', 'edge_index', 'edge_attr', 
                           'num_nodes', 'attn_bias', 'input_nodes', 'attn_edge_type', 
                           'in_degree', 'input_edges', 'out_degree', 'x']
        for k in gf_related_keys:
                # note: some keys are optional, so we need to check for them
                if k in spec_entry:
                    spec_entry.pop(k)
                else:
                    print(f"Warning: {k} not found in item")

        # now, let's handle the rest of the stuff
        spec_item = {k: [] for k in spec_entry.keys()}
        shapes = {}  # Dictionary to store shapes

        for k, v in spec_entry.items():
                    spec_item[k] = (v)
                    if isinstance(spec_item[k], th.Tensor):  # Check if it is a tensor
                        shapes[k] = spec_item[k].shape  # Store the shape for later

        # After the loop, print all shapes
        print(f"---------------- Shape of spec_item ----------------")
        for k, shape in shapes.items():
            print(f"Shape of {k}: {shape}")    
        return spec_item
        

    def forward(self, data):

        graph_data = self.get_graph_data(data)
        spec_data = self.get_spec_data(data)
        graph_embedding = self.graphormer_module(**graph_data)
        graph_embedding = graph_embedding.last_hidden_state[:, 0, :] # extract the embedding of the super node
        data_dict = {"graph_embed": graph_embedding, **spec_data}
        output = self.mlp_module(data_dict)
        print(output['pred'].shape)
        print(output["pred"])
        return output