# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert MPNet checkpoint."""


import argparse
import pathlib

import torch

from transformers import MPNetConfig, MPNetForMaskedLM
from transformers.utils import logging
from transformers.models.mpnet import MPNetLayer


logger = logging.get_logger(__name__)
logging.set_verbosity_info()


def convert_mpnet_checkpoint_to_pytorch(mpnet_checkpoint_path: str, pytorch_dump_folder_path: str):
    state_dicts = None
    try:
        import fairseq
        state_dicts = torch.load(mpnet_checkpoint_path)
    except:
        raise Exception("requires fairseq")

    mpnet_args = state_dicts['args']
    mpnet_weight = state_dicts['model']
    config = MPNetConfig(
        vocab_size=state_dicts['model']['decoder.sentence_encoder.embed_tokens.weight'].size(0),
        hidden_size=state_dicts['args'].encoder_embed_dim,
        num_hidden_layers=state_dicts['args'].encoder_layers,
        num_attention_heads=state_dicts['args'].encoder_attention_heads,
        intermediate_size=state_dicts['args'].encoder_ffn_embed_dim,
        max_position_embeddings=514,
    )
    model = MPNetForMaskedLM(config)

    tensor = model.mpnet.embeddings.word_embeddings.weight
    dim = config.hidden_size

    model.mpnet.embeddings.word_embeddings.weight.data = mpnet_weight['decoder.sentence_encoder.embed_tokens.weight'].type_as(tensor)
    model.mpnet.embeddings.position_embeddings.weight.data = mpnet_weight['decoder.sentence_encoder.embed_positions.weight'].type_as(tensor)
    model.mpnet.embeddings.LayerNorm.weight.data = mpnet_weight['decoder.sentence_encoder.emb_layer_norm.weight'].type_as(tensor)
    model.mpnet.embeddings.LayerNorm.bias.data = mpnet_weight['decoder.sentence_encoder.emb_layer_norm.bias'].type_as(tensor)

    model.lm_head.dense.weight.data = mpnet_weight['decoder.lm_head.dense.weight'].type_as(tensor)
    model.lm_head.dense.bias.data = mpnet_weight['decoder.lm_head.dense.bias'].type_as(tensor)
    model.lm_head.layer_norm.weight.data = mpnet_weight['decoder.lm_head.layer_norm.weight'].type_as(tensor)
    model.lm_head.layer_norm.bias.data = mpnet_weight['decoder.lm_head.layer_norm.bias'].type_as(tensor)
    model.lm_head.decoder.weight.data = mpnet_weight['decoder.lm_head.weight'].type_as(tensor)
    model.lm_head.decoder.bias.data = mpnet_weight['decoder.lm_head.bias'].type_as(tensor)

    # model.mpnet.encoder.relative_attention_bias.data = mpnet_weight['decoder.sentence_encoder.relative_attention_bias.weight'].type_as(tensor)

    for i in range(config.num_hidden_layers):
        layer : MPNetLayer = model.mpnet.encoder.layer[i]
        prefix = 'decoder.sentence_encoder.layers.{}.'.format(i)

        in_proj_weight = mpnet_weight[prefix + 'self_attn.in_proj_weight'].type_as(tensor)
        in_proj_bias = mpnet_weight[prefix + 'self_attn.in_proj_bias'].type_as(tensor)

        layer.attention.attn.q.weight.data = in_proj_weight[dim * 0 : dim * 1]
        layer.attention.attn.k.weight.data = in_proj_weight[dim * 1 : dim * 2]
        layer.attention.attn.v.weight.data = in_proj_weight[dim * 2 : dim * 3]
        layer.attention.attn.q.bias.data = in_proj_bias[dim * 0 : dim * 1] 
        layer.attention.attn.k.bias.data = in_proj_bias[dim * 1 : dim * 2] 
        layer.attention.attn.v.bias.data = in_proj_bias[dim * 2 : dim * 3] 

        layer.attention.attn.o.weight.data = mpnet_weight[prefix + 'self_attn.out_proj.weight'].type_as(tensor)
        layer.attention.attn.o.bias.data = mpnet_weight[prefix + 'self_attn.out_proj.bias'].type_as(tensor)
        layer.attention.LayerNorm.weight.data = mpnet_weight[prefix + 'self_attn_layer_norm.weight'].type_as(tensor)
        layer.attention.LayerNorm.bias.data = mpnet_weight[prefix + 'self_attn_layer_norm.bias'].type_as(tensor)

        layer.intermediate.dense.weight.data = mpnet_weight[prefix + 'fc1.weight'].type_as(tensor)
        layer.intermediate.dense.bias.data = mpnet_weight[prefix + 'fc1.bias'].type_as(tensor)
        layer.output.dense.weight.data = mpnet_weight[prefix + 'fc2.weight'].type_as(tensor)
        layer.output.dense.bias.data = mpnet_weight[prefix + 'fc2.bias'].type_as(tensor)

        layer.output.LayerNorm.weight.data = mpnet_weight[prefix + 'final_layer_norm.weight'].type_as(tensor)
        layer.output.LayerNorm.bias.data = mpnet_weight[prefix + 'final_layer_norm.bias'].type_as(tensor)

    pathlib.Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--mpnet_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_mpnet_checkpoint_to_pytorch(
        args.mpnet_checkpoint_path, args.pytorch_dump_folder_path,
    )