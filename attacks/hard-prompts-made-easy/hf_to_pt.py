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

import argparse
import os.path

import torch

from open_clip import create_model
from transformers import CLIPConfig, CLIPVisionConfig, CLIPTextConfig, CLIPModel


def copy_attn_layer(hf_attn_layer, pt_attn_layer):
    assert(hf_attn_layer.num_heads == pt_attn_layer.num_heads)
    # q_proj, k_proj, v_proj = pt_attn_layer.in_proj_weight.chunk(3, dim=0)
    # q_proj_bias, k_proj_bias, v_proj_bias = pt_attn_layer.in_proj_bias.chunk(3, dim=0)

    # hf_attn_layer.q_proj.weight.copy_(q_proj)
    # hf_attn_layer.q_proj.bias.copy_(q_proj_bias)

    # hf_attn_layer.k_proj.weight.copy_(k_proj)
    # hf_attn_layer.k_proj.bias.copy_(k_proj_bias)

    # hf_attn_layer.v_proj.weight.copy_(v_proj)
    # hf_attn_layer.v_proj.bias.copy_(v_proj_bias)

    pt_attn_layer.out_proj.weight.copy_(hf_attn_layer.out_proj.weight)
    pt_attn_layer.out_proj.bias.copy_(hf_attn_layer.out_proj.bias)

    pt_attn_layer.in_proj_weight = torch.nn.Parameter(torch.cat((hf_attn_layer.q_proj.weight, hf_attn_layer.k_proj.weight, hf_attn_layer.v_proj.weight), dim=0))
    pt_attn_layer.in_proj_bias = torch.nn.Parameter(torch.cat((hf_attn_layer.q_proj.bias, hf_attn_layer.k_proj.bias, hf_attn_layer.v_proj.bias), dim=0))


def copy_mlp(hf_mlp, pt_mlp):
    copy_linear(hf_mlp.fc1, pt_mlp.c_fc)
    copy_linear(hf_mlp.fc2, pt_mlp.c_proj)


def copy_linear(hf_linear, pt_linear):
    pt_linear.weight.copy_(hf_linear.weight)
    pt_linear.bias.copy_(hf_linear.bias)


def copy_layer(hf_layer, pt_layer):
    # copy layer norms
    copy_linear(hf_layer.layer_norm1, pt_layer.ln_1)
    copy_linear(hf_layer.layer_norm2, pt_layer.ln_2)

    # copy MLP
    copy_mlp(hf_layer.mlp, pt_layer.mlp)

    # copy attn
    copy_attn_layer(hf_layer.self_attn, pt_layer.attn)


def copy_layers(hf_layers, pt_layers):
    for hf_layer, pt_layer in zip(hf_layers, pt_layers):
        copy_layer(hf_layer, pt_layer)


def copy_encoder(hf_encoder, pt_model):
    # copy  embeds
    pt_model.token_embedding.weight.copy_(hf_encoder.embeddings.token_embedding.weight)
    pt_model.positional_embedding.copy_(hf_encoder.embeddings.position_embedding.weight)

    # copy layer norm
    copy_linear(hf_encoder.final_layer_norm, pt_model.ln_final)

    # copy hidden layers
    copy_layers(hf_encoder.encoder.layers, pt_model.transformer.resblocks)


def copy_text_model_and_projection(hf_model, pt_model):
    # copy projection
    pt_model.text_projection.copy_(hf_model.text_projection.weight.T)

    # copy text encoder
    copy_encoder(hf_model.text_model, pt_model)


def copy_vison_model_and_projection(hf_model, pt_model):
    # copy projection
    pt_model.visual.proj.copy_(hf_model.visual_projection.weight.T)

    # copy layer norms
    copy_linear(hf_model.vision_model.pre_layrnorm, pt_model.visual.ln_pre)
    copy_linear(hf_model.vision_model.post_layernorm, pt_model.visual.ln_post)

    # copy embeds
    pt_model.visual.conv1.weight.copy_(hf_model.vision_model.embeddings.patch_embedding.weight)
    pt_model.visual.class_embedding.copy_(hf_model.vision_model.embeddings.class_embedding)
    pt_model.visual.positional_embedding.copy_(hf_model.vision_model.embeddings.position_embedding.weight)

    # copy encoder
    copy_layers(hf_model.vision_model.encoder.layers, pt_model.visual.transformer.resblocks)


@torch.no_grad()
def convert_clip_checkpoint(model, pretrained, pytorch_dump_folder_path, config_path=None, hf_te_ckpt = None, suffix=''):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # if config_path is not None:
    #     config = CLIPConfig.from_pretrained(config_path)
    # else:
    #     # config = CLIPConfig(
    #     #     projection_dim=512,
    #     #     text_config_dict=dict(hidden_act='gelu'),
    #     #     vision_config_dict=dict(hidden_act='gelu'))
    #     config = CLIPConfig.from_pretrained('CompVis/stable-diffusion-v1-4')

        #CLIPVisionConfig()
        #CLIPTextConfig()

        # L14
        # config = CLIPConfig(
        #     projection_dim=768,
        #     text_config_dict=dict(
        #         hidden_act='gelu',
        #         hidden_size=768,
        #         intermediate_size=3072,
        #         num_attention_heads=12,
        #     ),
        #     vision_config_dict=dict(
        #         hidden_act='gelu',
        #         num_hidden_layers=24,
        #         patch_size=14,
        #         hidden_size=1024,
        #         intermediate_size=4096,
        #         num_attention_heads=16,
        #     ))

        ## H14
        #
        # config = CLIPConfig(
        #     projection_dim=1024,
        #     text_config_dict=dict(
        #         hidden_act='gelu',
        #         hidden_size=1024,
        #         intermediate_size=4096,
        #         num_attention_heads=16,
        #         num_hidden_layers=24,
        #     ),
        #     vision_config_dict=dict(
        #         hidden_act='gelu',
        #         num_hidden_layers=32,
        #         patch_size=14,
        #         hidden_size=1280,
        #         intermediate_size=5120,
        #         num_attention_heads=16,
        #     ))

        ## B16 / B16 plus
        # config = CLIPConfig(
        #     projection_dim=512,
        #     text_config_dict=dict(
        #         hidden_act='gelu',
        #     ),
        #     vision_config_dict=dict(
        #         hidden_act='gelu',
        #         num_hidden_layers=12,
        #         patch_size=16
        #     ))

        # config = CLIPConfig(
        #     projection_dim=640,
        #     text_config_dict=dict(
        #         hidden_act='gelu',
        #         hidden_size=640,
        #         intermediate_size=2560,
        #         num_attention_heads=10,
        #     ),
        #     vision_config_dict=dict(
        #         hidden_act='gelu',
        #         num_hidden_layers=12,
        #         patch_size=16,
        #         hidden_size=896,
        #         num_attention_heads=14,
        #         intermediate_size=3584,
        #         image_size=240,
        #     ))


        # ## g14
        # config = CLIPConfig(
        #     projection_dim=1024,
        #     text_config_dict=dict(
        #         hidden_act='gelu',
        #         hidden_size=1024,
        #         intermediate_size=4096,
        #         num_attention_heads=16,
        #         num_hidden_layers=24,
        #     ),
        #     vision_config_dict=dict(
        #         hidden_act='gelu',
        #         num_hidden_layers=40,
        #         patch_size=14,
        #         hidden_size=1408,
        #         intermediate_size=6144,
        #         num_attention_heads=16,
        #     ))


    # print(config)
    config = 'openai/clip-vit-large-patch14'
    hf_model = CLIPModel.from_pretrained(config).eval()

    # Load our specially trained te checkpoint
    if hf_te_ckpt is not None:
        print(f'checkpoint_path: {hf_te_ckpt}')
        te_checkpoint = torch.load(hf_te_ckpt, map_location=torch.device('cpu'))
        
        try:
            hf_model.text_model.load_state_dict(te_checkpoint['textEncoder_state_dict'])
        except:
            transformer_dict = {".".join(key.split(".")[1:]): value for key, value in te_checkpoint['textEncoder_state_dict'].items()}
            hf_model.text_model.load_state_dict(transformer_dict)

    print(hf_model)

    pt_model = create_model(model, pretrained=pretrained, precision='fp32')
    pt_model = pt_model.eval()
    print(pt_model)

    copy_text_model_and_projection(hf_model, pt_model)
    copy_vison_model_and_projection(hf_model, pt_model)
    hf_model.logit_scale = pt_model.logit_scale

    input_ids = torch.arange(0, 77).unsqueeze(0)
    pixel_values = torch.randn(1, 3, 224, 224)

    hf_image_embed = hf_model.get_image_features(pixel_values)
    hf_text_embed = hf_model.get_text_features(input_ids)

    pt_image_embed = pt_model.encode_image(pixel_values)
    pt_text_embed = pt_model.encode_text(input_ids)
    print((pt_image_embed - hf_image_embed).sum())
    print((pt_text_embed - hf_text_embed).sum())
    print((pt_text_embed - hf_text_embed).max(), (pt_text_embed - hf_text_embed).min())
    assert torch.allclose(hf_image_embed, pt_image_embed, atol=1e-4)
    assert torch.allclose(hf_text_embed, pt_text_embed, atol=1e-4)


    hf_logits_per_image, hf_logits_per_text = hf_model(
        input_ids=input_ids, pixel_values=pixel_values, return_dict=False
    )[:2]

    pt_image_features, pt_text_features, logit_scale = pt_model(pixel_values, input_ids)
    pt_logits_per_image = pt_image_features @ pt_text_features.T * logit_scale
    pt_logits_per_text = pt_logits_per_image.T

    assert torch.allclose(hf_logits_per_image, pt_logits_per_image, atol=1e-4)
    assert torch.allclose(hf_logits_per_text, pt_logits_per_text, atol=1e-4)

    if os.path.exists(pretrained):
        pretrained = os.path.splitext(os.path.basename(pretrained))[0]

    hf_model.save_pretrained(f'{model}-{pretrained}')

    # Check if the folder exists
    if not os.path.exists(f'{model}-{pretrained}-{suffix}'):
        # If the folder doesn't exist, create it
        os.makedirs(f'{model}-{pretrained}-{suffix}')
    torch.save(pt_model.state_dict(), f'{model}-{pretrained}-{suffix}/open_clip_pytorch_model.bin')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument('--hf_te_ckpt', default=None, type=str, help='Path to our hf textencoder checkpoint')
    parser.add_argument("--model", default="ViT-L-14", type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--pretrained", default="openai", type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--suffix", default="", type=str, help="Suffix for the file save directory")
    parser.add_argument("--config_path", default='stable-diffusion/configs/stable-diffusion/v1-inference.yaml', type=str, help="Path to hf config.json of model to convert")
    args = parser.parse_args()

    convert_clip_checkpoint(args.model, args.pretrained, args.pytorch_dump_folder_path, args.config_path, args.hf_te_ckpt, args.suffix)