# Authors: Anudeep Das, Vasisht Duddu, Rui Zhang, N Asokan
# Copyright 2025 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer
import argparse


dir_ = "CompVis/stable-diffusion-v1-4"
tokenizer = CLIPTokenizer.from_pretrained(dir_, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(dir_, subfolder="text_encoder").to('cuda')
num_samples = 5

parser = argparse.ArgumentParser(description='Calculating the clip scores using a file of prompts and a folder of images')
parser.add_argument('--experiment')
parser.add_argument('--anchor_text_path')
parser.add_argument('--target_text_path')
parser.add_argument('--target_img_path', default=None)
parser.add_argument('--anchor_img_path', default=None)
parser.add_argument('--adv_prompts_file', default=None)
parser.add_argument('--adv_images', default=None)
parser.add_argument('--ourclip_ckpt', default=None)

args = parser.parse_args()
ourclip_ckpt_given = False
if args.ourclip_ckpt is not None:
    te_checkpoint = torch.load(args.ourclip_ckpt)
    try:
        text_encoder.load_state_dict(te_checkpoint['textEncoder_state_dict'])
    except:
        try:
            te_checkpoint_renamed = {f'text_model.'+k:v for k,v in te_checkpoint['textEncoder_state_dict'].items()}
            text_encoder.load_state_dict(te_checkpoint_renamed)
        except:
            te_checkpoint_renamed.pop("text_model.embeddings.position_ids")
            text_encoder.load_state_dict(te_checkpoint_renamed)
    ourclip_ckpt_given = True


def read_promptfile(prompt_file, multiplier = 1):
    prompts = []
    with open(prompt_file, 'r') as f:
        for line in f.readlines():
            # print(line)
            # if args.experiment is not None:
            prompts += [line]*multiplier
            # else:
            #     prompts.append(line)
    print(len(prompts))
    return prompts

acc_prompts = read_promptfile(args.anchor_text_path)
unacc_prompts = read_promptfile(args.target_text_path)


unacc_text=[]
for _, row in enumerate(unacc_prompts):
    prompt = [f"{row}"]*num_samples
    text_input = tokenizer(prompt ,padding="max_length", max_length=77,  truncation=True,  return_tensors="pt")
    embed = text_encoder(text_input.input_ids.to('cuda'), return_dict=True)[0] 
    unacc_text.extend(embed.detach().cpu().numpy())
unacc_text = np.array(unacc_text)

acc_text = []
for _, row in enumerate(acc_prompts):
    prompt = [f"{row}"]*num_samples
    text_input = tokenizer(prompt ,padding="max_length", max_length=77,  truncation=True,  return_tensors="pt")
    embed = text_encoder(text_input.input_ids.to('cuda'), return_dict=True)[0] 
    acc_text.extend(embed.detach().cpu().numpy())
acc_text = np.array(acc_text)

vec = np.mean(unacc_text - acc_text, axis=0)
np.save(f'{args.experiment}_{ourclip_ckpt_given}_vector.npy', vec)
