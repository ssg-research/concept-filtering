#!/bin/bash

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

################ nudity #######################
input="assets/eval_prompts/nudity_eval.txt"
# input="assets/eval_prompts/i2p_nudity_prompts.txt"


while IFS= read -r line;
do
    python scripts/txt2img.py --prompt "$line" --experiment 'nudity' --outdir_suffix baseline --ourclip_ckpt /u1/a38das/generative-content-removal/model_checkpoints/L-patch-14/1/model_72_ce_nudity.pt 
done < "$input"






