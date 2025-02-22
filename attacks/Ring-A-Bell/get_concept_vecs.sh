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

python -m finetune_custom --anchor_text_path '/u1/test/generative-content-removal/stable-diffusion/assets/finetune_prompts/clothed.txt' \
    --target_text_path '/u1/test/generative-content-removal/stable-diffusion/assets/finetune_prompts/nudity.txt' \
    --anchor_img_path 'stable-diffusion/outputs/clothed/samples' \
    --target_img_path 'stable-diffusion/outputs/nudity_original/samples' \
    --adv_images 'stable-diffusion/oct23-nudity-12-loss1356/samples' \
    --adv_prompts_file 'attacks/hard-prompts-made-easy/adv_prompts_ours_official_train.txt' \
    --experiment 'nudity'
