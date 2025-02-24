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

python -m src.finetune_custom --acc_text_path 'stable-diffusion/assets/finetune_prompts/clothed.txt' \
    --unacc_text_path 'stable-diffusion/assets/finetune_prompts/nudity.txt' \
    --unacc_img_path '../stable-diffusion/outputs/nudity_original/samples' \
    --acc_img_path '../stable-diffusion/outputs/clothed/samples' \
    --experiment 'nudity' \
    --epochs 72    


