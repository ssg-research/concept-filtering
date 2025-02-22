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

python eval-scripts/generate-images.py --model_name='models/diffusers-nudity-ESDu1-UNET.pt' \
    --prompts_path '/u1/a38das/concept-filtering/stable-diffusion/assets/eval_prompts/i2p_nudity_prompts.txt' \
    --save_path 'outputs3/oct7_esd_nude_i2p' \
    --num_samples 20 \
