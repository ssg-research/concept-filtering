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

python main.py --target='sd' --method='rl' --reward_mode='clip' --threshold=0.26 --len_subword=10 --num_subword=1 --q_limit=60 --safety='custom_ti' --en=True   --prompts_path /u1/a38das/concept-filtering/stable-diffusion/assets/eval_prompts/i2p_nudity_prompts.txt --ourclip_ckpt /u1/test/generative-content-removal/model_checkpoints_preserve/L-patch-14/model_72_ce_nudity_counter_take1_loss1356_te.pt > april21_true_nud_i2p__ourclip.log






