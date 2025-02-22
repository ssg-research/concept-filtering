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

python clip_filter_test.py --images_dir stable-diffusion/coco-test/samples \
    --acc_prompt nude --unacc_prompt "clean" \
    --clip_ckpt model_checkpoints_final_april/L-patch-14/model_72_ce_nudity_counter_take1_loss1356_te.pt \

python clip_filter_test.py --images_dir stable-diffusion/nudity-12-loss13456-seed42-noti-_i2p-baseline/samples \
    --acc_prompt nude --unacc_prompt "clean" \
    --clip_ckpt model_checkpoints_final_april/L-patch-14/model_72_ce_nudity_counter_take1_loss1356_te.pt \
