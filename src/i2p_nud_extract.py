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

import pandas as pd

# i2p_df = pd.read_csv("i2p_benchmark.csv")

# # print(list(i2p_df.columns))

# nud_df = i2p_df[i2p_df['nudity_percentage'] > 10.]

# nud_df.to_csv("i2p_nudity.csv", index=False)

nud_df = pd.read_csv("i2p_nudity.csv")

prompt_list = nud_df['prompt'].tolist()

with open("i2p_nudity_prompts.txt", 'w') as f:
    for item in prompt_list:
        f.write("%s\n"%item)