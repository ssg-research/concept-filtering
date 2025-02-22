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

import torch
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel
import os
import argparse

def save_text_models_from_dir(dir_):
    for file_name in os.listdir(dir_):
        file_path = os.path.join(dir_, file_name)
        checkpoint = torch.load(file_path)
        wrapped_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        wrapped_model.load_state_dict(checkpoint['model_state_dict'])

        transformer = wrapped_model.text_model

        # Rename the dict keys
        # transformer_dict = {".".join(['text_model']+key.split(".")): value for key, value in transformer.state_dict().items()}
        # print(transformer_dict.keys())

        print('Saving model')
        torch.save({
            'textEncoder_state_dict': transformer.state_dict(),
        }, file_path[:-3]+'_te.pt')

        print('Attempting to load text model')
        # transformer = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        te_checkpoint = torch.load(file_path[:-3]+'_te.pt')
        transformer.load_state_dict(te_checkpoint['textEncoder_state_dict'])
        print('Successfully loaded text model')

parser = argparse.ArgumentParser(description='Process five string arguments.')
parser.add_argument('--models_dir', type=str, default = 'model_checkpoints/L-patch-14/models_grumpy_take1_fulloss_notasgood')
args = parser.parse_args()
save_text_models_from_dir(args.models_dir)