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

from datasets import load_dataset
from diffusers import StableDiffusionPipeline
import torch
import os
import json
from PIL import Image
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from I2P dataset")

    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--esd_checkpoint", type=str, help="Path to ESD checkpoint", default="")
    parser.add_argument("--special_token", type=str, help="Special token for image generation", default="")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    metadata = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)

    data = load_dataset('AIML-TUDA/i2p', split='train')

    print("Number of images: ", len(data))

    pipe = StableDiffusionPipeline.from_pretrained(args.model_path, safety_checker=None, torch_dtype=torch.float16).to(device)

    if(args.esd_checkpoint != ""):
        pipe.unet.load_state_dict(torch.load(args.esd_checkpoint))

    count = 0
    for item in data:
        if args.special_token != "":
            prompt = [item["prompt"]]
        else:
            prompt = [args.special_token + " " + item["prompt"]]
        seed = item["sd_seed"]

        gen = torch.Generator(device)
        gen.manual_seed(seed)
        
        out =  pipe(prompt=prompt, generator=gen, negative_prompt=["nudity"])

        for image in out.images:
            
            file_name = "img_{}.png".format(count)
            image.save(os.path.join(args.output_dir, "train", file_name))
            count += 1

            metadata.append({"file_name": file_name, "prompt": prompt})

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
