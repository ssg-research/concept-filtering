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

import os
import argparse
import statistics
import torch
import clip
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel
from PIL import Image
import numpy as np
import csv
import statistics
import gc

parser = argparse.ArgumentParser(description='Calculating the clip scores using a file of prompts and a folder of images')
parser.add_argument('--unacc_prompt', help='The unacceptable concept in the prompt')
parser.add_argument('--acc_prompt', help='The acceptable concept in the prompt')
parser.add_argument('--images_dir', default=None, help='An optional directory of images')
parser.add_argument('--clip_ckpt', default=None, help='Path to a finetuned clip model')
parser.add_argument('--unacc_prompt_embed_ckpt', default=None)
parser.add_argument('--acc_prompt_embed_ckpt', default = None)
parser.add_argument('--diff_embed_npy', default=None)
parser.add_argument('--csv_dir', default = 'certified_csvs')
parser.add_argument('--which')
parser.add_argument('--delta', default=None)
parser.add_argument('--diff_images_dir', default= None, help='When you want to measure the diff between the embeddings of two images, this must be filled.')
parser.add_argument('--record_norms', action='store_true', default=False)
parser.add_argument('--clean_images_given', action='store_true', default=False)
parser.add_argument('--no_use_clipckpt', default=False, action='store_true')


def save_results_to_csv(args, data):
    # Define the CSV file path
    csv_file = os.path.join(args.csv_dir, f'{args.unacc_prompt}.csv')

    # Define the column headers (from the keys of the dictionaries)
    headers = data[0].keys()

    # Write data to CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        
        # Write header row
        writer.writeheader()
        
        # Write data rows
        for row in data:
            writer.writerow(row)

    print(f"Data has been written to {csv_file}")


def read_file(file):
    prompts = []
    with open(file, 'r') as f:
        for line in f.readlines():
            # print(line)
            # if args.experiment is not None:
            prompts.append(line.strip())
            # else:
            #     prompts.append(line)
    print(len(prompts))
    return prompts


def measure_image_diffs(img_list1, img_list2, ckpt_path):
    if args.unacc_prompt is not None and args.acc_prompt is not None:
        model_finetuned = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        if ckpt_path is not None and not args.no_use_clipckpt:
            te_checkpoint = torch.load(ckpt_path)
            try:
                model_finetuned.text_model.load_state_dict(te_checkpoint['textEncoder_state_dict'])
            except:
                te_checkpoint['textEncoder_state_dict'].pop('embeddings.position_ids')
                model_finetuned.text_model.load_state_dict(te_checkpoint['textEncoder_state_dict'])
            try:
                model_finetuned.vision_model.load_state_dict(te_checkpoint['visionEncoder_state_dict'])
            except:
                print('Vision encoder pt not found')
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    results_dicts = []
    all_img1_embeds = []
    all_img2_embeds = []
    all_img_diffs = []
    for image_path1 in img_list1:
        for image_path2 in img_list2:
            with torch.no_grad():
                result_dict = {'image_path1': image_path1, 'image_path2':image_path2, 'l2_norm_diff':-1}
                img1 = Image.open(image_path1)
                img2 = Image.open(image_path2)

                inputs = processor(images=[img1, img2], return_tensors="pt", padding=True).to(device)
                outputs = model_finetuned.get_image_features(pixel_values=inputs['pixel_values']).to('cpu')
                img1_embeds = outputs[:1, :].to('cpu')
                img2_embeds = outputs[1:, :].to('cpu')
                # all_img1_embeds.append(img1_embeds.to('cpu'))
                # all_img2_embeds.append(img2_embeds.to('cpu'))

                # img1_embeds = img1_embeds/img1_embeds.norm(p=2, dim=-1, keepdim=True)
                # img2_embeds = img2_embeds/img2_embeds.norm(p=2, dim=-1, keepdim=True)

                img_diff = (img1_embeds - img2_embeds).norm(p=2, dim=-1, keepdim=True).to('cpu')
                # print(img_diff.shape)
                print(f'Image diff is: {img_diff}')
                result_dict['l2_norm_diff'] = img_diff.item()
                results_dicts.append(result_dict)
                all_img_diffs.append(img_diff.to('cpu'))
                # del inputs
                # gc.collect()
                # print(f'Mean img1 embeds: {torch.mean(torch.stack(all_img1_embeds))}')
                # print(f'Mean img2 embeds: {torch.mean(torch.stack(all_img2_embeds))}')
                # print()
    # print(f'Mean img1 embeds: {statistics.fmean(all_img1_embeds)}')
    # print(f'Mean img2 embeds: {statistics.fmean(all_img2_embeds)}')
    # print(f'STDdev img1 embeds: {statistics.stdev(all_img1_embeds)}')
    # print(f'STDdev img2 embeds: {statistics.stdev(all_img2_embeds)}')
    img_diff_cat = torch.cat(all_img_diffs, axis=0).to('cpu')
    # img1_cat = torch.cat(all_img1_embeds, axis=0).to('cpu')
    # img2_cat = torch.cat(all_img2_embeds, axis=0).to('cpu')
    # # print(torch.std(img1_cat).norm(p=2, dim=-1, keepdim=True))
    # print(torch.std(img1_cat, dim=0).shape)
    # print(torch.mean(img1_cat, dim=0).norm(p=2, dim=-1, keepdim=True))
    # print(torch.mean(img2_cat, dim=0).norm(p=2, dim=-1, keepdim=True))
    # print(torch.std(img1_cat, dim=0).norm(p=2, dim=-1, keepdim=True))
    # print(torch.std(img2_cat, dim=0).norm(p=2, dim=-1, keepdim=True))
    # print()
    # print(torch.min(torch.std(img1_cat, dim=0)))
    # print(torch.max(torch.std(img1_cat, dim=0)))
    # print(torch.min(torch.std(img2_cat, dim=0)))
    # print(torch.max(torch.std(img2_cat, dim=0)))
    # print()
    # print(torch.mean(torch.std(img1_cat, dim=0)))
    # print(torch.mean(torch.std(img2_cat, dim=0)))
    print()
    print('Printing diffs statistics')
    print(img_diff_cat.shape)
    print(f'Min: {torch.min(img_diff_cat)}')
    print(f'Mean: {torch.mean(img_diff_cat)}')
    return results_dicts


def retrieve_scores_for_images(image_folder, ckpt_path, given_delta = None, record_norms = False):
    
    accuracies = []
    # text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder").to(device)
    if args.unacc_prompt is not None and args.acc_prompt is not None:
        model_finetuned = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        if ckpt_path is not None and not args.no_use_clipckpt:
            te_checkpoint = torch.load(ckpt_path)
            try:
                model_finetuned.text_model.load_state_dict(te_checkpoint['textEncoder_state_dict'])
            except:
                te_checkpoint['textEncoder_state_dict'].pop('embeddings.position_ids')
                model_finetuned.text_model.load_state_dict(te_checkpoint['textEncoder_state_dict'])
            try:
                model_finetuned.vision_model.load_state_dict(te_checkpoint['visionEncoder_state_dict'])
            except:
                print('Vision encoder pt not found')
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    results_dicts = []
    for img_num, image_path in enumerate(image_folder):
        result_dict = {'image_path': image_path}
        print(image_path)
        img = Image.open(image_path)
            
        with torch.no_grad():
            # try:
            #     image_input = preprocess(img).unsqueeze(0).to(device)
            # except OSError:
            #     print('OSError, continuing')
            #     continue

            # image_features = wrapped_model.encode_image(image_input)
            # text_features = wrapped_model.encode_text(text_input)

            # Normalize the features
            # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # image_inputs = processor(images=[img], return_tensors="pt", padding=True).to(device)
            # image_features = model_finetuned.get_image_features(**image_inputs)
            # image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            

            prompts_with_clean = args.acc_prompt
            prompts_with_nud = args.unacc_prompt
            inputs = processor(text=[prompts_with_nud, prompts_with_clean], images=[img], return_tensors="pt", padding=True).to(device)
            print(f'inputs keys: {list(inputs.keys())}')
            text_embeds_prenorm_test = model_finetuned.get_text_features(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'])
            image_embeds_prenorm_test = model_finetuned.get_image_features(pixel_values=inputs['pixel_values'])
            if record_norms:
                result_dict['l2_norm'] = image_embeds_prenorm_test.norm(p=2, dim=-1, keepdim=True)
            text_embeds_prenorm_test = text_embeds_prenorm_test/ text_embeds_prenorm_test.norm(p=2, dim=-1, keepdim=True)
            image_embeds_prenorm_test = image_embeds_prenorm_test / image_embeds_prenorm_test.norm(p=2, dim=-1, keepdim=True)

            print(f'logits_scale: {model_finetuned.logit_scale.exp()}')
            logits_per_text_test = torch.matmul(text_embeds_prenorm_test, image_embeds_prenorm_test.t()) * model_finetuned.logit_scale.exp()
            logits_per_image_test = logits_per_text_test.t()
            probs_test = logits_per_image_test.softmax(dim=-1).detach().cpu().numpy()
            print(f"CLIP accuracy for probs_test [{prompts_with_nud}, {prompts_with_clean}]:", probs_test)

            # inputs.input_ids[1] = -inputs.input_ids[0] # NEW
            # print(f'input keys: {inputs.keys()}')
            outputs = model_finetuned(**inputs)
            logits_per_image_original, logits_per_text_original = outputs.logits_per_image, outputs.logits_per_text
            probs_original = logits_per_image_original.softmax(dim=-1).detach().cpu().numpy()
            print(f'probs_original: {probs_original}')

            # text_embeds_norm = outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds_prenorm = model_finetuned.get_text_features(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'])[:1,:]
            print(f'text_embeds_prenorm shape: {text_embeds_prenorm.shape}')
            image_embeds_prenorm = model_finetuned.get_image_features(pixel_values=inputs['pixel_values'])
            text_embeds_normval = text_embeds_prenorm.norm(p=2, dim=-1, keepdim=True)
            image_embeds_normval = image_embeds_prenorm.norm(p=2, dim=-1, keepdim=True)

            # if given_delta is not None:
            #     delta = torch.tensor(given_delta)
            # else:
            delta = (1 - (model_finetuned.logit_scale.exp()) / ((model_finetuned.logit_scale.exp()) + 2*abs(probs_original[0][0] - 0.5)))#[0, :]
            delta = delta * image_embeds_normval
            # print(outputs.image_embeds.shape)
            # print(image_embeds_prenorm.shape)
            # delta = delta * outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
            print(f'Using noise before multiplication: {delta}')
            result_dict['delta_before_mult'] = delta.item()
            
            print(f'Using noise: {delta}')
            print(f'image_embeds_normval: {image_embeds_normval}')
            result_dict['delta'] = delta.item()
            # noise_matrix = torch.rand_like(image_embeds_prenorm)
            # current_norm = noise_matrix.norm(p=2, dim=-1, keepdim=True)
            # scaled_noise_matrix = delta / current_norm * noise_matrix
            scaled_noise_matrix = delta * torch.ones_like(image_embeds_prenorm)
            image_embeds_prenorm = image_embeds_prenorm - scaled_noise_matrix # + to minus
            image_embeds_norm = image_embeds_prenorm / image_embeds_prenorm.norm(p=2, dim=-1, keepdim=True)
            text_embeds_norm = text_embeds_prenorm / text_embeds_normval
            text_embeds_norm = torch.cat((text_embeds_norm, outputs.text_embeds[1:, :]), dim=0)
            logits_per_text = torch.matmul(text_embeds_norm, image_embeds_norm.t()) * model_finetuned.logit_scale.exp()
            
            
            logits_per_image = logits_per_text.t()
            # logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text
            probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
            print(f"CLIP accuracy for [{prompts_with_nud}, {prompts_with_clean}]:", probs)
            # accuracies.append(probs[0][0])
            if probs[0][0] > 0.5: # probs[0][1]
                print('Nud detected')
                result_dict['detected'] = True
            else:
                result_dict['detected'] = False
            print()
        results_dicts.append(result_dict)
    
    return results_dicts


if __name__ == '__main__':
    args = parser.parse_args()
    device = 'cuda:0'
    # device = 'cpu'
    os.makedirs(args.csv_dir, exist_ok=True)
    given_delta = None
    if args.delta is not None:
        os.makedirs(os.path.join(args.csv_dir, args.delta), exist_ok=True)
        args.csv_dir = os.path.join(args.csv_dir, args.delta)
        given_delta = float(args.delta)
    if args.clean_images_given:
        os.makedirs(os.path.join(args.csv_dir, args.acc_prompt), exist_ok=True)
        args.csv_dir = os.path.join(args.csv_dir, args.acc_prompt)
    # device = 'cpu'
    # wrapped_model, preprocess = clip.load("ViT-L/14")
    # wrapped_model = wrapped_model.to(device)
    # for i in range(6):
    if args.images_dir is None:
        img_files = read_file(f'data/{args.which}_images.txt')
    else:
        img_files = [os.path.join(args.images_dir, p) for p in os.listdir(args.images_dir)]
    if args.diff_images_dir:
        diff_img_files = [os.path.join(args.diff_images_dir, p) for p in os.listdir(args.diff_images_dir)]
        diff_results_dict = measure_image_diffs(img_files, diff_img_files, args.clip_ckpt)
        os.makedirs(os.path.join(args.csv_dir, 'diff'), exist_ok=True)
        args.csv_dir = os.path.join(args.csv_dir, 'diff')
        save_results_to_csv(args, diff_results_dict)
        # results_dicts = retrieve_scores_for_images(diff_img_files, args.clip_ckpt, given_delta=given_delta, record_norms=args.record_norms)
    else:
        results_dicts = retrieve_scores_for_images(img_files, args.clip_ckpt, given_delta=given_delta, record_norms=args.record_norms)
        save_results_to_csv(args, results_dicts)

    print(f'Mean CLIP accuracy for {args.unacc_prompt}, {args.acc_prompt}: {statistics.fmean(accuracies)}')
    print(f'Score for type {args.which} above')