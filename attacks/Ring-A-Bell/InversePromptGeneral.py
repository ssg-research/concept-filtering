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

import argparse
import torch
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import numpy as np
import random
import csv


population_size = 200
generation = 3000
mutateRate = 0.25
crossoverRate = 0.5
length = 16 # for K = 77, please set length = 75
cof = 3
batch_size = 6



dir_ = "CompVis/stable-diffusion-v1-4" # all the erasure models built on SDv1-4
torch_device = device = 'cuda'
tokenizer = CLIPTokenizer.from_pretrained(dir_, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(dir_, subfolder="text_encoder").to(torch_device)
text_encoder.to(device)


parser = argparse.ArgumentParser(description='Calculating the clip scores using a file of prompts and a folder of images')
parser.add_argument('--prompt_file', default= None, help='The prompt file')
parser.add_argument('--multiplier', default = 1)
parser.add_argument('--experiment')
parser.add_argument('--acc_prompt')
parser.add_argument('--unacc_prompt')
parser.add_argument('--do_adaptive', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--i2p', default='')
parser.add_argument('--ourclip_ckpt', default = None)
args = parser.parse_args()

if args.ourclip_ckpt is not None:
    assert(args.do_adaptive)

torch.manual_seed(args.seed)

ourclip_ckpt_given = False
if args.ourclip_ckpt is not None:
    te_checkpoint = torch.load(args.ourclip_ckpt)
    try:
        text_encoder.load_state_dict(te_checkpoint['textEncoder_state_dict'])
    except:
        try:
            te_checkpoint_renamed = {f'text_model.'+k:v for k,v in te_checkpoint['textEncoder_state_dict'].items()}
            text_encoder.load_state_dict(te_checkpoint_renamed)
        except:
            te_checkpoint_renamed.pop("text_model.embeddings.position_ids")
            text_encoder.load_state_dict(te_checkpoint_renamed)
    ourclip_ckpt_given = True

# path_Nudity_vector = './Concept Vectors/Nudity_vector.npy'
if ourclip_ckpt_given:  
    path_concept_vector = f'./Concept Vectors/{args.experiment}_{ourclip_ckpt_given}_vector.npy'
else:
    path_concept_vector = f'./Concept Vectors/{args.experiment}_vector.npy'


text_input_viol = tokenizer([args.unacc_prompt]*batch_size, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
nudeEmbed = text_encoder(text_input_viol.input_ids.to(torch_device))[0] + cof * torch.from_numpy(np.load(path_concept_vector)).to(device)
# violEmbed = violEmbed / violEmbed.norm(p=2, dim=-1, keepdim=True)

text_input_peaceful = tokenizer([args.acc_prompt]*batch_size, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
cleanEmbed = text_encoder(text_input_peaceful.input_ids.to(torch_device))[0] + cof * torch.from_numpy(np.load(path_concept_vector)).to(device)
# peacefulEmbed = peacefulEmbed / peacefulEmbed.norm(p=2, dim=-1, keepdim=True)

def read_promptfile(prompt_file, multiplier = 1):
    prompts = []
    with open(prompt_file, 'r') as f:
        for line in f.readlines():
            # print(line)
            # if args.experiment is not None:
            prompts += [line]*multiplier
            # else:
            #     prompts.append(line)
    print(len(prompts))
    return prompts


def fitness(dummy_token, args = None):
    dummy_embed = text_encoder(dummy_token.to(device))[0] 
    loss = ((targetEmbed - dummy_embed) ** 2).sum() 
    return loss.item()


def adaptive_fitness(dummy_token, args= None, nudeEmbed = nudeEmbed, cleanEmbed = cleanEmbed):
    dummy_embed = text_encoder(dummy_token.to(device))[0] 
    loss_original = ((targetEmbed - dummy_embed) ** 2)/batch_size#.sum()  
    # print(targetEmbed.shape)
    # print(nudeEmbed.shape)

    # target_embeds = targetEmbed / targetEmbed.norm(p=2, dim=-1, keepdim=True)
    # nude_embeds = nudeEmbed / nudeEmbed.norm(p=2, dim=-1, keepdim=True)
    # clean_embeds = cleanEmbed / cleanEmbed.norm(p=2, dim=-1, keepdim=True)
    # # cosine similarity as logits
    # logit_scale = self.logit_scale.exp()
    # logits_per_text = torch.matmul(target_embeds, nude_embeds.t())# * logit_scale  # want this one to decrease (similarity)
    # logits_per_text_increase = torch.matmul(target_embeds, clean_embeds.t())  # want this one to increase
    logits_per_text = ((targetEmbed - nudeEmbed) ** 2)/batch_size#.sum() 
    logits_per_text_increase = ((targetEmbed - cleanEmbed) ** 2)/batch_size#.sum() 
   

    loss = (loss_original-logits_per_text+logits_per_text_increase).sum()

    return loss.item()


def crossover(parents, crossoverRate):
    new_population = []
    for i in range(len(parents)):
        new_population.append(parents[i])
        if random.random() < crossoverRate:
            idx = np.random.randint(0, len(parents), size=(1,))[0]
            crossover_point = np.random.randint(1, length+1, size=(1,))[0] ##Because idx 0 is 49406, random ids are from idx 1 to idx length +1.
            new_population.append(torch.concat((parents[i][:,:crossover_point],parents[idx][:,crossover_point:]), 1))
            new_population.append(torch.concat((parents[idx][:,:crossover_point],parents[i][:,crossover_point:]), 1))
            
    return new_population
        

def mutation(population, mutateRate):
    for i in range(len(population)):
        if random.random() < mutateRate:
            idx = np.random.randint(1, length+1, size=(1,)) ##Because idx 0 is 49406, random ids are from idx 1 to idx length +1.
            value = np.random.randint(1, 49406, size=(1))[0] # Choose meaningful (Avoid selecting Token Ids 0, 40496, and 40497.)
            population[i][:,idx] = value
    return population


prompts = read_promptfile(args.prompt_file, args.multiplier)

fitness_func = adaptive_fitness if args.do_adaptive else fitness

for _, row in enumerate(prompts):
    # categorical = row.categories.split(',')
    # if 'violence' in categorical:
    # if float(row.nudity_percentage) < 50 and float(row.inappropriate_percentage) > 50 and int(row.hard) == 1:
    #     # prompt = row.prompt
    #     if row.clothes != 'None':
    #         if row.people == 'None':
    #             prompt = [f"{row.nudity}, {row.location}"]*6
    #         else:
    #             prompt = [f"{row.nudity} {row.people}, {row.location}"]*6
    #     else:
    #         prompt = [f"{row.nudity} {row.people}, {row.location}"]*6
    prompt = [row]*6
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    targetEmbed = text_encoder(text_input.input_ids.to(torch_device))[0] + cof * torch.from_numpy(np.load(path_concept_vector)).to(device)
    targetEmbed = targetEmbed.detach().clone()
    population = [torch.concat((torch.from_numpy(np.array([[49406]])),torch.randint(low=1, high=49406, size=(1,length)),torch.tile(torch.from_numpy(np.array([[49407]])),[1,76-length])),1) for i in range(population_size)]
    for step in range(generation):
        score = [fitness_func(p) for p in population]
        idx = np.argsort(score)
        population = [population[index] for index in idx][:population_size//2] 
        if step != generation - 1:
            new_popu = crossover(population, crossoverRate)
            population = mutation(new_popu, mutateRate)
        if step % 50 == 0:
            print(f"[Info]: {args.experiment}{args.i2p}_cof_{cof}_length_{length}")
            print(f"Iteration {step+1}, minium loss: {score[idx[0]]}")

    with open(f'data/InvPrompt2/{args.experiment}{args.i2p}_ourclip{ourclip_ckpt_given}_{str(args.seed)}_{cof}_length_{length}_{args.do_adaptive}.csv', 'a') as f:
        writer = csv.writer(f)
        invPrompt = tokenizer.decode(population[0][0][1:length+1])
        print(invPrompt)
        writer.writerow([invPrompt])

