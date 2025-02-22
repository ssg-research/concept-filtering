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
import clip
from PIL import Image
import os
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor
from finetune import FrozenCLIPEmbedder, CocoDetection
from abc import ABC, abstractmethod


T=transforms.Compose([transforms.ToTensor()])
P=transforms.Compose([transforms.ToPILImage()])

ATTACK_STEPS = 7
BATCH_SIZE = 1

# device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
device = 'cpu'

# The following are the captions for the first 5 items, we are going to change them
original_map = [
'A man in a red shirt and a red hat is on a motorcycle on a hill side.',
'A little girl is getting ready to blow out a candle on a small dessert.',
'A man riding a bike past a train traveling along tracks.',
'Glasses and bottles are placed near a kitchen sink.',
'Wooden spoons are lined up on a table']

# caption_map = ['A man in a red shirt and a red hat is in a car on a hill side.', 
# 'A little girl is getting ready to blow out a candle on a small carrot.',
# 'A man riding a bike past a truck traveling along ice.',
# 'Glasses and bottles are placed near a bed.',
# 'Plastic forks are lined up on a table']
caption_map = ['a gray square' for t in original_map]




class Attacker(ABC):
    def __init__(self, model, config):
        """
        ## initialization ##
        :param model: Network to attack
        :param config : configuration to init the attack
        """
        self.config = config
        self.model = model
        self.clamp = (0,1)
    
    def _random_init(self, x):
        x = x + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * self.config['eps']
        x = torch.clamp(x,*self.clamp)
        return x

    def __call__(self, x,y):
        x_adv = self.forward(x,y)
        return x_adv


def do_attack(model, train_dataloader, targeted_attack = False):
    loss_img = torch.nn.CrossEntropyLoss()
    loss_txt = torch.nn.CrossEntropyLoss()
    loss_counter_txt = torch.nn.MSELoss()
    attack_lr = 2.
    attack_eps = 8.
    x_adv_images = []

    # add your own code to track the training progress.
    iterator = tqdm(range(ATTACK_STEPS))
        # for _ in range(self.config['attack_steps']):
        #     x_adv.requires_grad = True
        #     self.model.zero_grad()
        #     logits = self.model(x_adv) #f(T((x))
        #     if self.target is None:
        #         # Untargeted attacks - gradient ascent
                
        #         loss = F.cross_entropy(logits, y,  reduction="sum")
        #         loss.backward()                      
        #         grad = x_adv.grad.detach()
        #         grad = grad.sign()
        #         x_adv = x_adv + self.config['attack_lr'] * grad
        #     else:
        #         # Targeted attacks - gradient descent
        #         assert self.target.size() == y.size()           
        #         loss = F.cross_entropy(logits, self.target)
        #         loss.backward()
        #         grad = x_adv.grad.detach()
        #         grad = grad.sign()
        #         x_adv = x_adv - self.config['attack_lr'] * grad

        #     # Projection
        #     x_adv = x + torch.clamp(x_adv - x, min=-self.config['eps'], max=self.config['eps'])
        #     x_adv = x_adv.detach()
        #     x_adv = torch.clamp(x_adv, *self.clamp)

        # return x_adv
    items = 0
    for batch in train_dataloader:
        if items == 4:
            break
        image_path,texts = batch 
        # print(x)
        for epoch in iterator:
            model.zero_grad()

            image = Image.open(image_path[0])
            # if self.config['random_init'] :
                # x_adv = self._random_init(x_adv)
            x = torch.unbind(T(image))
            x_adv = x[0].detach().clone()
            x_adv.requires_grad = True
            image_adv = P(x_adv)
            # print(image_adv)
            inputs = processor(text=[caption_map[items]], images=image_adv, return_tensors="pt", padding=True)
            # print(f'inputs: {inputs}')
            inputs['pixel_values'].requires_grad = True
            print(f'x_adv.shape: {x_adv.shape}')
            print(f'inputs["pixel_values"].shape: {inputs["pixel_values"].shape}')
            outputs = model(**inputs)
            logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text

            if targeted_attack:
                counter_inputs = processor(text=[original_map[items]], images=image, return_tensors='pt', padding=True)
                counter_outputs = model(**counter_inputs)
                logits_per_counter_text = counter_outputs.logits_per_text
                print(f'logits_per_counter_text shape: {logits_per_counter_text.shape}')

            print(f'logits_per_text shape: {logits_per_text.shape}')
            print(f'logits_per_image shape: {logits_per_image.shape}')

            ground_truth = torch.arange(len(image_path),dtype=torch.long,device=device)
            print(f'ground_truth shape: {ground_truth.shape}')
            print()
            total_loss = -(loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2 
            if targeted_attack:
                total_loss += loss_txt(logits_per_text, logits_per_counter_text)/3

            total_loss.backward()
            print(f'x_adv: {x_adv}')
            print(f'image_adv: {image_adv}')
            grad = x_adv.grad.detach()
            grad = grad.sign()
            x_adv = x_adv + attack_lr * grad

            # Projection
            x_adv = x + torch.clamp(x_adv - x, min=-attack_eps, max=attack_eps)
            x_adv = x_adv.detach()
            x_adv = torch.clamp(x_adv, *self.clamp)

            # Put the image back
            x_adv_images.append(P(x_adv))

            # if device == "cpu":
            #     optimizer.step()
            # else : 
            #     convert_models_to_fp32(model)
            #     optimizer.step()
            #     clip.model.convert_weights(model)
            items += 1
        # iterator.set_description
        # print(f'total_loss: {total_loss}')



if __name__ == '__main__':
    # model_original, preprocess = clip.load("ViT-L/14",device=device,jit=False) #Must set jit=False for training
    # torch.save({
    #         # 'textEncoder_state_dict': textEncoder.state_dict(),
    #         'model_state_dict': model_original.state_dict(),
    #         }, f"model_checkpoints/L-14/model_original.pt")

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # torch.save({
    #         # 'textEncoder_state_dict': textEncoder.state_dict(),
    #         'model_state_dict': model.state_dict(),
    #         }, f"model_checkpoints/sd_cllip/model_original_new.pt")

    textEncoder = FrozenCLIPEmbedder()

    # use your own data
    train_dataset = CocoDetection(root='./coco-dataset/')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size = BATCH_SIZE) #Define your own dataloader


    if device == "cpu":
        model.float()
    else :
        clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

    x_adv_images = do_attack(model, train_dataloader)


    