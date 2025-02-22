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
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from autoattack import AutoAttack
import argparse
from PIL import Image


class CLIPWrapper(nn.Module):
    def __init__(self, model_name_or_path, prompts_with_nud, prompts_with_clean, device = 'cuda', ckpt_path = None):
        super(CLIPWrapper, self).__init__()
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)
        self.model = CLIPModel.from_pretrained(model_name_or_path).to(device)
        if ckpt_path is not None:
            te_checkpoint = torch.load(ckpt_path)
            try:
                self.model.text_model.load_state_dict(te_checkpoint['textEncoder_state_dict'])
            except:
                te_checkpoint['textEncoder_state_dict'].pop('embeddings.position_ids')
                self.model.text_model.load_state_dict(te_checkpoint['textEncoder_state_dict'])
        self.prompts_with_nud = prompts_with_nud
        self.prompts_with_clean = prompts_with_clean
        self.tensor_img_dict = {}
    
    def add_dict_entry(self, image_tensor, pil_image):
        self.tensor_img_dict[image_tensor] = pil_image

    def forward(self, images_tensor):
        # # Use the CLIP processor to encode text and images
        # print(f'image_tensors shape: {images_tensor.shape}')
        # images = self.tensor_img_dict[images_tensor]
        inputs = self.processor([self.prompts_with_nud, self.prompts_with_clean], return_tensors="pt", padding=True).to(self.device)
        inputs['pixel_values'] = images_tensor
        
        # Forward pass through the CLIP model
        outputs = self.model(**inputs)
        # print(f'logits shape: {outputs.logits_per_image.shape}')

        return outputs.logits_per_image
    

def do_attack(model, x, y, batch_size=1, device='cuda', eps=8/255):
    print(f'Using eps = {eps}')
    print(f'shape of x is :{x.shape}') 
    print(f'shape of y is: {y.shape}')
    # apgd-t, fab-t, square (these are the others besides apgd-ce) for attacks_to_run. Only apgd-ce and square apply since the other two have 9 target classes
    adversary = AutoAttack(model, norm='L2', eps=eps, device=device, verbose=True, version='custom', attacks_to_run=['apgd-ce', 'square'])
    # x = x.unsqueeze(1) # simulate single channel

    # x_adv = adversary.run_standard_evaluation(x, y, bs=x.shape[0])
    x_advs = []
    for i in range(1, len(y)):
        x_to_use = x[(i-1)*batch_size:i*batch_size].to(device)
        y_to_use = y[(i-1)*batch_size:i*batch_size].type(torch.LongTensor).to(device)
        print(f'len of x_to_use is: {len(x_to_use)} with shape {x_to_use.shape}')
        print(f'len of y_to_use is: {len(y_to_use)} with shape {y_to_use.shape}')
        if len(x_to_use) == 0:
            print('Finished')
            break
        x_adv = adversary.run_standard_evaluation(x_to_use, y_to_use, bs=x_to_use.shape[0])
        x_advs.append(x_adv.cpu())
        # print(x_adv)
        print('Done a run')

    return x_advs


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default = 'nudity')
parser.add_argument('--suffix', type=str, default=1)
parser.add_argument('--image_path', type=str, default = '/home/test/stable-diffusion/oct6-nudity-12-loss1356-baseline/samples')
parser.add_argument('--prompts_with_nud', type=str, default='nude')
parser.add_argument('--prompts_with_clean', type=str, default='clean')
parser.add_argument('--ckpt_path', default=None, help='Path to finetuned CLIP model')
args = parser.parse_args()

model_name_or_path = "openai/clip-vit-large-patch14"
device = 'cuda'
model = CLIPWrapper(model_name_or_path, args.prompts_with_nud, args.prompts_with_clean, ckpt_path=args.ckpt_path).to(device)


image_list = []
image_tensors = []
img_names = os.listdir(args.image_path)
img_names.sort()
for path in img_names:
    image_list.append(Image.open(os.path.join(args.image_path, path)))
    inputs = model.processor([args.prompts_with_nud, args.prompts_with_clean], image_list[-1:], return_tensors="pt", padding=True)
    # print(f"inputs['pixel_values']: {inputs['pixel_values'].shape}")
    # model.add_dict_entry(inputs['pixel_values'], image_list[-1:])
    image_tensors.append(inputs['pixel_values'].to(device))

image_tensor_full = torch.cat(image_tensors).to(device)
print(f'image_tensor_full shape: {image_tensor_full.shape}')
# label = torch.tensor([1,0]).to(device)
# labels = torch.stack([label for _ in range(image_tensor_full.shape[0])]).to(device)
labels = torch.zeros((image_tensor_full.shape[0]))

x_advs = do_attack(model, image_tensor_full, labels)
torch.save(x_advs, f'{args.experiment}_adv_{args.suffix}.pt')

# lower_limit, upper_limit = 0, 1
# def clamp(X, lower_limit, upper_limit):
#     return torch.max(torch.min(X, upper_limit), lower_limit)


# def attack_pgd(model, processor, criterion, X, target, text_tokens, alpha,
#                attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
#     delta = torch.zeros_like(X).cuda()
#     if norm == "l_inf":
#         delta.uniform_(-epsilon, epsilon)
#     elif norm == "l_2":
#         delta.normal_()
#         d_flat = delta.view(delta.size(0), -1)
#         n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
#         r = torch.zeros_like(n).uniform_(0, 1)
#         delta *= r / n * epsilon
#     else:
#         raise ValueError
#     delta = clamp(delta, lower_limit - X, upper_limit - X)
#     delta.requires_grad = True
#     for _ in range(attack_iters):
#         _images = processor(X + delta)
#         prompted_images = _images

#         output, _ = multiGPU_CLIP(model, prompted_images, text_tokens)

#         loss = criterion(output, target)

#         loss.backward()
#         grad = delta.grad.detach()
#         d = delta[:, :, :, :]
#         g = grad[:, :, :, :]
#         x = X[:, :, :, :]
#         if norm == "l_inf":
#             d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
#         elif norm == "l_2":
#             g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
#             scaled_g = g / (g_norm + 1e-10)
#             d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
#         d = clamp(d, lower_limit - x, upper_limit - x)
#         delta.data[:, :, :, :] = d
#         delta.grad.zero_()

#     return delta
