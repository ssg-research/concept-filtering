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
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPProcessor
import argparse
import random
import statistics
from torch.utils.data import Dataset, DataLoader


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def get_acc_unacc_prompt(args):
    if args.experiment == 'nudity':
        prompts_with_nud = f'nude'
        prompts_with_clean = f"clean"

    elif args.experiment == 'grumpy':
        prompts_with_nud = 'grumpy cat'
        prompts_with_clean = 'cat'

    elif args.experiment == 'vangogh':
        prompts_with_nud = 'in the style of van gogh'
        prompts_with_clean = 'in standard style'

    elif args.experiment == 'monet':
        prompts_with_nud = f'in the style of Monet'
        prompts_with_clean = 'in standard style'

    elif  args.experiment == 'dali':
        prompts_with_nud = f'in the style of Dali'
        prompts_with_clean = 'in standard style'

    elif args.experiment == 'gregrut':
        prompts_with_nud = f'in the style of Greg Rutowski'
        prompts_with_clean = 'in standard style'

    elif args.experiment == 'r2d2':
        prompts_with_nud = 'r2d2'
        prompts_with_clean = 'robot'

    elif args.experiment == 'marvel':
        prompts_with_nud = 'Captain Marvel'
        prompts_with_clean = 'female superhero'

    elif args.experiment == 'nemo':
        prompts_with_nud = 'nemo'
        prompts_with_clean = 'fish'

    elif args.experiment == 'musk':
        prompts_with_nud = 'Elon Musk'
        prompts_with_clean = 'man'

    elif args.experiment == 'pitt':
        prompts_with_nud = 'Brad Pitt'
        prompts_with_clean = 'man'

    elif args.experiment == 'swift':
        prompts_with_nud = 'Taylor Swift'
        prompts_with_clean = 'woman'
    
    elif args.experiment == 'jolie':
        prompts_with_nud = 'Angelina Jolie'
        prompts_with_clean = 'woman'
    
    elif args.experiment == 'snoopy':
        prompts_with_nud = 'Snoopy'
        prompts_with_clean = 'dog'
    
    elif args.experiment == 'violence' or args.experiment == 'peaceful':
        prompts_with_nud = 'violent'
        prompts_with_clean = 'peaceful'
    
    elif args.experiment == 'hateful':
        prompts_with_nud = 'hateful'
        prompts_with_clean = 'neutral'
    
    elif args.experiment == 'political' or args.experiment == 'casual':
        prompts_with_nud = 'political'
        prompts_with_clean = 'casual'
    
    elif args.experiment == 'hateful' or args.experiment == 'loving':
        prompts_with_nud = 'hateful'
        prompts_with_clean = 'loving'
    
    elif args.experiment == 'disturbing' or args.experiment == 'pleasant':
        prompts_with_nud = 'disturbing'
        prompts_with_clean = 'pleasant'
    
    return prompts_with_nud, prompts_with_clean

OPPOSING_CONCEPTS = ['violence', 'nudity']



# Custom dataset class
class QuadrupleDataset(Dataset):
    def __init__(self, target_text_chosen, anchor_text_chosen, image_paths_chosen, image_paths_target_chosen):
        self.target_text_chosen = target_text_chosen
        self.anchor_text_chosen = anchor_text_chosen
        self.image_paths_chosen = image_paths_chosen
        self.image_paths_target_chosen = image_paths_target_chosen

    def __len__(self):
        return len(self.target_text_chosen)

    def __getitem__(self, index):
        sample = {
            'target_text': self.target_text_chosen[index],
            'anchor_text': self.anchor_text_chosen[index],
            'anchor_img': self.image_paths_chosen[index],
            'target_img': self.image_paths_target_chosen[index]
        }
        # Perform any data preprocessing or transformations here
        # Return the sample as a dictionary or any other desired format
        return sample

 
class AbstractEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cpu", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)
        # print(f'outputs: {outputs}')
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)



EPOCHS = 100
BATCH_SIZE = 1

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model_original, preprocess = clip.load("ViT-L/14",device=device,jit=False) #Must set jit=False for training
textEncoder = FrozenCLIPEmbedder()


parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('--acc_text_path', type=str, default = None)
parser.add_argument('--unacc_text_path', type=str, default=None)
parser.add_argument('--unacc_img_path', type=str, default = None)
parser.add_argument('--acc_img_path', type=str, default = None)
parser.add_argument('--adv_prompts_file', type=str, default = None)
parser.add_argument('--adv_images', type=str, default = None)
parser.add_argument('--adv_counter_images', type=str, default=None)
parser.add_argument('--noreg', action='store_true', default = False)
parser.add_argument('--experiment', type=str, default='nemo')
parser.add_argument('--coco_valset', default = None)
parser.add_argument('--valset', default=None, help='Validation set with unacceptable images')
parser.add_argument('--epochs', default=None, type=int)
parser.add_argument('--alpha',type=float, default = 1)
parser.add_argument('--save_at', type=str, default = None)
parser.add_argument('--trial', type=int, default = 1)
args = parser.parse_args()


image_paths = []
base_img_dir = args.unacc_img_path
coco_valset = args.coco_valset
valset = args.valset
num = 0
if args.epochs is not None:
    EPOCHS = args.epochs
save_dir = os.path.join('model_checkpoints', 'L-patch-14', str(args.trial))
os.makedirs(save_dir, exist_ok=True)
anchor_img_names = os.listdir(base_img_dir)
anchor_img_names.sort()
for path in anchor_img_names:
    image_paths.append(os.path.join(base_img_dir, path))
    num += 1



anchor_text_nodup = []
with open(args.acc_text_path, 'r') as f:
    for line in f:
        anchor_text_nodup.append(line.strip())
        if len(anchor_text_nodup) >= len(image_paths):
            break

num_images_per_caption = 6
if args.experiment == 'nudity':
    num_images_per_caption = 3
anchor_text = [item for item in anchor_text_nodup for _ in range(num_images_per_caption)][:len(image_paths)]

target_text_nodup = []
with open(args.unacc_text_path, 'r') as f:
    for line in f:
        target_text_nodup.append(line.strip())
        if len(target_text_nodup) >= len(image_paths):
            break
target_text = [item for item in target_text_nodup for _ in range(num_images_per_caption)][:len(image_paths)]


image_paths_target = []
base_img_dir_target = args.acc_img_path
target_img_names = os.listdir(base_img_dir_target)
target_img_names.sort()
num = 0
for path in target_img_names:
    image_paths_target.append(os.path.join(base_img_dir_target, path))
    num += 1
    if len(image_paths_target) == len(image_paths) - 1:
        break
image_paths_target.append(os.path.join(base_img_dir_target, path))


adv_images = []
adv_counter_images = []
if args.adv_images is not None:
    adv_images = os.listdir(args.adv_images)
    for path in adv_images:
        adv_images.append(os.path.join(args.adv_images, path))

if args.adv_counter_images is not None:
    adv_counter_images = os.listdir(args.adv_counter_images)
    for path in adv_counter_images:
        adv_counter_images.append(os.path.join(args.adv_counter_images, path))
    

def select_from_list(target_text, anchor_text, image_paths, image_paths_target, num_to_take = 3):
    # Combine the lists using zip
    combined_lists = list(zip(target_text, anchor_text, image_paths, image_paths_target))
    combined_lists_final = []

    for i in range(num_to_take):
        combined_lists_parts = combined_lists[i::num_images_per_caption]
        combined_lists_final += combined_lists_parts
        print(len(combined_lists_final))

    # Unzip the shuffled lists
    target_text, anchor_text, image_paths, image_paths_target = zip(*combined_lists_final)

    return target_text, anchor_text, image_paths, image_paths_target


if args.experiment != 'nudity':
    num_to_take = 1
    target_text, anchor_text, image_paths, image_paths_target = select_from_list(target_text, anchor_text, image_paths, image_paths_target, num_to_take=num_to_take)
print(f'len image_paths: {len(image_paths)}')
print(f'len anchor_text: {len(anchor_text)}')
print(f'len caption map: {len(target_text)}')
print(f'len image_paths_target: {len(image_paths_target)}')
print(target_text)
print(image_paths)

def shuffle(target_text, anchor_text, image_paths, image_paths_target):
    # Combine the lists using zip
    combined_lists = list(zip(target_text, anchor_text, image_paths, image_paths_target))

    # Shuffle the combined lists
    random.shuffle(combined_lists)

    # Unzip the shuffled lists
    target_text, anchor_text, image_paths, image_paths_target = zip(*combined_lists)

    return target_text, anchor_text, image_paths, image_paths_target


def train_epoch(optimizer, target_text, anchor_text, images, images_target, processor, model, clip_scores = None, experiment = None, original_embeds = [], frozen_model = None):
    loss_img = torch.nn.CrossEntropyLoss()
    loss_txt = torch.nn.CrossEntropyLoss()
    loss_counter_txt = torch.nn.MSELoss(reduction='sum')

    if clip_scores is not None:
        clip_score_target, clip_score_anchor, clip_score_targettxt_anchorimg, clip_score_anchortxt_targetimg = clip_scores
        
        alpha_targettxt_anchorimg = 1-clip_score_targettxt_anchorimg

    else:

        alpha_targettxt_anchorimg = 0.99

        if args.experiment == 'nudity' or args.experiment == 'r2d2' or args.noreg:
            alpha_target, alpha_anchor, alpha_targettxt_anchorimg = 1., 1., 1.

    
    if experiment in OPPOSING_CONCEPTS:
        inputs = processor(text=target_text, images=images, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text


        counter_inputs = processor(text=anchor_text, images=images, return_tensors='pt', padding=True).to(device)
        counter_outputs = model(**counter_inputs)
        logits_per_counter_text = counter_outputs.logits_per_text

        counter_img_inputs = processor(text=target_text, images=images_target, return_tensors="pt", padding=True).to(device)
        counter_img_outputs = model(**counter_img_inputs)
        logits_per_image_target, logits_per_counter_txt_target = counter_img_outputs.logits_per_image, counter_img_outputs.logits_per_text

        print(f'logits_per_text shape: {logits_per_text.shape}')
        print(f'logits_per_image shape: {logits_per_image.shape}')

        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

        

        counter_text_embeds = counter_outputs.text_embeds / torch.linalg.norm(counter_outputs.text_embeds, dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / torch.linalg.norm(outputs.text_embeds, dim=-1, keepdim=True)
        image_embeds = outputs.image_embeds / torch.linalg.norm(outputs.image_embeds, dim=-1, keepdim=True)
        counter_image_embeds = counter_img_outputs.image_embeds / torch.linalg.norm(counter_img_outputs.image_embeds, dim=-1, keepdim=True)

        text_sim = torch.matmul(text_embeds, counter_text_embeds.t())
        image_sim = torch.matmul(image_embeds, counter_image_embeds.t())
        text_img_sim = torch.matmul(text_embeds, image_embeds.t())
        text_counter_img_sim = torch.matmul(text_embeds, counter_image_embeds.t())
        counter_text_counter_img_sim = torch.matmul(counter_text_embeds, counter_image_embeds.t())
        counter_text_img_sim = torch.matmul(counter_text_embeds, image_embeds.t())
        # print(f'text_sim: {text_sim}')
        print(image_embeds.shape)
        print(counter_image_embeds.shape)
        print(f'l2 norm text_sim: {torch.linalg.norm(text_sim)}')
        print(f'l2 norm text_embeds: {torch.linalg.norm(text_embeds)}')
        print(f'l2 norm counter_text_embeds: {torch.linalg.norm(counter_text_embeds)}')
        print(f'l2 norm text_embeds - counter_text_embeds: {torch.linalg.norm((text_embeds - counter_text_embeds))}')
        print()
        # print(f'image_sim: {image_sim}')
        print(f'l2 norm image_sim: {torch.linalg.norm(image_sim)}')
        print(f'l2 norm image_embeds: {torch.linalg.norm(image_embeds)}')
        print(f'l2 norm counter_image_embeds: {torch.linalg.norm(counter_image_embeds)}')
        print(f'l2 norm image_embeds - counter_image_embeds: {torch.linalg.norm((image_embeds - counter_image_embeds))}')
        print()
        print('Now checking drift of images and text')
        print(f'text_embeds img_embeds sim: {torch.linalg.norm(text_img_sim)}')
        print(f'image_embeds - text_embeds norm: {torch.linalg.norm((image_embeds - text_embeds))}')
        print(f'counter_text_embeds counter_img_embeds sim: {torch.linalg.norm(counter_text_counter_img_sim)}')
        print(f'counter_image_embeds - counter_text_embeds norm: {torch.linalg.norm((counter_image_embeds - counter_text_embeds))}')
        print(f'Now see how they compare to the opposite side')
        print(f'counter_text image sim: {torch.linalg.norm(counter_text_img_sim)}')
        print(f'image_embeds - counter_text_embeds norm: {torch.linalg.norm((image_embeds - counter_text_embeds))}')
        print(f'text counter_img sim: {torch.linalg.norm(text_counter_img_sim)}')
        print(f'counter_image_embeds - text_embeds norm: {torch.linalg.norm((counter_image_embeds - text_embeds))}')
        print('Now comparing with original if applicable')
        print()
        if len(original_embeds) > 0:
            original_text_embeds, original_counter_text_embeds, original_image_embeds, original_counter_image_embeds = original_embeds
            image_original_image_sim = torch.matmul(image_embeds, original_image_embeds.t())
            text_original_text_sim = torch.matmul(text_embeds, original_text_embeds.t())
            counter_img_original_sim = torch.matmul(counter_image_embeds, original_counter_image_embeds.t())
            counter_text_original_sim = torch.matmul(counter_text_embeds, original_counter_text_embeds.t())
            print(f'image_orginal_img sim: {torch.linalg.norm(image_original_image_sim)}')
            print(f'image_embeds now - original_image_embeds norm: {torch.linalg.norm((image_embeds - original_image_embeds))}')
            print(f'text_orginal_text sim: {torch.linalg.norm(text_original_text_sim)}')
            print(f'text_embeds now - original text_embeds: {torch.linalg.norm((text_embeds - original_text_embeds))}')
            print(f'counter_text_original sim: {torch.linalg.norm(counter_text_original_sim)}')
            print(f'counter_image_embeds now - original counter_image_embeds: {torch.linalg.norm((counter_image_embeds - original_counter_image_embeds))}')
            print(f'counter_img_original_sim: {torch.linalg.norm(counter_img_original_sim)}')
            print(f'counter_text_embeds now - original counter_text_embeds: {torch.linalg.norm(counter_text_embeds - original_counter_text_embeds)}')
            print()
        
        total_loss = (args.alpha * (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth)) \
                    - args.alpha * (loss_img(logits_per_image_target,ground_truth) + loss_txt(logits_per_text, ground_truth)) \
                    # Preserve
                    + args.alpha * (loss_img(logits_per_image_target,ground_truth) + loss_txt(logits_per_counter_text, ground_truth)) \
                    - args.alpha * (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_counter_text, ground_truth)) \
                    - args.alpha * loss_counter_txt(outputs.text_embeds, counter_outputs.text_embeds)) / logits_per_text.shape[0]

    else: 
        outputs, counter_outputs = None, None

        target_text = processor(text=target_text, return_tensors='pt', padding=True).to(device)
        anchor_text = processor(text=anchor_text, return_tensors='pt', padding=True).to(device)

        text_embeds = model.get_text_features(**target_text)
        counter_text_embeds = model.get_text_features(**anchor_text)
        text_direction = (text_embeds - counter_text_embeds)#.mean(axis=0, keepdim=True)
        text_direction = text_direction / text_direction.norm(dim=-1, keepdim=True)
        print(text_direction.shape)
        print(f'text_direction norm: {torch.linalg.norm(text_direction) / text_direction.shape[0]}')
        print(f'text_embeds norm: {torch.linalg.norm(text_embeds/text_embeds.norm(dim=-1, keepdim=True)) / text_embeds.shape[0]}')
        print(f'counter_text_embeds norm: {torch.linalg.norm(counter_text_embeds/counter_text_embeds.norm(dim=-1, keepdim=True)) / counter_text_embeds.shape[0]}')
        print(f'text_direction target text sim: {(text_embeds/text_embeds.norm(dim=-1, keepdim=True) @ text_direction.T).mean()}')
        print(f'text_direction anchor text sim: {(counter_text_embeds/counter_text_embeds.norm(dim=-1, keepdim=True) @ text_direction.T).mean()}')
        print()

        total_loss = -torch.linalg.norm(text_direction)

    
    total_loss.backward()
    if device == "cpu":
        optimizer.step()
    else : 
        optimizer.step()
    print(f'total_loss: {total_loss}')
    if outputs is not None and counter_outputs is not None:
        return processor, model, total_loss, text_embeds, counter_text_embeds, outputs.image_embeds, counter_outputs.image_embeds
    else:
        return processor, model, total_loss, text_embeds, counter_text_embeds, outputs, counter_outputs


def calculate_clip_score(img_path, prompt):
    imgs = [Image.open(image) for image in img_path]
    clip_scores = []
    with torch.no_grad():
        for i, img in enumerate(imgs):
            image_input = preprocess(img).unsqueeze(0).to(device)
            text_input = clip.tokenize([prompt[i]]).to(device)

            image_features = wrapped_model.encode_image(image_input)
            text_features = wrapped_model.encode_text(text_input)
        
            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate the cosine similarity to get the CLIP score
            clip_score = torch.matmul(image_features, text_features.T).item()
            print(f'CLIP score is: {clip_score}')

            clip_scores.append(clip_score)

    # May choose max instead
    return statistics.fmean(clip_scores)


if __name__ == '__main__':
    unacc_prompt, acc_prompt = get_acc_unacc_prompt(args)
    if coco_valset is not None:
        coco_val_img_files = [os.path.join(coco_valset, p) for p in os.listdir(coco_valset)]
    if valset is not None:
        val_img_files = [os.path.join(valset, p) for p in os.listdir(valset)]
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    te_state_dict = torch.load('base_clip.pt')['textEncoder_state_dict']
    try:
        model.text_model.load_state_dict(te_state_dict)
    except:
        te_state_dict['embeddings.position_ids'] = model.text_model.state_dict()['embeddings.position_ids']
        model.text_model.load_state_dict(te_state_dict)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    early_stopper = EarlyStopper(2)


    #https://github.com/openai/CLIP/issues/57
    def convert_models_to_fp32(model): 
        for p in model.parameters(): 
            p.data = p.data.float()
            if p.grad is not None: 
                p.grad.data = p.grad.data.float() 


    loss_img = torch.nn.CrossEntropyLoss()
    loss_txt = torch.nn.CrossEntropyLoss()
    # loss_counter_txt = torch.nn.MSELoss()
    loss_counter_txt = torch.nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    wrapped_model, preprocess = clip.load("ViT-L/14")
    wrapped_model = wrapped_model.to(device)

    iterator = tqdm(range(EPOCHS+1))
    single_batchers = []

    target_text_chosen, anchor_text_chosen, image_paths_chosen, image_paths_target_chosen = shuffle(target_text, anchor_text, image_paths, image_paths_target)
    batch_num = 10
    train_dataset = QuadrupleDataset(target_text_chosen, anchor_text_chosen, image_paths_chosen, image_paths_target_chosen)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_num, shuffle=True)
    
    all_original_embeds = [[] for _ in train_dataloader]
    for epoch in iterator:
        if args.experiment in single_batchers:
            target_text_chosen, anchor_text_chosen, image_paths_chosen, image_paths_target_chosen = shuffle(target_text, anchor_text, image_paths, image_paths_target)
            # Convert the tuples back to lists
            target_text_chosen = list(target_text_chosen)[:10]
            anchor_text_chosen = list(anchor_text_chosen)[:10]
            image_paths_chosen = list(image_paths_chosen)[:10]
            image_paths_target_chosen = list(image_paths_target_chosen)[:10]
            for i, image in enumerate(image_paths_chosen):
                optimizer.zero_grad()
                images = [Image.open(image)]
                images_target = [Image.open(image_paths_target_chosen[i])]
                processor, model, total_loss = train_epoch(optimizer, [target_text_chosen[i]], [anchor_text_chosen[i]], images, images_target, processor, model, experiment = args.experiment)
                
        else:            
            for j, batch in enumerate(train_dataloader):
                # The vison model should remain frozen
                for param in model.vision_model.parameters():
                    param.requires_grad = False

                target_text_chosen = batch['target_text']
                anchor_text_chosen = batch['anchor_text']
                image_paths_chosen = batch['anchor_img']
                image_paths_target_chosen = batch['target_img']

                clip_scores = None
                
                optimizer.zero_grad()
                images = [Image.open(ip) for ip in image_paths_chosen]
                images_target = [Image.open(ip) for ip in image_paths_target_chosen]
                processor, model, total_loss, original_text_embeds, original_counter_text_embeds, original_image_embeds, original_counter_image_embeds = train_epoch(optimizer, target_text_chosen, anchor_text_chosen, images, images_target, processor, model, clip_scores = clip_scores, experiment = args.experiment, original_embeds=all_original_embeds[j], frozen_model=None)
                if epoch == 0:
                    all_original_embeds[j].append(original_text_embeds)
                    all_original_embeds[j].append(original_counter_text_embeds)
                    all_original_embeds[j].append(original_image_embeds)
                    all_original_embeds[j].append(original_counter_image_embeds)
                if args.save_at is not None:
                    if int(epoch) == int(args.save_at):
                        torch.save({
                            'epoch': epoch,
                            'textEncoder_state_dict': model.text_model.state_dict(),
                            # 'visionEncoder_state_dict': model.vision_model.state_dict(),
                            # 'model_state_dict': model.state_dict(),
                            # 'optimizer_state_dict': optimizer.state_dict(),
                            'loss': total_loss,
                            }, os.path.join(save_dir, f"model_{epoch}_ce_{args.experiment}.pt")) #cat2 models use 5 prompts from either set
                        break
                else:
                    if epoch % 6 == 0 or (int(epoch) == EPOCHS):
                        torch.save({
                            'epoch': epoch,
                            'textEncoder_state_dict': model.text_model.state_dict(),
                            # 'visionEncoder_state_dict': model.vision_model.state_dict(),
                            # 'model_state_dict': model.state_dict(),
                            # 'optimizer_state_dict': optimizer.state_dict(),
                            'loss': total_loss,
                            }, os.path.join(save_dir, f"model_{epoch}_ce_{args.experiment}.pt"))

                    # if coco_valset is not None:
                    #     _, mean_num_detections = retrieve_scores_for_images(coco_val_img_files, unacc_prompt, acc_prompt, model)
                    #     print(f'Num predictions: {mean_num_detections}')
                    #     if early_stopper.early_stop(mean_num_detections):
                    #         break
                    # if valset is not None:
                    #     _, mean_num_detections = retrieve_scores_for_images(val_img_files, unacc_prompt, acc_prompt, model)
                    #     print(f'Num predictions: {mean_num_detections}')
                    #     if early_stopper.early_stop(mean_num_detections):
                    #         break

