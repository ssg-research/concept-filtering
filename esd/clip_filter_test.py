import os
import argparse
import statistics
import torch
import clip
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='Calculating the clip scores using a file of prompts and a folder of images')
parser.add_argument('--unacc_prompt', help='The unacceptable concept in the prompt')
parser.add_argument('--acc_prompt', help='The acceptable concept in the prompt')
parser.add_argument('--images_dir', default=None, help='An optional directory of images')
parser.add_argument('--clip_ckpt', default=None, help='Path to a finetuned clip model')
parser.add_argument('--unacc_prompt_embed_ckpt', default=None)
parser.add_argument('--acc_prompt_embed_ckpt', default = None)
parser.add_argument('--diff_embed_npy', default=None)
parser.add_argument('--which')



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


def retrieve_scores_for_images(image_folder, ckpt_path, unacc_prompt_embed_ckpt, acc_prompt_embed_ckpt, diff_embed_npy = None):
    
    accuracies = []
    # text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder").to(device)
    if args.unacc_prompt is not None and args.acc_prompt is not None:
        model_finetuned = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        if ckpt_path is not None:
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

    for img_num, image_path in enumerate(image_folder):
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
            image_inputs = processor(images=[img], return_tensors="pt", padding=True).to(device)
            image_features = model_finetuned.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            if unacc_prompt_embed_ckpt is not None and acc_prompt_embed_ckpt is not None:
                # print(image_features.shape)
                mean_unacc_embed = torch.load(unacc_prompt_embed_ckpt)
                mean_acc_embed = torch.load(acc_prompt_embed_ckpt)
                print(mean_acc_embed.shape)
                print(image_features.shape)
                # print(mean_unacc_embed.shape)

                unacc_sim = torch.matmul(image_features, mean_unacc_embed.t())
                acc_sim = torch.matmul(image_features, mean_acc_embed.t())        
                full_compare = torch.cat([unacc_sim, acc_sim], dim=-1).softmax(dim=-1).detach().cpu().numpy()
                print(full_compare.shape)
                if unacc_sim > acc_sim:
                    print(f'CLIP accuracy for [{args.unacc_prompt}, {args.acc_prompt}]: {full_compare}')
                    print('Nud detected')
                accuracies.append(unacc_sim)
            
            elif diff_embed_npy is not None:
                # # diff_embed = model_finetuned.text_model(torch.from_numpy(np.load(diff_embed_npy)).long().to(device))
                # diff_embed = torch.from_numpy(np.load(diff_embed_npy)).to(device)
                # diff_embed = diff_embed / diff_embed.norm(dim=-1, keepdim=True)
                # print(diff_embed.shape)
                # print(image_features.shape)

                prompts_with_clean = args.acc_prompt
                prompts_with_nud = args.unacc_prompt
                inputs = processor(text=[prompts_with_nud, prompts_with_clean], return_tensors="pt", padding=True).to(device)
                outputs = model_finetuned.get_text_features(**inputs)
                outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
                diff_embed = outputs[0] - outputs[1] # difference between the two text embeddings
                diff_embed = diff_embed / diff_embed.norm(dim=-1, keepdim=True)
                projected_img = torch.matmul(diff_embed, image_features.t()) * diff_embed
                # # project -nud onto the same vector
                # outputs[1] = torch.matmul(diff_embed, -outputs[0].t()) * diff_embed

                diff_to_unacc = torch.linalg.norm((projected_img - outputs[0]))
                diff_to_acc = torch.linalg.norm((projected_img - outputs[1]))
                print(f'Comparing [{args.unacc_prompt}, {args.acc_prompt}]:[{diff_to_unacc}, {diff_to_acc}]')
                # if (diff_to_unacc > diff_to_acc):
                #     print(f'Comparing [{args.unacc_prompt}, {args.acc_prompt}]:[{diff_to_unacc}, {diff_to_acc}]')

                # full_compare = torch.stack([-torch.abs(projected_img - outputs[0]), -torch.abs(projected_img - outputs[1])], dim=-1).softmax(dim=-1).detach().cpu().numpy()
                full_compare = torch.tensor([-torch.linalg.norm(projected_img - outputs[0]), -torch.linalg.norm(projected_img - outputs[1])]).softmax(dim=-1).detach().cpu().numpy()
                print(full_compare.shape)
                print(f'CLIP accuracy for [{args.unacc_prompt}, {args.acc_prompt}]: {full_compare}')
                if full_compare[0] > full_compare[1]:
                    print('Nud detected')
                accuracies.append(full_compare[0])
                

            
    
            elif args.unacc_prompt is not None and args.acc_prompt is not None:
                with torch.no_grad():
                    prompts_with_clean = args.acc_prompt
                    prompts_with_nud = args.unacc_prompt
                    inputs = processor(text=[prompts_with_nud, prompts_with_clean], images=[img], return_tensors="pt", padding=True).to(device)
                    # inputs.input_ids[1] = -inputs.input_ids[0] # NEW
                    outputs = model_finetuned(**inputs)
                    logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text
                    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
                    print(f"CLIP accuracy for [{prompts_with_nud}, {prompts_with_clean}]:", probs)
                    accuracies.append(probs[0][0])
                    if probs[0][0] > probs[0][1]:
                        print('Nud detected')
           

    return accuracies

if __name__ == '__main__':
    args = parser.parse_args()
    device = 'cuda:0'
    # device = 'cpu'
    wrapped_model, preprocess = clip.load("ViT-L/14")
    wrapped_model = wrapped_model.to(device)
    # for i in range(6):
    if args.images_dir is None:
        img_files = read_file(f'data/{args.which}_images.txt')
    else:
        img_files = [os.path.join(args.images_dir, p) for p in os.listdir(args.images_dir)]
    accuracies = retrieve_scores_for_images(img_files, args.clip_ckpt, args.unacc_prompt_embed_ckpt, args.acc_prompt_embed_ckpt, args.diff_embed_npy)
    print(f'Mean CLIP accuracy for {args.unacc_prompt}, {args.acc_prompt}: {statistics.fmean(accuracies)}')
    print(f'Score for type {args.which} above')