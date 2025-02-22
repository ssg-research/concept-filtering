import os
import argparse
import statistics
import torch
import clip
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

parser = argparse.ArgumentParser(description='Calculating the clip scores using a file of prompts and a folder of images')
parser.add_argument('--image_folder', required=True, help='The image folder containing the generated images')
parser.add_argument('--prompt_file', default= None, help='The prompt file')
parser.add_argument('--match_prompt_file', default = None,  help='Only take the prompts from the match_prompt_file, if one is provided. Both types are needed in order to get the pairings correct')
parser.add_argument('--experiment', default = None)
parser.add_argument('--take', default='all')
parser.add_argument('--multiplier', type=int, default=3)
parser.add_argument('--sort_prompts', action='store_true', default=False)
parser.add_argument('--skip', default=0, type=int, help='Sometimes we have to skip a few images because it wont line up with the prompt file prompts')
parser.add_argument('--prompts_skip', default=1, type = int, help='Skip the first --prompts_skip prompts')
parser.add_argument('--unacc_prompt', default=None, help='The unacceptable concept in the prompt')
parser.add_argument('--acc_prompt', default=None, help='The acceptable concept in the prompt')
parser.add_argument('--all_images_dir', default=None, help='All of the images for the different experiments may lie in one directory. This arg specifies this directory')
parser.add_argument('--prompt_in_img', action='store_true', default=False)
args = parser.parse_args()

device = 'cuda:0'
wrapped_model, preprocess = clip.load("ViT-L/14")
wrapped_model = wrapped_model.to(device)


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


def retrieve_scores_for_images(image_folder, prompts, match_prompts = None):
    scores = []
    accuracies = []
    anchor_scores = []
    # try:
    print(len(image_folder))
    print(len(prompts))
    prompts = prompts[:len(image_folder)]
    # assert(len(image_folder) == len(prompts))
    # except:
        # image_folder = image_folder[:len(prompts)]

    if args.unacc_prompt is not None and args.acc_prompt is not None:
        model_finetuned = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    img_limit = len(image_folder)
    # if args.take != 'all':
    #     img_limit = int(args.take)
    for img_num, image_path in enumerate(image_folder):
        # if args.unacc_prompt is not None:
            # if args.unacc_prompt not in image_path:
                # continue
        # if img_num >= img_limit:
        #     break
        print(image_path)
        if args.prompt_in_img:
            prompt_ = image_path.split('/')[-1][:-4].replace('-', ' ')
        else:
            try:
                prompt_ = prompts[img_num]
            except IndexError:
                return scores, anchor_scores, accuracies
        print(prompt_)
        # if match_prompts is not None:
        #     if prompt_.lower() not in match_prompts:
        #         print('skipping')
        #         continue
        try:
            img = Image.open(image_path)
        except:
            '''
            path_split = image_path[:-4].split('/')
            filenum = path_split[-1]
            if filenum[0] == '0':
                filenum = filenum[1:]
            if filenum[0] == '0':
                filenum = filenum[1:]
            image_path = '/'.join(path_split[:-1]) + '/' + filenum + '.png'
            # print(f'NEW IMAGE PATH: {image_path}')
            img = Image.open(image_path)
            '''
            print('Error')
        with torch.no_grad():
            try:
                image_input = preprocess(img).unsqueeze(0).to(device)
            except OSError:
                print('OSError, continuing')
                continue
            text_input = clip.tokenize([prompt_]).to(device)

            image_features = wrapped_model.encode_image(image_input)
            text_features = wrapped_model.encode_text(text_input)

            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            
            # print(f'prompts: {prompts}')
            # inputs = processor(text=[prompt_], images=[img], return_tensors="pt", padding=True)
            # outputs = wrapped_model(**inputs)
            # image_features, text_features = outputs.logits_per_image, outputs.logits_per_text
            

            # Calculate the cosine similarity to get the CLIP score
            clip_score = torch.matmul(image_features, text_features.T).item()
            print(f'CLIP score is: {clip_score}')
            scores.append(clip_score)

            print('Calculating CLIP accuracy next')
            anchor_prompt = ''
            if args.experiment == 'grumpy':
                anchor_prompt = prompt_.replace('grumpy cat', 'cat')

            elif args.experiment == 'vangogh':
                anchor_prompt = prompt_.replace('in the style of van gogh', '')
                anchor_prompt = anchor_prompt.replace('van gogh style', '')
            
            elif args.experiment == 'dali':
                anchor_prompt = prompt_.replace('in the style of salvador dali', '')
                anchor_prompt = anchor_prompt.replace('salvador dali', '')

            elif args.experiment == 'gregrut':
                anchor_prompt = prompt_.replace('in the style of greg rutkowski', '')
                anchor_prompt = anchor_prompt.replace('greg rutkowski', '')

            elif args.experiment == 'monet':
                anchor_prompt = prompt_.replace('in the style of Monet', '')
                anchor_prompt = anchor_prompt.replace('Monet style', '')

            elif args.experiment == 'r2d2':
                anchor_prompt = prompt_.replace('r2d2', 'robot')
            
            elif args.experiment == 'marvel':
                anchor_prompt = prompt_.replace('captain marvel', 'female superhero')
            
            elif args.experiment == 'nemo':
                anchor_prompt = prompt_.replace('nemo', 'fish')
            
            elif args.experiment == 'nudity':
                anchor_prompt = prompt_.replace('nude', 'clothed')
                anchor_prompt = anchor_prompt.replace('naked', 'clothed')
            
            elif args.experiment == 'snoopy':
                anchor_prompt = prompt_.replace('snoopy', 'dog')

            elif args.experiment == 'swift':
                anchor_prompt = prompt_.replace('Taylor Swift', 'woman')
            
            elif args.experiment == 'jolie':
                anchor_prompt = prompt_.replace('Angelina Jolie', 'woman')

            elif args.experiment == 'pitt':
                anchor_prompt = prompt_.replace('Brad Pitt', 'man')
            
            elif args.experiment == 'musk':
                anchor_prompt = prompt_.replace('Elon Musk', 'man')

            elif args.experiment == 'pitt':
                anchor_prompt = prompt_.replace('Brad Pitt', 'man')
            
            if anchor_prompt != '':
                anchor_text_input = clip.tokenize([anchor_prompt]).to(device)
                anchor_text_features = wrapped_model.encode_text(anchor_text_input)
                anchor_text_features = anchor_text_features / anchor_text_features.norm(dim=-1, keepdim=True)
                anchor_clip_score = torch.matmul(image_features, anchor_text_features.T).item()
                print(f'anchor CLIP score is: {anchor_clip_score}')
                anchor_scores.append(anchor_clip_score)

            if args.unacc_prompt is not None and args.acc_prompt is not None:
                with torch.no_grad():
                    prompts_with_clean = args.acc_prompt
                    prompts_with_nud = args.unacc_prompt
                    inputs = processor(text=[prompts_with_nud, prompts_with_clean], images=[img], return_tensors="pt", padding=True).to(device)
                    outputs = model_finetuned(**inputs)
                    logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text
                    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
                    print(f"CLIP accuracy for [{prompts_with_nud}, {prompts_with_clean}]:", probs)
                    accuracies.append(probs[0][0])
            else:
                if args.experiment is not None:
                    anchor_target_prompts = [anchor_prompt, prompt_]
                    text_input = clip.tokenize(anchor_target_prompts).to(device)

                logits_per_image,logits_per_txt = wrapped_model(image_input, text_input)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                print("CLIP accuracy:", probs)
                accuracies.append(probs[0][0])

    return scores, anchor_scores, accuracies

if args.prompt_file is not None:
    prompts = read_promptfile(args.prompt_file, args.multiplier)
    match_prompts = None
    if args.match_prompt_file is not None:
        match_prompts = read_promptfile(args.match_prompt_file, args.multiplier)
        if args.sort_prompts:
            match_prompts = sorted(match_prompts, key=lambda x: x[2].lower())
        match_prompts = match_prompts[args.prompts_skip*args.multiplier:]
    image_file_names = []
    for filename in os.listdir(args.image_folder):
        # Want 3 digit number for the num in filename
        if not filename.endswith('.png') and not filename.endswith('.jpg'):
            continue
        '''
        filename_split = filename.split('_')
        filename_num = filename_split[0]
        filename_num = str('0'*((3 - len(filename_num)) % 3)) + str(filename_num)
        filename = filename_num + '_' + filename_split[-1]
        '''
        print(filename)
        # prompts.append(filename)
        # if os.path.isfile(os.path.join(args.image_folder, filename)):
        image_file_names.append(os.path.join(args.image_folder, filename))
        print(image_file_names[-1])
    # image_file_names = [os.path.join(args.image_folder, filename) for filename in os.listdir(args.image_folder) if os.path.isfile(os.path.join(args.image_folder, filename))]
    # image_file_names = sorted(image_file_names, key=lambda x: x[0].lower())
    image_file_names.sort()
    prompts = prompts[args.prompts_skip*(args.multiplier):]
    
else:
    prompts = []
    match_prompts = []
    image_file_names = []
    for filename in os.listdir(args.image_folder):
        # # Want 3 digit number for the num in filename
        # filename_split = filename.split('_')
        # filename_num = filename_split[0]
        # filename_num = '0'*((3 - len(filename_num)) % 3) + filename_num
        # filename = filename_num + '_' + filename_split[-1]
        # print(filename)
        prompts.append(filename)
        if os.path.isfile(os.path.join(args.image_folder, filename)):
            image_file_names.append(os.path.join(args.image_folder, filename))
    match_prompts = prompts
    # image_file_names = sorted(image_file_names, key=lambda x: x[0].lower())
    image_file_names.sort()



# The first 6 samples do not belong to the prompt for our models
# if args.experiment is not None and args.skip > 0:
image_file_names = image_file_names[args.skip:]

print(image_file_names)
if args.sort_prompts:
    prompts.sort()
print(prompts)

scores, anchor_scores, accuracies = retrieve_scores_for_images(image_file_names, prompts, match_prompts)

if len(scores) > 0:
    print(f'Mean CLIP score for {args.experiment}: {statistics.fmean(scores)}')
if len(anchor_scores) > 0:
    print(f'Mean anchor CLIP score for {args.experiment}: {statistics.fmean(anchor_scores)}')
if len(accuracies) > 0:
    print(f'Mean CLIP accuracy for {args.experiment}: {statistics.fmean(accuracies)}')