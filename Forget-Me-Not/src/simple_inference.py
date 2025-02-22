import os
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import csv

from .patch_lora import safe_open, parse_safeloras_embeds, apply_learned_embed_in_clip
def patch_ti(pipe, ti_paths):
    for weight_path in ti_paths.split('|'):
        token = None
        idempotent_token = True

        safeloras = safe_open(weight_path, framework="pt", device="cpu")
        tok_dict = parse_safeloras_embeds(safeloras)

        print(f'tokenizer type: {type(pipe.tokenizer)}')

        apply_learned_embed_in_clip(
            tok_dict,
            pipe.text_encoder,
            pipe.tokenizer,
            token=token,
            idempotent=idempotent_token,
        )


def begin_test(prompts_path = None):
    starting_idx = 219
    prompts = []
    # if experiment == 'nemo':
        # starting_idx += 99
    # with open('../stable-diffusion/assets/eval-prompts/coco-dataset/test2014.csv', 'r') as f:
    #     csv_reader = csv.DictReader(f, delimiter='\t')
    #     for i, row in enumerate(csv_reader):
    #         if i < starting_idx:
    #             continue
    #         elif i >= (219 + 200):
    #             break
    #         prompt = row['title']
    #         prompts.append(prompt)

    if prompts_path is not None and prompts_path != 'coco':
        with open(prompts_path, 'r') as f:
        # with open('/u1/concept-ablation/assets/eval_prompts/nudity_eval.txt', 'r') as f:
            for i, line in enumerate(f):
                prompts.append(line.strip())

        return prompts

    if prompts_path == 'coco':
        print('Sending')
        with open('/u1/stable-diffusion/assets/eval-prompts/coco-dataset/test2014_subset.csv', 'r') as f:
            csv_reader = csv.DictReader(f, delimiter='\t')
            for i, row in enumerate(csv_reader):
                # if i < starting_idx:
                #     continue
                # elif i >= (219 + 200):
                #     break
                prompt = row['title']
                prompts.append(prompt)
        print(prompts)

    # with open('/u1/stable-diffusion/nudity_prompts_used_i2p.log', 'r') as f:
    # # with open('/u1/concept-ablation/assets/eval_prompts/nudity_eval.txt', 'r') as f:
    #     for i, line in enumerate(f):
    #         if i % 6 != 0:
    #             continue
    #         prompts.append(line.strip())

    return prompts


def main(args):

    prompts = []

    if args.patch_ti is not None:
        print(f"Inference using Ti {args.pretrained_model_name_or_path}")
        # model_id = "stabilityai/stable-diffusion-2-1-base"
        model_id = 'CompVis/stable-diffusion-v1-4'
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
            "cuda"
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)            

        patch_ti(pipe, f"{args.pretrained_model_name_or_path}/step_inv_{args.patch_ti.max_train_steps_ti}.safetensors")

        inverted_tokens = args.patch_ti.placeholder_tokens.replace('|', '')
        if args.patch_ti.use_template == "object":
            prompts += [f"a photo of {inverted_tokens}"]
        elif args.patch_ti.use_template == "style":
            prompts += [f"a photo in the style of {inverted_tokens}"]
        else:
            raise ValueError("unknown concept type!")          

    if args.multi_concept is not None:
        print(f"Inference using {args.pretrained_model_name_or_path}...")
        model_id = args.pretrained_model_name_or_path
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
            "cuda"
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)                    
        for c, t in args.multi_concept:
            c = c.replace('-', ' ')
            if args.prompts_path is not None:
                print(args.prompts_path)

                prompts = begin_test(args.prompts_path)
            else:
                if t == "object":
                    # prompts += [f"a photo of {c}"]
                    # prompts += [f"a photo of nemo"]
                    prompts = begin_test()
                    # break
                elif t == "style":
                    prompts += [f"a photo in the style of {c}"]
                else:
                    raise ValueError("unknown concept type!")        

            

    torch.manual_seed(int(args.seed))
    output_folder = f"{args.save_dir}/{args.seed}/generated_images"
    os.makedirs(output_folder, exist_ok=True)

    for prompt in prompts:
        print(f'Inferencing: {prompt}')
        images = pipe(prompt, num_inference_steps=50, guidance_scale=8, num_images_per_prompt=8).images
        for i, im in enumerate(images):
            im.save(f"{output_folder}/o_{prompt.replace(' ', '-')[:30]}_{i}.jpg") 
            print(f"Image saved at {output_folder}/o_{prompt.replace(' ', '-')[:30]}_{i}.jpg")
            with open('nudity_prompts_used_i2p.log', 'a') as f:
                f.write(f'{prompt}\n') 