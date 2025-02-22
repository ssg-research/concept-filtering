import sys
from PIL import Image
import os
# from diffusers import StableDiffusionPipeline


# if len(sys.argv) < 2:
#   sys.exit("""Usage: python run.py path-to-image [path-to-image-2 ...]
# Passing multiple images will optimize a single prompt across all passed images, useful for style transfer.
# """)

config_path = "sample_config.json"



# adv_prompt_file = 'adv_prompts_ours_official_tiattack_on_ourclip.txt'
# if len(sys.argv) > 1:
#   model_dir = sys.argv[1]
ourclip = False
if len(sys.argv) > 1:
  adv_prompt_file = sys.argv[1]
if len(sys.argv) > 2:
  experiment = sys.argv[2]
if len(sys.argv) > 3:
  image_paths_dir = sys.argv[3]
if len(sys.argv) > 4:
  ourclip = bool(sys.argv[4])

# image_paths = sys.argv[1:]
image_paths = []
if os.path.isdir(image_paths_dir):
  image_path_names = sorted(os.listdir(image_paths_dir))
  for path in image_path_names:
    image_paths.append(os.path.join(image_paths_dir, path))
# image_paths = image_paths[:6]

print(image_paths)

# load the target image
# images = [Image.open(image_path) for image_path in image_paths]

# defer loading other stuff until we confirm the images loaded
import argparse
import open_clip
from optim_utils import *

print("Initializing...")

# load args
args = argparse.Namespace()
args.__dict__.update(read_json(config_path))

# You may modify the hyperparamters here
args.print_new_best = True
args.experiment = experiment
args.seed = 42
args.ourclip = False
if ourclip:
  args.ourclip = True

# load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = 'ViT-L-14-openai'
model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrain, device=device)
# ckpt = torch.load(f'{model_dir}/open_clip_pytorch_model.bin')
# model.load_state_dict(ckpt)
print(f"Running for {args.iter} steps.")
if getattr(args, 'print_new_best', False) and args.print_step is not None:
  print(f"Intermediate results will be printed every {args.print_step} steps.")
print(type(model))

# try:
#   if args.ti_path is not None:
#     imagenet_templates_small = [
#       "a photo of a {}",
#       "a rendering of a {}",
#       "a cropped photo of the {}",
#       "the photo of a {}",
#       "a photo of a clean {}",
#       "a photo of a dirty {}",
#       "a dark photo of the {}",
#       "a photo of my {}",
#       "a photo of the cool {}",
#       "a close-up photo of a {}",
#       "a bright photo of the {}",
#       "a cropped photo of a {}",
#       "a photo of the {}",
#       "a good photo of the {}",
#       "a photo of one {}",
#       "a close-up photo of the {}",
#       "a rendition of the {}",
#       "a photo of the clean {}",
#       "a rendition of a {}",
#       "a photo of a nice {}",
#       "a good photo of a {}",
#       "a photo of the nice {}",
#       "a photo of the small {}",
#       "a photo of the weird {}",
#       "a photo of the large {}",
#       "a photo of a cool {}",
#       "a photo of a small {}",
#     ]
#     pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path=args.ti_path, safety_checker=None).to(device)
#     for i in range(6, len(images), 6):
#       all_prompts = []
#       for j in range(6):
#         text = random.choice(imagenet_templates_small).format("<s1>")
#         all_prompts.append(text)
#       prompt_embeds, _ = pipe.encode_prompt(all_prompts, device, 1, True)
#       prompt_ids = pipe.tokenizer(all_prompts, padding="max_length", max_length = pipe.tokenizer.model_max_length, truncation = True, return_tensors = "pt").input_ids
#       all_target_features = prompt_embeds[torch.arange(prompt_embeds.shape[0]), prompt_ids.argmax(dim=-1)] @ model.text_projection
#       all_target_features = all_target_features.to('cpu')
#       learned_prompt = optimize_prompt(model, preprocess, args, device, all_target_features=all_target_features)
#       print(learned_prompt)
#       with open(adv_prompt_file, "a") as f:
#         f.write(f"{learned_prompt}\n")
#   else:
#     for i in range(6, len(images)+6, 6):
#       for j in range(3):
#         # optimize prompt
#         learned_prompt = optimize_prompt(model, preprocess, args, device, target_images=images[i-6:i])
#         print(learned_prompt)
#         with open(adv_prompt_file, "a") as f:
#           f.write(f"{learned_prompt}\n")
# except:
for i in range(3, len(image_paths)+3, 3):
  # for j in range(3):
  # optimize prompt
  images = [Image.open(image_path) for image_path in image_paths[i-3:i]]
  learned_prompt = optimize_prompt(model, preprocess, args, device, target_images=images)
  print(learned_prompt)
  with open(adv_prompt_file, "a") as f:
    f.write(f"{learned_prompt}\n")
# finally:
#   for i in range(6, len(images)+6, 6):
#     for j in range(3):
#       # optimize prompt
#       learned_prompt = optimize_prompt(model, preprocess, args, device, target_images=images[i-6:i])
#       print(learned_prompt)
#       with open(adv_prompt_file, "a") as f:
#         f.write(f"{learned_prompt}\n")
