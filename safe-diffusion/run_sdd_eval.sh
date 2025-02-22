#!/bin/bash

# python generate.py \
#     --pipeline_type sdd  \
#     --prompt_path "/u1/test/generative-content-removal/stable-diffusion/assets/eval_prompts/nemo_eval.txt" --num_samples 20 \
#     --experiment nemo \
#     --safety_concept "an image of nemo" \
#     --use_fp16 --device "cuda:0"

# python generate.py \
#     --pipeline_type sdd  \
#     --prompt_path "/u1/test/generative-content-removal/stable-diffusion/assets/eval_prompts/nemo_eval.txt" --num_samples 20 \
#     --experiment nemo \
#     --safety_concept "an image of nemo" \
#     --use_fp16  --device "cuda:0"

# python generate.py \
#     --pipeline_type sdd  \
#     --prompt_path "/u1/test/generative-content-removal/stable-diffusion/assets/eval_prompts/nemo_eval.txt" --num_samples 20 \
#     --experiment nemo \
#     --safety_concept "an image of nemo" \
#     --use_fp16  --device "cuda:0"



# python generate.py \
#     --pipeline_type sdd  \
#     --prompt_path "/u1/test/generative-content-removal/stable-diffusion/assets/eval_prompts/marvel_eval.txt" --num_samples 20 \
#     --experiment marvel \
#     --safety_concept "an image of captain marvel" \
#     --use_fp16 --device "cuda:0"




# python generate.py \
#     --pipeline_type sdd  \
#     --prompt_path "/u1/test/generative-content-removal/stable-diffusion/assets/eval_prompts/r2d2_eval.txt" --num_samples 20 \
#     --experiment r2d2 \
#     --safety_concept "an image of r2d2" \
#     --use_fp16 --device "cuda:0"



# python generate.py \
#     --pipeline_type sdd  \
#     --prompt_path "/u1/test/generative-content-removal/stable-diffusion/assets/eval_prompts/snoopy_eval.txt" --num_samples 20 \
#     --experiment snoopy \
#     --safety_concept "an image of snoopy" \
#     --use_fp16 --device "cuda:0"




# python generate.py \
#     --pipeline_type sdd  \
#     --prompt_path "/u1/test/generative-content-removal/stable-diffusion/assets/eval_prompts/grumpy_cat_eval.txt" --num_samples 20 \
#     --experiment grumpy \
#     --safety_concept "an image of grumpy cat" \
#     --use_fp16 --device "cuda:0"




# python generate.py \
#     --pipeline_type sdd  \
#     --prompt_path "/u1/test/generative-content-removal/stable-diffusion/assets/eval_prompts/nudity_eval.txt" --num_samples 20 \
#     --experiment nudity \
#     --use_fp16 --device "cuda:0"



# python generate.py \
#     --pipeline_type sdd  \
#     --prompt_path "/u1/test/generative-content-removal/stable-diffusion/assets/finetune_prompts/violent.txt" --num_samples 20 \
#     --experiment violence \
#     --use_fp16 --device "cuda:0"



# python generate.py \
#     --pipeline_type sdd  \
#     --prompt_path "/u1/test/generative-content-removal/stable-diffusion/assets/eval_prompts/swift_eval.txt" --num_samples 20 \
#     --experiment swift \
#     --safety_concept "an image of Taylor Swift" \
#     --use_fp16 --device "cuda:0"




# python generate.py \
#     --pipeline_type sdd  \
#     --prompt_path "/u1/test/generative-content-removal/stable-diffusion/assets/eval_prompts/jolie_eval.txt" --num_samples 20 \
#     --experiment jolie \
#     --safety_concept "an image of Angelina Jolie" \
#     --use_fp16 --device "cuda:0"




# python generate.py \
#     --pipeline_type sdd  \
#     --prompt_path "/u1/test/generative-content-removal/stable-diffusion/assets/eval_prompts/pitt_eval.txt" --num_samples 20 \
#     --experiment pitt \
#     --safety_concept "an image of Brad Pitt" \
#     --use_fp16 --device "cuda:0"




# python generate.py \
#     --pipeline_type sdd  \
#     --prompt_path "/u1/test/generative-content-removal/stable-diffusion/assets/eval_prompts/musk_eval.txt" --num_samples 20 \
#     --experiment musk \
#     --safety_concept "an image of Elon Musk" \
#     --use_fp16 --device "cuda:0"







python generate.py \
    --pipeline_type sdd  \
    --prompt_path "/u1/test/generative-content-removal/stable-diffusion/assets/eval-prompts/i2p_nudity_prompts.txt" --num_samples 20 \
    --experiment nudity_i2p \
    --use_fp16 --device "cuda:0"




python generate.py \
    --pipeline_type sdd  \
    --prompt_path "/u1/test/generative-content-removal/stable-diffusion/assets/eval-prompts/i2p_violence_prompts.txt" --num_samples 20 \
    --experiment violence_i2p \
    --use_fp16 --device "cuda:0"

