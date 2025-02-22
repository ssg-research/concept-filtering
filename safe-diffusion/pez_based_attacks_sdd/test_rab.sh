#!/bin/bash

# attacks/Ring-A-Bell/data/InvPrompt/Nudity_3_length_16.csv

python generate.py --model_name='/u1/test/safe-diffusion/saved/nudity' \
    --prompt_path  attacks/Ring-A-Bell/data/InvPrompt_preserve/Nudity_i2p_42_3_length_16_False.csv\
    --save_path 'outputs_rab/stanford_sdd_nudity' \
    --num_samples 10 \
    --pipeline_type sdd --use_fp16 






# python generate.py --model_name='/u1/test/safe-diffusion/saved/marvel' \
#     --prompt_path attacks/Ring-A-Bell/data/InvPrompt/marvel_42_3_length_16_False.csv\
#     --save_path 'outputs_rab/stanford_sdd_marvel' \
#     --num_samples 10 \
#     --pipeline_type sdd --use_fp16 


    


# python generate.py --model_name='/u1/test/safe-diffusion/saved/nemo' \
#     --prompt_path 'attacks/Ring-A-Bell/data/InvPrompt/nemo_42_3_length_16_False.csv' \
#     --save_path 'outputs_rab/stanford_sdd_nemo' \
#     --num_samples 10 \
#     --pipeline_type sdd --use_fp16 





# python generate.py --model_name='/u1/test/safe-diffusion/saved/grumpy' \
#     --prompt_path 'attacks/Ring-A-Bell/data/InvPrompt/grumpy_44_3_length_16_False.csv' \
#     --save_path 'outputs_rab/stanford_sdd_grumpy_cat' \
#     --num_samples 10 \
#     --pipeline_type sdd --use_fp16 





# python generate.py --model_name='/u1/test/safe-diffusion/saved/r2d2' \
#     --prompt_path 'attacks/Ring-A-Bell/data/InvPrompt/r2d2_42_3_length_16_False.csv' \
#     --save_path 'outputs_rab/stanford_sdd_r2d2' \
#     --num_samples 10 \
#     --pipeline_type sdd --use_fp16 




# python generate.py --model_name='/u1/test/safe-diffusion/saved/snoopy' \
#     --prompt_path 'attacks/Ring-A-Bell/data/InvPrompt/snoopy_42_3_length_16_False.csv' \
#     --save_path 'outputs_rab/stanford_sdd_snoopy' \
#     --num_samples 10 \
#     --pipeline_type sdd --use_fp16 





# python generate.py --model_name='/u1/test/safe-diffusion/saved/jolie' \
#     --prompt_path 'attacks/Ring-A-Bell/data/InvPrompt/jolie_42_3_length_16_False.csv' \
#     --save_path 'outputs_rab/stanford_sdd_jolie' \
#     --num_samples 10 \
#     --pipeline_type sdd --use_fp16 





# python generate.py --model_name='/u1/test/safe-diffusion/saved/swift' \
#     --prompt_path 'attacks/Ring-A-Bell/data/InvPrompt/swift_42_3_length_16_False.csv' \
#     --save_path 'outputs_rab/stanford_sdd_swift' \
#     --num_samples 10 \
#     --pipeline_type sdd --use_fp16 





# python generate.py --model_name='/u1/test/safe-diffusion/saved/musk' \
#     --prompt_path 'attacks/Ring-A-Bell/data/InvPrompt/musk_42_3_length_16_False.csv' \
#     --save_path 'outputs_rab/stanford_sdd_musk' \
#     --num_samples 10 \
#     --pipeline_type sdd --use_fp16 





# python generate.py --model_name='/u1/test/safe-diffusion/saved/pitt' \
#     --prompt_path 'attacks/Ring-A-Bell/data/InvPrompt/pitt_42_3_length_16_False.csv' \
#     --save_path 'outputs_rab/stanford_sdd_pitt' \
#     --num_samples 10 \
#     --pipeline_type sdd --use_fp16 




# attacks/Ring-A-Bell/data/InvPrompt/Violence_3_length_16.csv
python generate.py --model_name='/u1/test/safe-diffusion/saved/violence' \
    --prompt_path 'attacks/Ring-A-Bell/data/InvPrompt_preserve/Violence_i2p_42_3_length_16_False.csv' \
    --save_path 'outputs_rab/stanford_sdd_violence' \
    --num_samples 10 \
    --pipeline_type sdd --use_fp16 


